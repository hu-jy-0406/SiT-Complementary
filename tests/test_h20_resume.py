import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from workflow import (
    experiment_status,
    finalize_handoff,
    portable_results,
    prepare_assets,
    stage_status,
)


class GpuProfileLockTest(unittest.TestCase):
    def test_output_root_cannot_mix_gpu_profiles(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            command = (
                'source workflow/h20_common.sh; '
                'handoff_lock_gpu_profile "$1" "$2"'
            )
            subprocess.run(
                ["bash", "-c", command, "bash", str(output), "a100-40gb"],
                cwd=repo_root,
                check=True,
            )
            subprocess.run(
                ["bash", "-c", command, "bash", str(output), "a100-40gb"],
                cwd=repo_root,
                check=True,
            )
            rejected = subprocess.run(
                ["bash", "-c", command, "bash", str(output), "h20"],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("locked to GPU_PROFILE=a100-40gb", rejected.stderr)


class LocalAssetReuseTest(unittest.TestCase):
    def test_verified_local_asset_never_calls_downloader(self):
        with tempfile.TemporaryDirectory() as temporary:
            local_dir = Path(temporary)
            payload = b"already downloaded and pinned"
            candidate = local_dir / "nested" / "asset.bin"
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(payload)

            with mock.patch.object(
                prepare_assets,
                "download",
                side_effect=AssertionError("network-capable downloader was called"),
            ):
                result = prepare_assets.ensure_asset(
                    "owner/repo",
                    "nested/asset.bin",
                    "revision",
                    local_dir,
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                )
            self.assertEqual(result, candidate)

    def test_local_only_rejects_corruption_without_downloader(self):
        with tempfile.TemporaryDirectory() as temporary:
            local_dir = Path(temporary)
            candidate = local_dir / "asset.bin"
            candidate.write_bytes(b"corrupt")
            with mock.patch.object(
                prepare_assets,
                "download",
                side_effect=AssertionError("network-capable downloader was called"),
            ):
                with self.assertRaises(RuntimeError):
                    prepare_assets.ensure_asset(
                        "owner/repo",
                        "asset.bin",
                        "revision",
                        local_dir,
                        len(b"expected"),
                        hashlib.sha256(b"expected").hexdigest(),
                        local_only=True,
                    )


class StageStatusTest(unittest.TestCase):
    def _write_eval(
        self,
        output: Path,
        asset: Path,
        variant: str,
        step: int,
        cfg: float,
        checkpoint: Path,
    ) -> None:
        raw = output / "training_results" / "raw" / variant
        raw.mkdir(parents=True, exist_ok=True)
        phase = (
            "final"
            if step == stage_status.FINAL_STEP
            else "resume-baseline"
            if variant == "conv" and step == 1_950_000
            else "periodic"
        )
        (raw / f"step-{step:07d}-cfg-{cfg:g}.json").write_text(
            json.dumps(
                {
                    **stage_status.EXACT_PROTOCOL,
                    "status": "ok",
                    "variant": variant,
                    "step": step,
                    "cfg": cfg,
                    "fid": 12.34,
                    "phase": phase,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_step": step,
                    "checkpoint_sha256": "a" * 64,
                    "reference": str(
                        (asset / "fid" / "VIRTUAL_imagenet256_labeled.npz").resolve()
                    ),
                }
            )
        )

    def test_periodic_stage_requires_checkpoint_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            asset = root / "assets"
            checkpoint = (
                output
                / "training"
                / stage_status.RUN_NAMES["rotation-head"]
                / "checkpoints"
                / "0250000.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"atomic checkpoint placeholder")
            self._write_eval(
                output,
                asset,
                "rotation-head",
                250_000,
                1.0,
                checkpoint,
            )

            self.assertEqual(
                stage_status.completion_issues(
                    "rotation-head", 250_000, output, asset
                ),
                [],
            )
            shard = (
                output
                / "training_results"
                / "raw"
                / "rotation-head"
                / "step-0250000-cfg-1.json"
            )
            shard.unlink()
            self.assertTrue(
                stage_status.completion_issues(
                    "rotation-head", 250_000, output, asset
                )
            )

    def test_protocol_drift_is_not_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            asset = root / "assets"
            checkpoint = (
                output
                / "training"
                / stage_status.RUN_NAMES["rotation-head"]
                / "checkpoints"
                / "0250000.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"atomic checkpoint placeholder")
            self._write_eval(
                output,
                asset,
                "rotation-head",
                250_000,
                1.0,
                checkpoint,
            )
            shard = (
                output
                / "training_results"
                / "raw"
                / "rotation-head"
                / "step-0250000-cfg-1.json"
            )
            record = json.loads(shard.read_text())
            record["world_size"] = 4
            shard.write_text(json.dumps(record))
            self.assertTrue(
                stage_status.completion_issues(
                    "rotation-head", 250_000, output, asset
                )
            )

    def test_final_stage_is_local_to_its_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            asset = root / "assets"
            checkpoint = (
                output
                / "training"
                / stage_status.RUN_NAMES["rotation-head"]
                / "checkpoints"
                / "4003200.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"final checkpoint")
            self._write_eval(
                output,
                asset,
                "rotation-head",
                4_003_200,
                1.0,
                checkpoint,
            )
            self._write_eval(
                output,
                asset,
                "rotation-head",
                4_003_200,
                4.0,
                checkpoint,
            )
            issues = stage_status.completion_issues(
                "rotation-head", 4_003_200, output, asset
            )
            self.assertEqual(issues, [])
            experiment_issues = experiment_status.experiment_issues(
                "rotation-head", output, asset
            )
            self.assertTrue(
                any("target 250000" in issue for issue in experiment_issues)
            )


class FinalizeHandoffTest(unittest.TestCase):
    def _run(self, output: Path, asset: Path) -> int:
        argv = [
            "finalize_handoff.py",
            "--active-variant",
            "conv",
            "--output-root",
            str(output),
            "--asset-root",
            str(asset),
            "--gpu-profile",
            "a100-40gb",
        ]
        with mock.patch("sys.argv", argv):
            return finalize_handoff.main()

    def test_first_finisher_builds_non_strict_pending_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(
                finalize_handoff,
                "experiment_issues",
                side_effect=lambda variant, *_: [] if variant == "conv" else ["wait"],
            ), mock.patch.object(
                finalize_handoff.build_results, "main", return_value=0
            ) as build:
                self.assertEqual(self._run(root / "output", root / "assets"), 0)
            self.assertNotIn("--strict", build.call_args.args[0])
            self.assertEqual(
                (root / "output" / ".finalize_passed-conv").read_text(),
                "pending\n",
            )

    def test_second_finisher_requires_strict_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(
                finalize_handoff, "experiment_issues", return_value=[]
            ), mock.patch.object(
                finalize_handoff.build_results, "main", return_value=0
            ) as build:
                self.assertEqual(self._run(root / "output", root / "assets"), 0)
            self.assertIn("--strict", build.call_args.args[0])
            self.assertEqual(
                (root / "output" / ".finalize_passed-conv").read_text(),
                "complete\n",
            )


class PortableResultsTest(unittest.TestCase):
    def _make_bundle(
        self, root: Path, variant: str, checkpoint_payload: bytes
    ) -> Path:
        output = root / f"{variant}-output"
        asset = root / f"{variant}-assets"
        bundle = root / f"{variant}-bundle"
        (output / ".gpu_profile").parent.mkdir(parents=True)
        (output / ".gpu_profile").write_text("a100-40gb\n")
        checkpoint = (
            output
            / "training"
            / stage_status.RUN_NAMES[variant]
            / "checkpoints"
            / f"{stage_status.FINAL_STEP:07d}.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(checkpoint_payload)
        checkpoint_hash = portable_results.sha256_file(checkpoint)
        raw = output / "training_results" / "raw" / variant
        raw.mkdir(parents=True)
        for name in portable_results.required_raw_names(variant):
            stem = name.removesuffix(".json")
            step = int(stem.split("-cfg-")[0].removeprefix("step-"))
            cfg = float(stem.split("-cfg-")[1])
            (raw / name).write_text(
                json.dumps(
                    {
                        "variant": variant,
                        "step": step,
                        "cfg": cfg,
                        "checkpoint": f"/isolated/{variant}/{step}.pt",
                        "checkpoint_sha256": checkpoint_hash,
                    }
                )
            )
        if variant == "conv":
            history = (
                asset
                / "huggingface"
                / "BlueSourceJY"
                / "SiT-Complementary"
                / "experiments"
                / "bs256_lr1e-4"
                / "conv-layer"
                / "fid_cfg1_50k.tsv"
            )
            history.parent.mkdir(parents=True)
            history.write_text("step\tfid\n")
        with mock.patch.object(portable_results, "experiment_issues", return_value=[]):
            self.assertEqual(
                portable_results.export_bundle(variant, output, asset, bundle), 0
            )
        return bundle

    def test_isolated_bundles_merge_and_relocate_final_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conv = self._make_bundle(root, "conv", b"conv-final")
            rotation = self._make_bundle(root, "rotation-head", b"rotation-final")
            merged = root / "merged"
            with mock.patch.object(
                portable_results.build_results, "main", return_value=0
            ) as build:
                self.assertEqual(
                    portable_results.merge_bundles(conv, rotation, merged), 0
                )
            self.assertIn("--strict", build.call_args.args[0])
            for variant in stage_status.RUN_NAMES:
                record = json.loads(
                    (
                        merged
                        / "training_results"
                        / "raw"
                        / variant
                        / f"step-{stage_status.FINAL_STEP:07d}-cfg-1.json"
                    ).read_text()
                )
                self.assertIn("portable_original_checkpoint", record)
                self.assertEqual(
                    Path(record["checkpoint"]),
                    merged
                    / "portable_import"
                    / variant
                    / f"{stage_status.FINAL_STEP:07d}.pt",
                )
            self.assertTrue((merged / ".portable_results_complete").is_file())

    def test_bundle_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._make_bundle(root, "rotation-head", b"rotation-final")
            raw = bundle / "raw" / portable_results.required_raw_names(
                "rotation-head"
            )[0]
            raw.write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                portable_results.validate_bundle(bundle, "rotation-head")


class SlurmSubmitterTest(unittest.TestCase):
    def test_skips_complete_stage_and_normalizes_parsable_job_id(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            asset = root / "assets"

            checkpoint = (
                output
                / "training"
                / stage_status.RUN_NAMES["conv"]
                / "checkpoints"
                / "2000000.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"completed target")
            conv_source = (
                asset
                / "huggingface"
                / "BlueSourceJY"
                / "SiT-Complementary"
                / "checkpoints"
                / "bs256_lr1e-4"
                / "conv-layer"
                / "1950000.pt"
            )
            conv_source.parent.mkdir(parents=True)
            conv_source.write_bytes(b"resume source")

            helper = StageStatusTest()
            helper._write_eval(
                output, asset, "conv", 1_950_000, 1.0, conv_source
            )
            helper._write_eval(
                output, asset, "conv", 2_000_000, 1.0, checkpoint
            )

            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "count=0\n"
                "[[ -f \"$FAKE_SBATCH_COUNTER\" ]] && "
                "count=\"$(<\"$FAKE_SBATCH_COUNTER\")\"\n"
                "count=$((count + 1))\n"
                "printf '%s\\n' \"$count\" > \"$FAKE_SBATCH_COUNTER\"\n"
                "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_CALLS\"\n"
                "printf '%s;test-cluster\\n' \"$((1000 + count))\"\n"
            )
            fake_sbatch.chmod(fake_sbatch.stat().st_mode | stat.S_IXUSR)

            calls = root / "calls.txt"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "IMAGENET_TRAIN": str(root / "imagenet"),
                    "OUTPUT_ROOT": str(output),
                    "ASSET_ROOT": str(asset),
                    "EXPERIMENT": "all",
                    "FAKE_SBATCH_COUNTER": str(root / "counter.txt"),
                    "FAKE_SBATCH_CALLS": str(calls),
                }
            )
            completed = subprocess.run(
                ["bash", "slurm/submit_h20_pipeline.sh"],
                cwd=repo_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            submitted = calls.read_text().splitlines()
            self.assertEqual(len(submitted), 26)
            self.assertIn("SKIPPED_COMPLETE variant=conv target=2000000", completed.stdout)
            self.assertIn("PREPARE_ASSETS=1", submitted[0])
            self.assertIn("--gres gpu:a100:8", submitted[0])
            self.assertIn("GPU_PROFILE=a100-40gb", submitted[0])
            self.assertTrue(all("PREPARE_ASSETS=0" in line for line in submitted[1:]))
            self.assertIn("afterok:1001", submitted[1])
            self.assertNotIn("test-cluster", "\n".join(submitted))
            self.assertIn("FINAL_JOB_ID=1026", completed.stdout)

    def test_selects_one_experiment_and_optional_node(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_CALLS\"\n"
                "count=$(wc -l < \"$FAKE_SBATCH_CALLS\")\n"
                "printf '%s\\n' \"$((2000 + count))\"\n"
            )
            fake_sbatch.chmod(fake_sbatch.stat().st_mode | stat.S_IXUSR)
            calls = root / "calls.txt"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "IMAGENET_TRAIN": str(root / "imagenet"),
                    "OUTPUT_ROOT": str(root / "output"),
                    "ASSET_ROOT": str(root / "assets"),
                    "EXPERIMENT": "conv",
                    "SLURM_NODELIST": "a100-node-01",
                    "FAKE_SBATCH_CALLS": str(calls),
                }
            )
            subprocess.run(
                ["bash", "slurm/submit_h20_pipeline.sh"],
                cwd=repo_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            submitted = calls.read_text().splitlines()
            self.assertEqual(len(submitted), 10)
            self.assertTrue(all("PIPELINE_VARIANT=conv" in line for line in submitted))
            self.assertTrue(all("--nodelist a100-node-01" in line for line in submitted))


if __name__ == "__main__":
    unittest.main()
