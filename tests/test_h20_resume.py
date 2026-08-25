import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from workflow import prepare_assets, stage_status


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

    def test_stale_complete_summary_does_not_hide_missing_raw_shards(self):
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
            result_root = output / "training_results"
            for filename in stage_status.FINAL_ARTIFACTS:
                path = result_root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"placeholder")
            (result_root / "training_results.json").write_text(
                json.dumps({"status": "COMPLETE"})
            )

            issues = stage_status.completion_issues(
                "rotation-head", 4_003_200, output, asset
            )
            self.assertTrue(any("missing evaluation shard" in issue for issue in issues))


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
            self.assertTrue(all("PREPARE_ASSETS=0" in line for line in submitted[1:]))
            self.assertIn("afterok:1001", submitted[1])
            self.assertNotIn("test-cluster", "\n".join(submitted))
            self.assertIn("FINAL_JOB_ID=1026", completed.stdout)


if __name__ == "__main__":
    unittest.main()
