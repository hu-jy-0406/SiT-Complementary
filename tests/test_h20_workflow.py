import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from workflow import build_results, evaluate_checkpoint, preflight


class PreflightGpuProfileTest(unittest.TestCase):
    def test_accepts_supported_eight_gpu_profiles(self):
        preflight.validate_gpu_profile(
            ["NVIDIA A100-SXM4-40GB"] * 8,
            [39.5] * 8,
            "a100-40gb",
        )
        preflight.validate_gpu_profile(
            ["NVIDIA H20"] * 8,
            [95.0] * 8,
            "h20",
        )

    def test_rejects_wrong_model_or_memory_tier(self):
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            preflight.validate_gpu_profile(
                ["NVIDIA H20"] * 8,
                [95.0] * 8,
                "a100-40gb",
            )
        with self.assertRaisesRegex(RuntimeError, "visible memory"):
            preflight.validate_gpu_profile(
                ["NVIDIA A100-SXM4-80GB"] * 8,
                [79.2] * 8,
                "a100-40gb",
            )


class EvaluateCheckpointTest(unittest.TestCase):
    def test_reference_is_hashed_and_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "reference.npz"
            reference.write_bytes(b"test-reference")
            actual = hashlib.sha256(b"test-reference").hexdigest()
            with mock.patch.object(evaluate_checkpoint, "REFERENCE_SHA256", actual):
                self.assertEqual(
                    evaluate_checkpoint.verify_reference(reference), actual
                )
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                evaluate_checkpoint.verify_reference(reference)

    def test_cache_reuse_compares_every_identity_field(self):
        identity = {
            field: f"value-{field}"
            for field in evaluate_checkpoint.EVALUATION_IDENTITY_FIELDS
        }
        existing = {**identity, "status": "ok", "fid": 12.5}
        self.assertTrue(
            evaluate_checkpoint.evaluation_already_complete(existing, identity)
        )
        for field in evaluate_checkpoint.EVALUATION_IDENTITY_FIELDS:
            changed = dict(existing)
            changed[field] = f"different-{field}"
            self.assertFalse(
                evaluate_checkpoint.evaluation_already_complete(changed, identity),
                field,
            )

    def test_protocol_requires_exact_topology_and_fid_version(self):
        args = SimpleNamespace(
            nproc=8,
            per_proc_batch_size=64,
            fid_batch_size=128,
            num_workers=8,
        )
        evaluate_checkpoint.require_exact_protocol(args, "0.3.0")
        args.nproc = 4
        with self.assertRaisesRegex(RuntimeError, "--nproc must be 8"):
            evaluate_checkpoint.require_exact_protocol(args, "0.3.0")
        args.nproc = 8
        with self.assertRaisesRegex(RuntimeError, "exactly 0.3.0"):
            evaluate_checkpoint.require_exact_protocol(args, "0.3.1")


class BuildResultsTest(unittest.TestCase):
    def make_checkpoint(
        self,
        root: Path,
        variant: str,
        *,
        state_step: int = build_results.FINAL_STEP,
        suffix: str = "",
    ) -> tuple[Path, str]:
        checkpoint_dir = root / "checkpoints" / variant
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"final{suffix}.pt"
        args = SimpleNamespace(
            model="SiT-S/2",
            epochs=800,
            max_train_steps=build_results.FINAL_STEP,
            global_batch_size=256,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            global_seed=0,
            vae="ema",
            path_type="Linear",
            prediction="velocity",
            image_size=256,
            num_classes=1000,
            restart_deterministic_data=True,
        )
        torch.save(
            {
                "model": {"marker": f"{variant}{suffix}"},
                "ema": {},
                "opt": {
                    "state": {},
                    "param_groups": [
                        {
                            "lr": 1e-4,
                            "weight_decay": 0,
                            "betas": (0.9, 0.999),
                        }
                    ],
                },
                "args": args,
                "train_steps": state_step,
                "resume_state_version": 1,
                "rng_states": [
                    {
                        "python": (3, (), None),
                        "numpy": ("MT19937", (), 0, 0, 0.0),
                        "torch_cpu": torch.tensor([], dtype=torch.uint8),
                        "torch_cuda": torch.tensor([], dtype=torch.uint8),
                    }
                    for _ in range(8)
                ],
                "training_lineage": (
                    {
                        "schema_version": 1,
                        "mode": "resume",
                        "initial_step": build_results.CONV_RESUME_STEP,
                        "initial_checkpoint": "/assets/conv/1950000.pt",
                        "initial_checkpoint_sha256": build_results.CONV_RESUME_SHA256,
                    }
                    if variant == "conv"
                    else {
                        "schema_version": 1,
                        "mode": "scratch",
                        "initial_step": 0,
                        "initial_checkpoint": None,
                        "initial_checkpoint_sha256": None,
                    }
                ),
                "data_state": {
                    "dataset_size": 1_281_167,
                    "steps_per_epoch": build_results.STEPS_PER_EPOCH,
                    "world_size": 8,
                    "global_batch_size": 256,
                    "gradient_accumulation_steps": 1,
                    "global_seed": 0,
                    "image_size": 256,
                    "num_classes": 1000,
                    "restart_deterministic_data": True,
                },
            },
            checkpoint,
        )
        return checkpoint, build_results._sha256_file(checkpoint)

    def make_record(
        self,
        variant: str,
        step: int,
        cfg: float,
        fid: float,
        *,
        checkpoint: Path | None = None,
        checkpoint_sha256: str | None = None,
    ) -> dict:
        if checkpoint is None:
            checkpoint = Path(f"/checkpoints/{variant}/{step:07d}.pt")
        if checkpoint_sha256 is None:
            checkpoint_sha256 = hashlib.sha256(
                f"{variant}-{step}".encode()
            ).hexdigest()
        if step == build_results.FINAL_STEP:
            phase = "final"
        elif variant == "conv" and step == build_results.CONV_RESUME_STEP:
            phase = "resume-baseline"
        else:
            phase = "periodic"
        return {
            "record_schema": build_results.RECORD_SCHEMA,
            "variant": variant,
            "step": step,
            "epoch": step / build_results.STEPS_PER_EPOCH,
            "phase": phase,
            "cfg": cfg,
            "fid": fid,
            "status": "ok",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": step,
            "num_requested": 50000,
            "num_png": 50176,
            "seed": 0,
            "sampler": "euler",
            "sampling_method": "euler",
            "sampling_steps": 250,
            "world_size": 8,
            "per_proc_batch_size": 64,
            "fid_impl": build_results.PYTORCH_FID_IMPL,
            "fid_batch_size": 128,
            "fid_num_workers": 8,
            "reference": "/assets/fid/VIRTUAL_imagenet256_labeled.npz",
            "reference_sha256": build_results.REFERENCE_SHA256,
            "protocol_id": build_results.PROTOCOL_ID,
        }

    def write_record(self, root: Path, record: dict) -> Path:
        variant_dir = root / "raw" / record["variant"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        label = f"{record['cfg']:g}"
        path = variant_dir / f"step-{record['step']:07d}-cfg-{label}.json"
        path.write_text(json.dumps(record))
        return path

    def populate_complete(
        self,
        output: Path,
        storage: Path,
        *,
        final_state_step: int = build_results.FINAL_STEP,
        omitted: tuple[str, int, float] | None = None,
    ) -> dict[str, tuple[Path, str]]:
        final_checkpoints = {
            variant: self.make_checkpoint(
                storage, variant, state_step=final_state_step
            )
            for variant in build_results.VARIANTS
        }

        def maybe_write(record: dict) -> None:
            key = (record["variant"], record["step"], float(record["cfg"]))
            if key != omitted:
                self.write_record(output, record)

        maybe_write(
            self.make_record("conv", build_results.CONV_RESUME_STEP, 1.0, 45.0)
        )
        for step in range(
            build_results.CONV_PERIODIC_START,
            build_results.PERIODIC_END + 1,
            build_results.PERIODIC_INTERVAL,
        ):
            maybe_write(
                self.make_record("conv", step, 1.0, 50.0 - step / 1e6)
            )
        for step in range(
            build_results.ROTATION_HEAD_PERIODIC_START,
            build_results.PERIODIC_END + 1,
            build_results.PERIODIC_INTERVAL,
        ):
            maybe_write(
                self.make_record(
                    "rotation-head", step, 1.0, 60.0 - step / 1e6
                )
            )
        for variant in build_results.VARIANTS:
            checkpoint, digest = final_checkpoints[variant]
            for cfg in (1.0, 4.0):
                maybe_write(
                    self.make_record(
                        variant,
                        build_results.FINAL_STEP,
                        cfg,
                        20.0 + cfg,
                        checkpoint=checkpoint,
                        checkpoint_sha256=digest,
                    )
                )
        return final_checkpoints

    def test_strict_complete_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            output = storage / "training_results"
            self.populate_complete(output, storage)

            return_code = build_results.main(
                [
                    "--output-dir",
                    str(output),
                    "--gpu-profile",
                    "a100-40gb",
                    "--strict",
                ]
            )
            self.assertEqual(return_code, 0)
            summary = json.loads((output / "training_results.json").read_text())
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertTrue(summary["models"]["conv"]["resume_baseline_complete"])
            self.assertEqual(
                summary["training_protocol"]["gpu_topology"],
                "8x NVIDIA A100 40GB",
            )
            report = (output / "TRAINING_RESULTS.md").read_text()
            self.assertIn("SHA-256", report)
            self.assertIn("8×NVIDIA A100 40GB", report)
            for filename in build_results.OUTPUT_FILES:
                self.assertTrue((output / filename).is_file(), filename)

    def test_strict_incomplete_returns_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "training_results"
            return_code = build_results.main(
                ["--output-dir", str(output), "--strict"]
            )
            self.assertEqual(return_code, 2)
            summary = json.loads((output / "training_results.json").read_text())
            self.assertEqual(summary["status"], "INCOMPLETE")

    def test_legacy_conv_history_cannot_fill_strict_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            output = storage / "training_results"
            missing = ("conv", build_results.CONV_PERIODIC_START, 1.0)
            self.populate_complete(output, storage, omitted=missing)
            history = storage / "conv-history.tsv"
            with history.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    delimiter="\t",
                    fieldnames=["variant", "step", "cfg", "fid", "status"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "variant": "conv",
                        "step": build_results.CONV_PERIODIC_START,
                        "cfg": 1,
                        "fid": 42,
                        "status": "ok",
                    }
                )
            return_code = build_results.main(
                [
                    "--output-dir",
                    str(output),
                    "--conv-history",
                    str(history),
                    "--strict",
                ]
            )
            self.assertEqual(return_code, 2)
            summary = json.loads((output / "training_results.json").read_text())
            self.assertIn(
                build_results.CONV_PERIODIC_START,
                summary["models"]["conv"]["missing_periodic_steps"],
            )

    def test_strict_rejects_protocol_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            output = storage / "training_results"
            self.populate_complete(output, storage)
            shard = (
                output
                / "raw"
                / "conv"
                / f"step-{build_results.CONV_PERIODIC_START:07d}-cfg-1.json"
            )
            record = json.loads(shard.read_text())
            record["fid_impl"] = "pytorch-fid 0.3.1"
            shard.write_text(json.dumps(record))
            self.assertEqual(
                build_results.main(["--output-dir", str(output), "--strict"]),
                2,
            )

    def test_strict_rejects_final_checkpoint_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            output = storage / "training_results"
            self.populate_complete(output, storage)
            for cfg in (1, 4):
                shard = (
                    output
                    / "raw"
                    / "conv"
                    / f"step-{build_results.FINAL_STEP:07d}-cfg-{cfg}.json"
                )
                record = json.loads(shard.read_text())
                record["checkpoint_sha256"] = "0" * 64
                shard.write_text(json.dumps(record))
            self.assertEqual(
                build_results.main(["--output-dir", str(output), "--strict"]),
                2,
            )

    def test_strict_rejects_checkpoint_with_wrong_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            output = storage / "training_results"
            self.populate_complete(
                output,
                storage,
                final_state_step=build_results.PERIODIC_END,
            )
            self.assertEqual(
                build_results.main(["--output-dir", str(output), "--strict"]),
                2,
            )

    def test_strict_rejects_final_checkpoint_training_protocol_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            output = storage / "training_results"
            checkpoints = self.populate_complete(output, storage)
            checkpoint, _ = checkpoints["rotation-head"]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["args"].model = "SiT-XL/2"
            payload["data_state"]["world_size"] = 4
            payload["training_lineage"]["mode"] = "resume"
            torch.save(payload, checkpoint)
            new_digest = build_results._sha256_file(checkpoint)
            for cfg in (1, 4):
                shard = (
                    output
                    / "raw"
                    / "rotation-head"
                    / f"step-{build_results.FINAL_STEP:07d}-cfg-{cfg}.json"
                )
                record = json.loads(shard.read_text())
                record["checkpoint_sha256"] = new_digest
                shard.write_text(json.dumps(record))
            return_code = build_results.main(
                ["--output-dir", str(output), "--strict"]
            )
            self.assertEqual(return_code, 2)
            summary = json.loads((output / "training_results.json").read_text())
            issues = "\n".join(summary["issues"])
            self.assertIn("args.model", issues)
            self.assertIn("data_state.world_size", issues)

    def test_strict_rejects_different_cfg_final_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            output = storage / "training_results"
            self.populate_complete(output, storage)
            alternate, alternate_sha = self.make_checkpoint(
                storage, "conv", suffix="-alternate"
            )
            shard = (
                output
                / "raw"
                / "conv"
                / f"step-{build_results.FINAL_STEP:07d}-cfg-4.json"
            )
            record = json.loads(shard.read_text())
            record["checkpoint"] = str(alternate)
            record["checkpoint_sha256"] = alternate_sha
            shard.write_text(json.dumps(record))
            self.assertEqual(
                build_results.main(["--output-dir", str(output), "--strict"]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
