#!/usr/bin/env python3
"""Build deterministic training-result artifacts from per-evaluation JSON records.

The evaluator is expected to atomically place JSON records below
``training_results/raw/{conv,rotation-head}``.  This script treats those JSON
files as the source of truth, optionally merges a legacy Conv TSV, validates
the expected evaluation schedule, and atomically rebuilds the human- and
machine-readable result bundle.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable

try:
    from .checkpoint_tool import inspect as inspect_checkpoint
except ImportError:  # Support ``python workflow/build_results.py``.
    from checkpoint_tool import inspect as inspect_checkpoint


VARIANTS = ("conv", "rotation-head")
VARIANT_LABELS = {
    "conv": "Conv",
    "rotation-head": "Rotation-head",
}
VARIANT_COLORS = {
    "conv": "#1f77b4",
    "rotation-head": "#d62728",
}
GPU_PROFILE_METADATA = {
    "a100-40gb": {
        "json_topology": "8x NVIDIA A100 40GB",
        "markdown_topology": "8×NVIDIA A100 40GB",
    },
    "h20": {
        "json_topology": "8x NVIDIA H20",
        "markdown_topology": "8×NVIDIA H20",
    },
}

RECORD_SCHEMA = "sit-fid-evaluation-v1"
PROTOCOL_ID = "sit-imagenet256-pytorch-fid-euler250-seed0-padded50176-v1"
REFERENCE_SHA256 = "b32732719497e42660a9affb4a966068cba0855ac449b82015e34ec376d20758"
CONV_RESUME_SHA256 = "f3724fa2651c6fbb3a624664f057c1dd56c658d68fb356c52cf51a7684ae7548"
PYTORCH_FID_IMPL = "pytorch-fid 0.3.0"
FINAL_STEP = 4_003_200
STEPS_PER_EPOCH = 5_004
CONV_RESUME_STEP = 1_950_000
CONV_PERIODIC_START = 2_000_000
ROTATION_HEAD_PERIODIC_START = 250_000
PERIODIC_END = 4_000_000
PERIODIC_INTERVAL = 250_000

EXACT_EVALUATION_PROTOCOL: dict[str, Any] = {
    "record_schema": RECORD_SCHEMA,
    "protocol_id": PROTOCOL_ID,
    "num_requested": 50_000,
    "num_png": 50_176,
    "seed": 0,
    "sampler": "euler",
    "sampling_method": "euler",
    "sampling_steps": 250,
    "world_size": 8,
    "per_proc_batch_size": 64,
    "fid_impl": PYTORCH_FID_IMPL,
    "fid_batch_size": 128,
    "fid_num_workers": 8,
    "reference_sha256": REFERENCE_SHA256,
}

TSV_FIELDS = (
    "record_schema",
    "variant",
    "step",
    "epoch",
    "phase",
    "cfg",
    "fid",
    "status",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_step",
    "num_requested",
    "num_png",
    "seed",
    "sampler",
    "sampling_method",
    "sampling_steps",
    "world_size",
    "per_proc_batch_size",
    "fid_impl",
    "fid_batch_size",
    "fid_num_workers",
    "reference",
    "reference_sha256",
    "protocol_id",
    "timestamp_utc",
    "git_commit",
    "wandb_run_name",
    "origin",
    "source",
)

OUTPUT_FILES = (
    "fid_results.tsv",
    "training_results.json",
    "TRAINING_RESULTS.md",
    "conv_fid_cfg1_curve.png",
    "rotation_head_fid_cfg1_curve.png",
    "fid_cfg1_training_curves.png",
)


class RecordError(ValueError):
    """Raised when an input record cannot be normalized safely."""


def _nested_get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _first(record: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        value = _nested_get(record, path)
        if value is not None and value != "":
            return value
    return None


def _as_int(value: Any, name: str, *, required: bool = False) -> int | None:
    if value is None or value == "":
        if required:
            raise RecordError(f"missing required field {name}")
        return None
    if isinstance(value, bool):
        raise RecordError(f"{name} must be an integer, not a boolean")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise RecordError(f"invalid integer for {name}: {value!r}") from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(converted)
    if not math.isfinite(numeric) or numeric != converted:
        raise RecordError(f"non-integral value for {name}: {value!r}")
    return converted


def _as_float(value: Any, name: str, *, required: bool = False) -> float | None:
    if value is None or value == "":
        if required:
            raise RecordError(f"missing required field {name}")
        return None
    if isinstance(value, bool):
        raise RecordError(f"{name} must be numeric, not a boolean")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise RecordError(f"invalid number for {name}: {value!r}") from exc
    if not math.isfinite(converted):
        raise RecordError(f"non-finite value for {name}: {value!r}")
    return converted


def _normalize_variant(value: Any) -> str:
    if value is None:
        raise RecordError("missing model variant")
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "conv": "conv",
        "conv-layer": "conv",
        "convolution": "conv",
        "rot-head": "rotation-head",
        "rotation-head": "rotation-head",
        "rotationhead": "rotation-head",
    }
    if normalized not in aliases:
        raise RecordError(f"unsupported model variant: {value!r}")
    return aliases[normalized]


def _normalize_status(value: Any, fid: float | None) -> str:
    if value is None or value == "":
        return "ok" if fid is not None else "unknown"
    normalized = str(value).strip().lower()
    if normalized in {"ok", "success", "complete", "completed", "pass", "passed"}:
        return "ok"
    return normalized


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _protocol_id(record: dict[str, Any], normalized: dict[str, Any]) -> str:
    supplied = _first(record, "protocol_id", "protocol.id", "evaluation.protocol_id")
    if supplied is not None:
        return str(supplied)

    fingerprint = {
        key: normalized.get(key)
        for key in (
            "sampler",
            "sampling_method",
            "sampling_steps",
            "seed",
            "num_requested",
            "num_png",
            "world_size",
            "per_proc_batch_size",
            "fid_impl",
            "fid_batch_size",
            "reference_sha256",
        )
        if normalized.get(key) not in (None, "")
    }
    if not fingerprint:
        return "unspecified"
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_record(
    record: dict[str, Any],
    *,
    default_variant: str,
    source: str,
    origin: str,
    steps_per_epoch: int,
    final_step: int,
    conv_resume_step: int,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RecordError("record must be a JSON object")

    embedded_variant = _first(record, "variant", "model_variant", "model.variant")
    variant = _normalize_variant(embedded_variant or default_variant)
    expected_variant = _normalize_variant(default_variant)
    if embedded_variant is not None and variant != expected_variant:
        raise RecordError(
            f"record variant {variant!r} conflicts with source directory "
            f"{expected_variant!r}"
        )

    step = _as_int(
        _first(
            record,
            "step",
            "optimizer_step",
            "train_steps",
            "checkpoint.step",
            "result.step",
        ),
        "step",
        required=True,
    )
    assert step is not None
    if step < 0:
        raise RecordError(f"step must be non-negative, got {step}")

    cfg = _as_float(
        _first(record, "cfg", "cfg_scale", "protocol.cfg", "protocol.cfg_scale"),
        "cfg",
        required=True,
    )
    assert cfg is not None

    fid = _as_float(_first(record, "fid", "result.fid", "metrics.fid"), "fid")
    status = _normalize_status(_first(record, "status", "result.status"), fid)
    if status == "ok" and fid is None:
        raise RecordError("status is ok but fid is missing")

    checkpoint_value = record.get("checkpoint")
    if isinstance(checkpoint_value, dict):
        checkpoint = _string(
            _first(record, "checkpoint.path", "checkpoint.local_path", "checkpoint.uri")
        )
        checkpoint_sha256 = _string(
            _first(record, "checkpoint.sha256", "checkpoint.digest")
        )
    else:
        checkpoint = _string(
            _first(record, "checkpoint", "checkpoint_path", "ckpt", "result.checkpoint")
        )
        checkpoint_sha256 = _string(
            _first(record, "checkpoint_sha256", "checkpoint_digest")
        )

    if not checkpoint_sha256:
        checkpoint_sha256 = _string(
            _first(record, "checkpoint_sha256", "checkpoint.sha256")
        )

    phase = _first(record, "phase", "evaluation.phase")
    if phase is None:
        if step == final_step:
            phase = "final"
        elif variant == "conv" and step == conv_resume_step:
            phase = "resume_baseline"
        else:
            phase = "periodic"

    epoch = _as_float(_first(record, "epoch", "result.epoch"), "epoch")
    if epoch is None:
        epoch = step / steps_per_epoch

    normalized: dict[str, Any] = {
        "record_schema": _string(
            _first(record, "record_schema", "schema", "evaluation.record_schema")
        ),
        "variant": variant,
        "step": step,
        "epoch": epoch,
        "phase": str(phase),
        "cfg": cfg,
        "fid": fid,
        "status": status,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": _as_int(
            _first(record, "checkpoint_step", "checkpoint.step"),
            "checkpoint_step",
        ),
        "num_requested": _as_int(
            _first(record, "num_requested", "protocol.num_requested"),
            "num_requested",
        ),
        "num_png": _as_int(_first(record, "num_png", "protocol.num_png"), "num_png"),
        "seed": _as_int(_first(record, "seed", "protocol.seed"), "seed"),
        "sampler": _string(_first(record, "sampler", "protocol.sampler")),
        "sampling_method": _string(
            _first(record, "sampling_method", "method", "protocol.sampling_method", "protocol.method")
        ),
        "sampling_steps": _as_int(
            _first(
                record,
                "sampling_steps",
                "num_sampling_steps",
                "protocol.sampling_steps",
                "protocol.num_steps",
            ),
            "sampling_steps",
        ),
        "world_size": _as_int(
            _first(record, "world_size", "protocol.world_size"),
            "world_size",
        ),
        "per_proc_batch_size": _as_int(
            _first(
                record,
                "per_proc_batch_size",
                "protocol.per_proc_batch_size",
                "protocol.per_proc_batch",
            ),
            "per_proc_batch_size",
        ),
        "fid_impl": _string(
            _first(record, "fid_impl", "protocol.fid_impl", "protocol.implementation")
        ),
        "fid_batch_size": _as_int(
            _first(record, "fid_batch_size", "protocol.fid_batch_size"),
            "fid_batch_size",
        ),
        "fid_num_workers": _as_int(
            _first(record, "fid_num_workers", "protocol.fid_num_workers"),
            "fid_num_workers",
        ),
        "reference": _string(
            _first(
                record,
                "reference",
                "reference_path",
                "protocol.reference",
                "protocol.reference_path",
            )
        ),
        "reference_sha256": _string(
            _first(record, "reference_sha256", "protocol.reference_sha256")
        ),
        "timestamp_utc": _string(
            _first(record, "timestamp_utc", "timestamp", "result.timestamp_utc")
        ),
        "git_commit": _string(_first(record, "git_commit", "provenance.git_commit")),
        "wandb_run_name": _string(
            _first(record, "wandb_run_name", "wandb.run_name", "provenance.wandb_run_name")
        ),
        "origin": origin,
        "source": source,
    }
    normalized["protocol_id"] = _protocol_id(record, normalized)
    return normalized


def _payload_records(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return [payload]
    for key in ("records", "evaluations"):
        if isinstance(payload.get(key), list):
            return payload[key]
    if isinstance(payload.get("record"), dict):
        return [payload["record"]]
    return [payload]


def _load_raw_records(
    raw_root: Path,
    *,
    steps_per_epoch: int,
    final_step: int,
    conv_resume_step: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    issues: list[str] = []

    for variant in VARIANTS:
        variant_dir = raw_root / variant
        if not variant_dir.is_dir():
            issues.append(f"raw directory is missing: {variant_dir}")
            continue
        for source_path in sorted(variant_dir.glob("*.json")):
            try:
                with source_path.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"cannot read {source_path}: {exc}")
                continue

            payload_records = _payload_records(payload)
            for index, raw_record in enumerate(payload_records):
                source = str(source_path)
                if len(payload_records) > 1:
                    source += f"#{index + 1}"
                try:
                    records.append(
                        _normalize_record(
                            raw_record,
                            default_variant=variant,
                            source=source,
                            origin="raw",
                            steps_per_epoch=steps_per_epoch,
                            final_step=final_step,
                            conv_resume_step=conv_resume_step,
                        )
                    )
                except (RecordError, TypeError) as exc:
                    issues.append(f"invalid record {source}: {exc}")

    return records, issues


def _load_conv_history(
    path: Path | None,
    *,
    steps_per_epoch: int,
    final_step: int,
    conv_resume_step: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    if path is None:
        return [], []
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        return [], [f"cannot read Conv history {path}: {exc}"]

    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            return [], [f"Conv history has no TSV header: {path}"]
        for row_number, row in enumerate(reader, start=2):
            source = f"{path}#{row_number}"
            history_row: dict[str, Any] = dict(row)
            history_row.setdefault("variant", "conv")
            history_row.setdefault("phase", "history")
            try:
                records.append(
                    _normalize_record(
                        history_row,
                        default_variant="conv",
                        source=source,
                        origin="conv_history",
                        steps_per_epoch=steps_per_epoch,
                        final_step=final_step,
                        conv_resume_step=conv_resume_step,
                    )
                )
            except (RecordError, TypeError) as exc:
                issues.append(f"invalid Conv history row {source}: {exc}")
    return records, issues


def _cfg_key(cfg: float) -> float:
    return round(cfg, 9)


def _record_key(record: dict[str, Any]) -> tuple[str, int, float]:
    return record["variant"], record["step"], _cfg_key(record["cfg"])


def _expected_phase(
    variant: str, step: int, *, final_step: int, conv_resume_step: int
) -> str:
    if step == final_step:
        return "final"
    if variant == "conv" and step == conv_resume_step:
        return "resume-baseline"
    return "periodic"


def _validate_workflow_records(
    records: Iterable[dict[str, Any]],
    *,
    steps_per_epoch: int,
    final_step: int,
    conv_resume_step: int,
) -> list[str]:
    """Validate shards that claim to have been emitted by this workflow.

    Legacy Conv TSV rows intentionally do not pass through this validator and
    therefore can be shown on a curve but can never establish strict coverage.
    """

    issues: list[str] = []
    sha_pattern = re.compile(r"[0-9a-f]{64}")
    for record in records:
        if record.get("origin") != "raw":
            continue
        source = record["source"]
        if "#" in source:
            issues.append(
                f"workflow evaluation shard must contain one record only: {source}"
            )

        for field, expected in EXACT_EVALUATION_PROTOCOL.items():
            actual = record.get(field)
            if actual != expected:
                issues.append(
                    f"workflow protocol mismatch in {source}: {field} must be "
                    f"{expected!r}, got {actual!r}"
                )

        if record.get("status") != "ok":
            issues.append(f"workflow record is not successful: {source}")
        fid = record.get("fid")
        if fid is None or not math.isfinite(float(fid)) or float(fid) < 0:
            issues.append(f"workflow record has invalid FID: {source}")
        if _cfg_key(record["cfg"]) not in {_cfg_key(1.0), _cfg_key(4.0)}:
            issues.append(f"workflow record has unsupported CFG value: {source}")
        if not record.get("checkpoint"):
            issues.append(f"workflow record has no checkpoint path: {source}")
        checkpoint_sha = record.get("checkpoint_sha256", "")
        if not sha_pattern.fullmatch(checkpoint_sha):
            issues.append(
                f"workflow record has invalid checkpoint SHA-256: {source}"
            )
        if record.get("checkpoint_step") != record["step"]:
            issues.append(
                f"workflow record checkpoint step mismatch in {source}: expected "
                f"{record['step']}, got {record.get('checkpoint_step')!r}"
            )
        if not record.get("reference"):
            issues.append(f"workflow record has no FID reference path: {source}")

        expected_phase = _expected_phase(
            record["variant"],
            record["step"],
            final_step=final_step,
            conv_resume_step=conv_resume_step,
        )
        if record.get("phase") != expected_phase:
            issues.append(
                f"workflow record phase mismatch in {source}: expected "
                f"{expected_phase!r}, got {record.get('phase')!r}"
            )
        expected_epoch = record["step"] / steps_per_epoch
        if not math.isclose(
            float(record["epoch"]), expected_epoch, rel_tol=0.0, abs_tol=1e-12
        ):
            issues.append(
                f"workflow record epoch mismatch in {source}: expected "
                f"{expected_epoch}, got {record['epoch']}"
            )

        source_path = Path(source)
        expected_name = (
            f"step-{record['step']:07d}-cfg-{record['cfg']:g}.json"
        )
        if source_path.name != expected_name:
            issues.append(
                f"workflow record filename mismatch: expected {expected_name}, "
                f"got {source_path.name}"
            )
    return issues


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(container: Any, name: str) -> Any:
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def _checkpoint_training_protocol_issues(
    checkpoint: dict[str, Any], *, variant: str, label: str, path: Path
) -> list[str]:
    """Check that the persisted final state supports the published protocol."""

    issues: list[str] = []
    args = checkpoint.get("args")
    expected_args = {
        "model": "SiT-S/2",
        "epochs": 800,
        "max_train_steps": FINAL_STEP,
        "global_batch_size": 256,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "global_seed": 0,
        "vae": "ema",
        "path_type": "Linear",
        "prediction": "velocity",
        "image_size": 256,
        "num_classes": 1000,
        "restart_deterministic_data": True,
    }
    if args is None:
        issues.append(f"{label} final checkpoint has no args: {path}")
    else:
        for field, expected in expected_args.items():
            actual = _field(args, field)
            if actual != expected:
                issues.append(
                    f"{label} final checkpoint args.{field} must be "
                    f"{expected!r}, got {actual!r}"
                )

    expected_data_state = {
        "dataset_size": 1_281_167,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "world_size": 8,
        "global_batch_size": 256,
        "gradient_accumulation_steps": 1,
        "global_seed": 0,
        "image_size": 256,
        "num_classes": 1000,
        "restart_deterministic_data": True,
    }
    data_state = checkpoint.get("data_state")
    if not isinstance(data_state, dict):
        issues.append(f"{label} final checkpoint has no valid data_state: {path}")
    else:
        for field, expected in expected_data_state.items():
            actual = data_state.get(field)
            if actual != expected:
                issues.append(
                    f"{label} final checkpoint data_state.{field} must be "
                    f"{expected!r}, got {actual!r}"
                )

    if checkpoint.get("resume_state_version") != 1:
        issues.append(
            f"{label} final checkpoint resume_state_version must be 1: {path}"
        )
    rng_states = checkpoint.get("rng_states")
    required_rng_fields = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(rng_states, list) or len(rng_states) != 8:
        issues.append(
            f"{label} final checkpoint must contain RNG state for eight ranks: {path}"
        )
    else:
        for rank, state in enumerate(rng_states):
            if not isinstance(state, dict) or not required_rng_fields.issubset(state):
                issues.append(
                    f"{label} final checkpoint has incomplete RNG state for rank "
                    f"{rank}: {path}"
                )

    lineage = checkpoint.get("training_lineage")
    if not isinstance(lineage, dict) or lineage.get("schema_version") != 1:
        issues.append(f"{label} final checkpoint has no valid training lineage: {path}")
    elif variant == "conv":
        expected_lineage = {
            "mode": "resume",
            "initial_step": CONV_RESUME_STEP,
            "initial_checkpoint_sha256": CONV_RESUME_SHA256,
        }
        for field, expected in expected_lineage.items():
            if lineage.get(field) != expected:
                issues.append(
                    f"{label} final checkpoint lineage.{field} must be "
                    f"{expected!r}, got {lineage.get(field)!r}"
                )
    else:
        expected_lineage = {
            "mode": "scratch",
            "initial_step": 0,
            "initial_checkpoint": None,
            "initial_checkpoint_sha256": None,
        }
        for field, expected in expected_lineage.items():
            if lineage.get(field) != expected:
                issues.append(
                    f"{label} final checkpoint lineage.{field} must be "
                    f"{expected!r}, got {lineage.get(field)!r}"
                )

    optimizer = checkpoint.get("opt")
    param_groups = optimizer.get("param_groups") if isinstance(optimizer, dict) else None
    if not isinstance(param_groups, list) or not param_groups:
        issues.append(
            f"{label} final checkpoint has no optimizer parameter group: {path}"
        )
    else:
        optimizer_expected = {
            "lr": 1e-4,
            "weight_decay": 0,
            "betas": (0.9, 0.999),
        }
        for field, expected in optimizer_expected.items():
            actual = param_groups[0].get(field)
            if actual != expected:
                issues.append(
                    f"{label} final checkpoint optimizer {field} must be "
                    f"{expected!r}, got {actual!r}"
                )
    return issues


def _validate_final_checkpoints(
    records: Iterable[dict[str, Any]], *, final_step: int
) -> list[str]:
    """Verify both final CFG records against one real, valid checkpoint."""

    issues: list[str] = []
    ok_records = _ok_record_map(records)
    hashes: dict[Path, str] = {}
    inspected_steps: dict[Path, int] = {}
    for variant in VARIANTS:
        cfg1 = ok_records.get((variant, final_step, _cfg_key(1.0)))
        cfg4 = ok_records.get((variant, final_step, _cfg_key(4.0)))
        if cfg1 is None or cfg4 is None:
            continue

        path_values = {cfg1.get("checkpoint", ""), cfg4.get("checkpoint", "")}
        sha_values = {
            cfg1.get("checkpoint_sha256", ""),
            cfg4.get("checkpoint_sha256", ""),
        }
        label = VARIANT_LABELS[variant]
        if "" in path_values or len(path_values) != 1:
            issues.append(
                f"{label} final CFG=1 and CFG=4 must reference the same checkpoint path"
            )
            continue
        if "" in sha_values or len(sha_values) != 1:
            issues.append(
                f"{label} final CFG=1 and CFG=4 must record the same checkpoint SHA-256"
            )
            continue

        checkpoint = Path(path_values.pop()).expanduser()
        try:
            checkpoint = checkpoint.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            issues.append(f"{label} final checkpoint does not exist: {checkpoint} ({exc})")
            continue
        if not checkpoint.is_file():
            issues.append(f"{label} final checkpoint is not a file: {checkpoint}")
            continue

        try:
            if checkpoint not in hashes:
                hashes[checkpoint] = _sha256_file(checkpoint)
            actual_sha = hashes[checkpoint]
        except OSError as exc:
            issues.append(f"cannot hash {label} final checkpoint {checkpoint}: {exc}")
            continue
        recorded_sha = sha_values.pop()
        if actual_sha != recorded_sha:
            issues.append(
                f"{label} final checkpoint SHA-256 mismatch: recorded "
                f"{recorded_sha}, actual {actual_sha}"
            )
            continue

        try:
            if checkpoint not in inspected_steps:
                inspected_steps[checkpoint] = inspect_checkpoint(checkpoint)
            actual_step = inspected_steps[checkpoint]
        except Exception as exc:
            issues.append(f"{label} final checkpoint is invalid: {checkpoint} ({exc})")
            continue
        if actual_step != FINAL_STEP:
            issues.append(
                f"{label} final checkpoint must be optimizer step {FINAL_STEP}, "
                f"got {actual_step}"
            )
            continue

        try:
            import torch

            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            issues.append(
                f"cannot read {label} final checkpoint metadata: {checkpoint} ({exc})"
            )
            continue
        if not isinstance(checkpoint_payload, dict):
            issues.append(f"{label} final checkpoint is not a mapping: {checkpoint}")
            continue
        issues.extend(
            _checkpoint_training_protocol_issues(
                checkpoint_payload,
                variant=variant,
                label=label,
                path=checkpoint,
            )
        )
    return issues


def _deduplicate_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    groups: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(_record_key(record), []).append(record)

    selected: list[dict[str, Any]] = []
    conflicts: list[str] = []
    for key, group in groups.items():
        successful = [record for record in group if record["status"] == "ok"]
        candidates = successful or group
        raw_candidates = [record for record in candidates if record["origin"] == "raw"]
        pool = raw_candidates or candidates

        if successful:
            comparable = [record for record in pool if record["status"] == "ok"]
            signatures = {
                (round(float(record["fid"]), 12), record["protocol_id"])
                for record in comparable
            }
            if len(signatures) > 1:
                variant, step, cfg = key
                sources = ", ".join(sorted(record["source"] for record in comparable))
                conflicts.append(
                    f"conflicting successful records for {variant} step={step} "
                    f"cfg={cfg:g}: {sources}"
                )

        winner = max(
            pool,
            key=lambda record: (record.get("timestamp_utc", ""), record["source"]),
        )
        selected.append(winner)

    variant_order = {variant: index for index, variant in enumerate(VARIANTS)}
    selected.sort(
        key=lambda record: (
            variant_order[record["variant"]],
            record["step"],
            record["cfg"],
        )
    )
    return selected, conflicts


def _expected_steps(start: int, end: int, interval: int) -> list[int]:
    if start < 0 or end < start or interval <= 0:
        raise ValueError("invalid periodic evaluation schedule")
    if (end - start) % interval != 0:
        raise ValueError(
            f"periodic range {start}..{end} is not divisible by interval {interval}"
        )
    return list(range(start, end + 1, interval))


def _ok_record_map(
    records: Iterable[dict[str, Any]],
) -> dict[tuple[str, int, float], dict[str, Any]]:
    return {
        _record_key(record): record
        for record in records
        if record["status"] == "ok" and record["fid"] is not None
    }


def _build_coverage(
    records: list[dict[str, Any]],
    *,
    schedules: dict[str, list[int]],
    final_step: int,
    conv_resume_step: int,
) -> tuple[dict[str, Any], list[str]]:
    ok_records = _ok_record_map(records)
    coverage: dict[str, Any] = {}
    missing_messages: list[str] = []

    for variant in VARIANTS:
        expected = schedules[variant]
        observed = [
            step
            for step in expected
            if (variant, step, _cfg_key(1.0)) in ok_records
        ]
        missing = [step for step in expected if step not in observed]
        final_metrics: dict[str, Any] = {}
        missing_final: list[float] = []
        for cfg in (1.0, 4.0):
            record = ok_records.get((variant, final_step, _cfg_key(cfg)))
            if record is None:
                missing_final.append(cfg)
                final_metrics[f"cfg_{cfg:g}"] = None
            else:
                final_metrics[f"cfg_{cfg:g}"] = {
                    "fid": record["fid"],
                    "checkpoint": record["checkpoint"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "protocol_id": record["protocol_id"],
                    "source": record["source"],
                }

        coverage[variant] = {
            "expected_periodic_steps": expected,
            "observed_periodic_steps": observed,
            "missing_periodic_steps": missing,
            "periodic_complete": not missing,
            "final_step": final_step,
            "final_metrics": final_metrics,
            "missing_final_cfg": missing_final,
            "final_complete": not missing_final,
        }
        cfg1_final = ok_records.get((variant, final_step, _cfg_key(1.0)))
        cfg4_final = ok_records.get((variant, final_step, _cfg_key(4.0)))
        if cfg1_final is not None and cfg4_final is not None:
            final_hashes = {
                cfg1_final.get("checkpoint_sha256", ""),
                cfg4_final.get("checkpoint_sha256", ""),
            }
            final_paths = {
                cfg1_final.get("checkpoint", ""),
                cfg4_final.get("checkpoint", ""),
            }
            if "" in final_hashes or len(final_hashes) != 1:
                coverage[variant]["final_complete"] = False
                missing_messages.append(
                    f"{VARIANT_LABELS[variant]} final CFG records do not share "
                    "one non-empty checkpoint SHA-256"
                )
            if "" in final_paths or len(final_paths) != 1:
                coverage[variant]["final_complete"] = False
                missing_messages.append(
                    f"{VARIANT_LABELS[variant]} final CFG records do not share "
                    "one checkpoint path"
                )
        if missing:
            missing_messages.append(
                f"{VARIANT_LABELS[variant]} missing periodic CFG=1 steps: "
                + ", ".join(str(step) for step in missing)
            )
        if missing_final:
            missing_messages.append(
                f"{VARIANT_LABELS[variant]} missing final step {final_step} CFG: "
                + ", ".join(f"{cfg:g}" for cfg in missing_final)
            )

    conv_baseline = ok_records.get(
        ("conv", conv_resume_step, _cfg_key(1.0))
    )
    coverage["conv"]["resume_baseline_step"] = conv_resume_step
    coverage["conv"]["resume_baseline_complete"] = conv_baseline is not None
    if conv_baseline is None:
        missing_messages.append(
            f"Conv missing resume-baseline CFG=1 at step {conv_resume_step}"
        )

    return coverage, missing_messages


def _tsv_value(field: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if field == "epoch":
            return f"{value:.8f}"
        return f"{value:.15g}"
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _write_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=list(TSV_FIELDS),
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {field: _tsv_value(field, record.get(field)) for field in TSV_FIELDS}
            )


def _format_fid(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.6f}"


def _format_steps(steps: list[int]) -> str:
    if not steps:
        return "none"
    return ", ".join(f"{step:,}" for step in steps)


def _protocol_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "record_schema",
        "protocol_id",
        "sampler",
        "sampling_method",
        "sampling_steps",
        "seed",
        "num_requested",
        "num_png",
        "world_size",
        "per_proc_batch_size",
        "fid_impl",
        "fid_batch_size",
        "fid_num_workers",
        "reference",
        "reference_sha256",
    )
    result: dict[str, Any] = {}
    valid = [record for record in records if record["status"] == "ok"]
    for field in fields:
        values = sorted(
            {
                str(record[field])
                for record in valid
                if record.get(field) not in (None, "")
            }
        )
        if not values:
            result[field] = None
        elif len(values) == 1:
            result[field] = values[0]
        else:
            result[field] = values
    return result


def _record_for(
    records: list[dict[str, Any]], variant: str, step: int, cfg: float
) -> dict[str, Any] | None:
    key = (variant, step, _cfg_key(cfg))
    for record in records:
        if _record_key(record) == key and record["status"] == "ok":
            return record
    return None


def _markdown_report(summary: dict[str, Any]) -> str:
    status = summary["status"]
    strict = summary["strict_requested"]
    lines = [
        "# Conv and Rotation-head Training Results",
        "",
        f"> **Status: {status}**",
        "",
    ]
    if not strict:
        lines.extend(
            [
                "This is a non-strict draft. Re-run the builder with `--strict` after "
                "all required evaluations finish to publish a COMPLETE report.",
                "",
            ]
        )
    elif status != "COMPLETE":
        lines.extend(
            [
                "Strict validation failed. Missing or invalid inputs are listed below.",
                "",
            ]
        )

    lines.extend(
        [
            f"Generated at `{summary['generated_at_utc']}`. Lower FID is better.",
            "",
            "## Training protocol",
            "",
            "| Field | Value |",
            "| --- | --- |",
            "| Model | `SiT-S/2`, ImageNet 256×256 |",
            f"| Topology | {summary['training_protocol']['markdown_topology']}, "
            "global batch 256, per-GPU batch 32 |",
            "| Optimizer | AdamW, LR `1e-4`, weight decay 0, default betas |",
            "| Duration | 800 epochs / 4,003,200 optimizer steps |",
            "| Transport / VAE / seed | Linear velocity / SD VAE EMA / 0 |",
            "| Conv initialization | Resume step 1,950,000 from `BlueSourceJY/SiT-Complementary` |",
            "| Rotation-head initialization | From scratch |",
            "",
            "## Final checkpoint FID",
            "",
        "| Model | Final step | Epoch | CFG=1 FID | CFG=4 FID | Checkpoint | SHA-256 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for variant in VARIANTS:
        model = summary["models"][variant]
        cfg1 = model["final_metrics"]["cfg_1"]
        cfg4 = model["final_metrics"]["cfg_4"]
        checkpoint = ""
        checkpoint_sha256 = ""
        for metric in (cfg1, cfg4):
            if metric and metric.get("checkpoint"):
                checkpoint = metric["checkpoint"]
                checkpoint_sha256 = metric.get("checkpoint_sha256", "")
                break
        checkpoint_text = f"`{checkpoint}`" if checkpoint else "—"
        sha_text = f"`{checkpoint_sha256}`" if checkpoint_sha256 else "—"
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {model['final_step']:,} | "
            f"{summary['schedule']['final_epoch']:.0f} | "
            f"{_format_fid(cfg1['fid'] if cfg1 else None)} | "
            f"{_format_fid(cfg4['fid'] if cfg4 else None)} | {checkpoint_text} | {sha_text} |"
        )

    lines.extend(
        [
            "",
            "## CFG=1 training curves",
            "",
            "The Conv curve covers the resumed interval produced by this workflow. "
            "Any merged legacy points "
            "are visually distinguished from measurements produced by this workflow. "
            "No missing point is interpolated.",
            "",
            "![Conv CFG=1 FID curve](conv_fid_cfg1_curve.png)",
            "",
            "![Rotation-head CFG=1 FID curve](rotation_head_fid_cfg1_curve.png)",
            "",
            "![Combined Conv and Rotation-head CFG=1 FID curves](fid_cfg1_training_curves.png)",
            "",
            "## Evaluation coverage",
            "",
            "| Model | Periodic CFG=1 | First expected step | Last expected step | Resume baseline | Final CFG=1/4 |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for variant in VARIANTS:
        model = summary["models"][variant]
        expected = model["expected_periodic_steps"]
        observed = model["observed_periodic_steps"]
        final_text = "complete" if model["final_complete"] else "missing"
        baseline_text = "n/a"
        if variant == "conv":
            baseline_text = (
                "complete" if model.get("resume_baseline_complete") else "missing"
            )
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {len(observed)}/{len(expected)} | "
            f"{expected[0]:,} | {expected[-1]:,} | {baseline_text} | {final_text} |"
        )

    missing_any = any(
        model["missing_periodic_steps"]
        or model["missing_final_cfg"]
        or (variant == "conv" and not model.get("resume_baseline_complete"))
        for variant, model in summary["models"].items()
    )
    if missing_any:
        lines.extend(["", "### Missing required evaluations", ""])
        for variant in VARIANTS:
            model = summary["models"][variant]
            if model["missing_periodic_steps"]:
                lines.append(
                    f"- {VARIANT_LABELS[variant]} periodic CFG=1: "
                    f"{_format_steps(model['missing_periodic_steps'])}"
                )
            if model["missing_final_cfg"]:
                values = ", ".join(f"CFG={cfg:g}" for cfg in model["missing_final_cfg"])
                lines.append(
                    f"- {VARIANT_LABELS[variant]} final step "
                    f"{model['final_step']:,}: {values}"
                )
            if variant == "conv" and not model.get("resume_baseline_complete"):
                lines.append(
                    f"- Conv resume-baseline CFG=1: "
                    f"{model['resume_baseline_step']:,}"
                )

    protocol = summary["evaluation_protocol"]
    lines.extend(
        [
            "",
            "## Evaluation protocol recorded by inputs",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
    )
    for field, value in protocol.items():
        if isinstance(value, list):
            display = ", ".join(value)
        elif value is None:
            display = "not recorded"
        else:
            display = str(value)
        lines.append(f"| `{field}` | `{display}` |")

    if summary["issues"]:
        lines.extend(["", "## Input issues", ""])
        lines.extend(f"- {issue}" for issue in summary["issues"])

    lines.extend(
        [
            "",
            "## Machine-readable records",
            "",
            "- Normalized evaluations: `fid_results.tsv`",
            "- Full summary and validation state: `training_results.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to build result plots; install it in the "
            "training environment"
        ) from exc
    return plt


def _cfg1_by_step(
    records: list[dict[str, Any]], variant: str
) -> dict[int, dict[str, Any]]:
    return {
        record["step"]: record
        for record in records
        if record["variant"] == variant
        and record["status"] == "ok"
        and _cfg_key(record["cfg"]) == _cfg_key(1.0)
    }


def _pillow_star(cx: float, cy: float, outer: float = 8, inner: float = 3.5):
    points: list[tuple[float, float]] = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = outer if index % 2 == 0 else inner
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _draw_pillow_panel(
    draw: Any,
    *,
    bounds: tuple[int, int, int, int],
    records: list[dict[str, Any]],
    variant: str,
    expected_steps: list[int],
    final_step: int,
    conv_resume_step: int,
    shared_y_range: tuple[float, float] | None = None,
) -> None:
    """Small dependency-light plotting fallback used when matplotlib is absent."""
    from PIL import ImageColor, ImageFont

    left, top, right, bottom = bounds
    plot_left, plot_top = left + 76, top + 52
    plot_right, plot_bottom = right - 24, bottom - 62
    font = ImageFont.load_default()
    bold_font = font
    values = _cfg1_by_step(records, variant)

    if shared_y_range is None:
        finite_fids = [float(record["fid"]) for record in values.values()]
        if finite_fids:
            y_min, y_max = min(finite_fids), max(finite_fids)
        else:
            y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = shared_y_range
    if y_max <= y_min:
        y_min -= 0.5
        y_max += 0.5
    padding = max((y_max - y_min) * 0.08, 0.2)
    y_min -= padding
    y_max += padding
    x_max = max(4.08, final_step / 1_000_000 + 0.04)

    def x_pixel(step: int) -> float:
        return plot_left + (step / 1_000_000) / x_max * (plot_right - plot_left)

    def y_pixel(fid: float) -> float:
        return plot_bottom - (fid - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    if variant == "conv":
        resume_x = x_pixel(conv_resume_step)
        draw.rectangle(
            (plot_left, plot_top, resume_x, plot_bottom), fill=(246, 246, 246)
        )
        for y in range(plot_top, plot_bottom + 1, 6):
            draw.line((resume_x, y, resume_x, min(y + 3, plot_bottom)), fill="#777777")

    for tick_index in range(6):
        fid = y_min + tick_index * (y_max - y_min) / 5
        y = y_pixel(fid)
        draw.line((plot_left, y, plot_right, y), fill="#e2e2e2", width=1)
        label = f"{fid:.1f}"
        label_box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (plot_left - (label_box[2] - label_box[0]) - 8, y - 6),
            label,
            fill="#333333",
            font=font,
        )
    for tick in range(5):
        x = plot_left + tick / x_max * (plot_right - plot_left)
        draw.line((x, plot_top, x, plot_bottom), fill="#eeeeee", width=1)
        label = str(tick)
        label_box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (x - (label_box[2] - label_box[0]) / 2, plot_bottom + 8),
            label,
            fill="#333333",
            font=font,
        )

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#222222", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#222222", width=2)
    draw.text(
        (left + 10, top + 8),
        f"{VARIANT_LABELS[variant]} CFG=1 FID",
        fill="#111111",
        font=bold_font,
    )
    x_label = "Optimizer step (millions)"
    x_label_box = draw.textbbox((0, 0), x_label, font=font)
    draw.text(
        (
            (plot_left + plot_right - (x_label_box[2] - x_label_box[0])) / 2,
            bottom - 25,
        ),
        x_label,
        fill="#222222",
        font=font,
    )
    draw.text(
        (left + 5, top + 30),
        "FID-50K (lower is better)",
        fill="#444444",
        font=font,
    )

    color = ImageColor.getrgb(VARIANT_COLORS[variant])
    previous: tuple[float, float] | None = None
    for step in expected_steps:
        record = values.get(step)
        if record is None:
            previous = None
            continue
        point = (x_pixel(step), y_pixel(float(record["fid"])))
        if previous is not None:
            draw.line((*previous, *point), fill=color, width=3)
        draw.ellipse(
            (point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4),
            fill=color,
            outline="white",
            width=1,
        )
        previous = point

    extra_steps = sorted(
        step for step in values if step not in expected_steps and step != final_step
    )
    history_points: list[tuple[float, float]] = []
    for step in extra_steps:
        record = values[step]
        point = (x_pixel(step), y_pixel(float(record["fid"])))
        if record.get("origin") == "conv_history":
            history_points.append(point)
            draw.ellipse(
                (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
                fill="#777777",
            )
        else:
            draw.polygon(
                (
                    (point[0], point[1] - 5),
                    (point[0] + 5, point[1]),
                    (point[0], point[1] + 5),
                    (point[0] - 5, point[1]),
                ),
                fill="#9467bd",
            )
    for first, second in zip(history_points, history_points[1:]):
        draw.line((*first, *second), fill="#888888", width=2)

    final_record = values.get(final_step)
    if final_record is not None:
        final_point = (x_pixel(final_step), y_pixel(float(final_record["fid"])))
        draw.polygon(
            _pillow_star(*final_point),
            fill="#ff7f0e",
            outline="#111111",
        )

    missing = [step for step in expected_steps if step not in values]
    if missing:
        draw.text(
            (plot_left + 8, plot_bottom - 18),
            f"Missing periodic points: {len(missing)}",
            fill="#b22222",
            font=font,
        )
    if not values:
        message = "No valid CFG=1 records"
        message_box = draw.textbbox((0, 0), message, font=font)
        draw.text(
            (
                (plot_left + plot_right - (message_box[2] - message_box[0])) / 2,
                (plot_top + plot_bottom) / 2,
            ),
            message,
            fill="#666666",
            font=font,
        )


def _write_plots_pillow(
    stage_dir: Path,
    *,
    records: list[dict[str, Any]],
    schedules: dict[str, list[int]],
    final_step: int,
    conv_resume_step: int,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "plot generation requires matplotlib or Pillow; neither is installed"
        ) from exc

    single_names = {
        "conv": "conv_fid_cfg1_curve.png",
        "rotation-head": "rotation_head_fid_cfg1_curve.png",
    }
    for variant in VARIANTS:
        image = Image.new("RGB", (1000, 640), "white")
        draw = ImageDraw.Draw(image)
        _draw_pillow_panel(
            draw,
            bounds=(18, 18, 982, 622),
            records=records,
            variant=variant,
            expected_steps=schedules[variant],
            final_step=final_step,
            conv_resume_step=conv_resume_step,
        )
        image.save(stage_dir / single_names[variant], format="PNG")

    all_values = [
        float(record["fid"])
        for record in records
        if record["status"] == "ok"
        and record["fid"] is not None
        and _cfg_key(record["cfg"]) == _cfg_key(1.0)
    ]
    shared_y_range = (
        (min(all_values), max(all_values)) if all_values else (0.0, 1.0)
    )
    image = Image.new("RGB", (1600, 650), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (500, 10),
        "CFG=1 FID during Conv resume and Rotation-head training",
        fill="#111111",
    )
    _draw_pillow_panel(
        draw,
        bounds=(10, 28, 795, 640),
        records=records,
        variant="conv",
        expected_steps=schedules["conv"],
        final_step=final_step,
        conv_resume_step=conv_resume_step,
        shared_y_range=shared_y_range,
    )
    _draw_pillow_panel(
        draw,
        bounds=(805, 28, 1590, 640),
        records=records,
        variant="rotation-head",
        expected_steps=schedules["rotation-head"],
        final_step=final_step,
        conv_resume_step=conv_resume_step,
        shared_y_range=shared_y_range,
    )
    image.save(stage_dir / "fid_cfg1_training_curves.png", format="PNG")


def _plot_panel(
    ax: Any,
    *,
    records: list[dict[str, Any]],
    variant: str,
    expected_steps: list[int],
    final_step: int,
    conv_resume_step: int,
) -> None:
    color = VARIANT_COLORS[variant]
    ok_cfg1 = {
        record["step"]: record
        for record in records
        if record["variant"] == variant
        and record["status"] == "ok"
        and _cfg_key(record["cfg"]) == _cfg_key(1.0)
    }

    x_expected = [step / 1_000_000 for step in expected_steps]
    y_expected = [
        ok_cfg1[step]["fid"] if step in ok_cfg1 else math.nan
        for step in expected_steps
    ]
    ax.plot(x_expected, y_expected, color=color, linewidth=1.8, alpha=0.9)
    present_steps = [step for step in expected_steps if step in ok_cfg1]
    if present_steps:
        ax.scatter(
            [step / 1_000_000 for step in present_steps],
            [ok_cfg1[step]["fid"] for step in present_steps],
            color=color,
            edgecolor="white",
            linewidth=0.55,
            s=34,
            zorder=3,
            label="Periodic CFG=1",
        )

    extra_steps = sorted(
        step
        for step, record in ok_cfg1.items()
        if step not in expected_steps and step != final_step
    )
    history_steps = [
        step
        for step in extra_steps
        if ok_cfg1[step].get("origin") == "conv_history"
    ]
    baseline_steps = [step for step in extra_steps if step not in history_steps]
    if history_steps:
        ax.plot(
            [step / 1_000_000 for step in history_steps],
            [ok_cfg1[step]["fid"] for step in history_steps],
            color="#777777",
            marker="o",
            markersize=3.5,
            linewidth=1.1,
            linestyle="--",
            label="Merged Conv history",
        )
    if baseline_steps:
        ax.scatter(
            [step / 1_000_000 for step in baseline_steps],
            [ok_cfg1[step]["fid"] for step in baseline_steps],
            color="#9467bd",
            marker="D",
            s=38,
            zorder=4,
            label="Additional/baseline eval",
        )

    if final_step in ok_cfg1:
        ax.scatter(
            [final_step / 1_000_000],
            [ok_cfg1[final_step]["fid"]],
            color="#ff7f0e",
            marker="*",
            edgecolor="black",
            linewidth=0.45,
            s=145,
            zorder=5,
            label="Final checkpoint CFG=1",
        )

    missing = [step for step in expected_steps if step not in ok_cfg1]
    if missing:
        ax.text(
            0.015,
            0.02,
            f"Missing periodic points: {len(missing)}",
            transform=ax.transAxes,
            fontsize=8,
            color="#b22222",
            verticalalignment="bottom",
        )

    if variant == "conv":
        ax.axvspan(0, conv_resume_step / 1_000_000, color="#999999", alpha=0.08)
        ax.axvline(
            conv_resume_step / 1_000_000,
            color="#555555",
            linewidth=1,
            linestyle=":",
            label=f"Resume ({conv_resume_step / 1_000_000:g}M)",
        )

    if not ok_cfg1:
        ax.text(
            0.5,
            0.5,
            "No valid CFG=1 records",
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            color="#666666",
        )

    ax.set_title(f"{VARIANT_LABELS[variant]} CFG=1 FID")
    ax.set_xlabel("Optimizer step (millions)")
    ax.set_ylabel("FID-50K (lower is better)")
    ax.set_xlim(0, max(4.08, final_step / 1_000_000 + 0.04))
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
        ax.legend(unique.values(), unique.keys(), fontsize=8, loc="best")


def _write_plots(
    stage_dir: Path,
    *,
    records: list[dict[str, Any]],
    schedules: dict[str, list[int]],
    final_step: int,
    conv_resume_step: int,
) -> None:
    try:
        plt = _load_matplotlib()
    except RuntimeError:
        _write_plots_pillow(
            stage_dir,
            records=records,
            schedules=schedules,
            final_step=final_step,
            conv_resume_step=conv_resume_step,
        )
        return

    single_names = {
        "conv": "conv_fid_cfg1_curve.png",
        "rotation-head": "rotation_head_fid_cfg1_curve.png",
    }
    for variant in VARIANTS:
        figure, axis = plt.subplots(figsize=(9, 5.5))
        _plot_panel(
            axis,
            records=records,
            variant=variant,
            expected_steps=schedules[variant],
            final_step=final_step,
            conv_resume_step=conv_resume_step,
        )
        figure.suptitle("SiT-S/2 ImageNet-256 training evaluation", fontsize=13)
        figure.tight_layout()
        figure.savefig(
            stage_dir / single_names[variant],
            format="png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    for axis, variant in zip(axes, VARIANTS):
        _plot_panel(
            axis,
            records=records,
            variant=variant,
            expected_steps=schedules[variant],
            final_step=final_step,
            conv_resume_step=conv_resume_step,
        )
    figure.suptitle("CFG=1 FID during Conv resume and Rotation-head training", fontsize=14)
    figure.tight_layout()
    figure.savefig(
        stage_dir / "fid_cfg1_training_curves.png",
        format="png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _write_artifacts(
    output_dir: Path,
    *,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    schedules: dict[str, list[int]],
    final_step: int,
    conv_resume_step: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".build-results-", dir=output_dir))
    try:
        _write_tsv(stage_dir / "fid_results.tsv", records)
        with (stage_dir / "training_results.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        (stage_dir / "TRAINING_RESULTS.md").write_text(
            _markdown_report(summary), encoding="utf-8"
        )
        _write_plots(
            stage_dir,
            records=records,
            schedules=schedules,
            final_step=final_step,
            conv_resume_step=conv_resume_step,
        )

        missing_staged = [name for name in OUTPUT_FILES if not (stage_dir / name).is_file()]
        if missing_staged:
            raise RuntimeError(
                "builder did not stage all expected outputs: " + ", ".join(missing_staged)
            )
        for name in OUTPUT_FILES:
            os.replace(stage_dir / name, output_dir / name)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Conv and Rotation-head FID records and build the final "
            "training report."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Raw JSON root (default: OUTPUT_DIR/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_results"),
        help="Directory receiving all rebuilt artifacts.",
    )
    parser.add_argument(
        "--conv-history",
        type=Path,
        default=None,
        help="Optional legacy Conv FID TSV to merge into the curve.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Require all periodic CFG=1 points and final step CFG=1/4 for both "
            "models; exit non-zero if incomplete."
        ),
    )
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    parser.add_argument("--conv-resume-step", type=int, default=CONV_RESUME_STEP)
    parser.add_argument("--conv-periodic-start", type=int, default=CONV_PERIODIC_START)
    parser.add_argument(
        "--rotation-head-periodic-start",
        type=int,
        default=ROTATION_HEAD_PERIODIC_START,
    )
    parser.add_argument("--periodic-end", type=int, default=PERIODIC_END)
    parser.add_argument("--periodic-interval", type=int, default=PERIODIC_INTERVAL)
    parser.add_argument("--final-step", type=int, default=FINAL_STEP)
    parser.add_argument(
        "--gpu-profile",
        choices=sorted(GPU_PROFILE_METADATA),
        default="h20",
        help="Hardware profile used for this eight-rank training run.",
    )
    return parser.parse_args(argv)


def _strict_configuration_issues(args: argparse.Namespace) -> list[str]:
    expected = {
        "steps_per_epoch": STEPS_PER_EPOCH,
        "conv_resume_step": CONV_RESUME_STEP,
        "conv_periodic_start": CONV_PERIODIC_START,
        "rotation_head_periodic_start": ROTATION_HEAD_PERIODIC_START,
        "periodic_end": PERIODIC_END,
        "periodic_interval": PERIODIC_INTERVAL,
        "final_step": FINAL_STEP,
    }
    return [
        f"strict schedule requires --{field.replace('_', '-')}={wanted}, got {getattr(args, field)}"
        for field, wanted in expected.items()
        if getattr(args, field) != wanted
    ]


def _gpu_profile_marker_issues(args: argparse.Namespace) -> list[str]:
    marker = args.output_dir.resolve().parent / ".gpu_profile"
    if not marker.exists():
        return []
    if not marker.is_file():
        return [f"GPU profile marker is not a regular file: {marker}"]
    try:
        recorded = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return [f"cannot read GPU profile marker {marker}: {exc}"]
    if recorded != args.gpu_profile:
        return [
            f"GPU profile mismatch: {marker} records {recorded!r}, but "
            f"--gpu-profile is {args.gpu_profile!r}"
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.steps_per_epoch <= 0:
        raise SystemExit("--steps-per-epoch must be positive")

    raw_root = args.raw_root or (args.output_dir / "raw")
    try:
        schedules = {
            "conv": _expected_steps(
                args.conv_periodic_start,
                args.periodic_end,
                args.periodic_interval,
            ),
            "rotation-head": _expected_steps(
                args.rotation_head_periodic_start,
                args.periodic_end,
                args.periodic_interval,
            ),
        }
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    raw_records, raw_issues = _load_raw_records(
        raw_root,
        steps_per_epoch=args.steps_per_epoch,
        final_step=args.final_step,
        conv_resume_step=args.conv_resume_step,
    )
    history_records, history_issues = _load_conv_history(
        args.conv_history,
        steps_per_epoch=args.steps_per_epoch,
        final_step=args.final_step,
        conv_resume_step=args.conv_resume_step,
    )
    records, conflicts = _deduplicate_records(raw_records + history_records)
    workflow_records = [
        record for record in records if record.get("origin") == "raw"
    ]
    coverage, missing_messages = _build_coverage(
        workflow_records,
        schedules=schedules,
        final_step=args.final_step,
        conv_resume_step=args.conv_resume_step,
    )
    workflow_issues = _validate_workflow_records(
        raw_records,
        steps_per_epoch=args.steps_per_epoch,
        final_step=args.final_step,
        conv_resume_step=args.conv_resume_step,
    )
    strict_issues: list[str] = []
    if args.strict:
        strict_issues.extend(_strict_configuration_issues(args))
        strict_issues.extend(
            _validate_final_checkpoints(workflow_records, final_step=args.final_step)
        )
    issues = (
        _gpu_profile_marker_issues(args)
        + raw_issues
        + history_issues
        + conflicts
        + workflow_issues
        + strict_issues
    )
    data_complete = not issues and not missing_messages
    publication_status = "COMPLETE" if args.strict and data_complete else "INCOMPLETE"
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": publication_status,
        "strict_requested": args.strict,
        "data_complete": data_complete,
        "generated_at_utc": generated_at,
        "inputs": {
            "raw_root": str(raw_root),
            "conv_history": str(args.conv_history) if args.conv_history else None,
        },
        "schedule": {
            "steps_per_epoch": args.steps_per_epoch,
            "conv_resume_step": args.conv_resume_step,
            "conv_periodic_start": args.conv_periodic_start,
            "rotation_head_periodic_start": args.rotation_head_periodic_start,
            "periodic_end": args.periodic_end,
            "periodic_interval": args.periodic_interval,
            "final_step": args.final_step,
            "final_epoch": args.final_step / args.steps_per_epoch,
        },
        "training_protocol": {
            "model": "SiT-S/2",
            "dataset": "ImageNet-1K 256x256",
            "gpu_profile": args.gpu_profile,
            "gpu_topology": GPU_PROFILE_METADATA[args.gpu_profile]["json_topology"],
            "markdown_topology": GPU_PROFILE_METADATA[args.gpu_profile][
                "markdown_topology"
            ],
            "global_batch_size": 256,
            "per_gpu_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0,
            "epochs": 800,
            "target_optimizer_steps": args.final_step,
            "transport": "Linear/velocity",
            "vae": "stabilityai/sd-vae-ft-ema",
            "seed": 0,
            "conv_initialization": "resume from step 1950000",
            "rotation_head_initialization": "from scratch",
        },
        "models": coverage,
        "evaluation_protocol": _protocol_summary(workflow_records),
        "records": records,
        "issues": issues,
        "missing_requirements": missing_messages,
        "artifacts": {
            "fid_tsv": "fid_results.tsv",
            "summary_json": "training_results.json",
            "report_markdown": "TRAINING_RESULTS.md",
            "conv_cfg1_curve": "conv_fid_cfg1_curve.png",
            "rotation_head_cfg1_curve": "rotation_head_fid_cfg1_curve.png",
            "combined_cfg1_curves": "fid_cfg1_training_curves.png",
        },
    }

    try:
        _write_artifacts(
            args.output_dir,
            records=records,
            summary=summary,
            schedules=schedules,
            final_step=args.final_step,
            conv_resume_step=args.conv_resume_step,
        )
    except (OSError, RuntimeError) as exc:
        print(f"build_results.py: failed to build artifacts: {exc}", file=sys.stderr)
        return 1

    print(
        f"Built {len(OUTPUT_FILES)} artifacts in {args.output_dir} "
        f"(status={publication_status}, records={len(records)})."
    )
    if issues:
        for issue in issues:
            print(f"INPUT_ISSUE: {issue}", file=sys.stderr)
    if missing_messages:
        for message in missing_messages:
            print(f"MISSING: {message}", file=sys.stderr)

    if args.strict and not data_complete:
        print("Strict result validation failed.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
