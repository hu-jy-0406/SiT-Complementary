"""Download and verify every non-ImageNet asset needed by the GPU workflow."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


MODEL_REPO = "BlueSourceJY/SiT-Complementary"
MODEL_REVISION = "8b0b8744b28ac0101e5528620b65ab86acb7be52"
CONV_FILENAME = "checkpoints/bs256_lr1e-4/conv-layer/1950000.pt"
CONV_HISTORY_FILENAME = "experiments/bs256_lr1e-4/conv-layer/fid_cfg1_50k.tsv"
CONV_SIZE = 526_985_776
CONV_SHA256 = "f3724fa2651c6fbb3a624664f057c1dd56c658d68fb356c52cf51a7684ae7548"
CONV_HISTORY_SIZE = 1_388
CONV_HISTORY_SHA256 = "4d829be0178ba4e45b56bad2f8cc66eefebb3c9aa3d547e7fcab8e5d53fb0dc1"

FID_REPO = "SII-PengZheng/discon"
FID_REVISION = "b796e2efe26d91a2936db5f2e82fc91a229b6801"
FID_FILENAME = "VIRTUAL_imagenet256_labeled.npz"
FID_SIZE = 2_037_122_530
FID_SHA256 = "b32732719497e42660a9affb4a966068cba0855ac449b82015e34ec376d20758"

VAE_REPO = "stabilityai/sd-vae-ft-ema"
VAE_REVISION = "f04b2c4b98319346dad8c65879f680b1997b204a"
VAE_CONFIG_SIZE = 547
VAE_CONFIG_SHA256 = "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e"
VAE_WEIGHTS_SIZE = 334_643_276
VAE_WEIGHTS_SHA256 = "32db726da04f06c1b6b14c0043ce115cc87a501482945c5add89a40d838fcb46"
ASSET_MARKER = ".h20-pinned-assets-v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Unexpected size for {path}: {actual_size}, expected {expected_size}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: {actual_sha256}, expected {expected_sha256}"
        )


def download(
    repo_id: str,
    filename: str,
    revision: str,
    local_dir: Path,
    *,
    force_download: bool = False,
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="model",
            local_dir=local_dir,
            force_download=force_download,
        )
    )


def ensure_asset(
    repo_id: str,
    filename: str,
    revision: str,
    local_dir: Path,
    expected_size: int,
    expected_sha256: str,
    *,
    local_only: bool = False,
) -> Path:
    """Use a verified local file without making any Hugging Face request."""

    candidate = local_dir / filename
    force_download = False
    if candidate.is_file():
        try:
            verify(candidate, expected_size, expected_sha256)
        except RuntimeError as exc:
            print(f"LOCAL_ASSET_INVALID={candidate}: {exc}")
            if local_only:
                raise RuntimeError(
                    f"Local-only verification failed for {candidate}"
                ) from exc
            force_download = True
        else:
            print(f"LOCAL_ASSET_REUSED={candidate}")
            return candidate

    if local_only:
        raise FileNotFoundError(
            f"Local-only verification requires pinned asset: {candidate}"
        )

    downloaded = download(
        repo_id,
        filename,
        revision,
        local_dir,
        force_download=force_download,
    )
    verify(downloaded, expected_size, expected_sha256)
    return downloaded


def write_marker(root: Path, assets: list[tuple[Path, int, str]]) -> Path:
    marker = root / ASSET_MARKER
    payload = {
        "schema_version": 1,
        "assets": [
            {
                "path": str(path.relative_to(root)),
                "size": size,
                "sha256": expected_sha256,
            }
            for path, size, expected_sha256 in assets
        ],
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    return marker


def validate_conv_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "ema", "opt", "args"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Conv checkpoint is missing keys: {sorted(missing)}")
    steps = {
        int(state["step"])
        for state in checkpoint["opt"].get("state", {}).values()
        if "step" in state
    }
    if steps != {1_950_000}:
        raise RuntimeError(f"Unexpected Conv optimizer steps: {sorted(steps)}")
    args = checkpoint["args"]
    expected = {
        "model": "SiT-S/2",
        "epochs": 800,
        "global_batch_size": 256,
        "global_seed": 0,
        "vae": "ema",
        "path_type": "Linear",
        "prediction": "velocity",
    }
    mismatches = {
        key: (getattr(args, key, None), value)
        for key, value in expected.items()
        if getattr(args, key, None) != value
    }
    optimizer_lr = checkpoint["opt"]["param_groups"][0]["lr"]
    if optimizer_lr != 1e-4:
        mismatches["optimizer_lr"] = (optimizer_lr, 1e-4)
    if mismatches:
        raise RuntimeError(f"Conv checkpoint configuration mismatch: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=Path("assets"))
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Verify local pinned files and fail without any download attempt.",
    )
    args = parser.parse_args()
    root = args.asset_root.resolve()
    # Never leave a stale readiness assertion behind if this verification run
    # discovers damage or is interrupted before all assets pass.
    (root / ASSET_MARKER).unlink(missing_ok=True)

    model_dir = root / "huggingface" / MODEL_REPO
    conv = ensure_asset(
        MODEL_REPO,
        CONV_FILENAME,
        MODEL_REVISION,
        model_dir,
        CONV_SIZE,
        CONV_SHA256,
        local_only=args.local_only,
    )
    history = ensure_asset(
        MODEL_REPO,
        CONV_HISTORY_FILENAME,
        MODEL_REVISION,
        model_dir,
        CONV_HISTORY_SIZE,
        CONV_HISTORY_SHA256,
        local_only=args.local_only,
    )
    validate_conv_checkpoint(conv)

    fid_dir = root / "fid"
    reference = ensure_asset(
        FID_REPO,
        FID_FILENAME,
        FID_REVISION,
        fid_dir,
        FID_SIZE,
        FID_SHA256,
        local_only=args.local_only,
    )

    verified_assets = [
        (conv, CONV_SIZE, CONV_SHA256),
        (history, CONV_HISTORY_SIZE, CONV_HISTORY_SHA256),
        (reference, FID_SIZE, FID_SHA256),
    ]

    if not args.skip_vae:
        vae_dir = root / "vae" / "sd-vae-ft-ema"
        vae_config = ensure_asset(
            VAE_REPO,
            "config.json",
            VAE_REVISION,
            vae_dir,
            VAE_CONFIG_SIZE,
            VAE_CONFIG_SHA256,
            local_only=args.local_only,
        )
        vae_weights = ensure_asset(
            VAE_REPO,
            "diffusion_pytorch_model.safetensors",
            VAE_REVISION,
            vae_dir,
            VAE_WEIGHTS_SIZE,
            VAE_WEIGHTS_SHA256,
            local_only=args.local_only,
        )
        verified_assets.extend(
            [
                (vae_config, VAE_CONFIG_SIZE, VAE_CONFIG_SHA256),
                (vae_weights, VAE_WEIGHTS_SIZE, VAE_WEIGHTS_SHA256),
            ]
        )
        marker = write_marker(root, verified_assets)
        print(f"ASSET_MARKER={marker}")

    print(f"CONV_CHECKPOINT={conv}")
    print(f"CONV_HISTORY={history}")
    print(f"FID_REFERENCE={reference}")
    print(f"ASSET_ROOT={root}")


if __name__ == "__main__":
    main()
