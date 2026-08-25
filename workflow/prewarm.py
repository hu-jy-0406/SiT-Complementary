"""Pre-download runtime weights before a long job loses internet access."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INCEPTION_SIZE = 95_628_359
INCEPTION_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
RUNTIME_MARKER = ".h20-runtime-weights-v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inception(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != INCEPTION_SIZE:
        raise RuntimeError(f"Unexpected Inception weight size: {path}")
    if sha256(path) != INCEPTION_SHA256:
        raise RuntimeError(f"Inception SHA-256 mismatch: {path}")


def write_marker(asset_root: Path, inception_path: Path) -> Path:
    marker = asset_root / RUNTIME_MARKER
    payload = {
        "schema_version": 1,
        "inception": {
            "path": str(inception_path.relative_to(asset_root)),
            "size": INCEPTION_SIZE,
            "sha256": INCEPTION_SHA256,
        },
    }
    temporary = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Verify the local Inception weight and never attempt a download.",
    )
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    # As with the pinned-asset marker, readiness is asserted only after the
    # complete verification below succeeds.
    (asset_root / RUNTIME_MARKER).unlink(missing_ok=True)
    os.environ["SIT_VAE_ROOT"] = str(asset_root / "vae")
    os.environ["TORCH_HOME"] = str(asset_root / "cache" / "torch")
    os.environ["HF_HOME"] = str(asset_root / "cache" / "huggingface")

    inception_path = (
        asset_root
        / "cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "pt_inception-2015-12-05-6726825d.pth"
    )
    try:
        validate_inception(inception_path)
    except (FileNotFoundError, RuntimeError) as exc:
        if args.local_only:
            raise RuntimeError(
                f"Local-only runtime-weight verification failed: {inception_path}"
            ) from exc
        if inception_path.exists():
            # pytorch-fid otherwise attempts to load the corrupt cache entry.
            inception_path.unlink()
        # All Hugging Face assets must already have been prepared.  The only
        # possible network access here is pytorch-fid's pinned Inception file.
        from vae_utils import load_vae
        from pytorch_fid.inception import InceptionV3

        load_vae("ema", device=None)
        InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
        validate_inception(inception_path)
    else:
        print(f"LOCAL_RUNTIME_WEIGHT_REUSED={inception_path}")
    marker = write_marker(asset_root, inception_path)
    print(f"RUNTIME_MARKER={marker}")
    print("RUNTIME_WEIGHTS_READY")


if __name__ == "__main__":
    main()
