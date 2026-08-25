"""Load the repository-local Stable Diffusion VAE weights when available."""

import os
from pathlib import Path

from diffusers.models import AutoencoderKL


_VAE_ROOT = Path(
    os.environ.get(
        "SIT_VAE_ROOT",
        Path(__file__).resolve().parent / "pretrained_models" / "vae",
    )
)


def get_vae_source(vae_name):
    local_path = _VAE_ROOT / f"sd-vae-ft-{vae_name}"
    if (local_path / "config.json").is_file():
        return str(local_path)
    return f"stabilityai/sd-vae-ft-{vae_name}"


def load_vae(vae_name, device=None):
    vae = AutoencoderKL.from_pretrained(
        get_vae_source(vae_name), use_safetensors=True
    )
    return vae.to(device) if device is not None else vae
