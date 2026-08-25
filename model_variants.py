"""Model registry selection shared by sampling utilities."""

from models import SiT_models as base_models
from models_conv import SiT_models as conv_models
from models_rot_head import SiT_models as rot_head_models
from models_rot_layer import SiT_models as rot_layer_models


MODEL_VARIANTS = {
    "base": base_models,
    "rot-layer": rot_layer_models,
    "rot-head": rot_head_models,
    "conv": conv_models,
}


def get_model_registry(variant):
    try:
        return MODEL_VARIANTS[variant]
    except KeyError as exc:
        choices = ", ".join(MODEL_VARIANTS)
        raise ValueError(
            f"Unknown model variant {variant!r}; choose one of: {choices}"
        ) from exc
