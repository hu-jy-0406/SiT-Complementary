#!/usr/bin/env bash

# Shared constants and cheap completion checks for the direct and Slurm
# launchers.  Full cryptographic verification is performed by prepare_assets.py
# before these ready markers are written.

H20_CONV_TARGETS=(
    2000000 2250000 2500000 2750000 3000000
    3250000 3500000 3750000 4000000 4003200
)
H20_ROTATION_HEAD_TARGETS=(
    250000 500000 750000 1000000 1250000 1500000 1750000 2000000
    2250000 2500000 2750000 3000000 3250000 3500000 3750000 4000000
    4003200
)

h20_file_has_size() {
    local path="$1"
    local expected_size="$2"
    [[ -f "$path" ]] && [[ "$(stat --format='%s' -- "$path")" == "$expected_size" ]]
}

h20_assets_ready() {
    local asset_root="$1"
    [[ -f "$asset_root/.h20-pinned-assets-v1.json" ]] || return 1
    [[ -f "$asset_root/.h20-runtime-weights-v1.json" ]] || return 1
    h20_file_has_size \
        "$asset_root/huggingface/BlueSourceJY/SiT-Complementary/checkpoints/bs256_lr1e-4/conv-layer/1950000.pt" \
        526985776 || return 1
    h20_file_has_size \
        "$asset_root/huggingface/BlueSourceJY/SiT-Complementary/experiments/bs256_lr1e-4/conv-layer/fid_cfg1_50k.tsv" \
        1388 || return 1
    h20_file_has_size \
        "$asset_root/fid/VIRTUAL_imagenet256_labeled.npz" \
        2037122530 || return 1
    h20_file_has_size \
        "$asset_root/vae/sd-vae-ft-ema/config.json" \
        547 || return 1
    h20_file_has_size \
        "$asset_root/vae/sd-vae-ft-ema/diffusion_pytorch_model.safetensors" \
        334643276 || return 1
    h20_file_has_size \
        "$asset_root/cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth" \
        95628359 || return 1
}

h20_stage_complete() {
    local variant="$1"
    local target="$2"
    local output_root="$3"
    local asset_root="$4"
    python workflow/stage_status.py \
        --variant "$variant" \
        --target "$target" \
        --output-root "$output_root" \
        --asset-root "$asset_root" \
        --quiet
}
