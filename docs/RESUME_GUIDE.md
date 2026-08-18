# Recreate and resume the backed-up SiT experiments

This guide restores the base, rotation-layer, or convolution-layer experiment on
a new GPU server. It assumes that the Hugging Face account has accepted the
ImageNet-1K access conditions and can read `BlueSourceJY/SiT-Complementary`.

## 1. Create the environment and obtain the code

```bash
git clone https://github.com/hu-jy-0406/SiT-Complementary.git
cd SiT-Complementary
conda env create -f environment.yml
conda activate SiT
pip install pyarrow pytorch-fid huggingface_hub
huggingface-cli login
```

Install a CUDA/PyTorch build appropriate for the new server if the environment
file's default build does not match its driver. Eight visible GPUs are assumed by
the commands below.

## 2. Download ImageNet-1K and convert it to ImageFolder

The gated Hugging Face ImageNet export is parquet. Keep that download separate
from the ImageFolder output (the latter is what training uses).

```bash
export DATA_ROOT=/data/imagenet-1k
huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset \
  --local-dir "${DATA_ROOT}-hf" --include 'data/*.parquet' 'classes.py' 'README.md'
python tools/convert_imagenet_parquet_to_imagefolder.py \
  "${DATA_ROOT}-hf" "$DATA_ROOT" "${DATA_ROOT}-hf/classes.py"
find "$DATA_ROOT/train" -mindepth 1 -maxdepth 1 -type d | wc -l  # must be 1000
```

The training `--data-path` is `"$DATA_ROOT/train"`. The converter also creates
`validation/`, `class_to_idx.json`, and `conversion_summary.json` under
`$DATA_ROOT`.

## 3. Download the experiment backup and FID reference

```bash
export EXP_ROOT=/data/sit-backup
huggingface-cli download BlueSourceJY/SiT-Complementary \
  --local-dir "$EXP_ROOT" \
  --include 'checkpoints/bs256_lr1e-4/*' 'experiments/bs256_lr1e-4/*'

mkdir -p /data/evaluation/reference/discon-download
huggingface-cli download SII-PengZheng/discon \
  VIRTUAL_imagenet256_labeled.npz \
  --local-dir /data/evaluation/reference/discon-download
```

Before resuming, compare each downloaded file's SHA-256 with
`$EXP_ROOT/experiments/bs256_lr1e-4/validation/checkpoint_validation.json`. The training
checkpoint format contains `model`, `ema`, `opt`, and `args`; resume restores all
four, including optimizer state.

## 4. One-image inference check

Use the model implementation that matches the checkpoint. `learn_sigma=True` is
intentional for these checkpoints; the model returns the velocity half internally.

```bash
# Base (replace the checkpoint path for the other variants)
torchrun --nproc_per_node=1 sample_ddp.py ODE \
  --model SiT-S/2 --image-size 256 \
  --ckpt "$EXP_ROOT/checkpoints/bs256_lr1e-4/base/4003200.pt" \
  --num-fid-samples 1 --per-proc-batch-size 1 --num-sampling-steps 10 \
  --sample-dir /data/sit-smoke-test

# Rotation layer: prefix the same command with:
# SIT_MODEL_MODULE=models_rot_layer SIT_EXPECTED_MODEL_MODULE=models_rot_layer
# Conv layer: prefix the same command with:
# SIT_MODEL_MODULE=models_conv SIT_EXPECTED_MODEL_MODULE=models_conv
```

Inspect the generated PNG before spending GPU time. For a repeatable FID run,
use eight processes and the archived settings: `--num-fid-samples 50000`,
`--per-proc-batch-size 64`, `--num-sampling-steps 250`, and the required CFG.

## 5. Resume training

Set W&B variables if online logging is wanted; for offline logging set
`WANDB_MODE=offline`, then run `wandb sync <offline-run-dir>` later. Use a new
`--results-dir` on the new server, retain the variant-specific `--run-name`, and
pass a target total epoch count greater than the saved progress. Do not change
the model module, global batch size, learning rate, seed, or data preprocessing.

The portable launchers read paths and optional features from environment
variables. For example, to resume Rotation-layer without editing the script:

```bash
DATA_PATH="$DATA_ROOT/train" \
RESULTS_DIR=/data/sit-results/rotation-layer \
CKPT="$EXP_ROOT/checkpoints/bs256_lr1e-4/rotation-layer/4003200.pt" \
EPOCHS=1000 ENABLE_WANDB=1 ENABLE_FID=1 FID_REFERENCE="$REF" \
./run_train_rot_layer.sh
```

Use `run_train.sh` for Base and `run_train_conv.sh` for Conv-layer. Set
`ENABLE_WANDB=0` or `ENABLE_FID=0` to disable either optional feature. Any
extra command-line arguments are forwarded to the Python trainer and can
override launcher defaults.

```bash
export DATA_ROOT=/data/imagenet-1k
export EXP_ROOT=/data/sit-backup
export REF=/data/evaluation/reference/discon-download/VIRTUAL_imagenet256_labeled.npz
export ENTITY='<wandb entity>' PROJECT='<wandb project>' WANDB_KEY='<wandb key>'

# Base: continue 800-epoch final checkpoint to 1,000 total epochs.
torchrun --nnodes=1 --nproc_per_node=8 train.py \
  --model SiT-S/2 --epochs 1000 --data-path "$DATA_ROOT/train" \
  --results-dir /data/sit-results/base --global-batch-size 256 \
  --learning-rate 0.0001 --global-seed 0 --vae ema --num-workers 4 \
  --log-every 100 --ckpt-every 50000 --sample-every 10000 --cfg-scale 4.0 \
  --run-name SiT-S-2-Linear-velocity-None \
  --ckpt "$EXP_ROOT/checkpoints/bs256_lr1e-4/base/4003200.pt" \
  --fid-every-checkpoint --fid-every 250000 --fid-num-samples 50000 \
  --fid-reference "$REF" --fid-per-proc-batch-size 64 \
  --fid-inception-batch-size 128 --fid-num-workers 8 --fid-sampling-steps 250 \
  --fid-seed 0 --fid-stop-consecutive-increases 3 \
  --fid-stop-min-absolute-rise 0.25 --fid-stop-min-relative-rise 0.005 --wandb
```

For rotation-layer replace `train.py` with `train_rot_layer.py`, use the
`rotation-layer/4003200.pt` checkpoint, and choose an isolated results directory.
For conv-layer use `train_conv.py` and `conv-layer/1950000.pt`. Those wrappers
select `models_rot_layer.py` and `models_conv.py` before importing the shared
trainer, so never substitute plain `train.py` for them.

The trainer derives the current step from the numeric checkpoint filename and
skips consumed batches in the first resumed epoch. It will first establish or
reuse the matching CFG=1 FID baseline, then records each periodic result in
`fid_cfg1_50k.tsv`. It stops only after the configured sustained, material FID
regression rule is met.
