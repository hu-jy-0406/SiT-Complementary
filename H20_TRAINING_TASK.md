# Coding-agent task: Conv then Rotation-head on 8×H20

This file is the execution contract for a coding agent operating a fresh clone
of this repository. The goal is to train and evaluate two SiT-S/2 variants in
order on one node with eight NVIDIA H20 GPUs:

1. Resume **Conv-layer** from optimizer step 1,950,000 and finish 800 epochs.
2. Train **Rotation-head** from random initialization for 800 epochs.
3. Produce one strict result package containing both CFG=1 training curves and
   final-checkpoint CFG=1 and CFG=4 FIDs for both variants.

Do not declare success until the strict report check passes.

Canonical repository: <https://github.com/hu-jy-0406/SiT-Complementary>

## Non-negotiable experiment protocol

| Setting | Required value |
| --- | --- |
| Model | SiT-S/2, ImageNet 256×256 |
| GPUs | 8 NVIDIA H20 on one node |
| Global / per-GPU batch | 256 / 32 |
| Gradient accumulation | 1 |
| Optimizer | AdamW, LR `1e-4`, weight decay 0, default betas |
| Training target | 800 epochs = 4,003,200 optimizer steps |
| Transport | Linear path, velocity prediction |
| VAE | `stabilityai/sd-vae-ft-ema` |
| Seed | 0 |
| Logs / samples / checkpoints | every 100 / 10,000 / 50,000 steps |
| Periodic FID | CFG=1 every 250,000 steps |
| Final FID | CFG=1 and CFG=4 at step 4,003,200 |
| Sampling | Euler ODE, 250 steps, seed 0 |
| Samples | request 50,000; retain the full padded batch of 50,176 PNGs |
| Metric | `pytorch-fid==0.3.0`, batch 128 |

Do not enable AMP/BF16, change global batch, change the LR, add `torch.compile`,
or alter the sampler to increase H20 utilization. Those changes would make the
run incomparable to the completed Base and Rotation-layer experiments. The
H20 run preserves the original eight-rank topology and optimizer semantics.
The downloaded Conv checkpoint predates resumable RNG metadata, so its first
H20 continuation cannot reconstruct the old A100 random stream. New checkpoints
save Python, NumPy, CPU Torch, and current-device CUDA RNG state for every rank.
Horizontal flips remain 50%, but are keyed statelessly by epoch and distributed
sampler position so DataLoader worker scheduling and prefetch cannot change an
augmentation after a mid-epoch restart. CUDA kernel nondeterminism and the
legacy Conv boundary still mean bitwise identity with the A100 run is not
promised.

FID early-stop values from the old experiments may be reported as diagnostics,
but they must **not** stop this task: both variants must reach 800 epochs.

## Trusted downloads

The preparation script pins and verifies every training/evaluation asset before
loading it.

Conv resume checkpoint:

- Hugging Face repository: `BlueSourceJY/SiT-Complementary`
- Browser URL: <https://huggingface.co/BlueSourceJY/SiT-Complementary>
- File: `checkpoints/bs256_lr1e-4/conv-layer/1950000.pt`
- Pinned file URL: <https://huggingface.co/BlueSourceJY/SiT-Complementary/resolve/8b0b8744b28ac0101e5528620b65ab86acb7be52/checkpoints/bs256_lr1e-4/conv-layer/1950000.pt>
- Revision: `8b0b8744b28ac0101e5528620b65ab86acb7be52`
- Size: 526,985,776 bytes
- SHA-256: `f3724fa2651c6fbb3a624664f057c1dd56c658d68fb356c52cf51a7684ae7548`

The same pinned repository supplies Conv's historical CFG=1 FID points through
step 1,750,000. The new workflow evaluates the resume checkpoint at 1,950,000
and all later periodic points, so the final Conv curve covers the full run.

The FID reference is pinned from `SII-PengZheng/discon`, and the EMA VAE is
pinned from `stabilityai/sd-vae-ft-ema`. Their revisions and hashes live in
`workflow/prepare_assets.py`.

## Inputs the operator must provide

Do not guess these paths. Discover them safely or ask the server owner:

```bash
export IMAGENET_TRAIN=/absolute/path/to/ILSVRC/Data/CLS-LOC/train
export OUTPUT_ROOT=/absolute/path/to/durable/sit-complementary-h20-runs
export ASSET_ROOT=/absolute/path/to/durable/sit-complementary-h20-assets
```

`IMAGENET_TRAIN` must be an ImageFolder with exactly 1,000 class directories
and 1,281,167 images. `OUTPUT_ROOT` should have at least 100 GiB free.
Use a new/empty `OUTPUT_ROOT` for the first launch; after that, reuse only that
same directory for retries. Pointing at another experiment could legitimately
cause the restart logic to resume checkpoints already present there.

W&B is optional. If requested, pass credentials only in the process/job
environment:

```bash
export WANDB_API_KEY='provided-out-of-band'
export WANDB_PROJECT=SiT-Complementary
# export WANDB_ENTITY=optional-entity
```

Never put a key in a file and never run `wandb login` on a shared account.

## Environment setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate sit-complementary
```

If the environment already exists, update it rather than creating a duplicate.
The workflow needs outbound Hugging Face/GitHub access during asset preparation.
Compute jobs may run offline after `workflow/prepare_assets.py` and
`workflow/prewarm.py` complete. Both scripts are local-first: when all pinned
files validate, they reuse them without a network request and refresh atomic
ready markers under `$ASSET_ROOT`.

## Required preflight

Run this inside a legitimate allocation or on an otherwise dedicated H20 node,
never on a shared login node:

```bash
export SIT_VAE_ROOT="$ASSET_ROOT/vae"
export TORCH_HOME="$ASSET_ROOT/cache/torch"
export HF_HOME="$ASSET_ROOT/cache/huggingface"
python workflow/prepare_assets.py --asset-root "$ASSET_ROOT"
python workflow/prewarm.py --asset-root "$ASSET_ROOT"
python workflow/preflight.py \
  --imagenet-train "$IMAGENET_TRAIN" \
  --output-root "$OUTPUT_ROOT" \
  --require-clean-git
```

Proceed only after `PREFLIGHT_PASS`. It verifies eight visible H20s, NCCL,
ImageNet contents, dependencies, batch arithmetic, and disk space. Also inspect
`nvidia-smi` and ensure the eight GPUs are assigned to this job and not used by
another process. The Git worktree must be clean so every FID shard can record
one unambiguous source commit; commit any necessary site adaptation first.

The pipeline then runs `workflow/smoke_h20.sh`: one real 8-rank Conv resume
optimizer step and one real 8-rank Rotation-head optimizer step at the fixed
batch configuration. Long training starts only after `.h20_smoke_passed` is
written under `OUTPUT_ROOT`.

## Execution option A: dedicated node/no wall-time limit

Run the idempotent pipeline from an activated environment:

```bash
mkdir -p "$OUTPUT_ROOT"
set -o pipefail
bash workflow/run_h20_pipeline.sh 2>&1 | tee "$OUTPUT_ROOT/pipeline.log"
```

It trains Conv in one continuous process, evaluates all saved Conv checkpoints,
then does the same for Rotation-head. Evaluation therefore does not force
artificial 250,000-step training restarts on a dedicated node. Re-run the same
command after interruption; valid checkpoints and atomic FID shards are reused.
If training itself was interrupted, new checkpoints restore every rank's
logical RNG stream and stateless augmentation position. The post-training
passes are evaluate-only: if an expected checkpoint is absent, the pipeline
fails instead of training a second branch and splicing its FID into the curve.
Completed checkpoint/evaluation stages are skipped on rerun. Set
`PREPARE_ASSETS=0` to require a fully offline run (and fail if any local asset
is missing or has the wrong SHA-256). Every executed stage cryptographically
revalidates all pinned assets before loading a model; ready markers are only a
cheap hint for submission planning. The default `auto` downloads only missing
or invalid assets and makes no network request when every local hash matches.

## Execution option B: Slurm with wall-time limits

The generic submitter creates an `afterok` chain of 8-GPU stages. Set the
site-specific options that exist on the target cluster:

```bash
export SLURM_PARTITION=your_gpu_partition
# export SLURM_ACCOUNT=your_account
# export SLURM_QOS=your_qos
# export SLURM_TIME=2-00:00:00  # override the one-day stage default if allowed
export SLURM_GRES=gpu:h20:8   # use gpu:8 if the cluster has no typed GRES
bash slurm/submit_h20_pipeline.sh
```

The submitter inspects durable checkpoint/evaluation artifacts and queues only
missing stages. Re-run it after a failure to resume the chain. If it must join
an existing external chain, set `SLURM_AFTEROK_JOB_ID` to that numeric job ID.
Only the first newly submitted stage gets `PREPARE_ASSETS=1`, and only when the
pinned asset/runtime ready markers are absent.

Submit from an activated Conda environment with `--export=ALL` support. If the
site limits the number of queued jobs, submit only the next stage manually:

```bash
sbatch --gres=gpu:h20:8 \
  --export=ALL,PIPELINE_VARIANT=conv,PIPELINE_TARGET=2000000,PREPARE_ASSETS=1 \
  slurm/h20_stage.slurm
```

Then advance through the target lists in `workflow/h20_common.sh`, using
`afterok` dependencies. The coding agent must monitor jobs, diagnose failures,
and re-run the submitter after a failed stage; it must not skip a required
evaluation point.

## Restart and storage behavior

- Checkpoints are written through a temporary file and atomically renamed.
- New checkpoints contain per-rank Python, NumPy, CPU Torch, and CUDA RNG state.
  Restoring those streams requires the same eight-rank topology. Stateless
  horizontal flips make resume independent of DataLoader worker prefetch.
- The old Conv step-1,950,000 checkpoint has no RNG state. Its first H20 segment
  is reproducible from the fixed seed but cannot recover the exact preceding
  A100 stream; every checkpoint newly written by this workflow is restartable.
- A restarted stage scans and validates the latest checkpoint below its target.
- A partial FID sample directory is never resumed; that evaluation is fully
  regenerated so RNG streams cannot duplicate earlier samples.
- The 50,176 PNGs are deleted only after FID is parsed and an atomic JSON result
  is committed. Set `KEEP_FID_SAMPLES=1` only for debugging.
- The unnecessary ~9.8 GB generated-sample NPZ is disabled.
- W&B is supplementary; local JSON shards are the source of truth.

## Required deliverables and completion check

The final job must leave this structure under `$OUTPUT_ROOT/training_results`:

```text
training_results/
├── TRAINING_RESULTS.md
├── training_results.json
├── fid_results.tsv
├── fid_cfg1_training_curves.png
├── conv_fid_cfg1_curve.png
├── rotation_head_fid_cfg1_curve.png
└── raw/
    ├── conv/*.json
    └── rotation-head/*.json
```

Run the strict check explicitly if needed:

```bash
python workflow/build_results.py \
  --output-dir "$OUTPUT_ROOT/training_results" \
  --conv-history "$ASSET_ROOT/huggingface/BlueSourceJY/SiT-Complementary/experiments/bs256_lr1e-4/conv-layer/fid_cfg1_50k.tsv" \
  --strict
```

Success requires:

- Conv and Rotation-head final checkpoints are both step 4,003,200.
- Their persisted lineage proves Conv originated from the pinned step-1,950,000
  SHA-256 and Rotation-head originated from scratch.
- Both final checkpoints have CFG=1 and CFG=4 FIDs.
- All required periodic CFG=1 points exist.
- `TRAINING_RESULTS.md` embeds both training curves and lists the four final
  FIDs and final checkpoint paths/hashes.
- The strict command exits zero and the report status is `COMPLETE`.

Finally, report the absolute path to `TRAINING_RESULTS.md`, the two final
checkpoint paths, and the strict-check outcome to the server owner.
