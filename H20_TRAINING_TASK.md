# Coding-agent task: Conv then Rotation-head on eight A100/H20 GPUs

This file is the execution contract for a coding agent operating a fresh clone
of this repository. The goal is to train and evaluate two SiT-S/2 variants in
order on one node with eight NVIDIA A100 40GB GPUs (preferred), or eight NVIDIA
H20 GPUs when A100s are unavailable:

1. Resume **Conv-layer** from optimizer step 1,950,000 and finish 800 epochs.
2. Train **Rotation-head** from random initialization for 800 epochs.
3. Produce one strict result package containing both CFG=1 training curves and
   final-checkpoint CFG=1 and CFG=4 FIDs for both variants.

Do not declare success until the strict report check passes.

Canonical repository: <https://github.com/hu-jy-0406/SiT-Complementary>

The historical filename `H20_TRAINING_TASK.md` is retained so existing links do
not break; this contract now governs both supported GPU profiles.

## Detect the execution environment first

Before installing dependencies, scanning ImageNet, or running any GPU command,
classify the target machine. Do not infer that Slurm is usable merely because a
client binary is installed. Run this lightweight check from the clone:

```bash
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_CLUSTER_NAME:-}" ]]; then
  EXECUTION_MODE=slurm
elif command -v sbatch >/dev/null 2>&1 && command -v squeue >/dev/null 2>&1; then
  if squeue -h -u "$(id -un)" >/dev/null 2>&1; then
    EXECUTION_MODE=slurm
  else
    EXECUTION_MODE=undetermined
  fi
elif ! command -v sbatch >/dev/null 2>&1 && ! command -v squeue >/dev/null 2>&1; then
  EXECUTION_MODE=standalone
else
  EXECUTION_MODE=undetermined
fi
export EXECUTION_MODE
printf 'EXECUTION_MODE=%s\n' "$EXECUTION_MODE"
```

Interpret the result conservatively:

- `slurm`: use **Execution option B**. Outside an allocation, the current host
  is a login/submission host; do not run preflight, smoke tests, training, or
  FID there. Submit those operations to compute nodes.
- `standalone`: this is only a candidate dedicated server. Confirm with the
  owner that the machine is not governed by another scheduler, inspect
  `nvidia-smi`, and verify exclusive use of the selected eight GPUs. Then use
  **Execution option A**.
- `undetermined`: stop and ask the server owner. This includes a partial Slurm
  installation or an installed client whose controller query failed. Never
  reinterpret this result as permission to run directly.

Make this decision once per server and do not mix the direct and Slurm launch
methods for the same active run. Site-specific paths remain environment
variables; do not edit tracked files to configure the server.

## Select exactly eight GPUs

Prefer eight A100 40GB GPUs and keep the original eight-rank topology:

```bash
export GPU_PROFILE=a100-40gb
export NPROC_PER_NODE=8
export GLOBAL_BATCH_SIZE=256
export GRADIENT_ACCUMULATION_STEPS=1
```

The completed Base and Rotation-layer experiments used eight A100 ranks. This
profile keeps `world_size=8`, per-GPU batch 32, and accumulation 1. Local tests
also showed that A100 40GB has ample memory: even per-GPU batch 64 used only
about 11 GiB for Conv and Rotation-head. Do **not** combine all 16 A100s into a
single 16-rank run; that would reduce the per-GPU batch, increase communication,
and change the saved RNG/data topology.

On a standalone host that physically contains 16 A100s, obtain an exclusive set
of eight from the owner and expose only those devices before preflight, for
example:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Do not guess device IDs or take GPUs used by another process. Slurm users must
request exactly eight A100s through GRES instead of setting device IDs on the
login node. All eight must be on one node; this handoff does not improvise a
multi-node topology. If eight same-node A100s cannot be allocated, the supported
fallback is:

```bash
export GPU_PROFILE=h20
```

Use a new `OUTPUT_ROOT` for each profile. The launcher records
`$OUTPUT_ROOT/.gpu_profile` and refuses to mix A100 and H20 artifacts.

## Non-negotiable experiment protocol

| Setting | Required value |
| --- | --- |
| Model | SiT-S/2, ImageNet 256×256 |
| GPUs | 8 NVIDIA A100 40GB on one node; 8 H20 supported as fallback |
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
or alter the sampler to increase accelerator utilization. Those changes would
make the run incomparable to the completed Base and Rotation-layer experiments.
The selected run preserves the original eight-rank topology and optimizer
semantics. The downloaded Conv checkpoint predates resumable RNG metadata, so
its first continuation cannot reconstruct the old A100 random stream. New
checkpoints save Python, NumPy, CPU Torch, and current-device CUDA RNG state for
every rank.
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
export OUTPUT_ROOT=/absolute/path/to/durable/sit-complementary-a100-runs
export ASSET_ROOT=/absolute/path/to/durable/sit-complementary-assets
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

Run this inside a legitimate Slurm allocation when `EXECUTION_MODE=slurm`, or
on the confirmed dedicated GPU server when `EXECUTION_MODE=standalone`. Never
run it on a shared login node:

For the Slurm branch, `slurm/submit_h20_pipeline.sh` schedules these same checks
inside its first compute job; do not execute the block manually on the login
node. It is shown explicitly so it can be run inside an interactive allocation
for diagnosis. For the standalone branch, run it directly on the dedicated
server before launching the persistent pipeline.

```bash
export SIT_VAE_ROOT="$ASSET_ROOT/vae"
export TORCH_HOME="$ASSET_ROOT/cache/torch"
export HF_HOME="$ASSET_ROOT/cache/huggingface"
python workflow/prepare_assets.py --asset-root "$ASSET_ROOT"
python workflow/prewarm.py --asset-root "$ASSET_ROOT"
python workflow/preflight.py \
  --imagenet-train "$IMAGENET_TRAIN" \
  --output-root "$OUTPUT_ROOT" \
  --gpu-profile "$GPU_PROFILE" \
  --require-clean-git
```

Proceed only after `PREFLIGHT_PASS`. It verifies exactly eight visible GPUs of
the selected profile (including the A100 40GB memory tier), NCCL, ImageNet
contents, dependencies, batch arithmetic, and disk space. Also inspect
`nvidia-smi` and ensure those GPUs are assigned to this job and not used by
another process. The Git worktree must be clean so every FID shard can record
one unambiguous source commit; commit any necessary site adaptation first.

The pipeline then runs `workflow/smoke_h20.sh`: one real 8-rank Conv resume
optimizer step and one real 8-rank Rotation-head optimizer step at the fixed
batch configuration. Long training starts only after the profile-specific
`.gpu_smoke_passed-$GPU_PROFILE` marker is written under `OUTPUT_ROOT`.

## Execution option A: standalone dedicated server

Use this option only when scheduler detection returned `standalone` and the
server owner confirmed exclusive access to the selected eight GPUs. Because the
full workflow runs for multiple weeks, do not leave it attached only to an SSH
or coding-agent session. Use the server's persistent process manager (`systemd`,
`tmux`, or an equivalent facility) when available. A portable `nohup` fallback
is shown below.

Run the idempotent pipeline from an activated environment:

```bash
mkdir -p "$OUTPUT_ROOT"
nohup bash -c \
  'set -o pipefail; bash workflow/run_h20_pipeline.sh 2>&1 | tee -a "$OUTPUT_ROOT/pipeline.log"' \
  >"$OUTPUT_ROOT/launcher.log" 2>&1 </dev/null &
PIPELINE_PID=$!
printf '%s\n' "$PIPELINE_PID" >"$OUTPUT_ROOT/pipeline.pid"
sleep 2
kill -0 "$PIPELINE_PID"
tail -n 20 "$OUTPUT_ROOT/launcher.log"
```

Record the PID and continue monitoring both logs and GPU health; `nohup` keeps
the process independent of a normal shell hangup but is not a substitute for
host-level failure monitoring. Before relaunching, verify that the recorded PID
is no longer running so two pipelines cannot write the same `OUTPUT_ROOT`.

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

## Execution option B: functioning Slurm cluster

Use this option only when scheduler detection returned `slurm`. Run submission
commands from a login/submission host; the generated jobs perform all heavy
preflight, smoke, training, and FID work on allocated compute nodes.

The generic submitter creates an `afterok` chain of 8-GPU stages. Set the
site-specific options that exist on the target cluster:

```bash
export SLURM_PARTITION=your_gpu_partition
# export SLURM_ACCOUNT=your_account
# export SLURM_QOS=your_qos
# export SLURM_TIME=2-00:00:00  # override the one-day stage default if allowed
export GPU_PROFILE=a100-40gb
export SLURM_GRES=gpu:a100:8  # use the exact A100 GRES reported by this site
# H20 fallback: GPU_PROFILE=h20 and SLURM_GRES=gpu:h20:8
bash slurm/submit_h20_pipeline.sh
```

If the cluster exposes only an untyped resource, the owner may specify
`SLURM_GRES=gpu:8`. Never request 16 A100s for this handoff.

The submitter inspects durable checkpoint/evaluation artifacts and queues only
missing stages. Re-run it after a failure to resume the chain. If it must join
an existing external chain, set `SLURM_AFTEROK_JOB_ID` to that numeric job ID.
Only the first newly submitted stage gets `PREPARE_ASSETS=1`, and only when the
pinned asset/runtime ready markers are absent.

Submit from an activated Conda environment with `--export=ALL` support. If the
site limits the number of queued jobs, submit only the next stage manually:

```bash
stage_exports=ALL,GPU_PROFILE=a100-40gb,PIPELINE_VARIANT=conv,\
PIPELINE_TARGET=2000000,PREPARE_ASSETS=1
sbatch --gres=gpu:a100:8 \
  --export="$stage_exports" \
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
- The old Conv step-1,950,000 checkpoint has no RNG state. Its first new segment
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
  --gpu-profile "$GPU_PROFILE" \
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
