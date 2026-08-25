# Coding-agent contract: Conv and Rotation-head training

This is the complete execution contract for a coding agent working from a
fresh clone of <https://github.com/hu-jy-0406/SiT-Complementary>. Read it before
running commands. The historical filename is retained for existing links; the
preferred hardware is now **two nodes with 8×A100 40GB each**.

## 1. Interpret the prompt literally

The operator will name the experiment to run. Do exactly that experiment:

| Prompt intent | Experiment selector | Action |
| --- | --- | --- |
| Run Conv | `conv` | Resume Conv from the pinned step-1,950,000 checkpoint |
| Run Rotation-head / rot-head | `rotation-head` | Train Rotation-head from scratch |
| Run both | both selectors | Launch one selector on each of two separate 8-GPU nodes |

Do not start the other experiment when the prompt names only one. If the prompt
says “run both”, run them concurrently on two nodes—not sequentially on one
node and not as one 16-rank distributed job. Each experiment always uses one
node and exactly eight GPUs.

The two runs are independent during training but must use the same absolute
`OUTPUT_ROOT` and `ASSET_ROOT` on storage visible from both nodes. The first run
to finish writes a valid partial report; the second automatically creates the
strict combined report. If shared storage is unavailable, stop and ask the
operator rather than inventing a file-copy workflow.

## 2. Required outcome

- Conv resumes from optimizer step 1,950,000 and reaches 800 epochs, step
  4,003,200.
- Rotation-head starts from random initialization and reaches the same target.
- CFG=1 FID is measured every 250,000 steps; each final checkpoint receives
  CFG=1 and CFG=4 FID.
- The combined result contains both CFG=1 curves and all four final FIDs.
- `training_results/training_results.json` has status `COMPLETE`.

Do not use an old FID threshold as early stopping and do not declare the joint
task complete while the result status is `INCOMPLETE`.

## 3. Fixed experiment protocol

| Setting | Required value |
| --- | --- |
| Model/data | SiT-S/2, ImageNet 256×256 |
| Topology | 1 node × 8 ranks per experiment |
| Preferred GPU | 8×NVIDIA A100 40GB |
| Fallback GPU | 8×NVIDIA H20 |
| Global/per-GPU batch | 256/32 |
| Gradient accumulation | 1 |
| Optimizer | AdamW, LR `1e-4`, weight decay 0, default betas |
| Training target | 800 epochs = 4,003,200 optimizer steps |
| Seed/transport | 0; Linear path, velocity prediction |
| VAE | `stabilityai/sd-vae-ft-ema` |
| Log/sample/checkpoint interval | 100/10,000/50,000 steps |
| Periodic evaluation | CFG=1 every 250,000 steps |
| Final evaluation | CFG=1 and CFG=4 at step 4,003,200 |
| FID sampling | Euler ODE, 250 steps, seed 0 |
| FID sample count | request 50,000; retain padded batch of 50,176 PNGs |
| Metric | `pytorch-fid==0.3.0`, batch 128 |

Keep `NPROC_PER_NODE=8`, `GLOBAL_BATCH_SIZE=256`, and
`GRADIENT_ACCUMULATION_STEPS=1`. Do not enable AMP/BF16, `torch.compile`, alter
the sampler, change the LR or batch, or combine both A100 nodes into a 16-rank
run. Eight ranks reproduce the completed Base/Rotation-layer topology. A100
40GB is sufficient; verified Conv and Rotation-head batch-64 tests used about
11 GiB per GPU.

New checkpoints preserve per-rank Python, NumPy, CPU Torch, and CUDA RNG state.
Horizontal flips are statelessly keyed by epoch and distributed sample
position. The legacy Conv checkpoint has no resumable RNG metadata, so the
first continuation is reproducible from seed 0 but cannot reconstruct its old
A100 RNG stream; bitwise identity is not promised.

## 4. Pinned Conv source and other assets

Conv source repository: `BlueSourceJY/SiT-Complementary`

- Browser: <https://huggingface.co/BlueSourceJY/SiT-Complementary>
- File: `checkpoints/bs256_lr1e-4/conv-layer/1950000.pt`
- Revision: `8b0b8744b28ac0101e5528620b65ab86acb7be52`
- Size: 526,985,776 bytes
- SHA-256: `f3724fa2651c6fbb3a624664f057c1dd56c658d68fb356c52cf51a7684ae7548`

`workflow/prepare_assets.py` pins and verifies that checkpoint, the historical
Conv CFG=1 points through step 1,750,000, the ImageNet FID reference, and the
VAE. The workflow additionally evaluates the resume checkpoint at step
1,950,000. Never replace a failed verified download with an unpinned file.

## 5. Discover the server mode before GPU work

Run this lightweight check from the clone:

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

- `slurm`: submit with section 8. Never run preflight, smoke, training, or FID
  on the login node.
- `standalone`: confirm with the owner that each selected machine is dedicated
  and that its eight GPUs are exclusive, then use section 7.
- `undetermined`: stop and ask the owner. An installed Slurm client without a
  working controller is not permission to run locally.

Do not mix standalone and Slurm launchers within one experiment.

## 6. One-time setup and site inputs

Create or update the environment from the repository root:

```bash
conda env create -f environment.yml   # or update the existing environment
conda activate sit-complementary
```

Obtain these paths from the operator; do not guess:

```bash
export IMAGENET_TRAIN=/absolute/path/to/ILSVRC/Data/CLS-LOC/train
export OUTPUT_ROOT=/shared/durable/path/sit-complementary-a100-run
export ASSET_ROOT=/shared/durable/path/sit-complementary-assets
export GPU_PROFILE=a100-40gb
```

Requirements:

- ImageNet contains exactly 1,000 class directories and 1,281,167 images.
- Both nodes see `OUTPUT_ROOT` and `ASSET_ROOT` at the same absolute paths.
- Use one new `OUTPUT_ROOT`, then reuse it for every retry of this task.
- Allow at least 100 GiB free. The workflow serializes shared asset preparation
  and report generation with file locks.
- One `OUTPUT_ROOT` may not mix A100 and H20 runs; `.gpu_profile` enforces this.
- Never launch two copies of the same experiment into the same output root.

W&B is optional. If requested, put credentials only in the process/job
environment; never commit them and never run `wandb login` on a shared account:

```bash
export WANDB_API_KEY='provided-out-of-band'
export WANDB_PROJECT=SiT-Complementary
# export WANDB_ENTITY=optional-entity
```

The launchers prepare pinned assets, validate exactly eight visible GPUs,
ImageNet, dependencies, NCCL, batch arithmetic, disk, and a clean Git tree,
then run one real eight-rank smoke optimizer step for the selected model. Long
training starts only after `PREFLIGHT_PASS` and its model-specific smoke marker.

## 7. Standalone execution: one node per agent

Use only on confirmed dedicated servers. Expose exactly the eight GPUs assigned
to that node; do not guess IDs or take devices used by another process:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nvidia-smi
```

For a prompt selecting Conv:

```bash
mkdir -p "$OUTPUT_ROOT"
nohup bash -c \
  'set -o pipefail; bash workflow/run_gpu_experiment.sh conv 2>&1 | tee -a "$OUTPUT_ROOT/conv.log"' \
  >"$OUTPUT_ROOT/conv-launcher.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$OUTPUT_ROOT/conv.pid"
```

For a prompt selecting Rotation-head, on the other node:

```bash
mkdir -p "$OUTPUT_ROOT"
nohup bash -c \
  'set -o pipefail; bash workflow/run_gpu_experiment.sh rotation-head 2>&1 | tee -a "$OUTPUT_ROOT/rotation-head.log"' \
  >"$OUTPUT_ROOT/rotation-head-launcher.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$OUTPUT_ROOT/rotation-head.pid"
```

Use `tmux`, `systemd`, or the site's persistent supervisor instead of `nohup`
when available. Record and monitor the PID, logs, disk, and GPU health. Before a
retry, confirm the old PID is dead. Re-run the same selector: valid atomic
checkpoints and FID shards are reused. Training runs continuously to the final
step before saved periodic checkpoints are evaluated.

`workflow/run_h20_pipeline.sh` remains a one-node sequential compatibility
entry point, but do not use it when the requested two-node layout is available.

## 8. Slurm execution: independent chains on two nodes

Run only lightweight submission commands on the login node. Configure the
site's valid settings:

```bash
export SLURM_PARTITION=your_gpu_partition
# export SLURM_ACCOUNT=your_account
# export SLURM_QOS=your_qos
# export SLURM_TIME=2-00:00:00
export SLURM_GRES=gpu:a100:8
```

For the Conv agent/prompt:

```bash
EXPERIMENT=conv bash slurm/submit_h20_pipeline.sh
```

For the Rotation-head agent/prompt:

```bash
EXPERIMENT=rotation-head bash slurm/submit_h20_pipeline.sh
```

These create two independent `afterok` chains, so Slurm can place one on each
8×A100 node concurrently. If the owner explicitly supplies node names and the
site permits node constraints, bind each chain separately:

```bash
EXPERIMENT=conv SLURM_NODELIST=a100-node-01 bash slurm/submit_h20_pipeline.sh
EXPERIMENT=rotation-head SLURM_NODELIST=a100-node-02 bash slurm/submit_h20_pipeline.sh
```

Do not invent node names. If the cluster uses untyped GRES, the owner may set
`SLURM_GRES=gpu:8`. H20 fallback requires both `GPU_PROFILE=h20` and the site's
H20 GRES. The submitter skips complete stages and can be re-run after a failed
job. `EXPERIMENT` is required; use `all` only for the legacy sequential chain.
`SLURM_AFTEROK_JOB_ID` may attach a selected chain to an existing numeric job
dependency. Monitor the jobs and resubmit only the affected selector.

## 9. Restart, concurrency, and result behavior

- Checkpoints and FID JSON shards are atomically committed.
- A restart validates the latest checkpoint at or below its target.
- A partial FID sample directory is regenerated, preventing duplicated RNG
  streams. The 50,176 PNGs are removed only after FID is safely recorded.
- `KEEP_FID_SAMPLES=1` is for debugging only; the default saves disk.
- W&B is supplementary; local JSON shards are authoritative.
- Asset preparation is locked under `ASSET_ROOT`; report generation is locked
  under `OUTPUT_ROOT`. The two experiments may otherwise run concurrently.
- Each completed experiment writes `.experiment_complete-conv` or
  `.experiment_complete-rotation-head`.
- The first finisher prints `JOINT_RESULTS_PENDING`; this is normal. The second
  prints `JOINT_RESULTS_COMPLETE` after the strict report passes.

If both experiment markers exist but the report was not finalized (for example
the second node stopped immediately after evaluation), recover with:

```bash
python workflow/finalize_handoff.py \
  --active-variant conv \
  --output-root "$OUTPUT_ROOT" \
  --asset-root "$ASSET_ROOT" \
  --gpu-profile "$GPU_PROFILE"
```

## 10. Completion and handoff

The final tree must include:

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

Verify `training_results.json` says `COMPLETE`. `TRAINING_RESULTS.md` must list
the two final checkpoint paths/hashes, four final CFG=1/CFG=4 FIDs, and both
CFG=1 curves. Report its absolute path, the final checkpoint paths, and the
strict completion status to the operator.
