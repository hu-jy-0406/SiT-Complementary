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

Do not assume that “two nodes” means one Slurm cluster. The deployment can be:

1. one Slurm cluster controlling two nodes;
2. two independent servers or clusters with a shared filesystem; or
3. two fully isolated servers with separate filesystems.

Shared-storage deployments combine results automatically. Isolated servers
export one verified portable bundle per experiment and merge the two bundles
later on a coordinator. Section 5 classifies the deployment; sections 7–9 give
the matching execution path.

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

## 5. Classify compute control and storage before GPU work

First determine whether both machines belong to one Slurm control plane. One
working `squeue` that lists allocations for both nodes means one cluster. Two
different login hosts/controllers, or no scheduler, means independent
execution endpoints. Do not use a node name from one cluster on another.

Run this lightweight check on every independent endpoint (once on the login
host is enough for a single Slurm cluster):

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
  on that cluster's login node.
- `standalone`: confirm with the owner that each selected machine is dedicated
  and that its eight GPUs are exclusive, then use section 7.
- `undetermined`: stop and ask the owner. An installed Slurm client without a
  working controller is not permission to run locally.

Do not mix standalone and Slurm launchers within one experiment. It is valid
for Conv and Rotation-head to use different execution modes when they run on
two independent systems, provided both still use the fixed eight-rank A100
profile.

Next classify storage independently:

- `shared`: the owner confirms that both compute nodes see the same filesystem,
  or a small probe file written under the proposed path from one node is visible
  at the same absolute path from the other. Use shared mode only after this is
  demonstrated.
- `isolated`: the nodes cannot see each other's files, the absolute paths
  differ, or access cannot be verified. Use the portable-bundle flow in section
  9. Treat uncertainty as `isolated`; training does not need to wait for this
  clarification.

## 6. One-time setup and site inputs

Create or update the environment from the repository root:

```bash
conda env create -f environment.yml   # or update the existing environment
conda activate sit-complementary
```

Obtain local paths from the operator; do not guess. For verified shared storage,
both experiments use the same values:

```bash
export IMAGENET_TRAIN=/absolute/path/to/ILSVRC/Data/CLS-LOC/train
export OUTPUT_ROOT=/shared/durable/path/sit-complementary-a100-run
export ASSET_ROOT=/shared/durable/path/sit-complementary-assets
export GPU_PROFILE=a100-40gb
```

Requirements:

- ImageNet contains exactly 1,000 class directories and 1,281,167 images.
- In shared mode, both nodes see `OUTPUT_ROOT` and `ASSET_ROOT` at the same
  absolute paths.
- In isolated mode, each server uses its own durable `OUTPUT_ROOT` and
  `ASSET_ROOT`; the paths may differ. Never point at a nonexistent “shared” path
  merely to make the strings match.
- Use one new `OUTPUT_ROOT`, then reuse it for every retry of this task.
- Allow at least 100 GiB free per training server. A later bundle coordinator
  needs roughly 2 GiB plus report space.
- Shared mode serializes asset preparation and report generation with file
  locks. Isolated mode has no cross-server writes.
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

## 8. Slurm execution: independent chains

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

On one Slurm cluster these create two independent `afterok` chains, so Slurm
can place one on each 8×A100 node concurrently. If the owner explicitly
supplies node names and the site permits node constraints, bind each chain
separately:

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

If the experiments live on two independent Slurm clusters, run the Conv command
on the Conv cluster's login host and the Rotation-head command on the other
cluster's login host. Configure each cluster separately and omit cross-cluster
`SLURM_NODELIST` values. Use section 9 unless shared storage has actually been
verified across the clusters.

## 9. Isolated-server export and merge

Skip this section when shared storage is verified. On isolated servers, wait
until each selected experiment has completed all evaluations, then export its
bundle locally. On the Conv server:

```bash
python workflow/portable_results.py export \
  --variant conv \
  --output-root "$OUTPUT_ROOT" \
  --asset-root "$ASSET_ROOT" \
  --bundle-dir /durable/transfer/conv-bundle
```

On the Rotation-head server:

```bash
python workflow/portable_results.py export \
  --variant rotation-head \
  --output-root "$OUTPUT_ROOT" \
  --asset-root "$ASSET_ROOT" \
  --bundle-dir /durable/transfer/rotation-head-bundle
```

An export succeeds only after that experiment's complete checkpoint/FID
schedule validates. Each bundle contains its final checkpoint, required raw FID
records, a SHA-256 manifest, and—in the Conv bundle—the pinned historical curve.
It does not copy every intermediate checkpoint. Expect roughly 0.6 GiB per
bundle. A completed bundle is immutable; choose a new directory for a different
run.

Transfer both entire bundle directories to one coordinator using an
operator-approved method such as `rsync -a`, `scp -r`, or offline storage. Do
not transfer a temporary directory whose export has not printed
`PORTABLE_BUNDLE_COMPLETE`. On the coordinator, use a fresh output directory:

```bash
export MERGED_OUTPUT_ROOT=/durable/path/sit-complementary-merged-results
python workflow/portable_results.py merge \
  --conv-bundle /received/conv-bundle \
  --rotation-head-bundle /received/rotation-head-bundle \
  --output-root "$MERGED_OUTPUT_ROOT"
```

The coordinator needs the repository's Conda environment and enough CPU RAM to
inspect checkpoints, but no GPU or ImageNet. The merge command verifies every
bundle hash, requires matching GPU profiles, safely relocates final checkpoint
paths, and invokes the original strict report builder. Never edit FID JSON files
or manifests manually. Success prints `PORTABLE_RESULTS_COMPLETE`; report that
coordinator path as the final output.

## 10. Restart, concurrency, and result behavior

- Checkpoints and FID JSON shards are atomically committed.
- A restart validates the latest checkpoint at or below its target.
- A partial FID sample directory is regenerated, preventing duplicated RNG
  streams. The 50,176 PNGs are removed only after FID is safely recorded.
- `KEEP_FID_SAMPLES=1` is for debugging only; the default saves disk.
- W&B is supplementary; local JSON shards are authoritative.
- In shared mode, asset preparation is locked under `ASSET_ROOT` and report
  generation under `OUTPUT_ROOT`. The two experiments otherwise run
  concurrently.
- In isolated mode, each experiment writes only local storage. Bundle merge is
  restart-safe and refuses conflicting imported files.
- Each completed experiment writes `.experiment_complete-conv` or
  `.experiment_complete-rotation-head`.
- With shared storage, the first finisher prints `JOINT_RESULTS_PENDING`; the
  second prints `JOINT_RESULTS_COMPLETE`. On isolated servers both local runs
  may remain pending until their bundles are merged; this is normal.

In shared mode only, if both experiment markers exist but the report was not
finalized, recover with:

```bash
python workflow/finalize_handoff.py \
  --active-variant conv \
  --output-root "$OUTPUT_ROOT" \
  --asset-root "$ASSET_ROOT" \
  --gpu-profile "$GPU_PROFILE"
```

## 11. Completion and handoff

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

This tree lives under the shared `OUTPUT_ROOT`, or under `MERGED_OUTPUT_ROOT`
for isolated servers. Verify `training_results.json` says `COMPLETE`.
`TRAINING_RESULTS.md` must list
the two final checkpoint paths/hashes, four final CFG=1/CFG=4 FIDs, and both
CFG=1 curves. Report its absolute path, the final checkpoint paths, and the
strict completion status to the operator.
