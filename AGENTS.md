# Agent Instructions

## Execution-environment detection and GPU safety

- This repository may be cloned onto either a shared Slurm cluster or a
  dedicated, non-Slurm GPU server. Never assume one environment from the
  repository layout alone.
- Before any GPU workload, large test, compilation, multiprocessing job, or
  large-directory scan, run the three-state scheduler detection documented in
  `H20_TRAINING_TASK.md`. Merely finding `squeue` or `sbatch` on `PATH` is not
  enough; a Slurm controller query must also succeed.
- If the result is `slurm`, treat the current shell as a login/submission shell
  unless it is inside a legitimate allocation. Keep heavy work off login nodes
  and use the Slurm execution branch in the task contract.
- If the result is `standalone`, first confirm that this is a dedicated server,
  inspect `nvidia-smi`, and obtain exclusive use of the selected eight GPUs
  before using the direct execution branch.
- If the result is `undetermined`, or another scheduler/shared-machine policy
  is present, stop and ask the server owner. Do not fall back to direct GPU use.
- Lightweight editing, Git operations, and small shell checks are allowed
  before the environment is classified.

## 8-GPU A100/H20 Training Handoff

- When asked to run the Conv + Rotation-head handoff, read
  `H20_TRAINING_TASK.md` completely before taking action and treat it as the
  task contract.
- Detect the execution environment first. Use `workflow/run_h20_pipeline.sh`
  only when detection returns `standalone`; use
  `slurm/submit_h20_pipeline.sh` when it returns `slurm`. Do not improvise a
  different training or FID protocol.
- Prefer `GPU_PROFILE=a100-40gb` with exactly eight visible A100 40GB GPUs. The
  project has verified that 40GB is ample, and eight ranks preserve the
  original topology. Do not use all 16 available A100s for one training run.
  Use `GPU_PROFILE=h20` only when the A100 allocation is unavailable.
- On a 16-GPU standalone host, set `CUDA_VISIBLE_DEVICES` to one exclusive set
  of eight devices before preflight. On Slurm, request eight GPUs through the
  site-specific GRES setting.
- The only required site-specific inputs are the ImageNet train path, a durable
  output path, and scheduler parameters when applicable. W&B is optional.
- Never commit, print, or persist `WANDB_API_KEY` or `HF_TOKEN`. Do not run
  `wandb login` on a shared account.
- Completion means that `workflow/build_results.py --strict` passes and the
  generated `training_results/TRAINING_RESULTS.md` contains both final
  checkpoints' CFG=1 and CFG=4 FIDs plus both CFG=1 training curves.
