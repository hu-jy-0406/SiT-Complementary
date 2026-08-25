# Agent Instructions

## Shared Slurm Cluster Safety

- This repository is hosted on a shared Slurm cluster. Never run GPU workloads, large tests, compilation, multiprocessing, large-directory scans, or other resource-intensive commands on a login node.
- Before performing GPU work, run `squeue` to find nodes already allocated to this account. SSH into an allocated node, check `nvidia-smi`, and use only idle GPUs without interfering with other users or jobs.
- Request new resources with `srun`, `salloc`, or `sbatch` only when necessary.
- Lightweight editing, Git operations, and small shell commands are allowed on login nodes.

## 8xH20 Training Handoff

- When asked to run the Conv + Rotation-head handoff, read
  `H20_TRAINING_TASK.md` completely before taking action and treat it as the
  task contract.
- Use `workflow/run_h20_pipeline.sh` on a dedicated 8-GPU node, or submit the
  restart-safe stages with `slurm/submit_h20_pipeline.sh`. Do not improvise a
  different training or FID protocol.
- The only required site-specific inputs are the ImageNet train path, a durable
  output path, and scheduler parameters when applicable. W&B is optional.
- Never commit, print, or persist `WANDB_API_KEY` or `HF_TOKEN`. Do not run
  `wandb login` on a shared account.
- Completion means that `workflow/build_results.py --strict` passes and the
  generated `training_results/TRAINING_RESULTS.md` contains both final
  checkpoints' CFG=1 and CFG=4 FIDs plus both CFG=1 training curves.
