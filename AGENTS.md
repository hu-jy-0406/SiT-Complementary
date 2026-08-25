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

## Conv / Rotation-head Training Handoff

- When asked to run Conv, Rotation-head, or both, read
  `H20_TRAINING_TASK.md` completely and treat it as the execution contract.
- Obey the experiment named in the prompt: `conv` resumes the pinned
  checkpoint; `rotation-head` starts from scratch. Do not start the other
  experiment unless the prompt requests both.
- Detect the execution environment first. Standalone runs use
  `workflow/run_gpu_experiment.sh {conv|rotation-head}`; Slurm runs use
  `EXPERIMENT={conv|rotation-head} bash slurm/submit_h20_pipeline.sh`.
- Do not assume that two machines belong to one Slurm cluster. Classify each
  independent execution endpoint and separately verify whether storage is
  shared, following the decision tree in `H20_TRAINING_TASK.md`.
- Prefer `GPU_PROFILE=a100-40gb` with exactly eight visible A100 40GB GPUs. The
  project has verified that 40GB is ample, and eight ranks preserve the
  original topology. With two 8×A100 nodes, run Conv on one node and
  Rotation-head on the other concurrently; never combine them into one
  16-rank job. Use H20 only when A100 is unavailable.
- With verified shared storage, both nodes use identical absolute
  `OUTPUT_ROOT` and `ASSET_ROOT`. With isolated storage, use local durable paths
  and export one bundle per experiment with `workflow/portable_results.py`;
  transfer and merge both verified bundles on a coordinator. Never fabricate a
  shared path or manually edit FID records.
- Never run two copies of the same experiment against one output root.
- Never commit, print, or persist `WANDB_API_KEY` or `HF_TOKEN`. Do not run
  `wandb login` on a shared account.
- A single experiment is done when its experiment-complete marker exists. The
  joint handoff—whether shared or bundle-merged—is done only when
  `training_results.json` says `COMPLETE` and `TRAINING_RESULTS.md` contains
  both CFG=1 curves and all four final FIDs.
