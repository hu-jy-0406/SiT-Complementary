# Project Memory

This file records durable project context that should remain available across tasks and sessions.

## Project Overview

- Project: SiT-Complementary
- Purpose: To be documented.

## Environment and Safety

- The repository is hosted on a shared Slurm cluster.
- Keep login-node activity lightweight: editing, Git operations, and small shell commands only.
- Run GPU workloads, large tests, compilation, multiprocessing, and large-directory scans on an allocated compute node.
- Before GPU work, inspect existing allocations with `squeue`, connect to an allocated node, and use `nvidia-smi` to confirm that a GPU is idle.
- Do not interfere with other users or jobs.

## Architecture and Conventions

- To be documented as the project is explored.

## Datasets

- ImageNet root: `/l/users/guangyi.chen/dataset/imagenet/ILSVRC/Data/CLS-LOC/`
- The ImageNet root contains the `train`, `val`, and `test` directories used for model training and evaluation.

## Decisions

- 2026-08-25: Created this file as the durable project-memory record.
- 2026-08-25: Added the portable 8×H20 Conv→Rotation-head handoff specified by
  `H20_TRAINING_TASK.md`. The handoff pins and verifies remote assets, preserves
  global batch 256 on eight ranks, and strictly generates Markdown/JSON/TSV/PNG
  results. On a dedicated node each model trains continuously before its saved
  periodic checkpoints are evaluated; Slurm stages remain restartable.
- 2026-08-25: New Conv/Rotation-head checkpoints save per-rank Python, NumPy,
  CPU Torch, and CUDA RNG state. The H20 workflow uses stateless 50% horizontal
  flips keyed by epoch and sampler position so mid-epoch DataLoader resume does
  not depend on worker prefetch. The legacy Conv step-1,950,000 checkpoint has
  no RNG metadata, so its first H20 continuation cannot reconstruct the old
  A100 random stream.

## Current Work

- The external-server workflow uses the public Hugging Face repository
  `BlueSourceJY/SiT-Complementary`, pinned at revision
  `8b0b8744b28ac0101e5528620b65ab86acb7be52`. Conv resumes from step 1,950,000;
  Rotation-head starts from scratch. Both must finish at step 4,003,200 and
  receive final CFG=1 and CFG=4 FID evaluations.

- Rotation-head training is configured to match the completed Base and
  Rotation-layer protocol: SiT-S/2, ImageNet-256, AdamW at `1e-4`, global batch
  256, 800 epochs / 4,003,200 optimizer steps, checkpoints every 50k, W&B
  logging, CFG=1 FID every 250k, and final CFG=4 FID. The operational guide is
  `docs/ROT_HEAD_TRAINING.md` and the Slurm entry point is
  `slurm/train_rot_head_4xa100_40g.slurm`.
- Rotation-head single-step training passed on an A100 40GB with micro-batch 64
  (10.74 GiB peak). Checkpoint resume plus EMA Euler-ODE inference passed, and
  the gradient-accumulation fallback (micro-batch 32, accumulation 2) passed at
  5.86 GiB peak.
- Rotation-head also passed a two-GPU DDP training step on `gpu-58` with
  micro-batch 64 per GPU, gradient accumulation 2, effective global batch 256,
  and 10.88 GiB peak memory per GPU. The test completed an optimizer step and
  saved a valid step-1 checkpoint without NCCL/DDP errors.

- Conv-layer training is being resumed from
  `pretrained_models/BlueSourceJY-SiT-Complementary/checkpoints/bs256_lr1e-4/conv-layer/1950000.pt`.
- The source checkpoint is at optimizer step 1,950,000 (approximately epoch
  389.7), with AdamW learning rate `1e-4`, global batch size 256, and EMA and
  optimizer state present.
- The resume target is step 4,003,200 (800 epochs).
- On four A100 40GB GPUs, use per-GPU batch 64 and gradient accumulation 1 for
  an effective global batch of 256. A tested fallback is per-GPU batch 32 with
  gradient accumulation 2.
- Stable single-GPU benchmarks measured 11.01 GiB and 128.52 samples/s for
  batch 64, versus 6.12 GiB and 127.32 samples/s for batch 32 with accumulation
  2. Both configurations also passed a two-GPU DDP resume step.
- Slurm job `180083` is the first formal segment (1,950,000 to 2,300,000) and
  is waiting for the shared account's GPU QOS limit. Job `180084` depends on it
  and continues to step 2,650,000. Later segments still need to be submitted as
  job-count quota becomes available.
- W&B authentication must use a process-only `WANDB_API_KEY`; never store the
  key in this repository or run `wandb login` on the shared account.

## Known Issues and Risks

- Changing distributed world size from 8 GPUs to 4 GPUs preserves the effective
  global batch, optimizer logic, and loss scaling, but cannot reproduce the
  exact sample order or random-number trajectory of the original run.
- Some cluster nodes (`gpu-12` and `gpu-24`) exposed unusable CUDA devices to
  Slurm jobs and are excluded from the conv resume scripts.

## Useful Commands

- Add only verified, safe, project-specific commands here.

## Maintenance Notes

- Record durable facts, decisions, constraints, and verified commands.
- Avoid secrets, credentials, transient logs, and speculative conclusions.
- Date important decisions and remove or amend stale information.
