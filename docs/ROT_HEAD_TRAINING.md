# Rotation-head training protocol

The Rotation-head run uses the same protocol as the completed Base and
Rotation-layer runs. Only the model implementation changes to
`models_rot_head.py`.

## Fixed configuration

| Setting | Value |
| --- | --- |
| Model | `SiT-S/2`, Rotation-head |
| Dataset | ImageNet-1K, 256 px |
| Optimizer | AdamW, LR `1e-4`, weight decay `0`, default betas |
| Global batch | 256 |
| Duration | 800 epochs / 4,003,200 optimizer steps |
| Transport | Linear path, velocity prediction |
| VAE | SD VAE EMA |
| Seed | 0 |
| Train logging | every 100 optimizer steps |
| Sample logging | every 10,000 optimizer steps, CFG 4 |
| Checkpoints | every 50,000 optimizer steps and at segment end |
| Periodic FID | every 250,000 steps, CFG 1 |
| Final FID | step 4,003,200, CFG 4 |
| FID sampler | ODE Euler, 250 steps, seed 0 |
| FID images | request 50,000; retain 50,176 distributed PNGs |
| FID implementation | PyTorch-FID, batch 128 |
| FID reference | `evaluation/VIRTUAL_imagenet256_labeled.npz` |
| FID early stop | 3 consecutive significant rises |

A rise is significant when it is at least both the effective threshold
`max(0.25, 0.005 * previous_fid)`. The evaluator records a durable
`EARLY_STOP` marker after three consecutive significant rises; later segment
jobs see the marker and finish without starting another training segment. These
values are taken directly from the completed Base checkpoint arguments.

On one four-GPU node, gradient accumulation 1 gives a micro-batch of 64 per
GPU and preserves the effective global batch of 256. If Rotation-head does not
fit after the smoke test, set `GRADIENT_ACCUMULATION_STEPS=2`; this changes the
micro-batch to 32 without changing optimizer-step semantics.

## Running in Slurm segments

The first segment starts from scratch and targets step 250,000:

```bash
sbatch --export=ALL,WANDB_API_KEY,SEGMENT_TARGET=250000 \
  slurm/train_rot_head_4xa100_40g.slurm
```

Continue from the checkpoint produced by the preceding segment:

```bash
sbatch --export=ALL,WANDB_API_KEY,SEGMENT_TARGET=500000,\
RESUME_CHECKPOINT=results-rot-head-4xa100-40g/SiT-S-2-RotationHead-bs256-lr1e-4-800ep/checkpoints/0250000.pt \
  slurm/train_rot_head_4xa100_40g.slurm
```

Use targets `250000, 500000, ..., 4000000, 4003200`. Each 250k boundary runs
the historical CFG=1 FID automatically; the final target also runs CFG=4 FID.
Submit the next segment with an `afterok` dependency or after confirming the
previous checkpoint and FID completed.

W&B uses `WANDB_API_KEY` only from the job environment. The scripts never call
`wandb login` and never persist the key.
