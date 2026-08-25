# Evaluation Results

## 2026-08-25: Base and Rotation-layer FID Comparison

Lower FID is better. The delta and relative change are calculated as the new
result minus the historical result.

| Variant | CFG | Historical FID | New FID | Delta (new - historical) | Relative change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 1 | 38.870827 | 38.715961 | -0.154866 | -0.40% |
| Base | 4 | 11.042093 | 11.640636 | +0.598544 | +5.42% |
| Rotation-layer | 1 | 40.273842 | 40.323977 | +0.050136 | +0.12% |
| Rotation-layer | 4 | 11.188732 | 11.269768 | +0.081036 | +0.72% |

### Interpretation

- The two rotation-layer results closely match their historical values, with
  absolute differences of only 0.05-0.08 FID.
- Base at CFG 1 also closely matches the historical result and improves by
  approximately 0.155 FID.
- Base at CFG 4 has the largest difference: approximately +0.599 FID, or
  +5.42%. This is the largest observed discrepancy, but this table alone does
  not identify its cause.
- At CFG 4, the historical results favored Base by approximately 0.147 FID,
  while the new results favor Rotation-layer by approximately 0.371 FID.
- The solver/sample-count differences below apply to both Base and
  Rotation-layer. They therefore cannot, by themselves, explain why only the
  Base CFG=4 cell moved by about 0.60.

### Protocol Notes

- Historical results used 250-step Euler sampling, seed 0, and 50,176 PNGs.
- New results used the default `dopri5` ODE solver, seed 0, and exactly 50,000
  PNGs.
- Historical CFG 1 results used the 4,000,000-step checkpoints; new CFG 1
  results used the backed-up 4,003,200-step checkpoints.
- CFG 4 results used the 4,003,200-step checkpoints in both evaluations.
- Both evaluations used PyTorch-FID and
  `evaluation/VIRTUAL_imagenet256_labeled.npz` as the reference statistics.

### Source Records

- Historical summary:
  `pretrained_models/BlueSourceJY-SiT-Complementary/experiments/bs256_lr1e-4/EXPERIMENT_RESULTS.md`
- New raw FID outputs: `evaluation/fid_results/`
- New generated samples and archives: `evaluation/generated_50k/`
