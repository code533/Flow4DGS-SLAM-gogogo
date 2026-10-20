# V2.2: RGB / Depth decoupled tracking (research prototype)

Feature branch: `feature/flow-rgbd-decoupled-v22`, based on V2.1. V1, V2 and V2.1 must all be **disabled** when V2.2 is enabled.

## Principle

Reuse detached V2.1 optical-flow, RGB residual and front-depth scores, but generate independent reliability maps:

- `S_rgb = p_flow * p_rgb` (flow + photometric disagreement).
- `S_depth = p_flow * p_front` (flow + front depth occlusion).
- Independent minimum scores, minimum candidate numbers and top-k budgets.
- `W_rgb` downweights RGB tracking residual only; `W_depth` downweights depth tracking residual only.
- Keep original camera-induced motion decomposition, original static hard mask, opacity and gradient masks, Mapping, and pose initialization.

**This is not UP-SLAM's learned DINO/MLP uncertainty estimator**. Scores are heuristic.

## Start with loss control experiment

In `configs/rgbd/bonn/config.yaml`, under `model_params`, set V1/V2/V2.1 `enabled: false` and V2.2:

```yaml
flow_rgbd_v22:
  enabled: true
  mode: all_ones
  diagnostic_only: false
  normalization: mass_preserving
  save_debug: true
  debug_every: 1
```

`all_ones` uses the new loss with unit weights in every Tracking iteration, and hence should reproduce Baseline within run-to-run variability. Compare with disabled baseline first! Next test `depth_only`, `rgb_only`, then `decoupled` with the same other parameters.

`mass_preserving` normalizes RGB and depth by their *own* effective confidence mass; `original_mean` maintains unnormalized original means. Both match native Tracking for unit weights.

## Run and inspect

```bash
python -m unittest discover -s tests -p 'test_flow_rgbd_fusion_v22.py' -v
python slam.py --inherit_from configs/rgbd/bonn/config.yaml --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic --exp_name=v22_all_ones_ --save_results=0
find results -name 'flow_rgbd_v22_*.npz'
python scripts/analyze_flow_rgbd_v22.py results
```

Archives in `<Results.save_dir>/tracking/flow_rgbd_v22_XXXXX.npz` contain independent weight maps, score maps, candidates, count diagnostics and a fallback reason. The saved gate represents only the most recent refresh, not average across Tracking iterations. The early-convergence case also saves a no-refresh record.

Unit tests are CPU-only. Full CUDA integration and ATE improvements are **unverified**.
