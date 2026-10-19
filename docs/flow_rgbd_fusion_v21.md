# Flow4DGS-SLAM V2.1 | Dual-branch flow / RGB-D diagnostics

V2.1 develops the research goal of stabilizing camera Tracking in dynamic
RGB-D scenes. It is **not** the original UP-SLAM DINO/MLP uncertainty system.
Its scores are uncalibrated reliability cues.

## Changes from V2

- `p_flow`: robust motion residual from the original camera-induced
  motion decomposition (the original hard motion mask stays intact).
- `p_rgb`, `p_front`: appearance mismatch and front-surface occlusion
  relative to **static-only** Gaussian rendering.
- `p_map = rgb_share*p_rgb + (1-rgb_share)*p_front`.
- `score_layered = flow_share*p_flow + (1-flow_share)*p_map`.
- In `layered` mode, require **both** a flow prior and a map prior above
  separate minima, followed by a fused score threshold. This avoids
  mistaking an RGB-only residual for motion while relaxing V2's product.
- In `product` mode, use the V2-style three-way product for ablation.
- Coverage, gradient detachment, minimum candidate count, minimum weight,
  and maximum downweighted fraction protect the original Tracking.
- Remains Tracking-only; does not update viewpoint.motion_mask or dynamic
  Mapping. Use `diagnostic_only: true` for a no-impact diagnostic run.

## Enable

On branch `feature/flow-rgbd-fusion-v21`, edit
`configs/rgbd/bonn/config.yaml` under `model_params`:

```yaml
rgbd_uncertainty:
  enabled: false
flow_rgbd_v2:
  enabled: false
flow_rgbd_v21:
  enabled: true
  fusion_mode: layered
  diagnostic_only: true
  save_debug: true
  debug_every: 1
```

Other V2.1 parameters are defined in the same YAML. Enable just one
experiment at a time. To affect Tracking, switch `diagnostic_only: false`.

## Run one scene

```bash
python -m unittest discover -s tests -p test_flow_rgbd_fusion_v21.py -v
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
  --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
  --exp_name=v21_diag_ --save_results=0
find results -name 'flow_rgbd_v21_*.npz'
python scripts/analyze_flow_rgbd_v21.py results
```

## Files

Each qualifying Tracking frame saves a compressed NPZ under its run's
`tracking/` directory with `p_flow`, `p_rgb`, `p_front`, `p_map`,
`score_product`, `score_layered`, `score_selected`,
`candidate_raw`, `candidate`, `weights`, `valid`, `rgb_uint8`,
the prior static mask, candidate counts, and reason.

`count_raw` tells how many pixels passed before the budget and minimum
count checks. `count_selected` is the selected number. `applied=False`
in diagnostic-only mode does not mean no candidates were detected.

If a frame converged before the first confidence refresh, a final diagnostic
snapshot will still be recorded with `no_refresh_before_convergence`, and
weights reset to 1. That snapshot **did not influence Tracking**.

## Limitations and evaluation

The result is a conservative candidate score, not a calibrated dynamic mask;
the flow prior is estimated only on originally-static valid pixels, so there
may be false negatives on motion already masked by the original system.
RGB residual may arise from map quality or illumination; motion residual
alone also cannot prove independent scene motion.

Compare `layered` with `product` in diagnostic-only mode first, then
compare original Baseline / V1 / V2 / V2.1 with identical configurations
and seeds. Assess ATE over several runs and visualize RGB overlay/branch
scores. No GPU SLAM benchmark improvements are claimed.
