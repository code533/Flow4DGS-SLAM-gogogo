# Flow4DGS-SLAM — V2: conservative flow + RGB-D residual Tracking

This V2 is built on the V1 branch but is **disabled by default**. It is an
experimental, UP-SLAM-inspired multi-cue residual fusion method, **not** a
reproduction of UP-SLAM's DINO features or learned uncertainty estimator.

## Main differences from V1

1. Flow Residual `results['resid_px']` from the ORIGINAL camera-induced motion decomposition is reused. The original flow/YOLO binary motion mask and camera initialization are unchanged.
2. A pixel is softened only if three cues agree: abnormal ego-motion optical-flow residual; observed depth substantially **in front of** the static-map rendered depth; and appearance mismatch. Depth edges are excluded.
3. Soft tracking weights are capped at a low affected fraction and bounded by `min_weight=0.70`; only originally static pixels with reliable map coverage can be modified. These scores are *not calibrated motion probabilities*.
4. Tracking alternates between pose refinement and **detached** confidence updates (`warmup_iters`, `refresh_interval`). An RGB/depth loss rescales by effective confidence mass separately so it reduces to the original loss when all confidence weights are 1.
5. Diagnostics are saved even when no update happened due to early convergence, with an explicit `reason`. The dynamic 4D Mapping receives the original viewpoint.motion_mask: **V2 is Tracking-only**.

## Enable V2

Within `configs/rgbd/bonn/config.yaml` or the TUM equivalent:

```yaml
model_params:
  rgbd_uncertainty:
    enabled: false   # DO NOT enable V1 simultaneously
  flow_rgbd_v2:
    enabled: true
    save_debug: true
    debug_every: 1
```

All remaining parameters are provided in the same YAML. The system raises an
error if both V1 and V2 are active.

## One scene: Bonn ballon2

```bash
python -m unittest discover -s tests -p 'test_flow_rgbd_fusion_v2.py' -v

python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
  --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
  --exp_name=v2_ --save_results=0
```

Find results:

```bash
find results -name 'flow_rgbd_v2_*.npz'
```

Saved in `<Results.save_dir>/tracking/flow_rgbd_v2_XXXXX.npz`, independent of
`--save_results=0`. Fields: `weights`, `joint_score`, `candidate`,
`valid`, `original_static`, `applied`, `reason`, `coverage`,
`effective_ratio`, `update_count`, `last_update_iteration`.

Common reasons: `no_refresh_before_convergence`, `missing_flow`,
`low_coverage`, `no_joint_candidates`, `fraction_budget_zero`, `applied`.
When `applied=False`, the frame falls back to the original Tracking loss.

## Suggested controlled study

Run Baseline (V1 off, V2 off), original V1 (V1 on, V2 off), and V2 (V1 off,
V2 on) on identical dataset and random seed; repeat at least 3 times.
Record ATE RMSE, tracking time and debug reasons, report fraction of frames
with `applied=True`, and candidate/edge overlays. If no frames are applied,
V2 was not effectively evaluated even though it was enabled.

Compare early-convergence frequency before changing thresholds. To trigger
more tests of the diagnostic path, set `save_debug=true` and
`debug_every=1`. Do not report a tracking gain until GPU SLAM benchmarks
have actually been run.

**Known limitations:** strict cue conjunction may miss motion; depth
occlusion reasoning requires an adequate static map; no optical-flow
uncertainty or object identity tracking is modeled. Changes to 4D Mapping
and learned DINO feature uncertainty are not included.
