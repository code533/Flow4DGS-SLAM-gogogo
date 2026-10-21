# V3.1a Same-frame Shadow Evaluation

Goal: compare original IRLS and V3.1 Geometry-Aware IRLS on identical RGB-D,
optical flow and original static mask **without changing Tracking or Mapping**.

## What is compared

- Original IRLS actually used by SLAM: `xi_legacy_actual`.
- Original IRLS recomputed on a **common depth-restricted input mask**:
  `xi_legacy_common`.
- V3.1 IRLS on that exact common input mask: `xi_geo`. Note V3.1
  intentionally has an internal stride/grid sampling method.
- Dense static-pixel and *unsampled* (stride holdout) residuals for each fit.
- Two hypothetical initial world-to-camera poses, each applying the same
  SE(3) exponential and the original translation/rotation clipping.
- Current/previous GT matrices saved for **offline** relative-pose metrics.

Only `xi_legacy_actual` drives the real SLAM pose. Neither `xi_geo`
nor the hypothetical poses are ever applied to Tracking. GT is never passed
to either motion estimator.

## Configure

On branch `feature/cimd-shadow-evaluation-v31a`, edit
`configs/rgbd/bonn/config.yaml` under `model_params`:

```yaml
flow_cam_v30:
  enabled: false
flow_cam_v31:
  enabled: false
flow_cam_v31a:
  enabled: true
  save_debug: true
  debug_every: 1
  sampling_stride: 4
```

Disable V1-V2.2 Tracking residual experiments. Other V3.1a parameters
already have defaults copied from V3.1.

```bash
python -m unittest discover -s tests -p test_camera_motion_v31a.py -v
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
  --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
  --exp_name=v31a_shadow_ --save_results=0
find results -name 'camera_motion_v31a_*.npz'
python scripts/analyze_camera_motion_v31a.py results/bonn/v31a_shadow_/YOUR_RUN_DIR \
  --csv results/v31a_per_frame.csv
```

Logs should show `[CIMD-V3.1a] frame=.. status=ok ...`.
The diagnostic NPZs live under `<Results.save_dir>/tracking/`.

Do not interpret an extra diagnostic solve as improving ATE. Check paired
translation/rotation metrics and held-out flow RMSE. Frame-to-frame GT
comparison assumes correct synchronized world-to-camera pose conventions.
The held-out subset excludes V3.1's deterministic stride grid but is not
independent of the original full-pixel IRLS fit. CPU tests do not establish
CUDA end-to-end correctness.
