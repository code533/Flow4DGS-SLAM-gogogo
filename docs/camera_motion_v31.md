# V3.1 Geometry-Aware Robust IRLS for Camera-Induced Motion Decomposition

**Scope**: optical-flow 6DoF motion fitting, not Tracking-loss reweighting. Based
on V3.0. Existing original Flow4DGS-SLAM flow residual/MAD segmentation,
SE(3) motion initialization, and dynamic 4D Gaussian Mapping are preserved.

## Mechanism

The solver shares a Cauchy IRLS weight across the horizontal/vertical
**2D optical-flow residual** at each pixel. Additional non-learning
observation reliability terms are computed from:
- relative depth discontinuity (default reliable weight 0.35 at edges)
- optical-flow local discontinuity (default weight 0.60 at edges)
- spatial grid balancing blended with uniform support

Use a deterministic pixel sampling stride to reduce fitting cost. A
column-normalized weighted 6x6 normal matrix is condition-checked every
iteration. If support is insufficient or the system is ill conditioned,
the calling code **falls back to the original fit_twist_weighted**.
No ground truth and no learned UP-SLAM features enter the solver.

These geometric reliabilities are heuristics, not calibrated uncertainty.
Flow discontinuities can occur at useful static object edges too.

## Configuration

Set all V1/V2/V2.1/V2.2 tracking gates disabled. For a fair comparison,
use `flow_cam_v30.enabled: false` (the unmodified legacy translation rule)
or enable V3.0 with `translation_mode: legacy` to log extra motion diagnostics.

Under `model_params` in `configs/rgbd/bonn/config.yaml`:

```yaml
flow_cam_v31:
  enabled: true
  stage: refit_only     # refit_only | first_only | both
  save_debug: true
  debug_every: 1
```

The complete parameters, including sampling stride, grid controls, and
geometry edge weights, are provided in the YAML.

- `first_only`: replace initial fit used for residual/MAD dynamic mask,
  keep original static inlier re-fit.
- `refit_only`: keep original dynamic mask, replace just camera motion
  re-fit on static candidate pixels; recommended first experiment.
- `both`: replace both fits independently.
- `enabled: false`: exact original camera-motion path.

## Test commands

```bash
python -m unittest discover -s tests -p test_camera_motion_v31.py -v
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
  --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
  --exp_name=v31_refit_ --save_results=0
find results -name 'camera_motion_v31_*.npz'
python scripts/analyze_camera_motion_v31.py results
```

Each diagnostic NPZ is saved under `<Results.save_dir>/tracking/`.
Per-fit statuses show `ok`, `low_points`, `low_grid_coverage`,
`ill_conditioned`, `singular`, or other explicit fallback reasons.

## Required ablation

Run Baseline (V3.1 disabled), refit_only, first_only, both under the *same*
V3.0 translation mode and identical seeds/datasets. Repeat several times;
compare ATE RMSE **and** RPE, fitting failure rates and tracking runtime.
Next isolate geometric components using `depth_edge_weight: 1`,
`flow_edge_weight: 1`, `grid_balance: 0`. Changing these to 1/0
disables their respective influence without changing segmentation logic.

**Limitations:** independent CUDA end-to-end validation and ATE improvement
are not established by CPU synthetic tests. No DINO/UP-SLAM learned
uncertainty or static-map weighting is included yet.
