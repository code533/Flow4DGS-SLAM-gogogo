# V4.0 — Motion / map residual observability (diagnostic-only)

Parent branch: `feature/cimd-real-flow-geometry-v32b`.
Branch: `feature/v40-motion-reliability-diagnostics`.

## Purpose

Before designing new dynamic classification algorithms, quantify how optical-flow
motion residuals relate to RGB-D static-map reconstruction residuals, coverage
and the existing semantic/flow masks. **This is not a new detection method.**
It does not alter camera motion fitting, poses, Tracking loss, Mapping masks,
Gaussian propagation, insertion, or optimizers.

The frontend saves one pre-Tracking snapshot after the original flow-based pose
initialization, and one post-Tracking snapshot. Both contain **static-only**
Gaussian rendering, using exposure-corrected RGB and the original depth.
It uses the existing reverse (current->previous) RAFT flow residual, and
separately records the original prior-static and final-static masks.
IMPORTANT: in this repo `viewpoint.motion_mask=True` means static/keep, NOT dynamic.

`map_covered` is opacity >= threshold with positive finite rendered depth,
**not** a calibrated confidence score. RGB/Depth residual is not ground-truth
motion. Prior-static is YOLO/preexisting prior, not semantic GT.

## Run

```bash
git fetch origin
git switch feature/v40-motion-reliability-diagnostics
python -m unittest tests.test_v40_diagnostics -v
```

Set in `configs/rgbd/bonn/config.yaml` (or the appropriate
inherited base configuration):

```yaml
model_params:
  flow_mask: true
  flow_cam: true
  v40_diagnostics:
    enabled: true
    every: 10
    opacity_min: 0.95
    save_maps: true
```

All V1–V3 optional modifications should be **disabled**, unless explicitly
running a controlled comparison; baseline model controls flow_mask/flow_cam
remain on. Run with your **existing** command for Bonn balloon2 (refer to
`docs/camera_motion_v32b.md` for dataset configuration naming). Use a fresh
results directory, and retain logs and ATE.

Expected outputs:
- `<Results.save_dir>/v40_diagnostics/v40_per_frame.csv`
- `<Results.save_dir>/v40_diagnostics/v40_NNNNN.npz`

Then run:
```bash
python scripts/analyze_v40_diagnostics.py \
  --input <Results.save_dir>/v40_diagnostics
```

Reports `v40_groups.csv`, `v40_summary.txt`, `v40_overview.png`.

## Interpret correctly

- Prior mask: True = static. Flow mask: True = dynamic outlier.
- Pre/post residuals differ partly because the camera pose changes; post
  residual should not be interpreted as GT dynamic evidence.
- `flow_valid` is valid depth AND prior static. Thus dynamic-prior group
  may have **zero** flow-valid pixels, by construction.
- No extra RAFT pass is performed; forward-backward flow reliability is **not**
  available in this initial online mode. `flow_cycle_error_px` is supplied as
  an independently testable helper for a later explicitly enabled replay.
- Bonn camera calibration / depth alignment need geometric validation as
  already described in V3.2b; errors there confound both flow and map residuals.
- ATE changes or extra runtime from logging do not demonstrate better SLAM.
- No ground truth dynamic labels are produced: IoU/F1 cannot be inferred.

## Safety / reproducibility

V4.0 disabled by default. On sampled frames two detached renders and NPZ I/O
add runtime and memory overhead. Use matched baseline configs and disable
the diagnostic for final FPS benchmarks. NPZ maps are stored as float16 to
reduce disk usage; online per-frame CSV uses full-resolution statistics.
