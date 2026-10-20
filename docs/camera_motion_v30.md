# Camera-Induced Motion Decomposition: V3.0 diagnosis and controlled translation cap

**Purpose:** diagnose the first/second robust flow-based motion fits and isolate a suspected translation clipping issue. No new uncertainty weights or Mapping changes. Based on V2.2; set all V1/V2/V2.1/V2.2 enabled=false.

## Enable on Bonn ballon2
Under `model_params` in `configs/rgbd/bonn/config.yaml`:

```yaml
flow_cam_v30:
  enabled: true
  translation_mode: legacy
  save_debug: true
  debug_every: 1
  condition_stride: 8
  save_residual_maps: false
```

`legacy` keeps original scaling; `capped` replaces only translation scale with min(1, max_translation / norm(delta_t)). Both retain the legacy rotation path and original Tracking objective. The diagnostic-only legacy mode still adds timing overhead.

Run:

```bash
python -m unittest discover -s tests -p test_camera_motion_v30.py -v
python slam.py --inherit_from configs/rgbd/bonn/config.yaml --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic --exp_name=v30_legacy_ --save_results=0
python scripts/analyze_camera_motion_v30.py results
```

Use a separate `--exp_name=v30_capped_` for the capped run. For formal comparisons first run: Baseline disabled, V3 legacy diagnostics, V3 capped; use matched seeds and repeat three or more times.

NPZ location: `<Results.save_dir>/tracking/camera_motion_v30_XXXXX.npz`. It includes first/refined twist, 4x4 previous/raw/tracking-initial/tracking-optimized poses, optional GT pose for **offline evaluation only**, step scales, pixel inlier fraction, two flow residual median statistics, and geometry condition number (the condition is of a *column normalized*, spatially sampled design matrix; not the IRLS weighted Hessian). Flow residuals before and after are evaluated with different fits; reductions do **not** prove pose accuracy. The saved optimized pose is after Tracking.

This V3.0 does not guarantee ATE improvement. Legacy output should behave as Baseline apart from logging overhead. We have not completed a full CUDA dataset test.
