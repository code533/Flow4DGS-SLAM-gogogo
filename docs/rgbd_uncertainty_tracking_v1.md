# Flow4DGS-SLAM — RGB-D Uncertainty Fusion (Tracking V1)

This experimental implementation keeps the official **Camera-Induced Motion Decomposition**, RAFT, YOLO prior, camera pose initialization, and dynamic 4D Mapping unchanged. It adds *a detached RGB-D residual gate* to down-weight unreliable pixels during **static-Gaussian Tracking**. Inspired by UP-SLAM's RGB-D multi-modal inconsistency cue; this is **not** UP-SLAM's DINO/MLP uncertainty predictor.

## Modified files
- `utils/slam_frontend.py`: compute the gate once after `warmup_iters` pose refinement iterations, using the existing static Gaussian RGB/depth/opacity render, and feed its floating-point weights through the existing Tracking loss `mask` argument. Does not overwrite `viewpoint.motion_mask`.
- `utils/uncertainty_fusion.py`: torch-only residual and coverage-aware confidence computation.
- `configs/rgbd/{bonn,tum}/config.yaml`: feature flag and parameters (OFF by default).
- `tests/test_rgbd_uncertainty.py`: fast CPU tests.

**Important:** in Flow4DGS-SLAM, `viewpoint.motion_mask=True` represents *static* valid pixels. The original hard mask remains active. The experimental score is a heuristic *inconsistency score*, **not a calibrated motion probability**; unobserved static-map regions are not classified as dynamic.

## How to run

Clone/check out the feature branch, install the official repository dependencies, then set

```yaml
model_params:
  rgbd_uncertainty:
    enabled: true
    save_debug: true
    debug_every: 30
```

in the relevant `configs/rgbd/bonn/config.yaml` or TUM config and run the original scripts (`bash scripts/train_bonn.sh` or `bash scripts/train_tum.sh`).

Debug archives named `rgbd_uncertainty_XXXXX.npz` are written in the configured results directory's `tracking/` subfolder. Fields are `uncertainty`, `weights`, `valid`, `dynamic_candidate`, `original_static`, `coverage` and `applied`.

Run CPU module tests:

```bash
python -m unittest discover -s tests -p 'test_rgbd_uncertainty.py' -v
```

## Experimental protocol and cautions

1. `enabled: false` is the original baseline; `enabled: true` is V1.
2. Compare identical datasets, frame budgets, seed, ATE RMSE, tracking time, and candidate/static mask statistics; start with Bonn `ballon2`.
3. The gate uses the first static rendering after `warmup_iters=3` tracking steps and stays fixed for that frame. If convergence occurs earlier, it is not applied.
4. Reliable pixels require valid RGB/depth, static-map opacity >= `opacity_min=0.95`, and coverage >= `min_coverage=0.08`. Otherwise the weight remains 1.
5. Initially use weights 0.40 (RGB) and 0.60 (relative depth), score threshold 1.60 and min weight 0.10. **These are starting heuristics, not fitted hyperparameters.**
6. Low static-map quality, misregistration, depth discontinuities, and occlusions can also yield high residuals; validate before treating diagnostic candidates as true dynamic objects.
7. No full GPU-based SLAM integration, benchmark ATE, or dynamic-mask IoU results have been established by these changes. This PR needs CUDA/benchmark verification.
