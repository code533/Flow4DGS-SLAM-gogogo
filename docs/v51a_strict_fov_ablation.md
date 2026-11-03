# V5.1a — Strict-FOV Gaussian Motion Ablation

Branch: `feature/v51a-strict-fov-ablation`, forked from V5.0.
Research objective: test whether **out-of-image pixel clamping in the
existing G0 motion correspondence path** adversely affects dynamic 3D
Gaussian movement and reconstruction.

## A very specific intervention, not an occlusion predictor

The unmodified `BackEnd.get_uv_coordinates()` admits finite NDC
coordinates outside [-1,1] and then clamps the resulting pixel index
to the image border. No code in that global helper is changed.

V5.1a adds a **local check only in `apply_flow_offset()`**:
- Source projection must be finite and inside NDC [-1,1] × [-1,1],
  **AND** clip homogeneous W > 1e-7.
- Target correspondence position must be finite and in NDC [-1,1].
  **Important caveat**: target here is source-projected NDC plus optical
  flow, as in the existing pipeline, *not* a separate target-camera
  projection. This is a source/warped-flow image-FOV consistency test.
- Existing 0.3 m Gaussian offset update gate remains identical.
- Existing raw depth filling, Flow4DGS optical flow, depth check,
  source/target camera poses, target projection approximation and
  segmentation remain unchanged.

For Strict-FOV, invalid motion samples are removed BEFORE smoothing:
only valid points are fed to the original G0 Gaussian-KNN smoothing
routine with identical weighting, radius and sigma. This prevents an
out-of-FOV neighbor from altering an in-FOV candidate. Gaussians without
a selected new motion increment simply **inherit the prior keyframe
offset**. They are not assigned a zero world-space velocity and not
extrapolated by any new history model.

## Modes

- `legacy`: same online instrumented G0 propagation as V5.0; compare
  FOV counts only (no additional Strict KNN recomputation).
- `shadow`: compute both Legacy and Strict candidates, log the
  differences, but update GaussianModel with **Legacy G0 only**.
- `strict`: rebuild KNN on strict surviving candidates; update Gaussian
  offsets with Strict subset and values. **Only this mode changes Mapping**.
- `off`: no V5.1a processing; the code's original behavior.

All V4.3 candidate-selection methods are disabled. V5.0 histogram
logging is disabled in this ablation to prevent mixing two diagnostics.

## Setup

```bash
git fetch origin
git switch feature/v51a-strict-fov-ablation
git pull --ff-only

python scripts/check_v51a_config.py
python -m unittest discover -s tests -p "test_v51a_strict_projection.py" -v
python -m unittest discover -s tests -p "test_v51a_config.py" -v
```

### Phase A — no-commit candidate / shadow audit

```bash
python slam.py \
  --config configs/rgbd/bonn/v51a_balloon2_shadow.yaml \
  --eval --dynamic --exp_name=v51a_shadow --save_results=0

python scripts/analyze_v51a_fov.py \
  --input "<shadow run>/v51a_projection"
```

Confirm **nonzero excluded originally accepted** candidate count, and
that the shared in-FOV candidates exhibit a plausible Strict-vs-Legacy
KNN displacement difference. If Strict removes nearly all candidate
motions in otherwise well-observed frames, investigate the camera
projection convention before online ablation.

### Phase B — online ablation

Use exactly the same settings in both configs except the `mode`:

```bash
python slam.py \
  --config configs/rgbd/bonn/v51a_balloon2_legacy.yaml \
  --eval --dynamic --exp_name=v51a_legacy_run1 --save_results=0

python slam.py \
  --config configs/rgbd/bonn/v51a_balloon2_strict.yaml \
  --eval --dynamic --exp_name=v51a_strict_run1 --save_results=0
```

Repeat **at least three independent runs per mode** if the smoke and
online runs succeed. Asynchronous keyframe scheduling and random
initialization may change ATE across runs. Do not infer improvement
from one numeric ATE difference.

### Metrics

1. Tracking ATE (mean/std over repeated seeds/runs).
2. Full-frame and dynamic-region PSNR/SSIM and RGB-D residuals, with
   identical scoring masks and all other options fixed.
3. Source/warped target FOV exclusion counts, number of originally
   accepted (<0.3m) motions removed.
4. Number of Gaussian updates and changed KNN-smoothed motions among
   shared in-FOV candidates.
5. Representative out-of-view and re-entry keyframes (e.g., around
   balloon2 KF 342–382); compare rendered depth/RGB visually.

**Outputs**:
```
<run>/v51a_projection/v51a_per_keyframe.csv
<run>/v51a_projection/v51a_summary.csv       # after offline script
<run>/v51a_projection/v51a_summary.txt       # after offline script
```
Optional Gaussian-level NPZ snapshots, if in common config:
```yaml
model_params:
  v51a_projection:
    save_points: true
    detailed_frames: [212,357]
```

The Shadow/Strict CSV includes how many original candidates were
removed, which were still eligible for Legacy's 0.3m gate, and how
KNN recomputation changes motions on common visible candidate IDs.
The `legacy` CSV cannot report Strict KNN motion differences.

## Stopping conditions

1. If the exclusion is mostly from non-front-facing projection
   conventions (clip-W sign mismatch), STOP: verify coordinate system.
2. If Strict FOV causes widespread loss of genuinely valid scene
   correspondences, STOP: camera projection / optical flow units may
   be inconsistent and need checking.
3. If repeated online ATE and dynamic reconstruction show no advantage,
   DO NOT introduce additional thresholds as an attempt to rescue the
   method. Continue to depth-provenance / actual occlusion tracking
   diagnosis instead.
4. If beneficial, Strict-FOV alone is fundamentally an implementation
   correction, not sufficient as a novel paper method. Next investigate
   **history-aware, uncertainty-aware updates** for unobserved Gaussians,
   tested by independent frame reconstruction and actual Mapping metrics.

The present implementation has been source-integrated and checked by
unit-test code, but not validated on the user's GPU/runtime. Always run
the shadow sanity check first.
