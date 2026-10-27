# V4.0a — Complete flow residual and map-coverage attribution (diagnostic-only)

**Parent:** `feature/v40-motion-reliability-diagnostics`
**Branch:** `feature/v40a-complete-observation-diagnostics`

This experiment answers the limited question: does the *existing* Flow4DGS
dynamic candidate mask correspond to high RGB-D map residuals, and do
semantic-prior pixels or map-uncovered regions behave differently? These are
diagnostics, not GT dynamic labels and not a new detection model.

## Changes

- Legacy segmentation is left unmodified. Its `resid_px` is zeroed outside
  the fitted prior-static mask and remains archived as
  `flow_legacy_residual_px`.
- The first-stage fitted twist (`results["xi"]`) is **re-used**, with no
  refitting, to predict rigid flow at all depth-valid pixels.
  `flow_full_residual_px` contains NaN where depth/flow is invalid;
  the separately stored `flow_full_valid` makes it usable on prior-dynamic
  pixels too. This is model disagreement, not optical-flow ground-truth error.
- `pre_depth_raw` and `post_depth_raw` retain depth differences when both
  observed and rendered depth are positive and finite, even if opacity is below
  the coverage threshold. Such low-opacity residuals are **diagnostic only**.
- Compare RGB/Depth residual before and after Tracking on the EXACT same
  pixels with valid observed depth and both pre/post map coverage.
- Optional extra RAFT inference on sampled frames provides previous->current
  flow; compare with existing current->previous flow using bilinear sampling
  and `flow_cycle_error_px`. Occlusion and motion discontinuities can cause
  high cycle error even for correctly predicted optical flow.
- Save RGB and static rendering images in NPZ to enable interpretable
  visualization at specific diagnostic frames.

**No changes** to Tracking loss, camera fit, mapping, the motion mask, or
Gaussian primitives. V4.0a adds sampled render and RAFT runtime overhead.

## Test, configure, run

```bash
git fetch origin
git switch feature/v40a-complete-observation-diagnostics
python -m unittest discover -s tests -p "test_v40a_diagnostics.py" -v
```

Edit `configs/rgbd/bonn/config.yaml`:

```yaml
model_params:
  v40_diagnostics:
    enabled: false
  v40a_diagnostics:
    enabled: true
    every: 10
    opacity_min: 0.95
    save_maps: true
    forward_backward: true
```

Keep baseline `flow_mask: true` and `flow_cam: true`, and disable all
earlier optional V1–V3 Tracking gates. Use the SAME dataset/checkpoints and
baseline settings as V4.0.

```bash
python slam.py \
 --inherit_from configs/rgbd/bonn/config.yaml \
 --config configs/rgbd/bonn/ballon2.yaml \
 --eval --dynamic --exp_name=v40a_diagnostics --save_results=0
```

Find results:

```bash
find results/bonn/v40a_diagnostics -name v40a_per_frame.csv
```

Run offline analysis **after** the SLAM job ends:

```bash
python scripts/analyze_v40a_diagnostics.py \
 --input /path/to/run/v40a_diagnostics \
 --frames 60,140,220,230,310,390
```

Output `v40a_per_frame.csv` (online), `v40a_NNNNN.npz` (online),
`v40a_per_frame_offline.csv`, `v40a_summary.txt`,
`v40a_visual_NNNNN.png` (offline).

## Important interpretation limitations

1. The real RAFT flow uses current->previous normalized grid for tracking.
   V4.0a first converts this to pixel displacement. The additional forward
   RAFT pass produces native pixel displacement; the two can be compared
   without changing original Tracking.
2. `prior_static=True` means eligible static pixels; `flow_dynamic=True`
   is a motion outlier in the legacy fit, not semantic GT.
3. The full residual reuses the **first** fitted twist, not the refitted
   motion estimate used for final pose initialization. This is deliberate:
   it isolates the prior-selected fit and reproduces the first-stage mask.
4. Current->previous optical flow is geometric frame-to-frame displacement;
   forward/backward error does not establish true object motion or error on
   occluded pixels.
5. The image and geometry may still suffer from depth rectification issues
   discussed in V3.2b. Check coordinate alignment before strong conclusions.
6. Prior low-opacity rendered depth can be physically meaningless. Never
   interpret `depth_raw_uncovered` as a calibrated geometry error.
7. For ATE comparisons, disable diagnostic recording, and use paired seeds.
8. This version does not compute dynamic IoU/F1; those require annotated GT.
