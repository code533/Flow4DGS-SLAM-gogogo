# V4.0b: flow calibration, boundary stratification and controlled replay

Branch: `feature/v40b-flow-reliability-calibration` (from V4.0a).

## Scientific question

How often do high independent-motion residuals coincide with (a) low
current/previous optical-flow cycle errors or (b) depth discontinuities
and invalid correspondences? This experiment measures associations,
**not** calibrated optical-flow accuracy, dynamic ground truth or
ground-truth occlusion.

## Technical correction

In `Camera.generate_flow()`, native RAFT pixel flow is converted to:
`F_norm = F_native / [W,H] * 2`. For forward-backward diagnostic
comparison, we must invert it using `F_native = F_norm * [W/2,H/2]`.
Older V4.0a used `[(W-1)/2,(H-1)/2]` for one direction, a mismatch.
V4.0b keeps this original grid scaling for the unmodified motion
estimator but uses native pixels for BOTH cycle directions.

Depth-edge proximity is a local threshold of 3x3 min/max depth with
optional dilation. Possible occlusion is a **proxy** based on:
out-of-frame flows, invalid corresponding depth, or depth-edge proximity
AND high cycle error. It is not GT occlusion. Do not compare depths at
the same pixel across moving cameras and call the difference occlusion.

## Installation / launch

```bash
git fetch origin
git switch feature/v40b-flow-reliability-calibration
python -m unittest discover -s tests -p "test_v40b_calibration.py" -v
```

Edit `configs/rgbd/bonn/config.yaml`, keeping all previously optional
V1-V3 modifications OFF and `flow_cam/flow_mask` enabled:

```yaml
model_params:
  v40_diagnostics:
    enabled: false
  v40a_diagnostics:
    enabled: false
  v40b_calibration:
    enabled: true
    every: 10
    opacity_min: 0.95
    save_maps: true
    forward_backward: true
    depth_edge_abs_m: 0.08
    depth_edge_rel: 0.06
    depth_edge_dilate: 2
    fb_bad_px: 1.0
    high_motion_px: 5.0
    low_fb_px: 0.5
```

```bash
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
  --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
  --exp_name=v40b_calibration --save_results=0
```

Output: `<run>/v40b_calibration/v40b_per_frame.csv` and
`v40b_XXXXX.npz`. Then:

```bash
RUN_DIR=$(find results/bonn/v40b_calibration -maxdepth 1 -type d \
 -name "rgbd_bonn_balloon2_*" | sort | tail -n 1)
python scripts/analyze_v40b_calibration.py \
 --input "$RUN_DIR/v40b_calibration" --frames 60,140,220,230,310,390
```

Outputs: `v40b_per_frame_offline.csv`,
`v40b_summary.txt`, and `v40b_visual_NNNNN.png`.

## Separate runtime / run-to-run control

V4.0a had a substantial static coverage difference from V4.0.
The following additional experiments are necessary; **do not attribute**
the discrepancy to GPU timing or randomness without repetitions.

For run-to-run controls: all diagnostics OFF, same scene, checkpoints,
runtime settings and same source commit. Run at least 3 independent runs,
save each generated `config.yml`, `outputs.txt`, ATE and Gaussians count,
plus any determinism settings. Then repeat with only V4.0b enabled,
and compare same-frame coverage and trajectory / mapping outputs.
The repository does not currently implement a fixed-seed switch; exact
identical-run replay or paired-seed claims should NOT be made until a
seed-control mechanism is added and exercised.

If examining the overhead of an extra forward RAFT pass, compare
`forward_backward: true` and `false`; note that `false` eliminates
cycle and interior statistics and does not constitute full calibration.

V4.0b adds extra inference, rendering and file IO, and does not improve
ATE by design. FPS from a diagnostic run is not directly comparable to
a baseline algorithm run.

## Cautions

- `prior_static=True` means selected as static, NOT semantic GT.
- `flow_dynamic=True` means residual outlier among the eligible prior
  pixels, NOT actual GT motion.
- Boundary, visibility and occlusion metrics are only heuristic.
- The 'high motion' and 'low FB' cutoffs are exploratory; test sensitivity
  and ensure enough valid pixels in each group.
- `interior_correspondence` is a strict subset; report coverage and sample
  counts to avoid hidden selection bias.
- Historical BONN alignment / distortion conventions still require
  independent geometric checks (V3.2b).
- Use human-labeled masks and validated correspondences before claiming
  detection accuracy or optical-flow error calibration.
