# V5.0 — Occlusion/observability-aware 3D Gaussian motion attribution

**Branch**: `feature/v50-occlusion-motion-attribution`
(based on simplified-config V4.3a.4 branch).

## Scope / scientific question

V4.3 candidate-score ablations concluded with a preregistered
cross-sequence NO-GO. V5 starts a **different** research line:
*Which observations and geometry stages are currently preventing
accurate 3D dynamic Gaussian motion estimates?*

V5.0 is READ-ONLY. No changes to Mapping, Tracking, dynamic mask,
depth filling, G0 KNN, optical flow, 0.3m update gate, or optimizer.
It is therefore a diagnostic baseline, NOT a reconstruction improvement.

### Evidence groups (proxy categories, NOT ground truth)

Applied to dynamic Gaussians that *survive the existing G0* motion
candidate construction. Six mutually exclusive groups:
- `source_projection_invalid`: nonfinite or outside source image.
- `previous_depth_missing`: original source depth unavailable.
- `flow_correspondence_invalid`: forward/FB flow invalid/out of image.
- `target_observation_missing`: warped depth invalid OR **original**
  target-depth sample invalid, even when Mapping's filled depth is valid.
- `low_flow_confidence`: very low or below threshold `Q_F`.
- `observed_reliable_proxy`: remaining selected Gaussian candidates.

**Important:** the selected set generally excludes out-of-view or
invalid-projection Gaussians by construction. For that reason, the
per-group table is NOT a full population distribution. Every
original dynamic Gaussian is also counted in an **all-dynamic reason
histogram**, so failures cannot silently disappear from the summary.

A low Q, invalid target depth, or out-of-view flag is NOT confirmed
occlusion, motion error or dynamic-object ground truth. A depth hole
may be missing sensor measurement or segmentation issue. The
reliability score is an optical-flow consistency proxy.

### Propagation stages

At each sampled eligible dynamic keyframe, log counts after:
1. all prior dynamic Gaussians;
2. source image projection;
3. source depth consistency check (uses existing depth filling);
4. target optical-flow projected image bounds;
5. target **filled-depth valid** check;
6. original method's raw displacement <1m prefilter;
7. exact online G0 update gate norm <0.3m.

For surviving candidate Gaussians log:
- original unfilled target depth validity (whether a **real depth**
  observation supports the motion),
- raw and KNN-smoothed 3D motion magnitude (mean/p95),
- magnitude of the smoothing correction `norm(G0 - raw)`,
- fraction passing the exact 0.3m online gate,
- corresponding Q_F proxy statistics.

Missing-depth flags are not overwritten by filled depth. Importantly,
the original Mapping *still uses the filled depth*.

### Running Balloon2

```bash
git fetch origin
git switch feature/v50-occlusion-motion-attribution
git pull --ff-only

python -m unittest discover -s tests -p 'test_v50_motion_attribution.py' -v

python slam.py \
  --config configs/rgbd/bonn/v50_balloon2_diagnostic.yaml \
  --eval --dynamic --exp_name=v50_occlusion_diagnostic --save_results=0
```

No `--inherit_from` required. Preset inherits the 124-line clean Bonn
core and overrides Balloon2 path, original Balloon2 Training settings,
and selects instrumented G0. V4.3 candidate scoring remains off.

The default config is:
```yaml
model_params:
  v42a_propagation:
    mode: 'g0'
  v50_attribution:
    enabled: true
    every_n_event: 1
    q_threshold: 0.30
    save_points: false
    detailed_frames: [212,357]
```

For a quick test, set `every_n_event: 5`. For spatial debugging
enable `save_points: true`: this writes selected-point NPZs only for
the explicitly listed detailed frames.

## Outputs

```
<run>/v50_attribution/
  v50_per_keyframe.csv
  v50_per_group.csv
  v50_summary.txt            # after offline script
  v50_group_summary.csv      # after offline script
  v50_00212.npz              # only if save_points=true and frame exists
```

Run:
```bash
python scripts/analyze_v50_attribution.py \
  --input "<run>/v50_attribution"
```

Useful columns:
- `all_reason_projection_outside_image_fraction`
  and `all_reason_previous_depth_invalid_fraction`;
- `stage_target_filled_depth_valid` versus
  `selected_original_target_depth_valid_fraction`;
- `selected_smoothing_correction_p95_m`;
- `g0_update_gate_pass_fraction`.

## Interpretation and decision

1. If major population mass is outside the viewport or missing
   source depth, no method can obtain direct current 3D measurements
   for those points. Next candidate: history-aware or visibility-aware
   prediction with uncertainty and later re-observation.
2. If selected Gaussians rely mostly on depth filling rather than
   original measurements, investigate provenance-aware depth
   association and occlusion boundaries.
3. If original depth is available but the raw/G0 displacement
   difference is large, investigate incorrect KNN smoothing/mixing
   different local object motions.
4. If most raw displacements survive but G0 outputs exceed the
   0.3m gate, inspect smoothing and gate interaction before adding
   motion assumptions.

Next steps must be driven by counts **and** actual supporting RGB-D
geometry. This diagnostic alone cannot identify the true 3D error
or prove occlusion, and does not by itself provide a publishable
V5 motion estimation algorithm.

Technical caveats:
- all-dynamic Gaussian histograms and selected-point groups have
  different denominators; **never compare their fractions directly**;
- repeated keyframe counts are not independent Gaussian identities;
- source/target depth mapping uses Flow4DGS' original projection and
  hole-filling conventions, not a new geometry model;
- Gaussian model is modified by Mapping *after* this diagnostic, as
  before. GPU validation is required on your machine.
