# V4.3a.2 — Spatial failure attribution and frozen-map temporal persistence

Branch `feature/v43a2-spatial-temporal-frozen-validation`, based on V4.3a.1.
No Tracking / Mapping / segmentation / optimizer changes. Actual online
displacement update **remains G0**.

## Research questions

A. For candidate differences at important keyframes (207, 237, 247,
252, 317, 357, 412), where are dynamic RGB / depth improvements and
tradeoffs located?

B. Does a candidate that fits the source frame better still fit the
**next incoming keyframe's independent RGB-D observation** better when
the same pre-Mapping map and source displacement offsets are held frozen?

## Causal comparison and limitation

A captured source map is cloned **before source-keyframe G0 propagation,
Gaussian insertion and Mapping optimization**. The three source
hypotheses G0/G3-Support/G4 all use the same map snapshot and source
observation. At the next keyframe message, before any new mapping
operation, each candidate's source displacement (with the exact
norm<0.3m update gate) is rendered with that future keyframe's camera
and observation. Gaussian appearance, shape and geometry are frozen.
GMM opacity/rotation are queried at the future keyframe UID identically
for all candidates.

This is a **zero-order-hold offset persistence** experiment. The
motion displacements are not extrapolated from source to future, and
no new future optical flow is estimated. Consequently, the result
tests persistence of a fixed source-state candidate, **not a fully
valid future motion forecasting model**. In particular, a next-keyframe
difference can arise from simply moving objects beyond the range in
which a frozen offset remains useful.

The camera pose is derived from the running SLAM; it is fixed for the
three candidates within the same future keyframe, but is not
independent GT. Source and future RGB-D are separate observations.
No claim of held-out *sequence* generalization or ATE improvement is
supported by this diagnostic alone.

**Memory warning:** a temporary deep-copied GaussianModel is retained
until the next keyframe. Enabling many temporal source IDs increases
the frequency of potentially large GPU memory copies. There is at
most one outstanding snapshot at once. Start with [237] only.

## Setup

```bash
git fetch origin
git switch feature/v43a2-spatial-temporal-frozen-validation
git pull --ff-only
python -m unittest discover -s tests -p 'test_v43a2_temporal.py' -v
python -m unittest discover -s tests -p 'test_v43a1_region.py' -v
python -m unittest discover -s tests -p 'test_v43a_observation.py' -v
```

In `configs/rgbd/bonn/config.yaml` under `model_params`:

```yaml
v42a_propagation:
  mode: 'g0'
  g34_shadow: false
  g4_ablation: false
  attribute_failures: false
  save_points: false
v43a_observation:
  enabled: true
  every_n_keyframe: 1
  detailed_frames: [207, 237, 247, 252, 317, 357, 412]
  save_maps: true
  opacity_threshold: 0.1
  uncovered_penalty_m: 0.5
  region_analysis: true
  grid_rows: 6
  grid_cols: 8
  min_dynamic_rgb_pixels: 64
  min_dynamic_depth_pixels: 16
  rgb_tolerance: 0.00001
  depth_tolerance_m: 0.0001

  temporal_enabled: true
  temporal_source_frames: [237]   # smoke first; expand to seven frames after
  temporal_max_frame_gap: 12
```

Keep other Flow4DGS training/model parameters and `Training.flow_ds=1`
unchanged. Keyframe IDs are stochastic; a source ID must be produced as
an eligible dynamic propagation keyframe in the fresh run to trigger.

```bash
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
  --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
  --exp_name=v43a2_frozen_single_source --save_results=0
```

Then, if GPU RAM and logs are healthy, set:

```yaml
temporal_source_frames: [207, 237, 247, 252, 317, 357, 412]
```

and a unique `exp_name` to run the full experiment.

## Output files

- `<run>/v43a_observation/v43a_per_keyframe.csv`: source frame
  G0/G3-Support/G4 paired render scores for all eligible source
  keyframes.
- `<run>/v43a_observation/v43a_XXXXX.npz`: full rendered RGB, depth,
  opacity, fixed masks and pixelwise errors for requested detailed frames
  (requires `save_maps: true`).
- `<run>/v43a1_regions/v43a1_per_keyframe.csv`, `v43a1_per_region.csv`:
  spatial improvements and conflicting modes.
- `<run>/v43a2_temporal/v43a2_temporal.csv`: source/future error
  deltas and both-RGB-and-depth persistence flags for successfully
  followed keyframes. The file need not exist if no configured source
  occurred or all future gaps exceeded the configured threshold.

```bash
python scripts/analyze_v43a_observation.py --input "<run>/v43a_observation"
python scripts/analyze_v43a1_regions.py --input "<run>/v43a1_regions"
python scripts/analyze_v43a2_temporal.py --input "<run>/v43a2_temporal"
python scripts/visualize_v43a2_rgbd.py --input "<run>/v43a_observation"
```

The last script produces PNG visualizations with candidate-minus-G0
residuals; blue/red have consistent signed error scales within each
modality. Report dynamic mask and true depth-valid pixel counts along
with paired scores.

## Decision criteria

- Verify spatial origin of high-improvement and high-regression cases,
  especially boundaries and occlusion.
- Check whether candidate source-frame winners retain a negative
  candidate-minus-G0 error on the next keyframe, for RGB and depth
  separately. Report **counts and magnitude**, not just winner ratios.
- A source positive with future negative effect indicates likely
  short-term overfit or inadequacy of zero-order hold. It is *not*
  proof that the candidate is intrinsically wrong.
- If future hold performance is inconclusive, do not add an online
  selection mechanism. Consider a genuine two-frame transport and
  held-out benchmark next.
- Keyframes are temporally correlated and candidate states share model
  parameters; per-pixel counts cannot be treated as independent
  statistical trials.

Source integration and tests are supplied; full GPU validation must
be performed in your environment.
