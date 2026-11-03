# V4.3a.4 — Preregistered Robust Score Predictiveness on Next Keyframe

Base: `feature/v43a3-robust-rgbd-audit`.
Research goal: decide whether a ROBUST source-keyframe depth score predicts
which of G0, G3-Support and G4 has lower next-keyframe *ordinary dynamic
depth MAE* under identical frozen pre-Mapping state. Online Mapping remains
G0; no optimizer, tracking, dynamic mask, or propagation rule is altered.

## Core evaluation validity controls

1. Source selection is based on **eligible dynamic-propagation event count**:
   first event, then every `predictive_source_every_n_event` events. It
   does not use frame UID modulo and does not cherry-pick known success
   frames. Source count capped by `predictive_max_sources`.
2. Only the source frame decides the policy's candidate. All policies
   evaluate the **same** frozen-source map, source target RGB-D, and
   next arriving keyframe RGB-D/camera. No future RGB-D enters selection.
3. Each candidate's motion offsets are applied with the original strict
   `norm(dx)<0.3m` gate to independent temporary displacement tensors.
4. Snapshot is held unchanged until the *next keyframe*, not updated by
   intervening Mapping. **Zero-order-hold** motion means this is a
   short-horizon persistence diagnostic, NOT actual future flow estimation.
5. Use exactly the same *future* scoring function for all policies:
   unbounded ordinary depth MAE on estimated dynamic pixels with valid
   observed depth and fixed penalty for missing prediction (from V4.3a).
   In particular, the winner is not judged by its own robust loss.
6. A source margin of 0.2mm over G0 is required to choose another mode,
   otherwise preserve G0. Source candidates must have depth coverage no
   lower than G0 by 0.5 percentage points. Source/future coverage >=95%;
   at least 256 dynamic valid depth pixels required.
7. Source/future frame masks can differ **across time** but must be
   candidate-independent **within time**. We report RGB/coverage effects
   even though the source selection policies use depth only.
8. Non-independent samples: successive frames from one sequence are
   correlated. Treat N pairs as descriptive evidence, do not compute
   binomial p-values pretending independent trials.
9. Compare two stages: (i) source policy selection, (ii) next-keyframe
   result and oracle regret. A policy that always chooses G0 should have
   zero future gain, forming a negative control. Future oracle is
   *diagnostic upper bound*, never available to source selector.

### Preregistered source rules

| Policy | Source-frame score | Interpretation |
|---|---|---|
| `always_g0` | ignore scores | safe baseline |
| `mae` | mean observed dynamic depth absolute error | standard |
| `clipped50` | mean min(depth residual, 0.05 m) | bounded |
| `huber20` | mean Huber(depth residual, 0.02m) / 0.02m | linear-tail robust |
| `interior_clipped50` | clipped depth MAE on estimated dynamic-mask interior after 5-px erosion | boundary-sensitive |

No policy uses paired delta clipping; it is NOT a separable candidate
scoring function. A missing/too-small interior => G0 fallback.
None of these policies is claimed to be the correct motion score a priori.

### Setup

```bash
git fetch origin
git switch feature/v43a4-robust-heldout-prediction
git pull --ff-only
python -m unittest discover -s tests -p 'test_v43a4_predictive.py' -v
python -m unittest discover -s tests -p 'test_v43a2_temporal.py' -v
python -m unittest discover -s tests -p 'test_v43a_observation.py' -v
```

In `configs/rgbd/bonn/config.yaml`, keep `model_params.flow_offset:
true`, `use_knn: true`, `dynamic_model: offset`,
`Training.flow_ds: 1`. Configure:

```yaml
model_params:
  v42a_propagation:
    mode: 'g0'
    g34_shadow: false
    g4_ablation: false
    attribute_failures: false
    save_points: false
    # retain existing reliability/G4 thresholds
  v43a_observation:
    enabled: true
    every_n_keyframe: 1
    detailed_frames: []   # do not save all NPZ: save_maps=false
    save_maps: false
    region_analysis: false
    opacity_threshold: 0.1
    uncovered_penalty_m: 0.5
    temporal_enabled: true
    temporal_source_frames: [237] # optional case study, separate from systematic sample
    temporal_max_frame_gap: 12

    predictive_enabled: true
    predictive_source_every_n_event: 10
    predictive_max_sources: 25
    predictive_min_dynamic_depth_pixels: 256
    predictive_min_interior_depth_pixels: 64
    predictive_interior_erosion_px: 5
    predictive_clip_m: 0.05
    predictive_huber_delta_m: 0.02
    predictive_decision_margin_m: 0.0002
    predictive_future_tie_tol_m: 0.0001
    predictive_coverage_tolerance: 0.005
    predictive_min_coverage_fraction: 0.95
    predictive_save_future_maps: false
```

Note `detailed_frames: []` means **any frame is considered eligible for
detail saving**, but `save_maps: false` safely disables NPZ saving. You
may set `temporal_source_frames: [9999]` to disable the legacy
handpicked case-study without violating temporal source validation.

Run:

```bash
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
  --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
  --exp_name=v43a4_predictive_ballon2 --save_results=0
```

First smoke test with `predictive_max_sources: 2` and
`predictive_source_every_n_event: 5`. For the broader balloon2 run,
set `predictive_source_every_n_event: 4` and
`predictive_max_sources: 25` (roughly 24 systematic source events
if 95 eligible keyframes arise again). The code may retain a full copy
of GaussianModel on GPU until the next keyframe; memory and rendering
overhead can alter asynchronous SLAM scheduling. Do not interpret ATE
differences from this diagnostic run as an improvement.

Outputs:

- `<run>/v43a4_predictive/v43a4_policy_per_pair.csv`: 5 policy rows
  per valid source/future pair. Includes selected mode, fixed future
  depth improvement vs G0, future oracle regret, RGB and coverage deltas.
- `<run>/v43a4_predictive/v43a4_candidate_scores.csv`: 6 rows
  per source/future pair (3 candidates x 2 timepoints).
- `<run>/v43a2_temporal/v43a2_temporal.csv`: legacy MAE diagnostic.
- No detailed future NPZ by default. For visual audit set
  `predictive_save_future_maps: true`, with a small source count.

```bash
python scripts/analyze_v43a4_predictive.py \
  --input "<run>/v43a4_predictive/v43a4_policy_per_pair.csv"
```

Produces `v43a4_policy_summary.csv`.

## Decision rule, before looking at results

Do not upgrade to online V4.3b based only on low source error. Compare
the *same eligible pairs* and future raw depth target for every policy.
Prefer robust score only if it reduces future regret and has better
average/median future gains than ordinary MAE, without large RGB or
coverage regressions. A few lucky keyframes do not establish
repeatability. Confirm the result with independent sequences, and
stronger future transport/optimization (zero-order-hold is limited).

If robust scores do not win, stop candidate selection and investigate
motion estimation, data association, geometry and occlusion instead.
