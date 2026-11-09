# V5.1c — Controlled dynamic reconstruction attribution

**Branch:** `feature/v51c-controlled-motion-attribution`, based on V5.1b.

## Why this experiment

V5.1a had a lower mean tracking ATE in five trials (0.04138m Legacy,
0.03658m Strict). However, the one-run V5.1b final-map comparison
showed +1.494mm mean dynamic depth MAE for Strict, with a strong gain
in dynamic RGB around keyframes 342–382. Two runs used DIFFERENT
keyframe sets (182 distinct keyframe-status mismatches out of 468
evaluated frames), so the full-run comparison is causally ambiguous.

V5.1c performs two complementary **read-only** analyses, with limits
stated explicitly.

## Experiment A: within-state pre-Mapping intervention (new GPU render)

Sample every 10th eligible dynamic propagation event, with at most 12
events. The SAME pre-Mapping Gaussian model, map geometry, current
camera pose, RGB-D observations, segmentation and GMM time are shared
by three local hypothetical motion states:

1. `hold`: inherit previous-keyframe dynamic offsets; no new increment.
2. `legacy`: apply ordinary G0 displacement to original candidate
   Gaussian IDs, using exact `norm(dx)<0.3m` update gate.
3. `strict`: apply strict-FOV filtered candidate IDs and re-smoothed
   G0 displacement, with the same gate.

Only temporary offset **tensors** are constructed: no model copies,
optimizer update, Tracking/Mapping mutation or keyframe changes. Three
extra renderer calls per sampled event, not per sequence frame.
The actual online SLAM path remains V5.1a **shadow** (unmodified G0).

For every candidate write matched dynamic RGB MAE, PSNR, fixed-target
penalized depth MAE, depth coverage, dynamic interior/boundary errors,
and counts/hashes for identical observed masks. All values are from
the SAME selection frame, so they are **in-sample local sensitivity**,
NOT proof of beneficial future dynamic geometry or ATE.

Additional useful paired diagnostic: `strict - hold`, `legacy - hold`,
and `strict - legacy`. If Strict preserves earlier displacement and
that candidate immediately fits worse than both alternatives, it is a
candidate mechanism for short-term loss of dynamic motion updates.
But self-fit is not motion truth.

### Historical activity proxy (NOT real temporal lag)

For strict-excluded candidate Gaussian indices, compare adjacent
previous keyframe *optimized* offset parameters:

```
||delta_xyz(prior_kf) - delta_xyz(prior_prior_kf)||
```

This records how much a candidate had changed historically before the
new update was withheld. It is **not** 3D motion ground truth, an actual
unobserved streak length, or an integrated position error. Gaussian
identities and topology change during densification, so index-based
multi-keyframe streaks would be unreliable unless persistent identities
are introduced. The module intentionally does not claim those streaks.

## Experiment B: retrospective observational stratification (no GPU)

Use the prior V5.1b `v51b_matched_per_frame.csv` to stratify Strict-minus-
Legacy final-map differences into: both keyframe, neither keyframe,
Legacy-only keyframe and Strict-only keyframe.

This quantifies the magnitude of differing keyframe selection, but
**does not identify the causal impact of keyframe decisions**. Different
trajectory estimates, local optimization, window membership, Gaussian
densification and long-term state histories are all confounders.

If Experiment A and final-map metrics disagree, it suggests an
indirect effect *somewhere* in the online estimation/Mapping loop,
not a proven single mechanism.

## Run (one short shadow SLAM pass)

```bash
git fetch origin
git switch feature/v51c-controlled-motion-attribution
git pull --ff-only
python -m unittest discover -s tests -p 'test_v51c_controlled.py' -v

python slam.py \
  --config configs/rgbd/bonn/v51c_balloon2_shadow.yaml \
  --eval --dynamic --exp_name=v51c_controlled_shadow --save_results=0
```

The preset inherits the original V5.1a shadow configuration; V4.3,
V5.0 and V5.1b final per-frame diagnostics are not required. It uses:

```yaml
model_params:
  v51c_controlled:
    enabled: true
    every_n_event: 10
    max_sources: 12
    opacity_threshold: 0.1
    uncovered_penalty_m: 0.5
    boundary_width_px: 5
    history_min_m: 0.005
```

For quick smoke test, set `max_sources: 2` (not a full study).

**One note about runtime:** extra GPU renders in `apply_flow_offset()`
may affect asynchronous keyframe scheduling or run time even though
they are read-only. V5.1c therefore compares three candidates **within
the same** sampled state, not V5.1c ATE against older runs.

Outputs:

```
<run>/v51c_controlled/v51c_frozen_per_keyframe.csv
<run>/v51a_projection/v51a_per_keyframe.csv
```

Analyze with:

```bash
python scripts/analyze_v51c_attribution.py \
  --frozen "<run>/v51c_controlled/v51c_frozen_per_keyframe.csv" \
  --matched-v51b "/path/to/v51b_matched_per_frame.csv" \
  --output "<run>/v51c_attribution"
```

The earlier V5.1b CSV is optional; if omitted, only same-state analysis
runs. New files:

- `v51c_fixed_map_contrasts.csv`
- `v51c_exclusion_history_proxy.csv`
- `v51c_crossrun_keyframe_status.csv` (when prior CSV available)
- `v51c_attribution_report.md`

## Interpretation and next experiment

**Case 1:** Strict has lower same-state dynamic RGB and depth errors but
lower final-map quality. This suggests the local FOV gate is not the
sole reason for final-map regression; changed Mapping/keyframes/poses
and non-observed motion histories may matter.

**Case 2:** Strict degrades the same-state dynamic interior and the
history-activity proxy is high for the excluded population.
This raises the *hypothesis* that skipping unseen motion causes lag,
but cannot establish it. Next: compare future observations against
an actual history-motion baseline, using controlled shared states.

**Case 3:** Strict clearly improves current dynamic boundary metrics
but not interior. Then avoid treating overall dynamic MAE as a
general geometry improvement.

To isolate **causal keyframe-selection effects**, a further replay
must force identical keyframe IDs, pose estimates and map optimization
schedule across online modes. This branch does NOT implement that
replay, and it is not claimed to do so. Similarly, to prove physical
lag requires correspondence GT or a validated future RGB-D forward
prediction, not merely differences between optimized offsets.

The experiment has been source-integrated and unit tests added, but
full GPU validation must be performed in the user's environment.
