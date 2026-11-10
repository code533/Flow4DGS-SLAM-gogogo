# V5.2a — Visibility-Conditioned Historical Motion (shadow-only)

Branch: `feature/v52a-shadow-visibility-conditioned-history`
Based on V5.1d frozen-map persistence branch.

## Hypothesis and scope

A high-exclusion source (e.g., KF342) showed increasing
Strict-minus-Legacy future depth MAE, but the overall V5.1d 1/3/5
keyframe pattern was not monotone, and a source around KF397 showed
the opposite trend. Therefore **not all unseen dynamic Gaussians
should receive historical extrapolation**.

Test whether a *conservative, eligibility-gated* historical motion
step improves held-future RGB-D relative to simple Strict-FOV HOLD,
on the SAME frozen pre-Mapping map, source camera and future tracked
camera. No V5.2a motion is written to the live GaussianModel.

## Four source states

1. `hold`: inherit offset from prior keyframe.
2. `legacy`: original G0 displacement and 0.3m acceptance gate.
3. `strict`: only valid source/warped target FOV motion candidates,
   recompute original G0 KNN on those points and apply 0.3m gate.
4. `strict_history`: Strict above, plus a bounded historical offset
   step **only** on candidate Gaussians excluded by Strict-FOV
   which have reliable-looking historical changes.

The new candidate is based ONLY on the *source* map and past
keyframe offsets. Future RGB-D observations and source-frame
reconstruction metrics are NOT used to decide whether to extrapolate.

### History eligibility and geometry

For a rejected dynamic Gaussian i, let previous sorted keyframe UIDs
be k0 < k1 < k2 < source_uid, and let stored optimized offsets be
d0_i, d1_i, d2_i. Estimate per-image-frame offset change:

```
v_old = (d1_i - d0_i) / (k1-k0)
v_new = (d2_i - d1_i) / (k2-k1)
predicted_increment = v_new * (source_uid-k2)
```

Apply only if:
- i appears in the original Legacy-G0 candidate set, but not in the
  source/target FOV-valid Strict subset.
- All three previous stored offset tensors have the SAME topology.
- The Gaussian's reported creation-keyframe ID <= k0.
- All image-frame gaps are strictly positive and <= 20.
- Both past displacement norms >= 0.005 m.
- Cosine(v_old,v_new) >= 0.5.
- Speed ratio new/old lies within [0.5,2].
- New speed <= 0.04 m/image-frame.
- Source incremental offset is magnitude-clipped at 0.10 m.

The candidate uses exactly the source-created Strict offsets plus
these increments; it does not modify the original Strict candidate
or Legacy baseline.

**Caveat:** GaussianModel's `unique_kfIDs` is a creation-keyframe
metadata field, not a persistent physical identity across split/
merge/densification. Historical offsets may have changed through
Mapping optimization. These guards reduce unreliable extrapolation
but do NOT prove 3D velocity is correct.

For source/target invalid Gaussians *without* eligible coherent
history, strict_history is **identical to strict**.

### Future protocol

V5.1d source-sampling and frozen-map semantics are unchanged.
On incoming future keyframe order 1/3/5, render all FOUR source
states using the frozen Gaussian map and that keyframe's observed
camera/GT-RGB-D. All four keep source offsets frozen; no new future
velocity is calculated. Save actual gap in image-frame units.
Compare:
- Strict+History minus Strict: **primary** future dynamic Depth MAE.
- Secondary dynamic RGB MAE and PSNR, dynamic interior/boundary
  Depth MAE, and predicted Depth coverage.
- Strict+History minus Legacy, Strict minus Legacy as interpretive
  comparisons.
- Source eligibility count `n_history_predicted` and clipping count.

Do not optimize thresholds using these focus frames, and do not
mistake a better held-future RGB-D fit for motion ground truth.
The source focus KFs [342,347,352,357,362,372,382] are NOT
independent heldout data (they were chosen from earlier experiments).
Full-map frozen renders have growing absolute depth residual at
longer horizons.

## Run commands

```bash
git fetch origin
git switch feature/v52a-shadow-visibility-conditioned-history
git pull --ff-only

python -m unittest discover -s tests -p 'test_v52a_conditioned_history.py' -v
python -m unittest discover -s tests -p 'test_v51d_temporal.py' -v

# Smoke: one source, check history eligibility and memory
python slam.py \
  --config configs/rgbd/bonn/v52a_balloon2_smoke.yaml \
  --eval --dynamic --exp_name=v52a_smoke --save_results=0

# Full: up to twelve sources (one frozen map in memory at a time)
python slam.py \
  --config configs/rgbd/bonn/v52a_balloon2_full.yaml \
  --eval --dynamic --exp_name=v52a_full --save_results=0
```

There is NO `--inherit_from` CLI argument. Presets inherit
V5.1d's clean V5.1a shadow behavior, all thresholds set explicitly.
If a scene ends or focus source is skipped, the event log records
missing/censored horizons. Online ATE is not an V5.2a performance
measurement because no history prediction is applied to Mapping.

### Outputs

```text
<run>/v51d_temporal/
  v51d_event_log.csv
  v51d_source_per_keyframe.csv    # V5.1d three-way source (optional reference)
  v52a_source_per_keyframe.csv    # V5.2a four-way source + eligibility counters
  v52a_future_per_horizon.csv     # V5.2a four-way held-future RGB-D
```

Analyze:

```bash
python scripts/analyze_v52a_history.py \
  --input "<run>/v51d_temporal"
```

Output:

```text
<run>/v51d_temporal/v52a_analysis/
  v52a_four_way_paired.csv
  v52a_history_vs_strict_summary.csv
  v52a_missing_horizons.csv
  v52a_report.md
```

The analyzer checks four-way completeness and equal future
observation/segmentation masks. It separates:
- all available source/horizon pairs,
- sources with all three horizons,
- sources with at least one history-predicted Gaussian,
- complete sources with nonzero history eligibility.

**Go/no-go rule:** A positive signal requires repeatable lower future
dynamic Depth MAE with Strict+History on multiple complete *and*
history-eligible sources, no systematic deterioration of RGB/coverage
and replication on an independent scene. A one-source 342 improvement
is not sufficient. Do not tune the six gates to maximize Balloon2.

## Important limitation

This is V5.2a **shadow verification**, not an online replacement
for the Strict motion updater. The frozen state is itself an
approximate short-horizon zero-order-hold prediction. The model
does not establish true occlusion age, world-frame velocity,
physical correspondence identity or future extrapolation beyond
the single source increment.

The backend snapshot is a full GaussianModel deep clone and three
to four GPU renders per evaluated frame. Ensure GPU memory is
sufficient and test the one-source smoke version first.
