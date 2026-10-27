# V4.1a Motion–Reliability Decoupling: OFFLINE SHADOW

Branch: `feature/v41a-motion-reliability-shadow` (parent V4.0b).

## Scope

This branch **does not modify SLAM Tracking or Mapping**. It takes archived
`v40b_XXXXX.npz` files and produces three *heuristic, uncalibrated*
observation scores:
- `p_dynamic` (P_D*): motion and map discrepancy evidence, NOT posterior probability.
- `q_flow` (Q_F*): correspondence agreement based on native-pixel FB cycle
  residual, boundary proximity, temporal depth support, and suspected occlusion.
- `q_map` (Q_M*): static-map availability based on opacity and valid RGB-D
  rendering, NOT estimated geometric accuracy.

`state_decoupled_3state`: 0 Static, 1 Dynamic, 2 Unknown.
Unknown covers missing evidence, weak evidence, and strong contradictory cues.
No labels or model training are used. All thresholds are preliminary.

**Important scientific caveat:** In this first interpretable prototype, a
large dynamic residual with a low map residual may yield Unknown or even
Static depending on reliability weighting. This is deliberately a testable
baseline, not a validated classifier. Likewise using the same FB score to
define Q_F and report FB consistency would be circular validation: use
manually annotated motion masks and independently checked correspondences
to assess performance.

## Input requirements

Offline-only, no new SLAM run is necessary **if** you retained V4.0b maps:
`<run>/v40b_calibration/v40b_00010.npz`, etc.

A `v40b_per_frame_offline.csv` or heatmap PNG is not enough for pixel-wise
inference. NPZs must contain keys defined by `utils/v41a_shadow.py`.

## Execute

```bash
git fetch origin
git switch feature/v41a-motion-reliability-shadow
python -m unittest discover -s tests -p "test_v41a_shadow.py" -v

RUN_DIR=$(find results/bonn/v40b_calibration -maxdepth 1 -type d \
 -name 'rgbd_bonn_balloon2_*' | sort | tail -n 1)

python scripts/run_v41a_shadow.py \
 --input "$RUN_DIR/v40b_calibration" \
 --output "$RUN_DIR/v41a_shadow" \
 --frames 60,140,220,230,310,390 \
 --save-maps selected
```

Results:
- `v41a_per_frame.csv`: state coverage and agreement against legacy mask
  (descriptive only, not precision/recall/IoU).
- `v41a_summary.txt`: mean frame statistics.
- `v41a_config.json`: exact evidence parameters.
- `v41a_visual_NNNNN.png`: RGB, three scores, four ablations.
- `v41a_NNNNN.npz`: selected scored masks, for later GT evaluation.

## Four in-sample ablations

1. Flow only: motion residual with no additional map information.
2. Naive RGB-D: fixed-weight flow plus map discrepancy, without reliability.
3. Reliability weighted binary: reliability-weighted flow and map discrepancy,
   with no explicit Unknown decision (invalid evidence remains Unknown).
4. Reliability weighted 3-state: same weighted score with explicit
   weak/contradictory evidence as Unknown.

The original `final_static` mask is read **only for descriptive agreement**.
Neither `prior_static` nor `flow_dynamic` enters inference. This prevents
the new method from trivially reproducing the original mask and calling it an
improvement.

## Reading output

- Higher `q_flow` means more internal FB consistency, not measured optical
  flow accuracy. It may be high for consistently wrong flows.
- Higher `q_map` means more covered map geometry, not proof of staticness.
- `p_dynamic` is a score from simple exponential transforms. It is not
  calibrated: do not compare numerical values to true event frequencies.
- All Unknown pixels must be included in coverage reporting, not silently
  excluded to inflate apparent accuracy.
- The visualization uses a fixed score scale [0,1] and states
  green Static / orange Dynamic / grey Unknown.
- The V4.0b dataset has no manual labels. No F1 / Precision / Recall
  or per-pixel accuracy can be claimed.
- Only BONN balloon2 has been explored; the same design must be evaluated
  on other sequences and evaluated against independent labels.
- V4.0, V4.0a, V4.0b map coverage variations make run-to-run ATE
  comparisons unsuitable for testing this OFFLINE model. The Shadow
  evaluations all use the **same** saved per-frame NPZ.

## Follow-up research design

After inspecting representative frames, prepare annotation masks including
Static / Dynamic / Ignore (occluded or ambiguous), and where possible an
independent flow correspondence error benchmark. Compare on *held-out frames*
and across sequences, report IoU/F1 alongside abstention/coverage curves.
Perform threshold sensitivity sweeps without tuning and testing on the same
frames. Only after those checks should V4.1b modify online Tracking.
