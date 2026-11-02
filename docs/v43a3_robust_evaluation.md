# V4.3a.3 — Robust RGB-D residual audit (offline, no SLAM update)

This branch inherits V4.3a.2 and adds only diagnostic code.
Run **without GPU or a new SLAM experiment** on previously saved
`v43a_XXXXX.npz` files and `v43a1_per_region.csv`.
No candidate propagation or Tracking/Mapping state is modified.

## Research question

Can ordinary dynamic-region depth MAE falsely suggest a broad candidate
benefit when only a few high-error pixels (especially around the *estimated*
dynamic mask boundary) contribute most of the net difference? Does the
winner change when residual contributions are bounded?

These are sensitivity diagnostics, **not** proof that rare large
geometry corrections are wrong. Correct occlusion boundaries often
produce large depth residual corrections.

## Run on current available material

```bash
git fetch origin
git switch feature/v43a3-robust-rgbd-audit
python -m unittest discover -s tests -p 'test_v43a3_robust.py' -v
python scripts/analyze_v43a3_robust.py \
  --npz /path/to/v43a_00212.npz /path/to/v43a_00357.npz \
  --regions /path/to/v43a1_per_region.csv \
  --output /tmp/v43a3_results
```

Outputs:
- `v43a3_pixel_robust.csv`: same observed dynamic mask for G0/Support/G4;
  dynamic-all, 3/5/10-pixel mask border and interior comparisons.
- `v43a3_tile_sensitivity.csv`: sensitivity of *tile-mean paired deltas*
  across 95 available keyframes; NOT per-pixel Huber or residual trimming.
- `v43a3_summary.txt`: representative paired differences.

## Distinguish two operations

For observed target pixels p with nonnegative depth residuals:

```
r0[p] = |D_render_G0[p] - D_gt[p]|
rh[p] = |D_render_h[p]  - D_gt[p]|
d[p] = rh[p] - r0[p]
```

Candidate-wise **bounded residual scoring** at cap c uses:
```
mean( min(rh,c) ) - mean( min(r0,c) )
```

This is a real *candidate scoring rule* with limited impact for pixels
having large residuals in both candidates.

By contrast **paired-delta clipping** uses:
```
mean( clip(d, -c,+c) )
```

It measures how much extreme candidate improvements/regressions dominate
the MAE *difference*, but it is not necessarily the difference of any
separately calculated candidate scores.

Normalized Huber `Huber(r;delta)/delta` is *linear* above delta and
therefore does not bound the influence of truly extreme residuals.
Do not expect standard Huber to eliminate extreme-error dominance.
The analysis reports several c, delta settings to avoid choosing a
threshold after looking at favorable results.

For each frame all candidates use **identical observed target masks**.
Depth scoring restricts to estimated motion mask intersect GT-valid
depth. Eroding that estimated mask defines *boundary* vs *interior*;
neither is a semantic boundary ground truth. The NPZ does NOT include
observed depth values/gradients or GT occlusion labels, so real
depth-discontinuity or occlusion analysis is **not possible** yet.

## Findings on provided V4.3a.2 files (2026-11-02)

KF212 / dynamic depth:
- ordinary mean delta: Support -2.593 mm, G4 -2.707 mm.
- paired-delta clipping at ±50mm: +0.021/+0.028 mm.
- candidate-wise residual saturation at 50mm: +0.008/+0.024 mm.
- mask interior (5px erosion): +0.266/+0.102 mm.
- 5px mask boundary: -22.703/-22.471 mm.

KF357 / dynamic depth:
- ordinary mean delta: Support +0.097 mm, G4 +0.367 mm.
- 5px mask interior: Support +0.192 mm, G4 +0.553 mm.

Full 95-keyframe tile CSV (1,490 eligible tiles):
- unbounded equal-tile mean paired depth delta:
  Support -0.867 mm, G4 -0.710 mm.
- clip each tile's *already averaged* paired delta to ±1mm:
  Support +0.014 mm, G4 +0.020 mm.

**Interpretation:** the observed mean residual gains are sensitive
to a limited number of larger errors, and not representative of the
typical dynamic-region pixel. This does NOT establish that robust
loss predicts physically correct motion or better future rendering.

## Conditions before developing V4.3b

- Keep G0 as fallback; do not pick modes by in-sample score alone.
- To validate actual predictor value, collect source AND subsequent
  detailed observations over an a priori defined set of keyframes,
  including stable, failing, boundary and non-boundary periods.
- Evaluate a frozen fixed-map holdout at the next frame first; note
  that zero-order-hold offsets are not true future flow predictions.
- Pre-register objective, mask policy, cap/Huber thresholds and
  minimum valid dynamic pixels. Compare all methods on the SAME
  holdout observations; use keyframe/sequence (not pixel) as the
  experimental unit. Bootstrap at sequence or independent-run level
  where possible.
- If no consistent held-out advantage over raw MAE and G0, stop
  candidate-scoring development and re-examine motion hypotheses.

No RGB-D frame in this archive provides actual GT moving-object
correspondences or observed depth discontinuities; do not call it
motion-accuracy ground truth.
