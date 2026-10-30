# V4.2a.1 — Read-only Gaussian propagation failure attribution

Branch `feature/v42a1-propagation-failure-attribution`, based on V4.2a.
Adds **diagnostics only** to the existing G0/G1/G2 candidates.
No modification to flow, Gaussian motion update, Tracking or Mapping loss.
Run in `shadow` mode so the map still uses original G0.

## Research questions

1. Why does the Gaussian-level Q become zero when image flow stays valid?
2. Are G1 outliers spatial KNN displacements driven by incompatible local
   motions and neighbor weights?
3. How does the original `max_motion=.3m` update gate change which
   Gaussians receive a displacement increment?

## Run

```bash
git fetch origin
git switch feature/v42a1-propagation-failure-attribution
python -m unittest discover -s tests -p 'test_v42a1_diagnostics.py' -v
python -m unittest discover -s tests -p 'test_v42a_propagation.py' -v
```

Edit `configs/rgbd/bonn/config.yaml`:

```yaml
model_params:
  flow_offset: true
  use_knn: true
  v42a_propagation:
    mode: shadow
    fb_scale_px: 1.0
    edge_abs_m: 0.08
    edge_rel: 0.06
    edge_penalty: 0.65
    min_support: 0.45
    min_neighbors: 3
    neighbor_q_min: 0.3
    save_points: false
    attribute_failures: true
    save_attribution_points: true
    detailed_frames: [22, 212, 357, 382]
    motion_mismatch_m: 0.10
```

Set all other optional V1–V4.1 diagnostics/experiments to OFF,
`flow_ds=1`, and keep original dynamic_model=offset.

```bash
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
 --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
 --exp_name=v42a1_shadow --save_results=0
```

Output:
- `<run>/v42a1_attribution/v42a1_per_keyframe.csv` on every
  instrumented dynamic keyframe.
- `v42a1_00022.npz`, etc only for selected frames actually triggered by
  propagation; choose `detailed_frames: []` to save all eligible frames
  (possibly much larger). The keyframe IDs in a fresh run may differ
  because mapping is asynchronous.
- Original `<run>/v42a_propagation/v42a_per_keyframe.csv` remains
  generated independently.

Then run:

```bash
python scripts/analyze_v42a1_attribution.py \
 --input "$RUN_DIR/v42a1_attribution"
```

where `RUN_DIR` is the actual V4.2a.1 experiment output directory,
**not** the previous V4.2a folder.

## Primary reason labels

Primary reasons are mutually exclusive and exhaustive over all dynamic
Gaussians, with priority order:
0. valid nonzero Q
1. nonfinite projection NDC
2. projected outside the NDC image
3. invalid previous-frame depth at the sampled pixel
4. invalid forward flow
5. out-of-frame forward correspondence
6. invalid bilinear-warped backward flow
7. invalid warped current-frame depth
8. Q <= 1e-6 despite a valid correspondence

Code 8 can be due to extreme FB error or low nonzero Q, including
configuration with zero edge penalty; it is **not** verified optical
flow error. Code 0 may have Q close to zero but greater than 1e-6.
The field `all_dynamic_...` uses the entire dynamic Gaussian set;
`selected_...` uses ONLY those points retained after the original
projection/depth/motion filters and entering KNN. Do not conflate these
denominators.

The two original V4.2a values `q_zero_fraction` (threshold <=1e-6)
and `g1_vs_g0_mean_m` remain unchanged. Detailed NPZ stores pointwise
reason, projection pixel, Q, 3D position, displacement, KNN confidence,
dominant neighbor index (index into selected KNN points), and 0.3 m
gate-pass decisions for G0/G1/G2.

## Motion mismatch proxy

`cross_motion_risk_proxy` flags low-Q Gaussians whose dominant,
high-weight neighbor differs in motion by >= 0.1m. This is a
**local disagreement heuristic**, not independent moving object or
cross-entity ground truth. The original KNN weights, neighbors and G0
smoother remain untouched. For true cross-object propagation, need
independently labeled motion entities or verified trajectory clusters.

## Gate interpretation

`update_delta_xyz(..., max_motion=.3)` checks strict
`norm(increment)<.3`. Failing Gaussian retains its previous stored
delta; it is NOT clamped to 0.3m. Effective increment summaries count
the accepted increment and zero otherwise, preserving this distinction.
Do not equate G0/G1 displacement disagreement with motion accuracy.

## Experimental boundaries

Since diagnostics add computation/IO in an asynchronous SLAM system, the
run may change timing/coverage despite the same update formulas. First
verify numerical consistency and causal reason breakdown; only then
compare full g0/g1/g2 experiments with repeated controlled runs.
This branch does not yet establish any PSNR/ATE improvement.
