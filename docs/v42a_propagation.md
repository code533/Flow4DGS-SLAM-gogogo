# V4.2a — Confidence-Guided Gaussian Propagation

Branch: `feature/v42a-reliability-gaussian-propagation`, based on V4.1b.

## Research goal

Test whether optical-flow **correspondence consistency** can improve
online dynamic Gaussian propagation. Unlike V4.1a, dynamic classification
is unchanged: the original YOLO/flow segmentation and Tracking are used.
No claim of 4D reconstruction improvement is made before controlled runs.

The backend already computes the two directions of full-resolution RAFT
flow at each dynamic keyframe. V4.2a reuses both; **no new RAFT inference**
is needed. `Q_F` is a heuristic, not calibrated flow accuracy.

## Ablation modes

The `model_params.v42a_propagation.mode` property is:

- `off`: original backend propagation (no extra algorithm).
- `shadow`: compute G0/G1/G2 from identical points, displacements, depth,
  flow, poses and KNN neighborhoods; **apply G0 only**.
- `g0`: compute all candidates, explicitly apply legacy G0.
- `g1`: use spatial KNN multiplied by flow-reliability Q.
- `g2`: G1 for supported neighborhoods; else revert to exact G0.

If no usable weighted neighbors exist, G1 also falls back to G0.
G2 does not assume a moving object is stationary. DynamicMask, Tracking,
flow losses, depth filling, keyframe selection and insertion remain unchanged.

Original KNN smoothing remains in the implementation so the G0 result
is exactly computed by the existing `smooth_deformation_knn_radius`.
Quality weights use normalized full-res RAFT flow converted to native
pixel coordinates. The original flow displacement projection is **not
modified** by this diagnostic branch.

## First test: Shadow only

```bash
git fetch origin
git switch feature/v42a-reliability-gaussian-propagation
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
    save_points: true
```

Run:
```bash
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
 --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
 --exp_name=v42a_shadow --save_results=0
```

Output: `<run>/v42a_propagation/v42a_per_keyframe.csv`, and
`v42a_NNNNN.npz` when `save_points: true`.

CSV includes counts, Q, FB error, propagation distances, fallback rates.
NPZ contains raw, G0, G1, G2 displacement candidates and per-point Q.
The NPZs are for **direct paired comparison of motion initializations**;
they are not 3D trajectory ground truth.

## Next experiments

When Shadow results are numerically valid and no indexing/shape error
occurs, repeat with `mode: g0`, then `g1`, `g2`, using otherwise matched
configurations. Report ATE, dynamic and static rendering metrics (if
suitable region masks are available), Gaussian counts, mapping time,
FB reliability histograms, and multiple independent repeated runs.
The backend is asynchronous: do not attribute one-run ATE changes to
G1/G2 without repeats and motion-specific evidence.

## Requirements and caveats

- `flow_ds=1` for the propagation Q computation, with flow/depth [H,W].
- `use_knn=true`, `flow_offset=true`, dynamic_model=`offset`.
- Q uses previous->current/ current->previous cycle agreement, depth
  validity at warped positions, and previous-frame depth discontinuity.
- All reliability weights are detached, no new gradients or loss terms.
- G2 low-confidence fallback chooses **original G0**. This intentionally
  prioritizes stability and may not improve performance. A truly
  different predictive prior requires a separate design/ablation.
- V4.0–V4.1 diagnostic switches should remain disabled.
- Saving NPZ for every mapping keyframe can consume significant disk space.
