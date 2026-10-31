# V4.2a.3 — Paired decomposition ablation of G4 (SHADOW ONLY)

Branch `feature/v42a3-g4-mechanism-ablation`, based on V4.2a.2.

## Scientific purpose

The V4.2a.2 result was conservative, but it was uncertain whether the
reduced motion changes came from (1) visibility, (2) trusted neighbor
requirements, (3) soft motion-consistency weighting, or (4) the hard
consensus-dispersion threshold. V4.2a.3 isolates these stages **using
the same raw point motion, same flow reliability, same KNN graph and
same per-Gaussian G0 reference in each keyframe**.

The four paired stage outputs are:

| Stage | Same visibility mask | Trusted neighbor selection | Soft compatibility weights | Hard dispersion cutoff |
| --- | --- | --- | --- | --- |
| `g3` | Yes | No | No | No |
| `support_only` | Yes | Yes | No | No |
| `soft_no_hard_dispersion` | Yes | Yes | Yes | No |
| `g4` | Yes | Yes | Yes | Yes |

The `soft_no_hard_dispersion` stage still requires finite positive soft
evidence and honors the existing optional maximum difference-from-G0
bound only in full G4, so stages differ in their **eligible population**
if soft evidence is absent. To isolate the *weighting* effect, use the
`soft_vs_support_shared_mean_m` metric: it compares two raw candidates
only on the common accepted support set, independent of extra rejections.
Full G4 is reproduced **bit-for-bit from the existing V4.2a.2 code**
and enforced by an assertion. No new optimizer parameters, no loss
term, and no change to the actual Mapping.

## Setup

```bash
git fetch origin
git switch feature/v42a3-g4-mechanism-ablation
git pull --ff-only
python -m unittest discover -s tests -p 'test_v42a3_ablation.py' -v
python -m unittest discover -s tests -p 'test_v42a2_shadow.py' -v
python -m unittest discover -s tests -p 'test_v42a1_diagnostics.py' -v
```

Change `configs/rgbd/bonn/config.yaml` under `model_params.v42a_propagation`:

```yaml
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
  attribute_failures: false
  save_attribution_points: false
  detailed_frames: [22, 212, 357, 382]
  motion_mismatch_m: 0.10

  g34_shadow: true
  g34_save_points: false
  g34_detailed_frames: [22, 212, 357, 382]
  g4_motion_sigma_m: 0.08
  g4_trusted_q: 0.30
  g4_min_neighbors: 3
  g4_min_support: 0.25
  g4_max_dispersion_m: 0.12
  g4_max_delta_from_g0_m: null

  g4_ablation: true
  g4_ablation_save_points: true
  g4_ablation_frames: [22, 212, 357, 382]
```

Keep `flow_offset: true`, `use_knn: true`, `flow_ds=1`.
Disable older unrelated tracking / segmentation experiments.

```bash
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
  --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
  --exp_name=v42a3_shadow --save_results=0
```

The `g4_ablation` option automatically activates the V4.2a.2 G3/G4
shadow calculation and V4.2a.1 visibility reasoning, even if older
`g34_shadow` or `attribute_failures` switches are false.
If `g4_ablation=true` while `mode` is not `shadow`, the backend
raises an error instead of changing the map.

## Result files

- `<run>/v42a3_ablation/v42a3_per_keyframe.csv`: four rows per
  eligible keyframe, stage, accepted fraction, G0 distance, >10cm
  changes, .3m update gate accept/reject numbers, effective difference.
- `v42a3_XXXXX.npz`: pointwise candidates and gate masks on selected
  keyframes. With `g4_ablation_frames: []`, all frames are saved;
  storage use can be substantial.
- `<run>/v42a2_shadow/v42a2_per_keyframe.csv` and older diagnostics
  are also produced.

```bash
python scripts/analyze_v42a3_ablation.py \
  --input "<run>/v42a3_ablation"
```

Produces `v42a3_summary.txt`.

## Reading the results

- `g3` is visibility-gated G1.
- `support_only` uses only trusted neighbors (same KNN graph, no
  motion consistency).
- `soft_no_hard_dispersion` adds motion similarity weights without
  the original G4's hard dispersion rejection.
- `g4` is byte-for-byte the original G4 candidate.

**Interpret accepted fraction alongside the movement statistics.**
A low G0 distance may merely mean more fallback, not better accuracy.
The soft weighting metric is computed on a shared accepted region and
is therefore more informative than G4-vs-G0 alone.

All outputs are uncalibrated motion hypotheses and internal diagnostics.
The Shadow run applies G0, so it does **not** establish ATE/PSNR benefit.
Full online Mapping experiments and repeated seeds/sequences are
required before any reconstruction improvement claim.
