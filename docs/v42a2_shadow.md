# V4.2a.2 — Visibility-aware and motion-consistent Gaussian propagation (Shadow)

Branch: `feature/v42a2-visibility-motion-shadow`; parent V4.2a.1.

## Purpose

Compare FIVE dynamic Gaussian movement candidates on IDENTICAL per-keyframe
observations: G0 original KNN, G1 flow-Q weighted, G2 conservative reliability
fallback, G3 visibility-aware, and G4 visibility + robust neighborhood motion
consensus. **V4.2a.2 cannot apply G3/G4 to the map**. With `mode: shadow`,
the actual Gaussian update is G0, exactly as in V4.2a.1.

### G3: visibility-aware

For each selected Gaussian, use the V4.2a.1 validity *reason*:
- reasons 0 and 8 have geometrically and temporally VALID correspondences;
- reasons 1–7 failed projection/validity checks.

G3 = G1 only when reason is 0 or 8; otherwise G3 = **exact G0**.

A zero-Q Gaussian (reason 8, numerical FB disagreement) is different from
an unobservable Gaussian (reason 1–7). G3 does not clip NDC projections or
invent a flow value. This is a baseline policy, not a verified motion prior.

### G4: trusted medoid and motion-consistency weights

For each Gaussian, find KNN neighbors that (i) are inside KNN radius,
(ii) themselves have valid correspondence, and (iii) Q >= trusted-Q.
Take their reliability-weighted **medoid** in raw 3D motion space as a
robust motion anchor when the TARGET has low Q. A target with high Q uses
its own raw motion as anchor. Multiply spatial-Q weights by:

```
C_ij = exp(-||raw_motion_j - anchor_i||^2 / (2 * sigma_m^2))
```

Only use robust propagation if enough trusted neighbors, sufficient weight
support and low trusted-neighbor dispersion; otherwise **G4 = exact G0**.
Also G4=G0 for invalid target correspondence. G4 never assumes an unseen
object is stationary.

**Important:** G4's consensus tests use the same raw motion estimates
being propagated; a smooth but incorrect flow can pass. A medoid is not
a ground truth motion entity label, nor does it establish that neighbors
belong to the same object. The method may be conservative and reject
actual useful improvements. Do not infer accuracy from smaller changes.

## Run

```bash
git fetch origin
git switch feature/v42a2-visibility-motion-shadow
python -m unittest discover -s tests -p 'test_v42a2_shadow.py' -v
python -m unittest discover -s tests -p 'test_v42a1_diagnostics.py' -v
python -m unittest discover -s tests -p 'test_v42a_propagation.py' -v
```

Edit `configs/rgbd/bonn/config.yaml` in the model_params section:

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
  g34_save_points: true
  g34_detailed_frames: [22, 212, 357, 382]
  g4_motion_sigma_m: 0.08
  g4_trusted_q: 0.30
  g4_min_neighbors: 3
  g4_min_support: 0.25
  g4_max_dispersion_m: 0.12
  g4_max_delta_from_g0_m: null
```

Keep all prior tracking/detection experiments OFF and
`flow_offset: true`, `use_knn: true`, `flow_ds: 1`.
Do NOT use `mode: g1` or `mode: g2` when `g34_shadow: true`:
the program raises a ValueError to prevent accidental map intervention.

```bash
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
 --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
 --exp_name=v42a2_shadow --save_results=0
```

Outputs:
- `<run>/v42a2_shadow/v42a2_per_keyframe.csv`: per-keyframe G3/G4
  fallback, movement, consensus, 0.3m gate-change statistics.
- `v42a2_XXXXX.npz`: detailed raw/G0/G1/G2/G3/G4 displacement and
  pointwise diagnostics for selected keyframes actually encountered.
- Prior `<run>/v42a_propagation/v42a_per_keyframe.csv` still produced.
- Prior V4.2a.1 full attribution CSV is also produced automatically when
  g34_shadow is enabled, unless no dynamic propagation was triggered.
  Disable `save_attribution_points` to reduce storage.

```bash
python scripts/analyze_v42a2_shadow.py --input <run>/v42a2_shadow
```

Produces `v42a2_summary.txt`.

## What to check before any full Mapping experiment

- G3 fallback proportion by visibility reason, especially KF357.
- G4 consensus accepted fraction, especially KF212.
- G3/G4 versus G0 **and** 0.3m-gated effective displacement changes.
- Counts of new reject/accept decisions at the .3m gate.
- Whether the output is trivially G0 everywhere (conservative settings),
  or has extreme local outliers.
- Repeat runs do not yield exactly matched Gaussian sets because
  multi-process SLAM may vary. Use within-keyframe shadow comparisons;
  do not compare different run's point arrays elementwise.

Only after these checks should G3/G4 become selectable online modes.
Even if Shadow diagnostics improve, this is NOT an ATE/PSNR test.
