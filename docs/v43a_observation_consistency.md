# V4.3a — Fixed-state RGB-D Motion Hypothesis Consistency (READ-ONLY)

Branch: `feature/v43a-rgbd-motion-hypothesis-evaluation` based on V4.2b.

## Scope and experimental design

**Hypothesis:** Different scene-flow propagation candidates may better fit the
observed RGB-D at the target keyframe. Do NOT select online motions, do NOT
change Tracking or Mapping, and do NOT claim generalization from the same
selection frames.

V4.3a intercepts `Backend.apply_flow_offset` BEFORE the normal
`GaussianModel.update_delta_xyz()` call and BEFORE the subsequent new Gaussian
insertion and iterative Mapping. The code evaluates **the SAME model,
camera and target observations** with candidate displacement proposals
G0, G3-Support, G4. The model's prior offset is taken from
`closest_viewpoint.uid`. Candidate offsets are built in temporary tensors
by exactly respecting the original `norm(displacement)<0.3m` gate.
Rejected selected points and all unselected Gaussians retain the PRIOR
offset. The renderer receives temporary `dx` tensors and fixed
`ds=0,dr=0` at the current target keyframe GMM timestamp.
No candidate is written to the map; V4.2b mode `g0` writes the native G0
after evaluation.

### Why a full render rather than nearest-pixel Gaussian residuals?

Occlusion, opacity and overlapping anisotropic Gaussians influence rendered
RGB-D. Full renders use original Gaussian color, scale, rotation and GMM
opacity instead of an approximate per-Gaussian reprojection.

### Metrics

- `all/dynamic/static_rgb_mae`: RGB [0,1] mean absolute error on fixed
  observed pixels (a valid target RGB independent of which candidate is used).
  In this repository `motion_mask=True` denotes **static**; dynamic is its
  complement. The motion mask is an existing **estimated** mask, not GT.
- `all/dynamic/static_depth_penalized_mae_m`: fixed observation-depth
  mask MAE; if candidate prediction is missing/opacity<0.1, use explicit
  `uncovered_penalty_m=0.5` in that pixel. This prevents artificially low
  depth errors from selectively not rendering.
- `all/dynamic/static_depth_pred_valid_mae_m`: depth MAE only on predicted
  covered & observed pixels (conditional metric, **not comparable alone**
  across hypotheses).
- `all/dynamic/static_depth_coverage`: fraction of GT valid depth pixels
  with predicted positive depth and opacity >=0.1. Report this alongside
  errors; depth threshold/penalty are configurable and require sensitivity checks.
- Count fields expose each metric's denominator, including zero-size regions.

The RGB error is penalized implicitly at uncovered regions by whatever
background the renderer uses. Separate RGB and depth scores are saved rather
than an arbitrary weighted total. Spatial masks remain the SAME within a
keyframe across candidates.

## Run

```bash
git fetch origin
git switch feature/v43a-rgbd-motion-hypothesis-evaluation
git pull --ff-only
python -m unittest discover -s tests -p 'test_v43a_observation.py' -v
python -m unittest discover -s tests -p 'test_v42b_online.py' -v
```

Edit `configs/rgbd/bonn/config.yaml`:

```yaml
model_params:
  flow_offset: true
  dynamic_model: offset
  use_knn: true
  v42a_propagation:
    mode: 'g0'
    g34_shadow: false
    g4_ablation: false
    attribute_failures: false
    save_points: false
    # retain the other pre-existing G4 settings
  v43a_observation:
    enabled: true
    every_n_keyframe: 5
    detailed_frames: [212, 357]
    save_maps: true
    opacity_threshold: 0.1
    uncovered_penalty_m: 0.5
```

Keep `Training.flow_ds: 1` and the other Flow4DGS defaults unchanged.
Use **only `mode: 'g0'`**; any other mode is rejected so that final Mapping
remains the original propagation.

```bash
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
 --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
 --exp_name=v43a_fixedmap_g0 --save_results=0
```

Outputs:
- `<run>/v43a_observation/v43a_per_keyframe.csv` (3 rows per evaluated
  target keyframe).
- `<run>/v43a_observation/v43a_00212.npz`, etc for requested **actual
  keyframes** that were eligible; the fresh run may select different keyframes.
- `<run>/v42b_online/v42b_per_keyframe.csv` remains present.
- Existing SLAM ATE output remains available but is **not the primary
  V4.3a measure**.

```bash
python scripts/analyze_v43a_observation.py \
  --input "<run>/v43a_observation"
```

This writes `v43a_summary.txt`.

### Safety and next decision

This experiment compares **pre-optimization initialization**, not the final
optimized maps or an unseen time. If Support/G4 repeatedly win on dynamic
RGB and depth with comparable coverage, a small, held-out-frame validation
can follow; if G0 consistently wins, stop candidate-selection development.
Do not choose a candidate online in V4.3a.

The new code is not yet GPU-validated in the user's environment.
Use the unit tests first and a short smoke run. Capturing extra renderings
can change wall-clock scheduling in asynchronous SLAM, even though actual
Gaussian displacement selection remains G0. Comparing models on a single
mapping state is the scientifically controlled comparison; comparing ATE
across separate runs remains stochastic.
