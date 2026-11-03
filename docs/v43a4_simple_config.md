# Simple Balloon2 configuration — V4.3a.4

**Use these two presets. Do not hand-edit the 369-line legacy config.**

## First test: 2 frozen source keyframes

```bash
git fetch origin
git switch feature/v43a4-simple-config
git pull --ff-only

python scripts/check_v43a4_config.py
python -m unittest discover -s tests -p "test_v43a4_config.py" -v
python -m unittest discover -s tests -p "test_v43a4_predictive.py" -v

python slam.py --config configs/rgbd/bonn/v43a4_balloon2_smoke.yaml \
  --eval --dynamic --exp_name=v43a4_simple_smoke --save_results=0
```

## Full sequence: systematic source sampling

After smoke succeeds:

```bash
python slam.py --config configs/rgbd/bonn/v43a4_balloon2_full.yaml \
  --eval --dynamic --exp_name=v43a4_simple_full --save_results=0

python scripts/analyze_v43a4_predictive.py \
  --input "<run>/v43a4_predictive/v43a4_policy_per_pair.csv"
```

**Do not supply `--inherit_from` on either run.** The two standalone
presets already include a complete, explicit inheritance chain:
```
v43a4_balloon2_smoke.yaml  --┐
                             -> v43a4_balloon2_common.yaml
v43a4_balloon2_full.yaml   --┘              |
                                           v
                               config.yaml (clean Flow4DGS core)
```

The original legacy `ballon2.yaml` is not touched. It has an
`inherit_from: configs/rgbd/bonn/base_config.yaml` line (not available
on this branch), but the new presets do **not** rely on that path.

## Files and when to edit them

| File | Role | Edit? |
| --- | --- | --- |
| `configs/rgbd/bonn/config.yaml` | 124-line Flow4DGS core; `v42a_propagation.mode='off'` | Normally no |
| `configs/rgbd/bonn/v43a4_balloon2_common.yaml` | Original Balloon2 calibration, scene overrides, V4.3a.4 analysis constants | Only for a new *research* protocol |
| `configs/rgbd/bonn/v43a4_balloon2_smoke.yaml` | Only two knobs: 5-event sampling and 2 snapshots | Run first |
| `configs/rgbd/bonn/v43a4_balloon2_full.yaml` | Only two knobs: 4-event sampling and cap 25 snapshots | Run after smoke |
| `configs/rgbd/bonn/archive/config_before_v43a4_simplification.yaml` | Unmodified 369-line legacy YAML, including all old switches | Archive; do not use for new work |

Old V1/V2 flow+RGB-D tracking weights, V3 IRLS/flow-camera diagnostic
switches, V4.0 calibration, V4.2 shadow point-dump options, and inactive
V4.3 region settings were **removed from the active YAML only**. No
algorithm implementation, Tracking/Mapping loss or mask code was
deleted. Earlier settings can be restored from the archive or their
original Git branches.

All new YAMLs use the existing `utils.config_utils.load_config` recursive
merge. The common preset explicitly sets `v42a_propagation.mode='g0'`
and enables V4.3a.4 read-only scoring. The actual online Gaussian
displacement remains G0. Other V4.2 thresholds use the existing
unchanged code defaults or retained G4 constants, rather than an
unmanaged list of stale flags.

**Technical caution:** Removing a disabled config block is expected
to preserve default-off behavior only where the code uses
`.get(...,default)`. It is not a substitute for end-to-end GPU
validation. The included configuration checker validates merged
settings and the source sampling mode, but does not test runtime
CUDA interaction, map timing, or all old features.

## To change the experiment without searching hundreds of lines

For a faster smoke test change **only**
`predictive_max_sources` under
`v43a4_balloon2_smoke.yaml`. The full preset's key controls are:

```yaml
model_params:
  v43a_observation:
    predictive_source_every_n_event: 4
    predictive_max_sources: 25
```

To compare on another scene, add a new scene preset rather than editing
shared Balloon2 calibration or the main `config.yaml`.

Logs:
- `v43a4_predictive/v43a4_policy_per_pair.csv`
- `v43a4_predictive/v43a4_candidate_scores.csv`
- `v43a4_predictive/v43a4_policy_summary.csv` (after analysis script)
