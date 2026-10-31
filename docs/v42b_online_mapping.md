# V4.2b — Online Gaussian Mapping propagation ablation

Branch `feature/v42b-online-gaussian-mapping-ablation` on top of V4.2a.3.

## What is implemented

The backend selects the **actual dynamic Gaussian displacement increment**
immediately before the existing `gaussians.update_delta_xyz(...,
max_motion=0.3)`. It does not modify dynamic masks, Tracking, mapping
losses, Gaussian insertion, deformation optimizer or the max-motion gate.

Modes under `model_params.v42a_propagation.mode`:

| Mode | Actual displacement written | Purpose |
| --- | --- | --- |
| `'off'` | original G0, no new reliability computation | native baseline |
| `'shadow'` | original G0; calculate candidate diagnostics | existing V4.2a methods |
| `g0` | original G0 with all G3/G4/Support candidates computed | *instrumented* baseline |
| `g3` | V4.2a.2 G3: observable => G1, else G0 | visibility ablation |
| `g3_support` | V4.2a.3 Support-only result | **primary simpler alternative** |
| `g4` | exact V4.2a.2 G4 | full motion-consistent algorithm |

The prior `g1` and `g2` modes remain available, unchanged.

**G0 comparison controls:**
- `off` is truly the original path. It may be faster than instrumented
  modes, affecting timings in asynchronous SLAM.
- `g0` computes the SAME G0/G1/G2/G3/Support/G4 candidates as other
  modes, then applies **only original G0**. This is the preferred
  primary algorithmic control for initial ATE and timing comparisons.
  The additional calculation and device->CPU logging still introduce
  overhead and can affect asynchronous scheduling; matched
  compute is not a guarantee of identical scheduling.
- All non-off instrumented online modes emit a per-keyframe CSV.
  The reported timer uses host `time.perf_counter` without CUDA
  synchronization; it is a host-side *approximate*, not trustworthy
  GPU-kernel latency.

## Run

```bash
git fetch origin
git switch feature/v42b-online-gaussian-mapping-ablation
git pull --ff-only
python -m unittest discover -s tests -p 'test_v42b_online.py' -v
python -m unittest discover -s tests -p 'test_v42a3_ablation.py' -v
python -m unittest discover -s tests -p 'test_v42a2_shadow.py' -v
```

Keep the existing `flow_offset: true`, `use_knn: true`,
`dynamic_model: offset`, and `Training.flow_ds: 1`.
Under `configs/rgbd/bonn/config.yaml`, change:

```yaml
model_params:
  v42a_propagation:
    mode: 'g0'            # g0, g3, g3_support, g4; 'off' for native
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
    g34_shadow: false
    g34_save_points: false
    g4_ablation: false
    g4_ablation_save_points: false

    g4_motion_sigma_m: 0.08
    g4_trusted_q: 0.30
    g4_min_neighbors: 3
    g4_min_support: 0.25
    g4_max_dispersion_m: 0.12
    g4_max_delta_from_g0_m: null
```

**Do not** set either `g34_shadow` or `g4_ablation` to true
in an online `g3`, `g3_support` or `g4` run; validation rejects it.

Use a UNIQUE experiment name on each run. Example for g0:

```bash
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
  --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
  --exp_name=v42b_g0_run01 --save_results=0
```

Then change ONLY `mode` and `exp_name` for g3, g3_support, g4.

The output includes:
- `<run>/config.yml`: the actual resolved run config.
- `<run>/outputs.txt`: console output.
- `<run>/v42b_online/v42b_per_keyframe.csv`: one row for each
  successful dynamic Gaussian propagation keyframe.
- Native pose/render evaluation files according to original Flow4DGS setup.

The online row includes Q validity, candidate means, selected motion
magnitude, selected vs G0, `max_motion=.3` gate pass numbers, added
reject/accept vs G0, and host-side approximate timing.

Postprocess:

```bash
python scripts/analyze_v42b_online.py --inputs \
  <run_g0>/v42b_online/v42b_per_keyframe.csv \
  <run_g3>/v42b_online/v42b_per_keyframe.csv \
  <run_support>/v42b_online/v42b_per_keyframe.csv \
  <run_g4>/v42b_online/v42b_per_keyframe.csv \
  --output v42b_online_summary.txt
```

Run `off` separately as a native-baseline sanity check; it
intentionally produces no V4.2b CSV.

## Evaluation plan

First **smoke test** a single g0 run, inspect outputs, then g3_support,
then g4; add g3. Evaluate ATE, whole-scene rendering (PSNR/SSIM/LPIPS
if original eval enables them), motion gate health, mapping time and
Gaussian count. Use *independent repeated runs*, ideally >= 5/mode,
and paired seeds when supported. A stochastic trajectory change on
one balloon2 run, especially mm-level ATE, is not evidence of gain.
Do not fabricate ATE improvements from the displacement-difference CSV.

Only consider follow-up benchmarks if the initial Bonn run shows
plausible behavior; then validate on multiple Bonn sequences and
other appropriate dynamic RGB-D sequences.

## Current limitations

V4.2b is source-integrated but **not GPU-validated in the user's
actual environment**. Its first use should be tested for tensor
indexing, frame availability, diagnostic cost, and whether configuration
inherits YAML options as intended.

`g4` uses O(N*K^2) medoid candidate calculations (K normally 16);
`g0` intentionally incurs the same calculation overhead. Host
timing is approximate and does not capture asynchronous CUDA work.

No independent Gaussian motion GT is provided: minimizing difference
from G0 is NOT the same as improving accuracy.
