# V5.1d — Frozen-map short-horizon motion persistence

Branch: `feature/v51d-frozen-short-horizon-persistence`
Base: V5.1c controlled motion attribution.

## Research question

Earlier V5.1c found the Strict-FOV candidate had **higher immediate
dynamic depth MAE in 9/10 samples**, while V5.1a repeatedly improved
tracking ATE and V5.1b showed mixed final-map reconstruction gains.
Does the source-time HOLD, Legacy or Strict offset hypothesis still
explain observations at **later keyframes**, and does the
Strict-minus-Legacy difference systematically change with horizon?

This experiment is **not** a new motion propagation method. It is
read-only and deliberately leaves actual online Mapping at **Legacy
G0**, using the existing V5.1a `shadow` mode.

## Exact meaning of the horizons

```yaml
future_keyframe_orders: [1,3,5]
```

These are the **first, third and fifth subsequently received keyframes**
following the sampled source keyframe, **not** individual RGB-D frames
t+1/t+3/t+5. The backend has tracked viewpoint objects for keyframes,
not for all intermediate video frames. The output logs
`elapsed_image_frames=future_uid-source_uid` for each observation.

At the source keyframe, before its flow update or Mapping optimization:

1. Generate the three gated dynamic-offset arrays: HOLD, Legacy, Strict.
2. Render current-frame triplet using the same pre-Mapping Gaussian map
   (V5.1c routine). Source must have >=256 observed dynamic depth pixels.
3. Deep-clone the GaussianModel **once**, and freeze the three source
   offset arrays. Only ONE snapshot is pending at a time.
4. Continue ordinary G0 Mapping. No V5.1d candidate is written online.

When future keyframes arrive, BEFORE new Gaussian insertion, source
flow updates or Mapping, re-render each frozen-source candidate at
that future keyframe's *tracked* camera pose and RGB-D observation.
Use the same frozen geometry/appearance parameters for all candidates,
the same original source displacement and exact <0.3m gate. The
renderer may still evaluate its existing time-dependent Gaussian
attributes at the **future UID**, identically for all candidates.

There is NO newly estimated future scene flow, motion extrapolation,
or progressive map optimization within any candidate. Accordingly,
this is a **zero-order-hold persistence** experiment, **NOT a full
prediction of a Gaussian's physical future position or true occlusion**.

The camera pose is taken from running SLAM; it can reflect the G0
online map, and is NOT independent ground-truth pose. Comparing the
three modes within each future frame is still paired/fair.

## Source sampling and memory protection

Default:
```yaml
every_n_event: 10
max_sources: 12
future_keyframe_orders: [1,3,5]
focused_source_frames: [342,347,352,357,362,372,382]
max_frame_gap: 45
```

The systematic source rule is eligible propagation event 1, 11, 21, ...
The listed focus frame IDs are prespecified from previous diagnostics.
Both strata are labeled: `systematic`, `focused`, or
`systematic_and_focused`. Keyframes are chosen stochastically; focus
IDs may never be eligible. **No fallback IDs are substituted.**

If a focus/source candidate arrives while a snapshot is pending, the
source is SKIPPED and logged. If a future frame gap exceeds 45 image
frames, the source is closed and logged; remaining horizons are
missing, not silently zero-filled. Likewise, the source cap, too few
valid dynamic depth pixels or end of sequence get distinct log codes.
There is at most **one GPU GaussianModel deep copy**, not an unbounded
list of snapshots. It may still use considerable GPU memory.

### Quick smoke test

```bash
git fetch origin
git switch feature/v51d-frozen-short-horizon-persistence
git pull --ff-only
python -m unittest discover -s tests -p 'test_v51d_temporal.py' -v
python -m unittest discover -s tests -p 'test_v51c_controlled.py' -v

python slam.py \
  --config configs/rgbd/bonn/v51d_balloon2_smoke.yaml \
  --eval --dynamic --exp_name=v51d_smoke --save_results=0
```

This smoke preset caps sources at ONE. It is not intended to produce
a statistically useful time trend. Check memory and the event logs.

### Full experiment

```bash
python slam.py \
  --config configs/rgbd/bonn/v51d_balloon2_full.yaml \
  --eval --dynamic --exp_name=v51d_full --save_results=0
```

**Do not supply `--inherit_from` on the CLI.**
Both configs inherit the known V5.1a Shadow and clean Bonn baseline.
V4.3 candidate selection, V5.0 histogram and V5.1c are off.
They are mutually excluded from V5.1d in the backend.

The extra full-map GPU snapshot and 3 future renders can affect
SLAM scheduling. **Do not use V5.1d ATE as a causal improvement**;
within-snapshot paired rendering is the object of this study.

## Outputs

```
<run>/v51d_temporal/
  v51d_event_log.csv
  v51d_source_per_keyframe.csv
  v51d_future_per_horizon.csv
```

- `v51d_source_per_keyframe.csv`: three matched source-frame rows for each frozen snapshot, useful when future observations are missing.\n- `v51d_event_log.csv`: source captures/skips, future evaluations,
  per-source completed/expired/stream-ended status and reasons.
- `v51d_future_per_horizon.csv`: three rows (hold/legacy/strict)
  per evaluated (source, future order) pair; future RGB, depth, PSNR,
  coverage, interior and boundary residuals; source residuals and
  excluded Gaussian history-activity proxy.

At most 12 *completed source snapshots* are attempted. If none yields
a valid future dynamic depth region, the future CSV need not exist.
Use event logs to see why; this is not a successful zero-error result.

Run:

```bash
python scripts/analyze_v51d_temporal.py \
  --input "<run>/v51d_temporal"
```

Analysis files:
- `analysis/v51d_paired_future_deltas.csv`: same-source/same-future
  Strict–Legacy, Strict–Hold and Legacy–Hold deltas.
- `analysis/v51d_horizon_summary.csv`: each horizon, all available
  sources vs **complete sources only**.
- `analysis/v51d_missing_horizons.csv`: explicit gaps/reasons.
- `analysis/v51d_history_vs_future.csv`: association between source
  past activity and the next-keyframe depth difference. Not
  ground-truth motion lag.
- `analysis/v51d_report.md`: compact primary findings.

## How to interpret

For RGB/Depth MAE:
```
delta = E(Strict) - E(Legacy)
negative => Strict has lower error.
```

For dynamic PSNR a **positive** delta is better.

Report:
1. N and actual elapsed image-frame gap for KF orders 1/3/5;
2. Dynamic depth MAE and dynamic RGB MAE deltas, separate interior,
   boundary and coverage;
3. Same-source triplets across all horizons (not shifting source sets);
4. Cases where high historic offset change accompanies relative
   Strict degradation, as a **hypothesis only**;
5. Absence/censoring of focus cases, and keyframe intervals.

The prespecified exit-window focus list was chosen **after** earlier
V5.0/5.1a observations. It is qualitative mechanism analysis, not a
held-out independent test of a new method. Valid time-trend evidence
requires multiple sources and ideally more than one independent scene.

### Stop/go

- If Strict–Legacy depth difference grows increasingly positive for
  complete-source trajectories with increasing elapsed image gap,
  the **hypothesis** that skipping new motion harms persistence is
  strengthened. But a frozen offset is itself a motion-model
  limitation, so this does NOT prove true position lag.
- If no systematic trend appears, don't add historical motion
  prediction purely to rescue V5.1c.
- If local vs final-map findings disagree strongly, conduct a
  controlled keyframe/pose/Mapping replay before designing V5.2.
- Real V5.2 must be evaluated with actual future motion estimation
  or heldout RGB-D geometries, tracking accuracy and dynamic metrics.
