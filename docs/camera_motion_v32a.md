# V3.2a: SE(3) composition and optical-flow sign consistency (offline)

V3.2a diagnoses **coordinate conventions**, not Tracking quality. It neither
changes the Flow4DGS-SLAM estimator nor applies any candidate pose to the SLAM
frontend or backend. Run it on the **original** `camera_motion_v31a_XXXXX.npz`
files from an existing V3.1a Shadow Mode experiment. The previously exported
`v31a_per_frame.csv` and `v31b_per_frame.csv` do **not** include both twists and
cannot replace the NPZ files. No CUDA or new SLAM run is needed.

## Hypotheses

For the identical estimated previous-frame world-to-camera matrix `Tprev`,
using the saved raw twist `xi_legacy_common` or `xi_geo`, compute

| Label | Hypothetical *uncapped* current w2c pose |
|---|---|
| `right_pos` | `Tprev @ Exp(xi)` (original Flow4DGS-SLAM composition) |
| `left_pos` | `Exp(xi) @ Tprev` |
| `right_neg` | `Tprev @ Exp(-xi)` |
| `left_neg` | `Exp(-xi) @ Tprev` |

These comparisons **do not** include the original translation normalization or
rotation step cap. V3.1a's `xi_legacy_actual`, if present, is also analyzed as
an additional reference; `xi_legacy_common` and `xi_geo` use the same explicit
input-validity mask. The saved `T_tracking_initial_actual` is evaluated
separately (this is a *clipped*, real initialization pose, **not** one of the
four uncapped hypotheses).

Two relative-transform conventions are output. Under **world-to-camera** input
poses, the current-camera-from-previous-camera motion is
`Tcurrent @ inv(Tprev)` (`--convention camera`, the physically meaningful
camera-to-camera frame). For compatibility with V3.1a's original offline CSV,
`inv(Tprev) @ Tcurrent` is also reported (`--convention right`). Both estimates
and GT always use the *same* selected convention. The alternatives are not
interchangeable. Note that `Tprev` is the **estimated**, not GT, previous pose;
the analysis is a one-step, fixed-previous-pose comparison, **not ATE**.

## Run

```bash
git fetch origin
git switch feature/cimd-coordinate-consistency-v32a
python -m unittest discover -s tests -p test_camera_motion_v32a.py -v

# No full SLAM run is required: point at the prior V3.1a run directory.
python scripts/analyze_camera_motion_v32a.py \
  results/bonn/v31a_shadow_/YOUR_RUN_DIRECTORY \
  --convention both \
  --min-gt-motion-mm 3.0 \
  --csv results/v32a_coordinate_analysis.csv
```

The script prints a synthetic geometry check, number of valid/failed NPZs and
paired median per-frame errors / counts for each hypothesis. The **best_counts**
values count the minimum reported error for each row, with ties separately
flagged. Do not infer strong ranking from counts dominated by near-ties.

Primary fields in CSV look like
`legacy_left_pos_camera_translation_m`,
`legacy_right_pos_camera_rotation_deg`,
`geo_left_neg_right_magnitude_m`, and
`legacy_left_right_pose_translation_separation_m`.

For GT relative translation below `--min-gt-motion-mm` (default 3 mm),
direction-angle metrics become NaN; vector/magnitude and rotation metrics
remain valid. Always inspect errors and sample counts before interpreting any
comparison. GT is used **only for offline scoring**, not selection at runtime.

## Independent synthetic reprojection sanity check

The script also creates a nonidentity previous w2c pose, infinitesimal `xi`,
and static current-frame 3D points. It compares the repository's
`build_interaction_matrix`-equivalent 2D Jacobian prediction for **current to
previous** flow with independently reprojected flow under all four candidate
poses. Under these stated conventions, `left_pos` is physically expected to
have the smallest error, with finite-motion linearization discrepancies.

Run only the synthetic check:

```bash
python scripts/analyze_camera_motion_v32a.py . --geometry-check-only
```

**Caveat:** These older NPZs do not store actual RAFT forward/backward flow
fields, so V3.2a cannot *empirically* verify the deployed flow direction. That
requires a separate, deliberately read-only flow replay from real frames.
Likewise, a `left_pos` advantage in synthetic geometry does not prove a real
ATE improvement. The next stage is to review GT pose orientation/time sync,
then inspect the actual saved per-frame errors and only after that consider an
isolated left-vs-right end-to-end controlled experiment.

## Validation

18 local CPU unit tests passed including synthetic 3D projection with a
nonidentity previous pose, sign, side, relative-transform conventions, bad NPZ
handling and CSV export. Real Bonn NPZ integration still needs user execution.
