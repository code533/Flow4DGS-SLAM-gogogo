# V3.1b — Offline evaluation before and after camera-motion clipping

V3.1b is a **NumPy-only, read-only analyzer** of the V3.1a `camera_motion_v31a_XXXXX.npz` files. It does not modify Camera-Induced Motion Decomposition, Tracking, Mapping, or the trajectory. You do **not** need to rerun SLAM.

## Requirements

- The original V3.1a per-frame NPZs (including `xi_legacy_common`, `xi_geo`, `T_previous_est`, `T_candidate_legacy_common`, `T_candidate_geo`, `T_previous_gt`, `T_current_gt`).
- Python 3.8+ and `numpy`.
- A valid frame with `status=ok`. Frames lacking GT are skipped with reasons. The previous `v31a_per_frame.csv` alone is **insufficient**: it did not save either 6DoF twist.

## Run

```bash
git fetch origin
git switch feature/cimd-clipping-analysis-v31b
python -m unittest discover -s tests -p test_camera_motion_v31b.py -v

# Prefer one exact V3.1a run directory, not all experiment folders.
python scripts/analyze_camera_motion_v31b.py \
  results/bonn/v31a_shadow_/YOUR_RUN_DIR \
  --convention both --min-gt-motion-mm 1.0 \
  --csv results/v31b_per_frame.csv
```

The script recursively discovers `camera_motion_v31a_*.npz` in the given directory.

## Three pose hypotheses per estimator

For each original IRLS estimate (`legacy`) and geometry-aware IRLS estimate (`geo`), compute:

1. **raw**: `T_prev @ SE3_exp(xi)`, before any frontend clipping.
2. **legacy_clip**: the V3.1a-saved frontend candidate, with original translation normalization to a fixed maximum, and original rotation clipping.
3. **capped_clip**: retain the saved original rotation clipping, but change *translation only* to `min(1, limit / ||delta_t||) * delta_t`. This never amplifies a small translation estimate.

Because V3.1a did not save `max_translation` explicitly, V3.1b **infers** the per-frame limit from the translation step of the saved legacy-clipped candidates. This relies on the original unbounded `s_t=max_translation/(||delta_t||+epsilon)` rule. It checks agreement between estimators and recomputed translation. When both raw motions are zero, the limit is not identifiable and is set to zero.

Rotation clipping is taken directly from the saved V3.1a candidates; no unknown per-frame rotation threshold is invented. The raw SE(3) uses the same Rodrigues and left-Jacobian expressions as the repository `pose_utils.SE3_exp`.

## Ground truth and relative-pose conventions

- `--convention right`: `inv(T_prev) @ T_current`, reproducing the V3.1a CSV metric. Both estimate and GT use this form.
- `--convention camera`: `T_current @ inv(T_prev)`, the previous-to-current *camera-to-camera transform* when the stored poses are world-to-camera.
- `--convention both` (default): prints both for transparency.

Verify time synchronization and whether each source pose is world-to-camera. These are **single-step relative-motion errors**, not ATE. Inconsistent GT conventions may make both comparisons misleading. The camera-mode relative translation is sensitive to the recovered relative rotation and pose origins; don't silently mix the two.

The translation-direction angle is set to NaN when GT inter-frame translation is smaller than `--min-gt-motion-mm` (default 1 mm), avoiding meaningless angles for near-stationary frames. Magnitude/vector/rotation errors are still included. Only complete, finite, successfully evaluated pairs are compared.

## Key outputs

For each model, mode and convention:

- `translation_error_m`: norm of estimated vs GT relative translation vector difference.
- `magnitude_error_m`: absolute difference in relative translation magnitude.
- `direction_error_deg`: relative translation direction angle where GT motion is sufficient.
- `rotation_error_deg`: geodesic SO(3) rotation difference.
- `predicted_translation_m`, `gt_translation_m`.

Also counts `legacy_amplifies` (original frontend increased a small translation) and `capped_reduces` (cap limited an oversized translation), and reports paired medians, mean differences and the share of frames improved. Positive **mean_delta** means the second approach is worse (for these errors).

## Interpretation

A geometry-aware estimator can be better before clipping but lose any advantage after original clipping. Conversely, worse preclip estimates point back to the motion solver. Results are shadow-based, using previous poses from the original trajectory; they do **not** establish what ATE would be after propagating a different motion rule across an entire SLAM run.

Synthetic CPU unit tests pass; full run validation on Bonn NPZ data remains to be done. If many files are skipped, inspect the five displayed error reasons before interpreting summaries.
