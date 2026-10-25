# B1 / B2 short-run comparison

Run from the repository root in the existing server SLAM environment. Set the
same Dataset.dataset_path in the shared parent configuration for all runs.
The configurations select frames [0, 100), keep semantic/flow processing, and
save diagnostics every 10 frames. Do not use --eval: it overrides output options.

```bash
python slam.py --config configs/experiments/b1_short.yaml --seed 0 --exp_name B1_short
python slam.py --config configs/experiments/b2_acc_short.yaml --seed 0 --exp_name B2_acc_short
python slam.py --config configs/experiments/b2_norm_short.yaml --seed 0 --exp_name B2_norm_short
```

The detector defaults to normalized depth for backward compatibility. B2-acc
selects accumulated depth; both definitions remain saved in every diagnostic
NPZ. Map validity uses the same normalized-depth checks in both modes to avoid
changing the comparison support. RGB and depth disagreement are combined by OR.
The extra mask affects tracking only, is held fixed after the warmup decision,
and does not overwrite motion_mask or change mapping loss code. Subsequent maps
can nevertheless change because the estimated trajectory changes.

Fallback uses the UNION of baseline RGB/depth tracking supports, requiring at
least 256 retained pixels and 25% retention. This replaces the older intersection
support, which did not represent either loss. RGB/depth statistics are also
reported separately; there is no per-modality retention guarantee. RGB counts
represent mask eligibility, not opacity-weighted gradient strength.

summary.jsonl fields:
- depth_definition: active detector convention.
- baseline_support_pixels / retained_support_pixels: union before / candidate after.
- candidate_excluded_pixels/fraction: proposed removal, even if fallback occurs.
- applied_retained_support_pixels / applied_excluded_pixels: effective counts.
- fallback / fallback_reason / mask_applied: selection result (tracking mode).
- rgb/depth/union_support_pixels, *_candidate_retained_pixels,
  *_applied_retained_pixels, *_applied_excluded_fraction: modality-specific counts.
- support_evaluation_pose: tracking counts are measured at the mask decision;
  observe counts are hypothetical at the final pose, with no applied exclusion.
- tracking_iterations: actual iterations used for the frame.

Dense NPZ and map_* summary metrics are evaluated AFTER tracking; they are not
necessarily the mask used earlier at mask_iteration. Support counts are a
snapshot, not counts recomputed at every subsequent iteration. Zero-denominator
fractions are JSON null.

Important experimental limitation: existing tracking mode delays early stopping
until the warmup mask decision; observe preserves baseline early stopping. A
B2-vs-B1 difference may therefore include extra optimization iterations. Compare
iteration counts; if early stopping differs materially, use an iteration-matched
no-filter control before attributing improvements to filtering. B2-acc and
B2-norm share the same stopping policy. Fixed seeds do not guarantee GPU
bitwise determinism. Repeat full-sequence evaluation after this smoke test.

Offline paired analysis (no dataset required):
```bash
python scripts/visualize_consistency.py --results /path/to/run --compare-depth --output /path/to/new_analysis
```

Compare trajectory ATE/RPE, failures, fallback frequency, actual support removal
and runtime. Lower mismatch rate alone is not evidence of better motion detection
or better tracking. Full CUDA/SLAM execution must be verified on the server.
