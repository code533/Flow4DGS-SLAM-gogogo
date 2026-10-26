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

## Decision-time diagnostics

The short-run parent now sets `StaticConsistency.save_tracking_decisions: true`.
The default for other configurations is false. Tracking mode writes a separate
`tracking_decisions/` directory immediately after selection/fallback, before the
first optimizer update using the mask. Observe mode does not create this directory
and does not add a warmup decision or change its stopping rule.

Every tracking frame gets a compressed NPZ with RGB/depth/union support masks,
candidate exclusions, actual exclusions, candidate keep and applied keep masks,
map validity, frame, zero-based mask iteration and the active depth definition.
On fallback the applied keep mask is all true and actual exclusions are empty;
candidate exclusions remain available. These masks are the EXTRA consistency
filter, not replacements for the semantic/flow motion mask. Actual exclusions
are restricted to the corresponding support at the decision instant.

At `save_every` frames (10 in the short configs), the same NPZ additionally
contains the full evidence dictionary, observed RGB (CHW float), and camera R/T
at the decision pose. Dense fields include observed/raw/normalized depths,
opacity, both signed residuals, tolerance and RGB score. `save_every: 0` disables
these dense additions but still saves decision masks when enabled. Each decision
also appends metadata and existing selection statistics to
`tracking_decisions/summary.jsonl`; `dense_diagnostics` marks dense frames.

These files differ from `static_consistency/`, whose evidence is rendered AFTER
tracking. The original offline `--compare-depth` mode expects dense post-tracking
exports, not this mixed sparse/dense decision directory. A dedicated decision
analysis can use the saved mask fields directly.

Saving synchronizes tensors to CPU and compresses files inside tracking, so
reported tracking time includes diagnostic I/O. Use recording-disabled runs for
runtime benchmarks. No loss, threshold, mask-selection or stopping rule changes
are introduced by this recorder. Full SLAM validation remains a server task.

## Compact server-side decision report

Run after a diagnostic tracking run completes (numpy, Pillow, matplotlib, PyYAML;
no Torch/CUDA or original dataset needed):

```bash
python scripts/analyze_tracking_decisions.py \
  --results /path/to/B2_norm_seed1_run \
  --output /path/to/new_seed1_analysis
```

Run separately for seed 2. Send only each output's `analysis_report.zip`.
The output directory must not already exist. Keep `visualize_consistency.py`
beside the new script: it supplies the shared depth-region definition.

The report validates NPZ masks against decision-time JSONL, reports frame-wise
RGB/depth/union removal, pools counts, and exports fixed 4x4 spatial cell counts.
Dense snapshots additionally yield depth-region counts and RGB overlays (default
frames 10,50,90; override with `--plot-frames 20 60`). Missing dense plot frames
are reported, never substituted silently. Region statistics use dense frames
only and include invalid sensor depth so RGB support is fully accounted for.
ATE statistics are collected from stats_final.json files under the run directory;
if absent the report explicitly requests them separately. This script does not
compute framewise trajectory error or RPE. Mask coverage does not quantify pose
observability. Runtime includes the earlier recorder overhead.

## B3 soft-weight comparison

Three configurations inherit B2-norm with identical thresholds, warmup, stopping
policy and diagnostic settings. Only `StaticConsistency.inconsistent_weight`
differs (default 0 preserves old hard filtering):

```bash
python slam.py --config configs/experiments/b3_control_short.yaml --seed 2 --exp_name B3_control_seed2
python slam.py --config configs/experiments/b3_soft_short.yaml --seed 2 --exp_name B3_soft_seed2
python slam.py --config configs/experiments/b3_hard_short.yaml --seed 2 --exp_name B3_hard_seed2
```

Control uses weight 1, soft uses 0.5, hard uses 0. All three use tracking mode;
the control is NOT the old observe-mode B1. Candidate masks are computed at the
same warmup iteration and held fixed. All variants use the same conservative
hard-candidate retention test (256 pixels / 25% union support). If it fails,
weights become 1 everywhere; no soft fallback threshold is retuned.

Weights multiply BOTH sides of the existing masked L1 RGB/depth residual.
For nonnegative weights this is exactly linear loss weighting, not squared
weighting. Original image-wide mean denominators and RGB/depth mixing remain
unchanged; there is no normalization by sum of weights. Semantic/flow masks,
RGB opacity weighting, and mapping loss code remain unchanged.

Decision NPZ files additionally contain `tracking_weights` every frame.
`applied_keep_mask` means nonzero weight. Thus soft observations remain supported
and applied exclusion counts are zero, even though their contribution is reduced.
`mask_applied` indicates an accepted nontrivial weight policy; it is false for
weight-1 control. `decision_accepted` reports the common fallback decision.
Per-scope `*_downweighted_pixels`, `*_weight_sum`, `*_mean_weight` report effective
weighting (not actual gradient strength). Candidate hard-exclusion counts remain
available. The compact analyzer now validates weights and plots downweighted
pixels in red; CSVs include both downweighted and fully excluded counts.

Run the three seed-2 experiments as an integration check, then repeat all three
with seeds 0 and 1. Do not reuse old hard-run scores as the only control. Compare
ATE RMSE, repeatability, iterations and weight statistics. Diagnostic I/O is
included in runtime. No full server SLAM validation has been performed locally.
