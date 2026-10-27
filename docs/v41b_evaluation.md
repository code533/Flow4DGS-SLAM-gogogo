# V4.1b — Unknown attribution and independent dynamic label evaluation

Branch `feature/v41b-unknown-attribution-gt-eval`, based on V4.1a.
**Offline only. No SLAM/Mapping changes and NO new ATE improvement claim.**

## 1. Data

Retain all `v40b_XXXXX.npz` files from the previously completed V4.0b
ballon2 run. These raw archives are essential; CSV/plots alone are
insufficient. No additional SLAM run is necessary.

```bash
git fetch origin
git switch feature/v41b-unknown-attribution-gt-eval
python -m unittest discover -s tests -p 'test_v41b_evaluation.py' -v
RUN_DIR=$(find results/bonn/v40b_calibration -maxdepth 1 -type d \
 -name 'rgbd_bonn_balloon2_*' | sort | tail -n 1)
python scripts/run_v41b_evaluation.py \
 --input "$RUN_DIR/v40b_calibration" \
 --output "$RUN_DIR/v41b_attribution" \
 --frames 60,140,220,230,310,390
```

Outputs `v41b_unknown_reasons.csv` and selected
`annotation_reference_XXXXX.png` (RGB only),
`unknown_reason_codes_XXXXX.png` and
`v41b_reason_legend.json`. Without independent labels it
**does not** output accuracy scores.

## 2. Why Unknown?

V4.1b exactly reproduces the V4.1a decision rule for A3 and assigns
five **disjoint reasons in priority order**:

1. INVALID_FLOW
2. LOW_TOTAL_EVIDENCE
3. LOW_FLOW_AND_MAP_RELIABILITY
4. FLOW_MAP_CONFLICT
5. AMBIGUOUS_SCORE_BAND

These categories **do not overlap**, and all Unknown pixels must have
exactly one reason. Additional cues such as map-uncovered, boundary-near
and possible-occlusion may overlap with each other. They are **diagnostic
causes/proxies**, not ground truth of physical error.

## 3. Manual annotation protocol

Open the saved *raw RGB reference* (not predicted mask heatmaps) and
if possible inspect temporal adjacent frames separately.
Decide whether visible pixels belong to genuinely moving objects
independently of the algorithm's prediction.

Use 3 labels:
- 0 = independently verified static surface;
- 1 = independently verified independently moving surface;
- 255 = uncertain, occluded, unobservable or unannotated.

**Do not automatically label every person as moving; do not copy YOLO or
any of the V4.x output masks into ground truth.** Annotating one frame of
static RGB alone can be insufficient to establish motion; use a sequence
or independent motion cues. The system cannot infer true dynamic labels
automatically.

Prepare one JSON per frame, e.g. `annotation_00140.json`:

```json
{
  "image": "v40b_00140.npz",
  "polygons": [
    {"label": "static", "points": [[4,4],[630,4],[630,470],[4,470]]},
    {"label": "dynamic", "points": [[100,100],[170,100],[170,250],[100,250]]},
    {"label": "ignore", "points": [[130,110],[138,110],[138,120],[130,120]]}
  ]
}
```

Coordinates above are **illustrative only**, NOT real scene labels.
Polygons overwrite previous ones in listed order; everything not
covered by a polygon is `Ignore=255`. For meticulous annotation use
several tight polygons, not approximate boxes. Existing manually drawn
binary segmentation masks may instead be converted to grayscale PNGs
with the SAME class codes and resolution.

Convert annotations:

```bash
mkdir -p "$RUN_DIR/v41b_gt"
python scripts/v41b_make_gt.py \
 --npz "$RUN_DIR/v40b_calibration/v40b_00140.npz" \
 --json annotation_00140.json \
 --output "$RUN_DIR/v41b_gt/gt_00140.png"
```

Repeat for other chosen frames. For final paper evaluation, reserve
held-out frames/sequences and avoid adjusting thresholds on test labels.

## 4. Ground-truth evaluation

```bash
python scripts/run_v41b_evaluation.py \
 --input "$RUN_DIR/v40b_calibration" \
 --gt-dir "$RUN_DIR/v41b_gt" \
 --output "$RUN_DIR/v41b_gt_eval"
```

Only frames with a matching `gt_XXXXX.png` are evaluated. Outputs:

- `v41b_gt_per_frame.csv`: five methods including original Flow4DGS mask.
- `v41b_gt_micro.csv`: pooled confusion counts, coverage, selective
  precision/recall/F1/IoU and **full dynamic recall** (rejects counted
  as missed detections).
- `v41b_risk_coverage.csv` and `v41b_risk_coverage.png`:
  selective error vs acceptance coverage, using score-margin threshold
  sweeps. The original baseline appears as a fixed coverage=100% point.
- `v41b_unknown_reasons.csv` and per-frame reference images.

**Critical:** selective F1/IoU exclude rejected pixels. They cannot be
interpreted alone, since rejecting difficult pixels boosts them.
Report accepted coverage, recall over ALL labeled dynamic pixels,
and penalty risk `(FP+FN+abstentions)/labeled GT pixels` as well.
If there is no labeled dynamic pixel, dynamic-class recall is NaN;
do not replace it with 100%.

The original Flow4DGS `final_static` mask is an evaluated baseline only,
never used in predictions or labels. `P_D^*, Q_F^*, Q_M^*` are NOT
calibrated probabilities. The curve is a confidence-margin heuristic,
not a formal calibrated uncertainty curve.

## 5. Experimental boundaries

This release does not optimize any thresholds on human labels.
A1/A2/A3 are heuristic; no performance claim before GT results.
Do not use adjacent highly-correlated frames as independent test cases.
For evidence suitable for a paper: annotate additional sequences, use
held-out testing, evaluate annotator agreement, and separately test
flow-correspondence quality with independent correspondence labels.
