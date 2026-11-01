# V4.3a.1 — Full-keyframe RGB-D reconstruction hypothesis analysis

Branch: `feature/v43a1-full-keyframe-region-analysis` based on V4.3a.

## Research aim

Validate the preliminary five-keyframe V4.3a results on **all eligible
dynamic-propagation keyframes**, and analyze RGB/depth improvements and
conflicts within estimated dynamic regions.

This release is **read-only**. Online Mapping still uses G0. It does not
choose a winner, modify Tracking, Mapping losses, dynamic masks, camera
poses, optimization, or Gaussian state. As in V4.3a, same-state candidate
scores are computed **before** the actual G0 propagation and mapping.

## Changes from V4.3a

1. Keyframe selection uses an **eligible dynamic-propagation event counter**.
   With `every_n_keyframe=1`, every eligible propagation event is evaluated.
   With `5`, events 1, 6, 11, 16 ... are evaluated. The old
   `viewpoint.uid % 5` logic accidentally selected very few frames.
2. Existing paired G0/G3-Support/G4 fixed-target RGB-D MAE, depth coverage,
   and observation masks remain unchanged.
3. Optional coarse, disjoint **6×8 image tiles** are scored using
   exactly the same target observation masks for all candidates. A tile
   is considered evaluable only if it has enough estimated dynamic
   RGB pixels and valid observed depth pixels.
4. Detailed RGB vs depth paired improvements are categorized as:
   both better, RGB better/depth worse, depth better/RGB worse, both
   worse, or tie/mixed. Pixel-level proportions use the **common
   observed-RGB-and-valid-depth dynamic mask**, independent of predicted
   coverage. Depth uncovered pixels continue receiving the fixed
   penalty from V4.3a. Tolerances are explicit.
5. Detailed candidate NPZs can still be saved for selected frame IDs;
   analysis itself uses temporary CPU score arrays but does **not**
   write a full-map NPZ every frame.

## Run

```bash
git fetch origin
git switch feature/v43a1-full-keyframe-region-analysis
git pull --ff-only
python -m unittest discover -s tests -p 'test_v43a1_region.py' -v
python -m unittest discover -s tests -p 'test_v43a_observation.py' -v
```

Set in `configs/rgbd/bonn/config.yaml`:

```yaml
model_params:
  v42a_propagation:
    mode: 'g0'
    g34_shadow: false
    g4_ablation: false
    attribute_failures: false
    save_points: false
  v43a_observation:
    enabled: true
    every_n_keyframe: 1
    detailed_frames: [212, 357]
    save_maps: false
    opacity_threshold: 0.1
    uncovered_penalty_m: 0.5
    region_analysis: true
    grid_rows: 6
    grid_cols: 8
    min_dynamic_rgb_pixels: 64
    min_dynamic_depth_pixels: 16
    rgb_tolerance: 0.00001
    depth_tolerance_m: 0.0001
```

Leave the existing G4 thresholds and `Training.flow_ds=1`,
`flow_offset=true`, `use_knn=true`, `dynamic_model=offset` unchanged.
For detailed images at specific frame IDs, set `save_maps: true`;
otherwise only CSVs are written.

```bash
python slam.py --inherit_from configs/rgbd/bonn/config.yaml \
  --config configs/rgbd/bonn/ballon2.yaml --eval --dynamic \
  --exp_name=v43a1_all_keyframes --save_results=0
```

### Outputs

```
<run>/v43a_observation/v43a_per_keyframe.csv
<run>/v43a1_regions/v43a1_per_keyframe.csv
<run>/v43a1_regions/v43a1_per_region.csv
```

The first CSV has three candidates per event. The V4.3a.1 keyframe CSV
has two comparisons versus G0 per event. Region rows contain tile
bounds, pixel counts, candidate errors, improvement sign classifications
and unique/tied RGB/depth winners.

```bash
python scripts/analyze_v43a_observation.py \
 --input "<run>/v43a_observation"
python scripts/analyze_v43a1_regions.py \
 --input "<run>/v43a1_regions"
```

Outputs: `v43a_summary.txt` and `v43a1_summary.txt`.
Check both CSVs for the number of fully paired frames.

## Interpretation and limitations

- These are **selection-frame pre-optimization scores**, not unseen
  frame generalization or ATE/PSNR/LPIPS improvements.
- The original `motion_mask` is an *estimated* segmentation, not
  dynamic ground truth. Boundary contamination remains possible.
- Tiles are coarse, spatially disjoint image regions, **not Gaussian
  motion entities**. Individual Gaussian changes may influence adjacent
  tiles due to opacity and occlusions.
- Missing render depth is penalized; coverage is separately reported.
  RGB/depth have different physical units. V4.3a.1 intentionally does
  NOT pick arbitrary weights or choose a single winner.
- Tile and pixel samples are spatially and temporally correlated. Do
  NOT treat thousands of pixels as independent significance samples.
- Scoring all eligible keyframes renders three candidate scenes per
  event and can increase computational burden and impact asynchronous
  SLAM scheduling. For initial testing set every_n_keyframe=5 before
  using 1 for the full study.
- The branch has been integrated in source but has **not been verified
  on the user's GPU environment**. Run tests and smoke first.
- Subsequent work should verify any candidate selection on genuinely
  independent time frames; no online selection is implemented here.
