"""V4.3a.1: paired RGB-D candidate comparison on disjoint image tiles.

A diagnostic only. RGB/Depth masks are fixed by TARGET observations;
dynamic pixels use the original estimated motion mask, not motion GT.
Every candidate is rendered from the SAME map before any Mapping update.
All methods' scores are evaluated on identical pixels. No online selection.
"""
import csv
from pathlib import Path

import numpy as np

METHODS = ("g0", "g3_support", "g4")


def eligible_keyframe(selected_event_number, every_n):
    """Select the FIRST eligible dynamic-propagation event, then every n-th."""
    if not isinstance(every_n, int) or every_n < 1:
        raise ValueError("every_n_keyframe must be a positive integer")
    if selected_event_number < 1:
        raise ValueError("eligible propagation event count must start at 1")
    return (selected_event_number - 1) % every_n == 0


def _mean(data, mask):
    if not np.any(mask):
        return float("nan")
    return float(np.mean(data[mask], dtype=np.float64))


def _fraction(mask, selected):
    if not np.any(selected):
        return float("nan")
    return float(np.mean(mask[selected], dtype=np.float64))


def _get_arrays(hypotheses):
    if set(hypotheses) != set(METHODS):
        raise ValueError("Need exactly g0, g3_support and g4")
    out = {}
    expected = None
    for method in METHODS:
        source = hypotheses[method]
        required = ("rgb_abs_mean", "depth_penalized_abs",
                    "prediction_covered", "dynamic_region",
                    "gt_depth_valid", "gt_rgb_valid")
        if not all(key in source for key in required):
            raise ValueError("Missing score map for " + method)
        d = {key: np.asarray(source[key]) for key in required}
        if expected is None:
            expected = d["rgb_abs_mean"].shape
        if any(v.shape != expected for v in d.values()):
            raise ValueError("Candidate score maps have inconsistent shapes")
        for key in ("dynamic_region", "gt_depth_valid", "gt_rgb_valid"):
            d[key] = d[key].astype(bool)
        d["prediction_covered"] = d["prediction_covered"].astype(bool)
        if (not np.isfinite(d["rgb_abs_mean"]).all()
                or not np.isfinite(d["depth_penalized_abs"]).all()):
            raise FloatingPointError("Nonfinite RGB-D score maps")
        out[method] = d
    for key in ("dynamic_region", "gt_depth_valid", "gt_rgb_valid"):
        if not all(np.array_equal(out["g0"][key],out[m][key]) for m in METHODS[1:]):
            raise ValueError("Candidate-specific GT masks are forbidden")
    return out


def analyze_paired_regions(hypotheses, *, frame_id, event_index,
                           grid_rows=6, grid_cols=8,
                           min_dynamic_rgb_pixels=64,
                           min_dynamic_depth_pixels=16,
                           rgb_tolerance=1e-5, depth_tolerance_m=1e-4):
    """Return (two frame-level comparison records, per-region hypothesis rows).

    Spatial grid is non-overlapping. Only tiles meeting BOTH GT target
    dynamic-RGB and dynamic-depth pixel counts are ranked. For comparable
    pixel classifications, use intersection of GT RGB and depth validity,
    not candidate-specific predicted coverage. Ties remain explicit.
    """
    if min(grid_rows,grid_cols,min_dynamic_rgb_pixels,
           min_dynamic_depth_pixels) < 1:
        raise ValueError("Grid and minimum region size must be positive")
    if rgb_tolerance < 0 or depth_tolerance_m < 0:
        raise ValueError("Tolerance must be nonnegative")
    maps=_get_arrays(hypotheses)
    baseline=maps["g0"]
    h,w=baseline["rgb_abs_mean"].shape
    if grid_rows>h or grid_cols>w:
        raise ValueError("Grid finer than image")
    dynamic=baseline["dynamic_region"]
    rgb_mask=dynamic & baseline["gt_rgb_valid"]
    depth_mask=dynamic & baseline["gt_depth_valid"]
    common=rgb_mask & depth_mask
    # Fixed common mask for paired pixel signs. Noncoverage still
    # incurs the fixed missing-depth penalty in depth_penalized_abs.
    regions=[]
    tile_info={}
    for row in range(grid_rows):
        y0,y1=(row*h)//grid_rows, ((row+1)*h)//grid_rows
        for col in range(grid_cols):
            x0,x1=(col*w)//grid_cols, ((col+1)*w)//grid_cols
            sub=np.zeros((h,w),dtype=np.bool_)
            sub[y0:y1,x0:x1]=True
            n_rgb=int(np.count_nonzero(rgb_mask&sub))
            n_depth=int(np.count_nonzero(depth_mask&sub))
            if n_rgb<min_dynamic_rgb_pixels or n_depth<min_dynamic_depth_pixels:
                continue
            tile_index=row*grid_cols+col
            scores={}
            for method in METHODS:
                cand=maps[method]
                scores[method]={
                    "rgb_mae":_mean(cand["rgb_abs_mean"],rgb_mask&sub),
                    "depth_penalized_mae_m":_mean(cand["depth_penalized_abs"],depth_mask&sub),
                    "depth_coverage":_mean(cand["prediction_covered"].astype(np.float32),depth_mask&sub)
                }
            min_rgb=min(scores[m]["rgb_mae"] for m in METHODS)
            min_depth=min(scores[m]["depth_penalized_mae_m"] for m in METHODS)
            # Exact unique winner only; strict tie leaves no unique winner.
            rgb_winners=[m for m in METHODS if np.isclose(scores[m]["rgb_mae"], min_rgb,rtol=0,atol=rgb_tolerance)]
            depth_winners=[m for m in METHODS if np.isclose(scores[m]["depth_penalized_mae_m"], min_depth,rtol=0,atol=depth_tolerance_m)]
            rgb_winner=rgb_winners[0] if len(rgb_winners)==1 else "tie"
            depth_winner=depth_winners[0] if len(depth_winners)==1 else "tie"
            tile_info[tile_index]=(rgb_winner,depth_winner)
            for method in METHODS:
                a=scores[method]
                b=scores["g0"]
                drgb=a["rgb_mae"]-b["rgb_mae"]
                ddepth=a["depth_penalized_mae_m"]-b["depth_penalized_mae_m"]
                rgb_better=drgb < -rgb_tolerance
                depth_better=ddepth < -depth_tolerance_m
                rgb_worse=drgb > rgb_tolerance
                depth_worse=ddepth > depth_tolerance_m
                outcome=("both_better" if rgb_better and depth_better else
                         "rgb_better_depth_worse" if rgb_better and depth_worse else
                         "depth_better_rgb_worse" if depth_better and rgb_worse else
                         "both_worse" if rgb_worse and depth_worse else "tie_or_mixed")
                regions.append(dict(
                    keyframe_id=int(frame_id), eligible_event_index=int(event_index),
                    tile_index=tile_index,tile_row=row,tile_col=col,
                    x0=x0,x1=x1,y0=y0,y1=y1,
                    n_dynamic_rgb=n_rgb,n_dynamic_depth=n_depth,
                    hypothesis=method,
                    rgb_mae=a["rgb_mae"],
                    depth_penalized_mae_m=a["depth_penalized_mae_m"],
                    depth_coverage=a["depth_coverage"],
                    delta_rgb_vs_g0=drgb,
                    delta_depth_vs_g0_m=ddepth,
                    rgb_winner=rgb_winner,
                    depth_winner=depth_winner,
                    outcome_vs_g0=outcome
                ))
    summaries=[]
    for method in METHODS[1:]:
        candidate=maps[method]
        drgb=(candidate["rgb_abs_mean"]-baseline["rgb_abs_mean"])
        ddepth=(candidate["depth_penalized_abs"]-baseline["depth_penalized_abs"])
        rgb_good=drgb< -rgb_tolerance
        rgb_bad=drgb>rgb_tolerance
        depth_good=ddepth< -depth_tolerance_m
        depth_bad=ddepth>depth_tolerance_m
        tile_rows=[r for r in regions if r["hypothesis"]==method]
        both=sum(r["outcome_vs_g0"]=="both_better" for r in tile_rows)
        conflict=sum(r["outcome_vs_g0"] in ("rgb_better_depth_worse","depth_better_rgb_worse") for r in tile_rows)
        both_bad=sum(r["outcome_vs_g0"]=="both_worse" for r in tile_rows)
        summaries.append(dict(
            keyframe_id=int(frame_id),eligible_event_index=int(event_index),
            hypothesis=method,
            image_height=h,image_width=w,
            n_dynamic_rgb=int(np.count_nonzero(rgb_mask)),
            n_dynamic_depth=int(np.count_nonzero(depth_mask)),
            n_common_dynamic=int(np.count_nonzero(common)),
            n_eligible_tiles=len(tile_rows),
            min_dynamic_rgb_pixels=int(min_dynamic_rgb_pixels),
            min_dynamic_depth_pixels=int(min_dynamic_depth_pixels),
            rgb_tolerance=rgb_tolerance,depth_tolerance_m=depth_tolerance_m,
            rgb_delta_dynamic=_mean(drgb,rgb_mask),
            depth_delta_dynamic_m=_mean(ddepth,depth_mask),
            candidate_dynamic_depth_coverage=_mean(
                candidate["prediction_covered"].astype(np.float32),depth_mask),
            g0_dynamic_depth_coverage=_mean(
                baseline["prediction_covered"].astype(np.float32),depth_mask),
            pixel_both_better_fraction=_fraction(rgb_good & depth_good,common),
            pixel_rgb_better_depth_worse_fraction=_fraction(rgb_good & depth_bad,common),
            pixel_depth_better_rgb_worse_fraction=_fraction(depth_good & rgb_bad,common),
            pixel_both_worse_fraction=_fraction(rgb_bad & depth_bad,common),
            pixel_other_fraction=_fraction(
                ~((rgb_good & depth_good)|(rgb_good & depth_bad)|
                  (depth_good & rgb_bad)|(rgb_bad & depth_bad)),common),
            tiles_both_better=both,
            tiles_conflict=conflict,
            tiles_both_worse=both_bad,
            tiles_rgb_winner=sum(r["rgb_winner"]==method for r in tile_rows),
            tiles_depth_winner=sum(r["depth_winner"]==method for r in tile_rows),
            tiles_same_rgb_depth_winner=sum(
                r["rgb_winner"]==method and r["depth_winner"]==method
                for r in tile_rows),
        ))
    return summaries, regions


def save_region_analysis(folder, frame_rows, tile_rows):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    for filename,rows in (("v43a1_per_keyframe.csv",frame_rows),
                          ("v43a1_per_region.csv",tile_rows)):
        # An empty tile collection is a valid result (no suitable region).
        if not rows:
            continue
        dest=folder/filename
        new=not dest.exists() or dest.stat().st_size==0
        with dest.open("a",encoding="utf-8",newline="") as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]))
            if new:wr.writeheader()
            wr.writerows(rows)
