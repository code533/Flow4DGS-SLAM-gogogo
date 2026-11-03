"""V4.3a.4: preregistered source-score -> frozen next-keyframe audit.

Source candidate selection uses ONLY observed RGB-D at time t.
Grounded next-keyframe outcome is fixed-mask *ordinary dynamic depth MAE*,
independent of the source scoring rule. Next-frame camera is held common.
The map is frozen and source offsets are held (zero-order hold); results are
predictive diagnostics, NOT physical motion GT or improved online SLAM.

Choices: G0, G3-Support, G4. A deterministic minimum source advantage
over G0 is required before changing candidate. RGB changes and depth
coverage are logged, NOT tuned post hoc into the selection objective.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

METHODS = ("g0", "g3_support", "g4")
POLICIES = ("always_g0", "mae", "clipped50", "huber20", "interior_clipped50")


def is_systematic_source(event_index, every_n_events):
    if not isinstance(every_n_events, int) or every_n_events < 0:
        raise ValueError("source_every_n_event must be a nonnegative integer")
    if event_index < 1:
        raise ValueError("Event indices start at 1")
    return every_n_events > 0 and (event_index-1) % every_n_events == 0


def validate_protocol(source_every_n_event, decision_margin_m, future_tie_tol_m,
                      clip_m, huber_delta_m, interior_erosion_px,
                      min_dynamic_depth_pixels, min_interior_depth_pixels,
                      min_coverage_fraction):
    if not isinstance(source_every_n_event, int) or source_every_n_event < 0:
        raise ValueError("Invalid systematic source interval")
    for val in (decision_margin_m, future_tie_tol_m):
        if not np.isfinite(val) or val < 0:
            raise ValueError("Invalid nonnegative margin/tolerance")
    if min(clip_m, huber_delta_m) <= 0:
        raise ValueError("Invalid robust depth scale")
    for x in (interior_erosion_px,min_dynamic_depth_pixels,min_interior_depth_pixels):
        if not isinstance(x,int) or x < 1:
            raise ValueError("Invalid dynamic mask configuration")
    if not 0 <= min_coverage_fraction <= 1:
        raise ValueError("min_coverage_fraction must be in [0,1]")


def huber_normalized(residual, delta):
    return np.where(residual <= delta, residual*residual/(2*delta),
                    residual-delta/2)


def _maps(maps):
    if set(maps) != set(METHODS):
        raise ValueError("Missing G0, Support, or G4 map")
    arrays = {}
    required = ("dynamic_region","gt_depth_valid","gt_rgb_valid",
                "depth_penalized_abs","rgb_abs_mean","prediction_covered")
    for name in METHODS:
        if any(k not in maps[name] for k in required):
            raise ValueError("Missing score map for "+name)
        arrays[name]={k:np.asarray(maps[name][k]) for k in required}
    base=arrays["g0"]
    shape=base["dynamic_region"].shape
    for name, d in arrays.items():
        for key,val in d.items():
            if val.shape!=shape:
                raise ValueError("Inconsistent map shapes")
            if key in ("dynamic_region","gt_depth_valid","gt_rgb_valid"):
                if not np.array_equal(base[key],val):
                    raise ValueError("Candidate-specific observation mask")
    return arrays


def score_from_maps(maps, *, clip_m=.05, huber_delta_m=.02,
                    interior_erosion_px=5,min_dynamic_depth_pixels=256,
                    min_interior_depth_pixels=64):
    """Per-candidate source scores with fixed target masks and denominators.

    Huber(r,delta)/delta keeps *meter* units and has a linear tail.
    Saturation min(r,cap) is bounded. No clipping of paired differences
    is used as a candidate score.
    """
    a=_maps(maps)
    base=a["g0"]
    dynamic=base["dynamic_region"].astype(bool)
    depth_mask=dynamic & base["gt_depth_valid"].astype(bool)
    rgb_mask=dynamic & base["gt_rgb_valid"].astype(bool)
    interior=binary_erosion(dynamic,iterations=interior_erosion_px)
    interior_depth=depth_mask&interior
    n=int(depth_mask.sum())
    ni=int(interior_depth.sum())
    if n < min_dynamic_depth_pixels or not rgb_mask.any():
        raise ValueError("Too few valid dynamic observations for predictor")
    result={}
    for name in METHODS:
        d=a[name]
        depth=d["depth_penalized_abs"][depth_mask].astype(np.float64)
        if not np.isfinite(depth).all() or np.any(depth<0):
            raise FloatingPointError("Invalid depth residual")
        rgb=d["rgb_abs_mean"][rgb_mask].astype(np.float64)
        if not np.isfinite(rgb).all() or np.any(rgb<0):
            raise FloatingPointError("Invalid RGB residual")
        inside = d["depth_penalized_abs"][interior_depth].astype(np.float64)
        scores={
            "mae":float(depth.mean()),
            "clipped50":float(np.minimum(depth,clip_m).mean()),
            "huber20":float(huber_normalized(depth,huber_delta_m).mean()),
            "interior_clipped50":float(np.minimum(inside,clip_m).mean())
                  if ni >= min_interior_depth_pixels else float("nan"),
            "rgb_mae":float(rgb.mean()),
            "depth_coverage":float(d["prediction_covered"][depth_mask].mean()),
            "n_dynamic_depth":n,
            "n_interior_depth":ni,
            "n_dynamic_rgb":int(rgb_mask.sum()),
        }
        result[name]=scores
    return result


def choose_by_source(scores, policy, *, decision_margin_m=.0002,
                     coverage_tolerance=0.005):
    """Return G0 unless a finite source score beats it by >= margin.

    If the policy lacks a valid object interior, it falls back to G0.
    Candidate depth coverage cannot be substantially below G0's.
    """
    if policy not in POLICIES:
        raise ValueError("Unknown source policy "+str(policy))
    if policy=="always_g0":
        return "g0"
    if not 0<=coverage_tolerance<=1 or decision_margin_m<0:
        raise ValueError("Invalid selection safeguards")
    g0=scores["g0"][policy]
    if not np.isfinite(g0):
        return "g0"
    candidate_names=METHODS[1:]
    cand=[name for name in candidate_names
          if np.isfinite(scores[name][policy])
          and scores[name]["depth_coverage"] >= scores["g0"]["depth_coverage"]-coverage_tolerance]
    if not cand:return "g0"
    best=min(cand,key=lambda name:(scores[name][policy],METHODS.index(name)))
    if scores[best][policy] < g0-decision_margin_m:
        return best
    return "g0"


def audit_source_future(source_scores, future_scores, *,
                        source_uid,future_uid,event_index,
                        decision_margin_m=.0002,future_tie_tol_m=.0001,
                        coverage_tolerance=.005):
    """Paired policy evaluation. Future best judged by RAW dynamic depth MAE.

    All rows share the same (source, future) pair and future target.
    This is an offline policy evaluation, NOT a causal intervention.
    """
    if future_uid <= source_uid:
        raise ValueError("Future frame must be strictly later")
    if set(source_scores)!=set(METHODS) or set(future_scores)!=set(METHODS):
        raise ValueError("Incomplete paired scores")
    if decision_margin_m<0 or future_tie_tol_m<0:
        raise ValueError("Invalid decision thresholds")
    oracle=min(future_scores[name]["mae"] for name in METHODS)
    second=sorted(future_scores[name]["mae"] for name in METHODS)[1]
    future_unique=(second-oracle)>future_tie_tol_m
    rows=[]
    for policy in POLICIES:
        selected=choose_by_source(source_scores,policy,
                                  decision_margin_m=decision_margin_m,
                                  coverage_tolerance=coverage_tolerance)
        fs=future_scores[selected]
        fg=future_scores["g0"]
        s=source_scores[selected]
        rows.append(dict(
            source_keyframe_id=int(source_uid),future_keyframe_id=int(future_uid),
            gap=int(future_uid-source_uid),eligible_event_index=int(event_index),
            policy=policy,selected=selected,
            temporal_model="frozen_map_zero_order_hold",
            decision_margin_m=float(decision_margin_m),
            future_tie_tolerance_m=float(future_tie_tol_m),
            source_selected_policy_score=float(s.get(policy,float("nan"))),
            source_g0_policy_score=float(source_scores["g0"].get(policy,float("nan"))),
            source_selected_depth_mae_m=float(s["mae"]),
            future_selected_depth_mae_m=float(fs["mae"]),
            future_g0_depth_mae_m=float(fg["mae"]),
            future_oracle_depth_mae_m=float(oracle),
            future_oracle_unique=bool(future_unique),
            future_near_oracle=bool(fs["mae"]<=oracle+future_tie_tol_m),
            future_gain_over_g0_m=float(fg["mae"]-fs["mae"]),
            future_regret_m=float(fs["mae"]-oracle),
            future_depth_win_vs_g0=bool(fg["mae"]-fs["mae"]>future_tie_tol_m),
            future_depth_loss_vs_g0=bool(fs["mae"]-fg["mae"]>future_tie_tol_m),
            future_rgb_change_vs_g0=float(fs["rgb_mae"]-fg["rgb_mae"]),
            future_coverage_change_vs_g0=float(fs["depth_coverage"]-fg["depth_coverage"]),
            source_depth_valid_n=int(s["n_dynamic_depth"]),
            future_depth_valid_n=int(fs["n_dynamic_depth"]),
            source_interior_depth_valid_n=int(s["n_interior_depth"]),
            future_interior_depth_valid_n=int(fs["n_interior_depth"])
        ))
    return rows


def save_policy_results(folder,rows,source_scores,future_scores,
                        future_maps=None):
    """Small, auditable CSVs + optional full-res future NPZ."""
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"v43a4_policy_per_pair.csv"
    new=not path.exists() or path.stat().st_size==0
    with path.open("a",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        if new:writer.writeheader()
        writer.writerows(rows)
    score_path=folder/"v43a4_candidate_scores.csv"
    create=not score_path.exists() or score_path.stat().st_size==0
    with score_path.open("a",newline="",encoding="utf-8") as handle:
        fields=["source_keyframe_id","future_keyframe_id","timepoint","hypothesis",
                "mae","clipped50","huber20","interior_clipped50",
                "rgb_mae","depth_coverage","n_dynamic_depth",
                "n_interior_depth","n_dynamic_rgb"]
        writer=csv.DictWriter(handle,fieldnames=fields)
        if create:writer.writeheader()
        for label,score_set in (("source",source_scores),("future",future_scores)):
            for name in METHODS:
                writer.writerow(dict(source_keyframe_id=rows[0]["source_keyframe_id"],
                            future_keyframe_id=rows[0]["future_keyframe_id"],
                            timepoint=label,hypothesis=name,**score_set[name]))
    if future_maps:
        src=int(rows[0]["source_keyframe_id"])
        fut=int(rows[0]["future_keyframe_id"])
        arrays={f"{name}__{key}":np.asarray(value)
                for name,m in future_maps.items() for key,value in m.items()}
        np.savez_compressed(folder/f"v43a4_source{src:05d}_future{fut:05d}.npz",
                            **arrays)
