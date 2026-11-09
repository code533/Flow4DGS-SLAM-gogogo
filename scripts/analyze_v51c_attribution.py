"""V5.1c read-only attribution: frozen same-map motion vs cross-run coupling.

Inputs:
* V5.1c shadow run's hold/legacy/strict PRE-MAPPING fixed-map pairs;
* Existing V5.1b matched Legacy-vs-Strict final-map CSV (optional).
The second comparison is observational; never infer cause from the
'keyframe mismatch' subgroups.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CHOICES=("hold","legacy","strict")
METRICS=("dynamic_rgb_mae","dynamic_depth_penalized_mae_m",
         "interior_depth_penalized_mae_m","boundary_depth_penalized_mae_m",
         "dynamic_psnr_db")


def frozen_summary(csv):
    df=pd.read_csv(csv)
    if df.empty:
        raise ValueError("No frozen source rows")
    if df.duplicated(["keyframe_id","mode"]).any():
        raise ValueError("Repeated frame/mode")
    if set(df["mode"])!=set(CHOICES):
        raise ValueError("Need all hold/legacy/strict")
    groups=df.groupby("keyframe_id").mode.apply(set)
    if any(group!=set(CHOICES) for group in groups):
        raise ValueError("Incomplete fixed-map triplet")
    fixed=["dynamic_mask_hash","depth_valid_mask_hash","dynamic_depth_n",
           "dynamic_rgb_n","source_keyframe_id","n_dynamic_gaussians"]
    for col in fixed:
        if df.groupby("keyframe_id")[col].nunique(dropna=False).max()!=1:
            raise ValueError("Candidate-specific target/map metadata: "+col)
    rows=[]
    for col in METRICS:
        p=df.pivot(index="keyframe_id",columns="mode",values=col)
        for mode in ("legacy","strict"):
            d=p[mode]-p["hold"]
            rows.append(dict(measure=col,comparison=mode+"_minus_hold",
                 n_frames=int(d.notna().sum()),mean_delta=float(d.mean()),
                 median_delta=float(d.median()),
                 lower_error_n=int((d<0).sum()),higher_error_n=int((d>0).sum())))
        d=p["strict"]-p["legacy"]
        rows.append(dict(measure=col,comparison="strict_minus_legacy",
                 n_frames=int(d.notna().sum()),mean_delta=float(d.mean()),
                 median_delta=float(d.median()),
                 lower_error_n=int((d<0).sum()),higher_error_n=int((d>0).sum())))
    p=df[df["mode"]=="hold"].copy()
    motion=p[["keyframe_id","source_keyframe_id","n_excluded",
             "n_excluded_legacy_gate_accepted","n_history_available",
             "excluded_prev_change_mean_m",
             "excluded_prev_change_gt_threshold_fraction"]]
    return pd.DataFrame(rows),motion


def cross_run_confounding(csv):
    d=pd.read_csv(csv)
    required=["frame_idx","frame_uid","is_keyframe_legacy","is_keyframe_strict",
              "delta_strict_minus_legacy_dynamic_depth_penalized_mae_m",
              "delta_strict_minus_legacy_dynamic_rgb_mae"]
    if any(x not in d.columns for x in required):
        raise ValueError("Missing V5.1b joined columns")
    if d.frame_idx.duplicated().any():
        raise ValueError("Duplicate image frame")
    left=d.is_keyframe_legacy.astype(int).astype(bool)
    right=d.is_keyframe_strict.astype(int).astype(bool)
    category=np.where(left&right,"both_keyframe",
        np.where(~left&~right,"neither_keyframe",
          np.where(left,"legacy_only_keyframe","strict_only_keyframe")))
    d=d.assign(keyframe_status=category)
    rows=[]
    for name,g in d.groupby("keyframe_status"):
        rows.append(dict(group=name,n_frames=len(g),
            dynamic_depth_mean_delta_m=float(
                g.delta_strict_minus_legacy_dynamic_depth_penalized_mae_m.mean()),
            dynamic_rgb_mean_delta=float(
                g.delta_strict_minus_legacy_dynamic_rgb_mae.mean())))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--frozen",required=True)
    parser.add_argument("--matched-v51b",default=None,
                        help="Prior V5.1b matched_per_frame CSV (optional)")
    parser.add_argument("--output",required=True)
    opts=parser.parse_args()
    folder=Path(opts.output)
    folder.mkdir(parents=True,exist_ok=True)
    scores,history=frozen_summary(opts.frozen)
    scores.to_csv(folder/"v51c_fixed_map_contrasts.csv",index=False)
    history.to_csv(folder/"v51c_exclusion_history_proxy.csv",index=False)
    report=[
        "# V5.1c controlled attribution",
        "",
        "Same map + camera + observation at pre-Mapping keyframe.",
        "Negative Strict-minus-Legacy MAE = current-frame lower residual.",
        "This fixed-map test is local in-sample, not optimized online ATE.",
        "",
        "## Frozen motion comparisons",
        scores.to_string(index=False,float_format=lambda x:f"{x:+.6f}"),
        "",
        f"Excluded candidate records (sum over {len(history)} keyframes): "
        f"{int(history.n_excluded.sum())}",
        "Excluded historical offset change is an activity proxy; "
        "not accumulated time-unobserved nor physical location error.",
    ]
    if opts.matched_v51b:
        d=cross_run_confounding(opts.matched_v51b)
        d.to_csv(folder/"v51c_crossrun_keyframe_status.csv",index=False)
        report.extend(["","## Final maps with *different* SLAM trajectories",
          d.to_string(index=False,float_format=lambda x:f"{x:+.6f}"),
          "Subgroup association is not a causal estimate of keyframe selection."])
    report.extend(["","## Decision guidance",
       "If fixed-map Strict gains differ from final-map results, Mapping/poses/",
       "keyframe/optimization may mediate differences; they are not identified",
       "individually. A future causal test must lock keyframe decisions and",
       "poses or replay identical states, then compare actual online Mapping.",
       "Do not claim the history proxy proves unseen Gaussians lag."])
    (folder/"v51c_attribution_report.md").write_text(
        "\n".join(report)+"\n",encoding="utf-8")
    print("\n".join(report))


if __name__=="__main__":main()
