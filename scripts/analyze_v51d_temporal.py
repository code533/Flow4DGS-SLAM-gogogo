"""V5.1d frozen-map temporal persistence audit.

Compare Hold/Legacy/Strict on EXACTLY the same source-map snapshot,
future tracked camera and RGB-D target. Group by future keyframe ORDER
1/3/5, retain actual image-frame gap, and report both unpaired horizon
summary and a *complete-source* subset for time-trend analysis.

These are short-horizon zero-order-hold renders, not motion GT, true
occlusion duration or temporally extrapolated Gaussian positions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODES=("hold","legacy","strict")
METRICS=("dynamic_depth_penalized_mae_m",
         "dynamic_rgb_mae","dynamic_psnr_db",
         "interior_depth_penalized_mae_m",
         "boundary_depth_penalized_mae_m",
         "dynamic_depth_coverage")
DEFAULT_HORIZONS=(1,3,5)


def paired_deltas(df):
    if df.empty:
        raise ValueError("No successfully evaluated future pairs")
    keys=["source_keyframe_id","future_keyframe_order"]
    if df.duplicated(keys+["mode"]).any():
        raise ValueError("Duplicate source/future/mode rows")
    if set(df["mode"])!=set(MODES):
        raise ValueError("Expected exactly Hold, Legacy and Strict")
    rows=[]
    for (src,order),group in df.groupby(keys,sort=True):
        if len(group)!=3 or set(group["mode"])!=set(MODES):
            raise ValueError(f"Incomplete future triplet {src} / {order}")
        invariants=["future_keyframe_id","elapsed_image_frames","dynamic_mask_hash",
                    "depth_valid_mask_hash","dynamic_depth_n","dynamic_rgb_n",
                    "interior_depth_n","boundary_depth_n","source_selection",
                    "source_n_excluded"]
        for col in invariants:
            if group[col].nunique(dropna=False)!=1:
                raise ValueError(f"Unmatched future target for {src}/{order}: {col}")
        indexed=group.set_index("mode")
        row=dict(
            source_keyframe_id=int(src),future_keyframe_order=int(order),
            future_keyframe_id=int(indexed.loc["hold","future_keyframe_id"]),
            elapsed_image_frames=int(indexed.loc["hold","elapsed_image_frames"]),
            source_selection=str(indexed.loc["hold","source_selection"]),
            source_n_excluded=int(indexed.loc["hold","source_n_excluded"]),
            source_n_excluded_legacy_gate_accepted=int(
                indexed.loc["hold","source_n_excluded_legacy_gate_accepted"]),
            source_excluded_prev_change_mean_m=float(
                indexed.loc["hold","source_excluded_prev_change_mean_m"]),
            dynamic_depth_n=int(indexed.loc["hold","dynamic_depth_n"]),
            interior_depth_n=int(indexed.loc["hold","interior_depth_n"]),
            boundary_depth_n=int(indexed.loc["hold","boundary_depth_n"]),
            dynamic_mask_hash=str(indexed.loc["hold","dynamic_mask_hash"]),
        )
        for metric in METRICS:
            for mode in MODES:
                row[f"{mode}_{metric}"]=float(indexed.loc[mode,metric])
            for candidate,baseline in (("legacy","hold"),("strict","hold"),("strict","legacy")):
                row[f"{candidate}_minus_{baseline}_{metric}"]=(
                    row[f"{candidate}_{metric}"]-row[f"{baseline}_{metric}"])
        rows.append(row)
    out=pd.DataFrame(rows)
    if out.duplicated(["source_keyframe_id","future_keyframe_order"]).any():
        raise ValueError("Duplicate paired sources")
    return out


def analyze(future_csv,event_csv,folder,horizons=DEFAULT_HORIZONS):
    df=pd.read_csv(future_csv)
    events=pd.read_csv(event_csv)
    if not horizons or tuple(horizons)!=tuple(sorted(set(horizons))):
        raise ValueError("Unsorted or duplicate horizons")
    pair=paired_deltas(df)
    if not set(pair.future_keyframe_order).issubset(set(horizons)):
        raise ValueError("Unexpected future horizon in source CSV")
    if events.empty or (events.action=="source_captured").sum()==0:
        raise ValueError("Missing captured source event log")
    captures=events.loc[events.action=="source_captured","source_keyframe_id"]
    if captures.duplicated().any():
        raise ValueError("Duplicated source captures")
    if not set(pair.source_keyframe_id).issubset(set(captures)):
        raise ValueError("Future row without captured source log")
    complete=pair.groupby("source_keyframe_id").future_keyframe_order.apply(
        lambda x:set(x)==set(horizons))
    full_ids=set(complete[complete].index)
    pair["has_all_requested_horizons"]=pair.source_keyframe_id.isin(full_ids)
    summary=[]
    for population,frame in (("all_available",pair),
                             ("complete_source_only",pair[pair.has_all_requested_horizons])):
        for order in horizons:
            subset=frame[frame.future_keyframe_order==order]
            for metric in METRICS:
                d=subset[f"strict_minus_legacy_{metric}"]
                n=int(d.notna().sum())
                summary.append(dict(
                    population=population,
                    horizon_keyframe_order=int(order),
                    metric=metric,n_source_pairs=len(subset),
                    n_valid_pairs=n,
                    mean_strict_minus_legacy=float(d.mean()) if n else float("nan"),
                    median_strict_minus_legacy=float(d.median()) if n else float("nan"),
                    strict_lower_n=int((d<0).sum()),
                    strict_higher_n=int((d>0).sum()),
                    mean_elapsed_image_frames=float(subset.elapsed_image_frames.mean())
                                                if len(subset) else float("nan"),
                    mean_strict_minus_hold=float(
                        subset[f"strict_minus_hold_{metric}"].mean()) if n else float("nan"),
                    mean_legacy_minus_hold=float(
                        subset[f"legacy_minus_hold_{metric}"].mean()) if n else float("nan")))
    summary=pd.DataFrame(summary)
    missing_rows=[]
    attempted=events[events.action=="source_captured"]
    for source in attempted.source_keyframe_id:
        present=set(pair[pair.source_keyframe_id==source].future_keyframe_order)
        for order in horizons:
            if order not in present:
                source_events=events[(events.source_keyframe_id==source)&
                                     (events.future_keyframe_order==order)]
                explanation="no_future_keyframe_record"
                if len(source_events):
                    explanation=";".join(source_events.action.astype(str).tolist())
                else:
                    closures=events[(events.source_keyframe_id==source)&
                                    (events.action=="snapshot_closed")]
                    if len(closures):
                        explanation=";".join(closures.reason.astype(str).tolist())
                missing_rows.append(dict(source_keyframe_id=int(source),
                                         future_keyframe_order=order,
                                         reason=explanation))
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    pair.to_csv(folder/"v51d_paired_future_deltas.csv",index=False)
    summary.to_csv(folder/"v51d_horizon_summary.csv",index=False)
    pd.DataFrame(missing_rows,columns=[
        "source_keyframe_id","future_keyframe_order","reason"]).to_csv(
        folder/"v51d_missing_horizons.csv",index=False)
    available=pair[pair.future_keyframe_order==1]
    lag=available[["source_keyframe_id","source_n_excluded_legacy_gate_accepted",
                   "source_excluded_prev_change_mean_m",
                   "strict_minus_legacy_dynamic_depth_penalized_mae_m"]].copy()
    lag.to_csv(folder/"v51d_history_vs_future.csv",index=False)
    text=[
      "# V5.1d frozen-map persistence report",
      "",
      f"Captured sources: {len(captures)}",
      f"Future source/horizon triplets: {len(pair)}",
      f"Complete 1/3/5 source trajectories: {len(full_ids)}",
      f"Explicitly missing source/horizons: {len(missing_rows)}",
      "",
      "## Future dynamic Depth MAE difference (Strict minus Legacy)",
      "",
    ]
    for pop in ("all_available","complete_source_only"):
        text.append("### "+pop)
        for order in horizons:
            sub=summary[(summary.population==pop)&
                        (summary.horizon_keyframe_order==order)&
                        (summary.metric=="dynamic_depth_penalized_mae_m")].iloc[0]
            text.append(
                f"future KF #{order}: N={int(sub.n_valid_pairs)}, "
                f"mean={sub.mean_strict_minus_legacy*1000:+.3f} mm, "
                f"median={sub.median_strict_minus_legacy*1000:+.3f} mm, "
                f"Strict better={int(sub.strict_lower_n)}, "
                f"mean actual image gap={sub.mean_elapsed_image_frames:.1f}")
    text.extend(["",
        "## Interpretation constraints",
        "- Future order is keyframe-arrival order, NOT video frame gap.",
        "- Missing or skipped horizons are explicit; only complete-source",
        "  subset is valid for a simple time-trend comparison.",
        "- Dynamic/valid depth masks match ACROSS CANDIDATES for each",
        "  future frame, but differ across time as new RGB-D is observed.",
        "- Frozen Gaussian appearance, geometry and source displacement",
        "  are held fixed. No future flow or 3D motion extrapolation.",
        "- No result here can by itself prove physical Gaussian location",
        "  lag, actual occlusion state, or online ATE improvement.",
    ])
    (folder/"v51d_report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print("\n".join(text))
    return pair,summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,
                    help="Folder containing v51d_future_per_horizon.csv and v51d_event_log.csv")
    ap.add_argument("--output",default=None)
    a=ap.parse_args()
    folder=Path(a.input)
    analyze(folder/"v51d_future_per_horizon.csv",
            folder/"v51d_event_log.csv",
            Path(a.output) if a.output else folder/"analysis")


if __name__=="__main__":
    main()
