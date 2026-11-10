"""V5.2a four-way frozen map future evaluation, with attrition safeguards.

Primary decision: Strict+History versus Strict on future dynamic-depth MAE,
same source and future camera, time orders 1/3/5. Secondary metrics are
dynamic RGB, boundary/interior depth, PSNR and coverage. Report full
completion subset independently from all-available samples. Also record
no-history-eligible cases; they must not count as algorithm improvement.

No hyperparameter search or one-frame threshold optimization.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODES=("hold","legacy","strict","strict_history")
HORIZONS=(1,3,5)
METRICS=("dynamic_depth_penalized_mae_m","dynamic_rgb_mae",
         "interior_depth_penalized_mae_m",
         "boundary_depth_penalized_mae_m","dynamic_psnr_db",
         "dynamic_depth_coverage")


def pairs(future,source):
    rows=pd.read_csv(future)
    src=pd.read_csv(source)
    if rows.empty or src.empty:
        raise ValueError("V5.2a files must contain future and source evaluation")
    keys=["source_keyframe_id","future_keyframe_order"]
    if rows.duplicated(keys+["mode"]).any():
        raise ValueError("Duplicate future source/horizon/mode combination")
    if src.duplicated(["source_keyframe_id","mode"]).any():
        raise ValueError("Duplicate source/mode combination")
    if set(rows["mode"])!=set(MODES) or set(src["mode"])!=set(MODES):
        raise ValueError("Missing Hold/Legacy/Strict/Strict+History mode")
    metadata=("future_keyframe_id","elapsed_image_frames",
             "dynamic_mask_hash","depth_valid_mask_hash",
             "dynamic_depth_n","dynamic_rgb_n","boundary_depth_n",
             "interior_depth_n")
    parsed=[]
    for (uid,order),grp in rows.groupby(keys,sort=True):
        if set(grp["mode"])!=set(MODES) or len(grp)!=4:
            raise ValueError(f"Incomplete future four-way set for source={uid}, order={order}")
        for item in metadata:
            if grp[item].nunique(dropna=False)!=1:
                raise ValueError(f"Inconsistent future observation {item} for source {uid}")
        if int(order) not in HORIZONS:
            raise ValueError("Unsupported future keyframe order")
        src_gr=src[src.source_keyframe_id==uid]
        if len(src_gr)!=4 or set(src_gr["mode"])!=set(MODES):
            raise ValueError(f"Incomplete source four-way set: {uid}")
        for item in ("dynamic_mask_hash","depth_valid_mask_hash","dynamic_depth_n"):
            if src_gr[item].nunique(dropna=False)!=1:
                raise ValueError(f"Inconsistent source observation {item}")
        s=src_gr.set_index("mode")
        g=grp.set_index("mode")
        n_pred=int(s.loc["strict_history","n_history_predicted"])
        out=dict(source_keyframe_id=int(uid),
            future_keyframe_order=int(order),
            future_keyframe_id=int(g.iloc[0]["future_keyframe_id"]),
            elapsed_image_frames=int(g.iloc[0]["elapsed_image_frames"]),
            source_selection=str(g.iloc[0]["source_selection"]),
            source_n_fov_excluded=int(s.loc["strict_history","n_fov_excluded"]),
            n_history_predicted=n_pred,
            n_history_clipped=int(s.loc["strict_history","n_history_clipped"]),
            history_eligible=int(n_pred>0),
            dynamic_depth_n=int(g.iloc[0]["dynamic_depth_n"]),
            boundary_depth_n=int(g.iloc[0]["boundary_depth_n"]),
            interior_depth_n=int(g.iloc[0]["interior_depth_n"]),
            dynamic_mask_hash=str(g.iloc[0]["dynamic_mask_hash"]),
            depth_valid_mask_hash=str(g.iloc[0]["depth_valid_mask_hash"]))
        for metric in METRICS:
            for mode in MODES:
                out[mode+"_"+metric]=float(g.loc[mode,metric])
            for cmp,base in (("strict_history","strict"),
                             ("strict_history","legacy"),
                             ("strict","legacy"),
                             ("strict_history","hold")):
                out[f"{cmp}_minus_{base}_{metric}"]=(
                    out[cmp+"_"+metric]-out[base+"_"+metric])
        parsed.append(out)
    data=pd.DataFrame(parsed)
    # No imputed horizons, retain actual observed sample count.
    full=data.groupby("source_keyframe_id").future_keyframe_order.apply(
        lambda x: set(x)==set(HORIZONS))
    data["complete_source"]=data.source_keyframe_id.isin(
        set(full[full].index))
    return data


def analyze(future,source,events,output):
    data=pairs(future,source)
    log=pd.read_csv(events)
    captures=log.loc[log.action=="source_captured","source_keyframe_id"]
    if captures.duplicated().any():
        raise ValueError("Duplicate capture in event log")
    if not set(data.source_keyframe_id).issubset(set(captures)):
        raise ValueError("Future scored source without source capture")
    summary=[]
    for population,part in (("all_available",data),
                            ("complete_only",data[data.complete_source]),
                            ("history_eligible",data[data.history_eligible==1]),
                            ("complete_and_history_eligible",
                                data[(data.complete_source)&(data.history_eligible==1)])):
        for horizon in HORIZONS:
            h=part[part.future_keyframe_order==horizon]
            for metric in METRICS:
                diff=h["strict_history_minus_strict_"+metric]
                valid=diff.dropna()
                summary.append(dict(
                    population=population,horizon_keyframe_order=horizon,
                    metric=metric,n_source_pairs=int(len(h)),
                    n_valid=int(len(valid)),mean_delta=float(valid.mean()) if len(valid) else float("nan"),
                    median_delta=float(valid.median()) if len(valid) else float("nan"),
                    strict_history_better_n=int((valid>0).sum()) if metric=="dynamic_psnr_db"
                      else int((valid<0).sum()),
                    strict_history_worse_n=int((valid<0).sum()) if metric=="dynamic_psnr_db"
                      else int((valid>0).sum()),
                    mean_image_gap=float(h.elapsed_image_frames.mean()) if len(h) else float("nan")))
    summary=pd.DataFrame(summary)
    source_ids=set(data.source_keyframe_id)
    missing=[]
    for source_id in captures:
        present=set(data.loc[data.source_keyframe_id==source_id,
                             "future_keyframe_order"])
        for h in HORIZONS:
            if h not in present:
                logrows=log[(log.source_keyframe_id==source_id)&
                            (log.future_keyframe_order==h)]
                close=log[(log.source_keyframe_id==source_id)&
                          (log.action=="snapshot_closed")]
                reason="missing_future_record"
                if len(logrows):reason=";".join(logrows.action.astype(str))
                elif len(close):reason=";".join(close.reason.astype(str))
                missing.append(dict(source_keyframe_id=int(source_id),
                                    future_keyframe_order=h,reason=reason))
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    data.to_csv(output/"v52a_four_way_paired.csv",index=False)
    summary.to_csv(output/"v52a_history_vs_strict_summary.csv",index=False)
    pd.DataFrame(missing,columns=("source_keyframe_id","future_keyframe_order",
                                  "reason")).to_csv(output/"v52a_missing_horizons.csv",index=False)
    report=[
        "# V5.2a frozen-map conditional history audit",
        "",
        f"Captured sources: {len(captures)}",
        f"Completed source-horizon four-way sets: {len(data)}",
        f"Full three-horizon sources: {data.loc[data.complete_source,'source_keyframe_id'].nunique()}",
        f"History eligible sources: {data.loc[data.history_eligible==1,'source_keyframe_id'].nunique()}",
        f"Missing source horizons: {len(missing)}",
        "",
        "Primary: future dynamic Depth MAE, strict_history - strict (negative is better).",
        "The same future observation/camera and frozen source map are shared.",
        "No history-eligible sources must not be interpreted as evidence.",
    ]
    for name in ("all_available","complete_only","complete_and_history_eligible"):
        report.extend(["","## "+name])
        for h in HORIZONS:
            sub=summary[(summary.population==name)&
                        (summary.horizon_keyframe_order==h)&
                        (summary.metric=="dynamic_depth_penalized_mae_m")].iloc[0]
            report.append(f"future keyframe #{h}: N={int(sub.n_valid)}, "
                f"Strict+History - Strict Depth={1000*sub.mean_delta:+.3f} mm, "
                f"better={int(sub.strict_history_better_n)}, worse={int(sub.strict_history_worse_n)}, "
                f"mean image gap={sub.mean_image_gap:.1f}")
    report.extend(["","## Scientific limits",
        "- Past optimized offset changes are not true 3D motion GT.",
        "- Gaussian creation KF IDs reduce but do not eliminate ID ambiguity.",
        "- History candidate is applied only at source, then frozen. It does not",
        "  predict extra motion at later target keyframes.",
        "- All source snapshots are evaluated on a single running SLAM model;",
        "  future tracked poses are not independent of Legacy online Mapping.",
        "- The 342–382 focus window was identified from previous Bonn trials;",
        "  confirmation requires independent heldout sequences and repeated runs.",
        "- No online ATE or dynamic reconstruction improvement is claimed."])
    (output/"v52a_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("\n".join(report))
    return data,summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",default=None)
    args=ap.parse_args()
    root=Path(args.input)
    analyze(root/"v52a_future_per_horizon.csv",
            root/"v52a_source_per_keyframe.csv",
            root/"v51d_event_log.csv",
            Path(args.output) if args.output else root/"v52a_analysis")


if __name__=="__main__":
    main()
