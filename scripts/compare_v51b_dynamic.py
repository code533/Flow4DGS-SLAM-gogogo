"""V5.1b: matched-frame reconstruction audit for Legacy vs Strict-FOV.

Consumes final eval_rendering CSVs. No rendering, GPU or model reload.
Refuses a comparison when the chosen same-index frame has a different
dynamic/static target mask or valid-depth mask. Do not mask away failures.
Each summary uses the SAME frames and fixed GT-pixel denominators.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REGIONS=("all","static","dynamic","interior","boundary")
METRICS=("rgb_mae","psnr_db","depth_penalized_mae_m",
         "depth_pred_valid_mae_m","depth_coverage")
WINDOWS={"all":None,"fov_transition":(317,382),"critical_fov":(342,382)}


def load_metrics(path,expected_mode):
    d=pd.read_csv(path)
    for key in ("frame_idx","frame_uid","mode","phase","dynamic_mask_hash",
                "depth_valid_mask_hash","n_dynamic_mask_pixels","dynamic_fraction",
                "is_keyframe","boundary_width_px","opacity_threshold",
                "uncovered_penalty_m"):
        if key not in d.columns:raise ValueError("CSV lacks "+key)
    if d.empty or d.frame_idx.duplicated().any():
        raise ValueError("Empty data or repeated frame_idx in "+str(path))
    if set(d["mode"])!={expected_mode}:
        raise ValueError("Wrong evaluation mode in "+str(path))
    if d["phase"].nunique()!=1:
        raise ValueError("Multiple stages in one file")
    for region in REGIONS:
        for metric in METRICS:
            if region+"_"+metric not in d.columns:
                raise ValueError("Missing "+region+"_"+metric)
        for k in ("rgb_n","depth_n","depth_pred_n"):
            if region+"_"+k not in d.columns:
                raise ValueError("Missing "+region+"_"+k)
    return d.set_index("frame_idx",drop=False).sort_index()


def compare(legacy_csv,strict_csv,folder,*,allow_partial=False):
    a=load_metrics(legacy_csv,"legacy")
    b=load_metrics(strict_csv,"strict")
    ia,ib=set(a.index),set(b.index)
    common=sorted(ia&ib)
    if not common:raise ValueError("No same-index frames")
    if not allow_partial and ia!=ib:
        raise ValueError("Frame sets differ: Legacy-only={} Strict-only={}".format(
            sorted(ia-ib),sorted(ib-ia)))
    a=a.loc[common]
    b=b.loc[common]
    # No silent swapping, different dynamic labels, missing targets or
    # inconsistent evaluation hyperparameters.
    fixed_cols=["frame_uid","dynamic_mask_hash","depth_valid_mask_hash",
                "n_dynamic_mask_pixels","boundary_width_px",
                "opacity_threshold","uncovered_penalty_m"]
    for region in REGIONS:
        fixed_cols.extend([region+"_rgb_n",region+"_depth_n"])
    mismatches={}
    for col in fixed_cols:
        mismatch=(a[col].astype(str)!=b[col].astype(str))
        if mismatch.any():
            mismatches[col]=[int(x) for x in a.index[mismatch]][:20]
    if mismatches:
        raise ValueError("Incomparable evaluation masks/target frames: "+str(mismatches))
    if a["phase"].iloc[0]!=b["phase"].iloc[0]:
        raise ValueError("Evaluation stages differ")
    paired=pd.DataFrame({
        "frame_idx":common,
        "frame_uid":a.frame_uid.to_numpy(),
        "phase":a.phase.to_numpy(),
        "is_keyframe_legacy":a.is_keyframe.to_numpy(),
        "is_keyframe_strict":b.is_keyframe.to_numpy(),
        "n_dynamic_mask_pixels":a.n_dynamic_mask_pixels.to_numpy(),
        "dynamic_fraction":a.dynamic_fraction.to_numpy(),
        "dynamic_mask_hash":a.dynamic_mask_hash.to_numpy(),
        "depth_valid_mask_hash":a.depth_valid_mask_hash.to_numpy(),
    })
    for region in REGIONS:
        for metric in METRICS:
            name=region+"_"+metric
            paired["legacy_"+name]=a[name].to_numpy()
            paired["strict_"+name]=b[name].to_numpy()
            paired["delta_strict_minus_legacy_"+name]=b[name].to_numpy()-a[name].to_numpy()
        paired["n_"+region+"_depth"]=a[region+"_depth_n"].to_numpy()
        paired["n_"+region+"_rgb"]=a[region+"_rgb_n"].to_numpy()
    rows=[]
    for window,span in WINDOWS.items():
        subset=paired if span is None else paired[
            (paired.frame_uid>=span[0])&(paired.frame_uid<=span[1])]
        for region in REGIONS:
            for metric in METRICS:
                key="delta_strict_minus_legacy_"+region+"_"+metric
                valid=subset[key].dropna()
                rows.append(dict(
                    window=window,region=region,metric=metric,
                    n_common_frames=len(subset),n_valid_frames=len(valid),
                    mean_delta=float(valid.mean()) if len(valid) else float("nan"),
                    median_delta=float(valid.median()) if len(valid) else float("nan"),
                    strict_lower_n=int((valid<0).sum()),
                    strict_higher_n=int((valid>0).sum()),
                    # For all of these except PSNR, lower is favorable.
                    better_direction="higher" if metric=="psnr_db" else "lower",
                    strict_better_n=int((valid>0).sum()) if metric=="psnr_db"
                       else int((valid<0).sum()),
                ))
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    paired.to_csv(folder/"v51b_matched_per_frame.csv",index=False)
    summary=pd.DataFrame(rows)
    summary.to_csv(folder/"v51b_comparison_summary.csv",index=False)
    lines=[
        "# V5.1b matched dynamic RGB-D reconstruction audit",
        "",
        f"Phase: {a.phase.iloc[0]}, matched frames: {len(common)}",
        f"Missing frames Legacy-only={len(ia-ib)}, Strict-only={len(ib-ia)}",
        "Per-frame dynamic mask AND depth-valid target hash equality checked.",
        "Dynamic mask True=estimated motion pixels, NOT moving-object GT.",
        "Metrics computed using individual-run estimated camera poses and",
        "Gaussian maps; identical target frames, not identical geometry/poses.",
        "RGB-D is measured on this same training sequence, not heldout scenes.",
        "Missing rendered depth penalized; coverage is separately logged.",
        "Negative Strict-minus-Legacy MAE is improvement; positive PSNR is improvement.",
    ]
    for window in WINDOWS:
        lines.append("\n## "+window)
        for region in ("dynamic","interior","boundary","all"):
            for metric in ("rgb_mae","depth_penalized_mae_m","depth_coverage","psnr_db"):
                r=summary[(summary.window==window)&(summary.region==region)
                          &(summary.metric==metric)].iloc[0]
                lines.append(f"{region} {metric}: N={r.n_valid_frames} "
                             f"mean Strict-Legacy={r.mean_delta:+.7f}, "
                             f"Strict better frames={r.strict_better_n}")
    (folder/"v51b_comparison_report.md").write_text(
        "\n".join(lines)+"\n",encoding="utf-8")
    return summary,paired


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--legacy",required=True)
    p.add_argument("--strict",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--allow-partial",action="store_true",
                   help="Explicitly compare intersection if frames missing")
    a=p.parse_args()
    summary,paired=compare(a.legacy,a.strict,a.output,
                           allow_partial=a.allow_partial)
    print("Matched frames",len(paired))
    print(summary[(summary.window=="all")&
                  (summary.region.isin(["all","dynamic","interior","boundary"]))&
                  (summary.metric.isin(["rgb_mae","depth_penalized_mae_m",
                                       "depth_coverage","psnr_db"]))].to_string(index=False))


if __name__=="__main__":
    main()
