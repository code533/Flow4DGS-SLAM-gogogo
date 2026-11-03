"""Summarize V5.0 motion-observation failure proxies across keyframes.

Do not interpret source/target depth missing as verified occlusion.
Input files are saved under <run>/v50_attribution/.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def summarize(folder):
    folder=Path(folder)
    frame=pd.read_csv(folder/"v50_per_keyframe.csv")
    group=pd.read_csv(folder/"v50_per_group.csv")
    if frame.empty or group.empty:
        raise ValueError("Empty V5.0 results")
    if frame.keyframe_id.duplicated().any():
        raise ValueError("Repeated keyframe id")
    if group.duplicated(["keyframe_id","group_name"]).any():
        raise ValueError("Repeated per-group row")
    counts=group.groupby("keyframe_id").n_selected_group.sum()
    f=frame.set_index("keyframe_id")
    if set(counts.index)!=set(f.index):
        raise ValueError("Frame and group samples differ")
    if not np.array_equal(counts.sort_index().to_numpy(),
                          f.n_selected_movements.sort_index().to_numpy()):
        raise ValueError("Group partition does not sum to selected points")
    for prefix in ["stage_source_projected",
                   "stage_source_depth_consistent",
                   "stage_target_projection_valid",
                   "stage_target_filled_depth_valid",
                   "stage_raw_motion_under_1m"]:
        if prefix not in f.columns:
            raise ValueError("Missing stage counter "+prefix)
    stages=["n_all_dynamic_gaussians","stage_source_projected",
            "stage_source_depth_consistent","stage_target_projection_valid",
            "stage_target_filled_depth_valid","stage_raw_motion_under_1m",
            "n_actual_g0_update_gate_pass"]
    totals=[float(f[key].sum()) for key in stages]
    if any(totals[i+1]>totals[i] for i in range(len(totals)-1)):
        raise ValueError("Pipeline stage counts must be non-increasing")
    weighted=(group.groupby("group_name").agg(
        group_count=("n_selected_group","sum"),
        n_keyframes=("keyframe_id","nunique")).reset_index())
    weighted["group_fraction"]=weighted.group_count/weighted.group_count.sum()
    weighted=weighted.sort_values("group_count",ascending=False)
    lines=[
      f"V5.0 read-only motion attribution: {len(frame)} eligible keyframes",
      "No confirmed occlusion / no 3D ground truth / G0 Mapping unchanged.",
      "",
      "Stage counts (sum over keyframes, not unique Gaussian identities):",
    ]
    for i,key in enumerate(stages):
        retained=totals[i]/totals[0] if totals[0] else float("nan")
        prev=totals[i]/totals[i-1] if i>0 and totals[i-1] else float("nan")
        lines.append(f"  {key}: {int(totals[i])} (share all={retained:.3%}, "
                     f"share prev stage={prev:.3%})")
    lines.extend(["","Selected-motion proxy groups:"])
    for r in weighted.itertuples():
        lines.append(f"  {r.group_name}: {int(r.group_count)} ({r.group_fraction:.3%})")
    lines.extend(["",
      f"Mean per-keyframe original target depth fraction: "
      f"{f.selected_original_target_depth_valid_fraction.mean():.4f}",
      f"Mean per-keyframe G0 gate pass rate: "
      f"{f.g0_update_gate_pass_fraction.mean():.4f}",
      f"Median smoothing correction mean per frame: "
      f"{f.selected_smoothing_correction_mean_m.median():.5f} m",
      "Interpretation: diagnosis of observation availability and propagation, "
      "not an accuracy comparison; select next V5 algorithm only after verifying "
      "dominant evidence groups and representative depth/flow visualizations."])
    (folder/"v50_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    weighted.to_csv(folder/"v50_group_summary.csv",index=False)
    print("\n".join(lines))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    a=ap.parse_args()
    summarize(a.input)


if __name__=="__main__":
    main()
