"""Analyze V4.2a.1 read-only attribution CSV and detailed Gaussian NPZs."""
import argparse
import csv
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,help="v42a1_attribution folder")
    p.add_argument("--output",default=None)
    args=p.parse_args()
    folder=Path(args.input)
    out=Path(args.output) if args.output else folder
    out.mkdir(parents=True,exist_ok=True)
    source=folder/"v42a1_per_keyframe.csv"
    if not source.is_file():
        p.error("No v42a1_per_keyframe.csv found; run V4.2a.1 shadow with attribute_failures=true")
    with source.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if not rows:p.error("Attribution CSV is empty")
    focus=("all_dynamic_reason_projection_outside_image_fraction",
           "selected_reason_projection_outside_image_fraction",
           "selected_reason_previous_depth_invalid_fraction",
           "selected_reason_forward_flow_invalid_fraction",
           "selected_reason_forward_correspondence_outside_image_fraction",
           "selected_reason_backward_flow_invalid_at_warp_fraction",
           "selected_reason_current_depth_invalid_at_warp_fraction",
           "selected_reason_q_near_zero_despite_valid_correspondence_fraction",
           "g0_to_g1_new_rejections_n","g0_to_g1_new_acceptances_n",
           "selected_cross_motion_risk_proxy_fraction",
           "g1_diff_p99_m")
    def num(r,k):
        return float(r[k])
    summary=[f"V4.2a.1: {len(rows)} keyframe records; diagnostic only."]
    for k in focus:
        values=np.array([num(r,k) for r in rows],dtype=float)
        finite=values[np.isfinite(values)]
        summary.append(f"{k}: mean={finite.mean():.7g} max={finite.max():.7g}" if len(finite)
                       else f"{k}: all NaN")
    files=sorted(folder.glob("v42a1_[0-9][0-9][0-9][0-9][0-9].npz"))
    for file in files:
        with np.load(file,allow_pickle=False) as z:
            q=np.asarray(z["projection_q_sample"])
            change=np.asarray(z["knn_g1_minus_g0_norm_m"])
            reason=np.asarray(z["projection_reason"])
            effective=np.asarray(z["knn_effective_g1_minus_g0_norm_m"])
            risk=np.asarray(z["knn_cross_motion_risk_proxy"])
        count=int((change>.1).sum())
        overlap=int(((change>.1)&risk).sum())
        summary.append(
            f"{file.name}: N={len(q)} Q<=1e-6={(q<=1e-6).mean():.4f} "
            f"G1change>10cm={count} overlap_cross_motion_proxy={overlap} "
            f"effective_mean_mm={effective.mean()*1000:.3f} "
            f"projection_outside={(reason==2).sum()}")
    output=out/"v42a1_summary.txt"
    output.write_text("\n".join(summary)+"\n",encoding="utf-8")
    print("\n".join(summary))


if __name__=="__main__":
    main()
