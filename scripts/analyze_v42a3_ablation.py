"""V4.2a.3: paired G3 / Support / Soft / Full G4 mechanism decomposition."""
import argparse,csv
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,help="v42a3_ablation directory")
    args=p.parse_args()
    root=Path(args.input)
    with (root/"v42a3_per_keyframe.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if not rows:raise SystemExit("Empty CSV")
    methods=["g3","support_only","soft_no_hard_dispersion","g4"]
    frames={}
    for row in rows:
        frames.setdefault(int(row["keyframe_id"]),{})[row["stage"]]=row
    for fid,methods_by in frames.items():
        if set(methods_by)!=set(methods):
            raise ValueError(f"Frame {fid}: missing modes {set(methods)-set(methods_by)}")
    output=[f"V4.2a.3 SHADOW: {len(frames)} paired keyframes. No GT and no accuracy claim."]
    for method in methods:
        group=[frames[fid][method] for fid in sorted(frames)]
        def mean(k):
            a=np.array([float(row[k]) for row in group])
            return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
        output.append(
            f"{method}: accepted={mean('accepted_fraction'):.4f}, "
            f"vs_g0={mean('vs_g0_mean_m')*1000:.3f}mm, "
            f"large_10cm={mean('change_over_10cm_fraction'):.4f}, "
            f"gate_new_reject={mean('new_gate_rejection_n'):.2f}/frame, "
            f"effective_diff={mean('effective_vs_g0_mean_m')*1000:.3f}mm")
    source=frames[sorted(frames)[0]]["g4"]
    for k in ("soft_vs_support_shared_mean_m","hard_dispersion_new_rejection_n",
              "delta_limit_new_rejection_n"):
        vals=np.array([float(frames[f]["g4"][k]) for f in frames])
        output.append(f"{k}: mean={np.nanmean(vals):.8g}")
    for fid in (22,212,357,382):
        if fid in frames:
            output.append(f"KF{fid}: "+", ".join(
                f"{name} accept={float(frames[fid][name]['accepted_fraction']):.3f} "
                f"diff={float(frames[fid][name]['vs_g0_mean_m'])*1000:.3f}mm"
                for name in methods))
    report=root/"v42a3_summary.txt"
    report.write_text("\n".join(output)+"\n",encoding="utf-8")
    print("\n".join(output))
if __name__=="__main__":main()
