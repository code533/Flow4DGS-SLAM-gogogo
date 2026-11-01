"""Summarize V4.3a fixed-state RGB-D motion candidate render scores."""
import argparse,csv
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,help="v43a_observation output folder")
    args=p.parse_args()
    root=Path(args.input)
    with (root/"v43a_per_keyframe.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    frames={}
    for row in rows:
        fid=int(row["keyframe_id"])
        if row["hypothesis"] in frames.setdefault(fid,{}):
            raise ValueError(f"Duplicate candidate {fid} {row['hypothesis']}")
        frames[fid][row["hypothesis"]]=row
    keys={"g0","g3_support","g4"}
    if not frames or any(set(g)!=keys for g in frames.values()):
        raise ValueError("Missing complete paired candidate frame(s)")
    metrics=["dynamic_rgb_mae","dynamic_depth_penalized_mae_m","dynamic_depth_coverage",
             "static_rgb_mae","static_depth_penalized_mae_m",
             "all_rgb_mae","all_depth_penalized_mae_m"]
    report=[f"V4.3a: {len(frames)} paired pre-Mapping RGB-D frames; selection-frame agreement only.",
            "No fixed weighted total score; RGB, depth and coverage evaluated separately."]
    for metric in metrics:
        fids=[f for f in frames if all(np.isfinite(float(frames[f][key][metric])) for key in keys)]
        report.append(f"{metric}: evaluable_frames={len(fids)}")
        if not fids:continue
        for key in ("g3_support","g4"):
            gain=np.asarray([float(frames[f][key][metric])-float(frames[f]["g0"][metric]) for f in fids])
            if metric.endswith("coverage"):
                improved=(gain>0).sum()
            else:improved=(gain<0).sum()
            report.append(f"  {key}-g0 mean={gain.mean():+.8g}, median={np.median(gain):+.8g}, improved={improved}/{len(fids)}")
    for fid in (212,357):
        if fid in frames:
            report.append(f"KF{fid} dynamic RGB/depth/coverage:")
            for key in ("g0","g3_support","g4"):
                r=frames[fid][key]
                report.append(f"  {key}: rgb={r['dynamic_rgb_mae']} depth={r['dynamic_depth_penalized_mae_m']} coverage={r['dynamic_depth_coverage']}")
    output=root/"v43a_summary.txt"
    output.write_text("\n".join(report)+"\n",encoding="utf-8")
    print("\n".join(report))
if __name__=="__main__":main()
