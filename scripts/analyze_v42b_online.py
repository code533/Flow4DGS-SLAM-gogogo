"""Summarize V4.2b online Mapping audit CSV(s), without treating change as accuracy."""
import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs",nargs="+",required=True,
                        help="One or more v42b_per_keyframe.csv files")
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    lines=["V4.2b ONLINE Mapping audit (candidate changes are NOT reconstruction accuracy)."]
    for src in args.inputs:
        p=Path(src)
        with p.open(newline="",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        if not rows:
            raise ValueError("Empty CSV: "+str(p))
        mode={r["mode"] for r in rows}
        if len(mode)!=1:
            raise ValueError("Mixed modes inside single run: "+str(p))
        if len({int(r["keyframe_id"]) for r in rows})!=len(rows):
            raise ValueError("Duplicate frame IDs: "+str(p))
        keys=("q_mean","support_selected_fraction","g4_accepted_fraction",
              "selected_vs_g0_mean_m","selected_vs_g0_gt_10cm_fraction",
              "selected_gate_pass_fraction","new_gate_rejections_n",
              "new_gate_acceptances_n","effective_vs_g0_mean_m",
              "candidate_compute_ms","gaussian_update_ms")
        lines.append(f"file={p} mode={next(iter(mode))} keyframes={len(rows)}")
        for key in keys:
            v=np.array([float(r[key]) for r in rows],dtype=np.float64)
            v=v[np.isfinite(v)]
            if v.size:
                lines.append(f"  {key}: mean={v.mean():.7g}, median={np.median(v):.7g}")
    result="\n".join(lines)+"\n"
    if args.output:
        Path(args.output).write_text(result,encoding="utf-8")
    print(result)


if __name__=="__main__":
    main()
