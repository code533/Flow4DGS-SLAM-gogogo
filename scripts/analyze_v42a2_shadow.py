"""Summarize G3/G4 Shadow keyframe CSV and detailed point NPZ."""
import argparse,csv
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True,help="v42a2_shadow folder")
    args=parser.parse_args()
    folder=Path(args.input)
    path=folder/"v42a2_per_keyframe.csv"
    if not path.is_file():
        parser.error("Missing v42a2_per_keyframe.csv")
    with path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if not rows:parser.error("Empty CSV")
    keys=["valid_observation_fraction","invalid_projection_fraction",
          "g3_visibility_fallback_fraction","g4_consensus_fallback_fraction",
          "g4_accepted_fraction","consensus_dispersion_mean_m",
          "g1_vs_g0_mean_m","g3_vs_g0_mean_m","g4_vs_g0_mean_m",
          "g3_diff_over_10cm_fraction","g4_diff_over_10cm_fraction",
          "g3_new_rejections_n","g4_new_rejections_n",
          "g3_effective_vs_g0_mean_m","g4_effective_vs_g0_mean_m"]
    report=[f"V4.2a.2 SHADOW: {len(rows)} keyframe records. NOT a reconstruction accuracy test."]
    for k in keys:
        values=np.asarray([float(row[k]) for row in rows])
        if not np.isfinite(values).all():
            report.append(f"{k}: has nonfinite statistics")
            continue
        report.append(f"{k}: mean={values.mean():.8g} median={np.median(values):.8g} max={values.max():.8g}")
    for path in sorted(folder.glob("v42a2_[0-9][0-9][0-9][0-9][0-9].npz")):
        with np.load(path,allow_pickle=False) as z:
            q=z["q"]; g0=z["g0"]; g3=z["g3"]; g4=z["g4"]
            risk=z["diag_consensus_dispersion_m"]
            accepted=z["diag_g4_accepted"]
        d3=np.linalg.norm(g3-g0,axis=1)
        d4=np.linalg.norm(g4-g0,axis=1)
        report.append(
            f"{path.name}: N={len(q)}, G3>10cm={int((d3>.1).sum())}, "
            f"G4>10cm={int((d4>.1).sum())}, G4accepted={accepted.mean():.4f}, "
            f"dispersion_p95={np.percentile(risk,95):.6f}m")
    dest=folder/"v42a2_summary.txt"
    dest.write_text("\n".join(report)+"\n",encoding="utf-8")
    print("\n".join(report))
if __name__=="__main__":
    main()
