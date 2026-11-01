"""Summarize same-frame vs next-keyframe frozen map motion-hypothesis differences."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,help="v43a2_temporal directory")
    args=p.parse_args()
    root=Path(args.input)
    data=pd.read_csv(root/"v43a2_temporal.csv")
    if data.duplicated(["source_keyframe_id","future_keyframe_id","hypothesis"]).any():
        raise ValueError("Duplicated source/future/method records")
    report=[f"Frozen-map ZERO-ORDER-HOLD: {data.source_keyframe_id.nunique()} source frames.",
        "Not an optimized future flow predictor, not independent motion ground truth."]
    for method,grp in data.groupby("hypothesis"):
        report.append(f"{method}: {len(grp)} observations")
        for metric in ("rgb","depth"):
            col="dynamic_rgb_delta" if metric=="rgb" else "dynamic_depth_delta_m"
            s=grp["source_"+col]; f=grp["future_"+col]
            report.append(f"  {metric}: source_improved={(s<0).sum()}/{len(grp)} "
                          f"future_improved={(f<0).sum()}/{len(grp)}, "
                          f"both_improved={((s<0)&(f<0)).sum()}, "
                          f"source_mean={s.mean():+.8g} future_mean={f.mean():+.8g}")
        report.append("  both_sign_retained={}/{}".format(
            int(grp.both_sign_retained.sum()),len(grp)))
    text="\n".join(report)+"\n"
    (root/"v43a2_summary.txt").write_text(text,encoding="utf-8")
    print(text)

if __name__=="__main__":main()
