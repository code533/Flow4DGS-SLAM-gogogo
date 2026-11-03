"""Compare preregistered selection policies on identical held-out pairs.

The winner is defined by FUTURE ordinary depth MAE, not the source scoring
rule; near-oracle accuracy handles ties. Reports depth regret, gain vs G0,
selection rate, RGB and coverage side effects. No independent frames assumed.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def summarize(df):
    if df.empty:raise ValueError("Empty policy results")
    fields=["source_keyframe_id","future_keyframe_id","policy"]
    if df.duplicated(fields).any():raise ValueError("Duplicate policy/pair")
    order=["always_g0","mae","clipped50","huber20","interior_clipped50"]
    paired=df.groupby(["source_keyframe_id","future_keyframe_id"]).policy.apply(set)
    if any(x!=set(order) for x in paired):
        raise ValueError("All policies must evaluate identical source/future pairs")
    rows=[]
    for policy in order:
        g=df[df.policy==policy]
        rows.append(dict(
            policy=policy,n_pairs=len(g),
            chose_g0_fraction=float((g.selected=="g0").mean()),
            future_depth_win_fraction=float(g.future_depth_win_vs_g0.mean()),
            future_depth_loss_fraction=float(g.future_depth_loss_vs_g0.mean()),
            future_near_oracle_fraction=float(g.future_near_oracle.mean()),
            mean_future_gain_over_g0_mm=float(g.future_gain_over_g0_m.mean()*1000),
            median_future_gain_over_g0_mm=float(g.future_gain_over_g0_m.median()*1000),
            mean_regret_mm=float(g.future_regret_m.mean()*1000),
            median_regret_mm=float(g.future_regret_m.median()*1000),
            mean_future_rgb_change=float(g.future_rgb_change_vs_g0.mean()),
            mean_future_coverage_change=float(g.future_coverage_change_vs_g0.mean()),
        ))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",default=None)
    args=ap.parse_args()
    path=Path(args.input)
    df=pd.read_csv(path)
    report=summarize(df)
    target=Path(args.output) if args.output else path.parent/"v43a4_policy_summary.csv"
    report.to_csv(target,index=False)
    print(report.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("NOTE: 1 scene / correlated keyframes; this is zero-order-hold")
    print("future rendering, not online ATE or independent motion GT.")

if __name__=="__main__":main()
