"""V4.3a.1 offline summary: paired RGB/Depth sign consistency and spatial tiles."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--input",required=True,help="v43a1_regions directory")
 args=p.parse_args();root=Path(args.input)
 f=root/"v43a1_per_keyframe.csv"
 if not f.exists():p.error("missing v43a1_per_keyframe.csv")
 d=pd.read_csv(f)
 if len(d)==0:p.error("empty CSV")
 if d.duplicated(["keyframe_id","hypothesis"]).any():p.error("duplicate pairs")
 expected={"g3_support","g4"}
 groups=d.groupby("keyframe_id").hypothesis.apply(set)
 if any(g!=expected for g in groups):p.error("incomplete paired candidates")
 lines=[f"V4.3a.1: {len(groups)} fully paired keyframes. Same-state diagnostic only; NOT held-out evaluation."]
 for name,x in d.groupby("hypothesis"):
  lines.append(f"{name}: frames={len(x)}")
  for k in ("rgb_delta_dynamic","depth_delta_dynamic_m",
            "pixel_both_better_fraction","pixel_rgb_better_depth_worse_fraction",
            "pixel_depth_better_rgb_worse_fraction","pixel_both_worse_fraction",
            "tiles_both_better","tiles_conflict","n_eligible_tiles"):
   a=x[k].dropna()
   if a.empty:lines.append(f"  {k}: unavailable")
   else:lines.append(f"  {k}: mean={a.mean():+.8g}, median={a.median():+.8g}")
  rg=x["rgb_delta_dynamic"];dep=x["depth_delta_dynamic_m"]
  lines.append(f"  rgb_improved={(rg<0).sum()}/{rg.notna().sum()}, depth_improved={(dep<0).sum()}/{dep.notna().sum()}, both_improved={((rg<0)&(dep<0)).sum()}")
 reg=root/"v43a1_per_region.csv"
 if reg.exists():
  t=pd.read_csv(reg)
  lines.append(f"Eligible method-tile rows: {len(t)}")
  for name,x in t.groupby("hypothesis"):
   if name=="g0":continue
   lines.append(f"{name} tile outcomes: {x.outcome_vs_g0.value_counts().to_dict()}")
   lines.append(f"{name} unique RGB winner tiles: {(x.rgb_winner==name).sum()}; unique depth winner tiles: {(x.depth_winner==name).sum()}")
 summary="\n".join(lines)+"\n"
 (root/"v43a1_summary.txt").write_text(summary,encoding="utf-8")
 print(summary)
if __name__=="__main__":main()
