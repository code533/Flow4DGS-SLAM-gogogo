"""V4.0b offline reliability stratification, no ground-truth labels required."""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.v40b_calibration import summarize_calibration

def median(x):
    x = np.asarray(x, float)
    return float(np.median(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",default=None)
    ap.add_argument("--frames",default="60,140,220,230,310,390")
    ap.add_argument("--high-motion-px",type=float,default=5.0)
    ap.add_argument("--low-fb-px",type=float,default=.5)
    args=ap.parse_args()
    folder=Path(args.input); out=Path(args.output) if args.output else folder
    out.mkdir(parents=True,exist_ok=True)
    files=sorted(folder.glob("v40b_[0-9][0-9][0-9][0-9][0-9].npz"))
    if not files: raise SystemExit("No v40b_NNNNN.npz found; enable save_maps.")
    selected = None if args.frames=="all" else set(int(v) for v in args.frames.split(",") if v.strip())
    rows=[]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in files:
        with np.load(f) as npz: m={k:npz[k] for k in npz.files}
        r=summarize_calibration(m,args.high_motion_px,args.low_fb_px)
        rows.append(r)
        if selected is not None and r["frame_id"] not in selected: continue
        fig,ax=plt.subplots(2,4,figsize=(15,8))
        images=[
          ("RGB", m["rgb_observed"],None),
          ("Motion residual (px)",m["flow_full_residual_px"],"magma"),
          ("Native FB cycle (px)",m["fb_error_px"],"magma"),
          ("Legacy FB cycle (px)",m.get("fb_error_legacy_grid_px",m["fb_error_px"]),"magma"),
          ("Prior dynamic",~m["prior_static"].astype(bool),"gray"),
          ("Depth boundary proximity",m["boundary_near"],"gray"),
          ("Possible occlusion (proxy)",m["possible_occlusion"],"gray"),
          ("Interior matched region",m["interior_correspondence"],"gray"),
        ]
        for a,(name,img,cmap) in zip(ax.flat,images):
            if np.asarray(img).dtype.kind=="f":
                finite=np.asarray(img)[np.isfinite(img)]
                vmax=max(1e-6,np.percentile(finite,98)) if finite.size else 1.0
                a.imshow(img,cmap=cmap,vmin=0,vmax=vmax)
            else: a.imshow(img,cmap=cmap)
            a.set_title(name,fontsize=10);a.axis("off")
        fig.suptitle("V4.0b frame %05d (proxy cues, not GT)" % r["frame_id"])
        fig.tight_layout()
        fig.savefig(out/("v40b_visual_%05d.png"%r["frame_id"]),dpi=125)
        plt.close(fig)
    csvpath=out/"v40b_per_frame_offline.csv"
    with csvpath.open("w",newline="",encoding="utf-8") as stream:
        w=csv.DictWriter(stream,rows[0].keys());w.writeheader();w.writerows(rows)
    keys=["map_covered_pre_fraction","boundary_near_fraction",
          "possible_occlusion_fraction","interior_fraction","fb_median_px"]
    for group in ("prior_dynamic","flow_dynamic","static_candidate"):
        keys += [group+"_n_common_interior",group+"_n_high_motion_interior",
                 group+"_flow_med_interior_px",group+"_fb_med_interior_px",
                 group+"_low_fb_given_high_motion"]
    summary=["V4.0b: %d frames (no dynamic/occlusion ground truth)"%len(rows)]
    for key in keys: summary.append("%s: median_over_frames=%.6g"%(key,median([r[key] for r in rows])))
    (out/"v40b_summary.txt").write_text("\n".join(summary)+"\n",encoding="utf-8")
    print("\n".join(summary));print("Output:",csvpath)
if __name__=="__main__":
    main()
