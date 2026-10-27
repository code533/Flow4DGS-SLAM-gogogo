"""V4.1b offline Unknown attribution + independent GT evaluation."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.v41a_shadow import infer_shadow,ShadowConfig
from utils.v41b_evaluation import (
 attribute_unknown,summarize_reasons,eval_frame,combine_confusions,
 selective_curve,STATE_KEYS,REASONS,validate_labels)

def write_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,help="V4.0b NPZ folder")
    p.add_argument("--gt-dir",help="Independent label PNG directory; no label = attribution only")
    p.add_argument("--output",required=True)
    p.add_argument("--frames",default="60,140,220,230,310,390")
    args=p.parse_args()
    source=Path(args.input);dest=Path(args.output)
    if not source.is_dir():p.error("Input directory not found")
    if source.resolve()==dest.resolve():p.error("Output must differ from input")
    dest.mkdir(parents=True,exist_ok=True)
    files=sorted(source.glob("v40b_[0-9][0-9][0-9][0-9][0-9].npz"))
    if not files:p.error("No V4.0b pixel NPZ found (CSV/PNG are insufficient)")
    selected=set(int(s) for s in args.frames.split(",") if s.strip())
    cfg=ShadowConfig()
    reason_rows=[];metric_rows=[];labeled=[]
    for fn in files:
        with np.load(fn,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        shadow=infer_shadow(raw,cfg)
        reason,cues=attribute_unknown(raw,shadow,cfg)
        fid=int(raw["frame_id"])
        row={"frame_id":fid,**summarize_reasons(reason,cues)}
        reason_rows.append(row)
        if fid in selected:
            from PIL import Image
            # no predicted-state overlay on the raw annotation reference
            raw_rgb=Image.fromarray(raw["rgb_observed"].astype(np.uint8))
            raw_rgb.save(dest/("annotation_reference_%05d.png"%fid))
            Image.fromarray((reason.astype(np.uint8)*42),mode="L").save(
                dest/("unknown_reason_codes_%05d.png"%fid))
        gt_fn=Path(args.gt_dir)/("gt_%05d.png"%fid) if args.gt_dir else None
        if gt_fn is not None and gt_fn.is_file():
            gt=np.asarray(Image.open(gt_fn))
            gt=validate_labels(gt,shadow["p_dynamic"].shape)
            all_metrics=eval_frame(raw,shadow,gt)
            for name,metrics in all_metrics.items():
                metric_rows.append({"frame_id":fid,"method":name,**metrics})
            labeled.append((raw,shadow,gt))
        elif args.gt_dir:
            print("[V4.1b] GT not found, skipping evaluation",gt_fn)
    write_csv(dest/"v41b_unknown_reasons.csv",reason_rows)
    (dest/"v41b_reason_legend.json").write_text(
        json.dumps(REASONS,indent=2)+"\n",encoding="utf-8")
    print("[V4.1b] attribution done",len(reason_rows),"frames")
    if labeled:
        write_csv(dest/"v41b_gt_per_frame.csv",metric_rows)
        micro=[]
        for name in STATE_KEYS:
            from utils.v41b_evaluation import confusion
            groups=[confusion(gt,(
                (~raw["final_static"]).astype(np.uint8) if name=="legacy_binary"
                else shadow[name])) for raw,shadow,gt in labeled]
            micro.append({"method":name,**combine_confusions(groups)})
        write_csv(dest/"v41b_gt_micro.csv",micro)
        curve=selective_curve(labeled)
        write_csv(dest/"v41b_risk_coverage.csv",curve)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        for name in STATE_KEYS:
            group=[r for r in curve if r["method"]==name]
            plt.plot([r["coverage"] for r in group],
                     [r["selective_risk"] for r in group],
                     marker=".",label=name)
        plt.xlabel("Accepted GT-labeled pixel coverage")
        plt.ylabel("Error among accepted GT-labeled pixels")
        plt.title("Selective risk vs coverage (not calibrated)")
        plt.legend(fontsize=8);plt.grid(alpha=.3);plt.tight_layout()
        plt.savefig(dest/"v41b_risk_coverage.png",dpi=160);plt.close()
        print("[V4.1b] independently labeled frames:",len(labeled))
    else:
        print("[V4.1b] No independent GT labels: no F1/IoU/risk claims.")
if __name__=="__main__":
    main()
