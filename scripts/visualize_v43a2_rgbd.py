"""Plot paired candidate RGB/depth residual maps from V4.3a saved NPZs.

Use --input <run>/v43a_observation and set save_maps=true for requested
keyframes. The plots are diagnostic; RGB and depth share the same observation.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_file(path,out):
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    needed=("g0","g3_support","g4")
    if any(h+"__rgb_abs_mean" not in data for h in needed):
        raise ValueError("Missing maps in "+str(path))
    base=data["g0__rgb_abs_mean"]
    mask=data["g0__dynamic_region"].astype(bool)
    depth_valid=data["g0__gt_depth_valid"].astype(bool)
    for h in needed:
        if not np.array_equal(data[h+"__dynamic_region"].astype(bool),mask):
            raise ValueError("Different dynamic GT masks")
    fig,axes=plt.subplots(3,3,figsize=(13,11))
    common=mask&depth_valid
    for col,h in enumerate(needed):
        rgb=data[h+"__rendered_rgb"]
        axes[0,col].imshow(np.moveaxis(rgb,0,-1).clip(0,1))
        axes[0,col].set_title(f"{h}: rendered RGB")
        delta_rgb=data[h+"__rgb_abs_mean"]-base
        if h=="g0":
            shown=np.where(mask,data[h+"__rgb_abs_mean"],np.nan)
            im=axes[1,col].imshow(shown,vmin=0,vmax=.2,cmap="magma")
            axes[1,col].set_title("G0 RGB MAE on estimated motion mask")
        else:
            shown=np.where(mask,delta_rgb,np.nan)
            im=axes[1,col].imshow(shown,vmin=-.025,vmax=.025,cmap="coolwarm")
            axes[1,col].set_title(f"{h} minus G0 RGB MAE")
        plt.colorbar(im,ax=axes[1,col],fraction=.046)
        delta_depth=data[h+"__depth_penalized_abs"]-data["g0__depth_penalized_abs"]
        if h=="g0":
            shown=np.where(common,data[h+"__depth_penalized_abs"],np.nan)
            im=axes[2,col].imshow(shown,vmin=0,vmax=.5,cmap="magma")
            axes[2,col].set_title("G0 depth abs error (m)")
        else:
            shown=np.where(common,delta_depth,np.nan)
            im=axes[2,col].imshow(shown,vmin=-.05,vmax=.05,cmap="coolwarm")
            axes[2,col].set_title(f"{h} minus G0 depth error (m)")
        plt.colorbar(im,ax=axes[2,col],fraction=.046)
        for ax in axes[:,col]:ax.axis("off")
    fig.suptitle(f"V4.3a.2 KF {path.stem[-5:]} / same-state pre-Mapping",fontsize=13)
    fig.tight_layout()
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/(path.stem+"_rgbd.png"),dpi=140)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",default=None)
    args=p.parse_args()
    base=Path(args.input)
    out=Path(args.output) if args.output else base/"diagnostic_images"
    files=sorted(base.glob("v43a_[0-9][0-9][0-9][0-9][0-9].npz"))
    if not files:
        raise SystemExit("No detailed v43a_XXXXX.npz; set save_maps=true")
    for file in files:
        run_file(file,out)
        print("Saved",out/(file.stem+"_rgbd.png"))

if __name__=="__main__":main()
