"""V5.1b lightweight read-only dynamic reconstruction metrics.

Runs on RGB/depth/opacity already produced by the existing final
eval_rendering() loop: NO additional rendering, flow, YOLO, or Mapping.
The *dataset evaluation mask* uses Flow4DGS conventions:
motion_mask == True denotes STATIC pixels; dynamic == ~motion_mask.
This is an estimated segmentation, NOT moving-object ground truth.

All candidate/render values use the same observed-target masks.
No confidence-driven pixel selection: missing depth predictions incur
a fixed penalty, separately reporting prediction coverage.

Frame-level hashes detect inconsistent segmentation/GT-depth-valid
regions across Legacy / Strict evaluation runs.
"""
import csv
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


def _as_2d(x, h, w, name, device):
    out=torch.as_tensor(x,device=device).squeeze()
    if out.shape!=(h,w):
        raise ValueError(name+" must have shape [H,W]")
    return out


def _digest(mask):
    # CPU hash for exact frame-level pair comparability; no RGB-D image dump.
    return hashlib.sha256(
        mask.detach().to("cpu",dtype=torch.uint8).contiguous().numpy().tobytes()
    ).hexdigest()


def _inner_mask(dynamic_mask, width):
    """Erosion with outside-image treated as NON-dynamic."""
    if width<1:
        raise ValueError("boundary_width_px must be >= 1")
    if not dynamic_mask.any():
        return torch.zeros_like(dynamic_mask)
    m=(~dynamic_mask).float()[None,None]
    padded=F.pad(m,(width,width,width,width),value=1.0)
    missing_neighbor=F.max_pool2d(padded,kernel_size=2*width+1,stride=1)[0,0]>0
    return dynamic_mask & ~missing_neighbor


def _mean(errors,mask):
    n=int(mask.sum().item())
    if n==0:
        return float("nan"),0
    return float(errors[mask].double().mean().item()),n


@torch.no_grad()
def measure_dynamic_reconstruction(
        *, frame_idx,frame_uid,phase,mode,is_keyframe,
        render_rgb,render_depth,render_opacity,gt_rgb,gt_depth,static_mask,
        opacity_threshold=0.1,uncovered_penalty_m=0.5,
        boundary_width_px=5):
    """Compute region metrics in one pass, with no train-time gradients.

    RGB MAE and PSNR use observed RGB pixels irrespective of predicted
    opacity; depth penalized MAE is evaluated on all valid observed
    depth pixels in each region (fixed denominator). Raw depth MAE
    on covered pixels is secondary and MUST be read with coverage.
    """
    if render_rgb.ndim!=3 or render_rgb.shape[0]!=3:
        raise ValueError("render_rgb must be [3,H,W]")
    if gt_rgb.shape!=render_rgb.shape:
        raise ValueError("RGB GT/renderer shapes differ")
    if not 0<=opacity_threshold<=1 or uncovered_penalty_m<=0:
        raise ValueError("invalid fixed opacity/coverage penalty")
    h,w=render_rgb.shape[-2:]
    device=render_rgb.device
    raw_depth=_as_2d(render_depth,h,w,"render_depth",device).float()
    alpha=_as_2d(render_opacity,h,w,"render_opacity",device).float()
    target_depth=_as_2d(gt_depth,h,w,"gt_depth",device).float()
    stat=_as_2d(static_mask,h,w,"static_mask",device).bool()
    dyn=~stat
    interior=_inner_mask(dyn,boundary_width_px)
    boundary=dyn & ~interior

    rgb_gt=gt_rgb.to(device=device,dtype=torch.float32)
    rgb_pre=render_rgb.detach().float()
    good_rgb=torch.isfinite(rgb_gt).all(0)
    good_dep=torch.isfinite(target_depth)&(target_depth>.01)
    covered=(torch.isfinite(raw_depth)&(raw_depth>0)&
             torch.isfinite(alpha)&(alpha>=opacity_threshold))
    # NEVER let predicted coverage change denominators.
    rgb_diff=(torch.nan_to_num(rgb_pre,nan=0.,posinf=0.,neginf=0.).clamp(0,1)-
              torch.nan_to_num(rgb_gt,nan=0.,posinf=0.,neginf=0.)).abs()
    rgb_abs=rgb_diff.mean(0)
    rgb_sq=(rgb_diff*rgb_diff).mean(0)
    raw_depth_error=(
        torch.nan_to_num(raw_depth,nan=0.,posinf=0.,neginf=0.)-
        torch.nan_to_num(target_depth,nan=0.,posinf=0.,neginf=0.)).abs()
    penalized=torch.where(covered,raw_depth_error,
                           torch.full_like(raw_depth_error,float(uncovered_penalty_m)))
    output=dict(frame_idx=int(frame_idx),frame_uid=int(frame_uid),
        phase=str(phase),mode=str(mode),
        is_keyframe=int(bool(is_keyframe)),
        eval_mask_source="dataset_static_mask_inverted",
        boundary_width_px=int(boundary_width_px),
        opacity_threshold=float(opacity_threshold),
        uncovered_penalty_m=float(uncovered_penalty_m),
        dynamic_mask_hash=_digest(dyn),
        depth_valid_mask_hash=_digest(good_dep),
        n_dynamic_mask_pixels=int(dyn.sum().item()),
        dynamic_fraction=float(dyn.float().mean().item()))
    regions=(("all",torch.ones_like(dyn)),("static",stat),
             ("dynamic",dyn),("interior",interior),("boundary",boundary))
    for label,region in regions:
        rgb_mask=region & good_rgb
        depth_mask=region & good_dep
        mae,n_rgb=_mean(rgb_abs,rgb_mask)
        mse,_=_mean(rgb_sq,rgb_mask)
        # 120 dB cap at MSE=1e-12 for deterministic finite CSVs.
        psnr_db=float(-10.*torch.log10(
            torch.tensor(max(mse,1e-12),dtype=torch.float64)).item()) if n_rgb else float("nan")
        depth_mae,n_depth=_mean(penalized,depth_mask)
        depth_valid_pred_mae,n_depth_pred=_mean(raw_depth_error,depth_mask&covered)
        coverage,_=_mean(covered.float(),depth_mask)
        output.update({
            label+"_rgb_mae":mae,label+"_rgb_mse":mse,
            label+"_psnr_db":psnr_db,label+"_rgb_n":n_rgb,
            label+"_depth_penalized_mae_m":depth_mae,
            label+"_depth_pred_valid_mae_m":depth_valid_pred_mae,
            label+"_depth_coverage":coverage,
            label+"_depth_n":n_depth,
            label+"_depth_pred_n":n_depth_pred,
        })
    return output


def write_frame_metrics(folder,phase,rows):
    if not rows:
        raise ValueError("No evaluation rows to save")
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/("v51b_"+str(phase)+"_per_frame.csv")
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
