"""V5.1a: STRICT-FOV observation validity ablation (G0 only).

Never changes the global get_uv_coordinates() implementation. A Gaussian
may sample flow/depth for motion ONLY when its source clip projection
has positive clip-w and source/warped NDC coordinates are within [-1,1].
Warped-NDC uses the legacy Flow4DGS motion approximation; it is NOT
reprojected with the target camera's own pose.

The strict candidate is rebuilt with the ORIGINAL G0 KNN smoother over
ONLY the surviving Gaussian motion samples. Shadow observes that candidate;
strict applies it, while rejected Gaussians inherit previous offset via
GaussianModel.update_delta_xyz(..., selected_idx=...).

A validity check is not ground-truth occlusion or motion accuracy.
"""
import csv
from pathlib import Path

import numpy as np
import torch

MODES=("off","legacy","shadow","strict")


def validate_mode(mode, *, online_mode, flow_ds, use_knn, flow_offset,
                  dynamic_model, v43_enabled=False, v50_enabled=False):
    if mode not in MODES:
        raise ValueError("v51a_projection.mode must be off/legacy/shadow/strict")
    if mode=="off":
        return
    if online_mode!="g0" or flow_ds!=1 or not use_knn or not flow_offset or dynamic_model!="offset":
        raise ValueError("V5.1a requires v42a_propagation.mode=g0, full-res flow, KNN and offset")
    if v43_enabled:
        raise ValueError("Disable old V4.3 RGB-D candidate scoring for V5.1a")
    if v50_enabled:
        raise ValueError("Disable V5.0 logging for V5.1a controlled ablation")


@torch.no_grad()
def strict_ndc_valid(uvz, *, clip_w=None, min_positive_w=1e-7):
    """Validity aligned with one NDC row per dynamic Gaussian.

    clip_w is available for the source full projection and prevents
    behind-camera source points from being admitted. Target 'NDC'
    represents legacy source-projection-plus-optical-flow, not a true
    new camera clip projection.
    """
    if uvz.ndim!=2 or uvz.shape[1]<2:
        raise ValueError("Expected NDC projected Gaussian coordinates [N,2+]")
    if min_positive_w<=0:
        raise ValueError("min_positive_w must be positive")
    xy=uvz[:,:2]
    valid=torch.isfinite(xy).all(-1)
    valid &= ((xy >= -1.) & (xy <= 1.)).all(-1)
    if clip_w is not None:
        if clip_w.ndim!=1 or clip_w.shape[0]!=uvz.shape[0]:
            raise ValueError("clip_w must have one scalar per Gaussian")
        valid &= torch.isfinite(clip_w) & (clip_w > min_positive_w)
    return valid


@torch.no_grad()
def strict_indices(legacy_indices, source_valid, target_valid):
    """Return ordered STRICT subset of the same pre-KNN G0 candidate set."""
    if legacy_indices.ndim!=1 or legacy_indices.dtype not in (torch.int64,torch.int32):
        raise ValueError("legacy_indices must be integer vector")
    if source_valid.ndim!=1 or target_valid.shape!=source_valid.shape:
        raise ValueError("source/target masks misaligned")
    if legacy_indices.numel() and (legacy_indices.min()<0 or
                                   legacy_indices.max()>=source_valid.numel()):
        raise IndexError("Candidate id outside dynamic Gaussian domain")
    if legacy_indices.unique().numel()!=legacy_indices.numel():
        raise ValueError("Duplicate legacy candidate id")
    keep=source_valid[legacy_indices.long()] & target_valid[legacy_indices.long()]
    return legacy_indices[keep],keep


def _mean(t):
    return float(t.float().mean().item()) if t.numel() else float("nan")


def _p95(t):
    return float(torch.quantile(t.float(),.95).item()) if t.numel() else float("nan")


@torch.no_grad()
def compare_candidates(*,frame_id,prior_keyframe_id,mode,
                       source_valid,target_valid,legacy_indices,legacy_raw,
                       legacy_g0,strict_g0=None,gate_m=.3):
    """One paired, before-Map-update frame record.

    The *exact same selected Gaussian IDs* are compared for legacy vs
    strict where both methods update. Removal and smoother effects are
    recorded separately. strict_g0 must be supplied for shadow/strict.
    """
    if mode not in ("legacy","shadow","strict"):
        raise ValueError("Invalid comparison mode")
    if source_valid.shape!=target_valid.shape or source_valid.ndim!=1:
        raise ValueError("Invalid full-population FOV masks")
    n=int(legacy_indices.numel())
    if legacy_raw.shape!=(n,3) or legacy_g0.shape!=(n,3):
        raise ValueError("Pre-KNN and post-KNN motion array shape mismatch")
    strict_idx,keep=strict_indices(legacy_indices,source_valid,target_valid)
    if strict_g0 is not None and strict_g0.shape!=(len(strict_idx),3):
        raise ValueError("Strict smoother must be computed on surviving candidates")
    if mode!="legacy" and strict_g0 is None:
        raise ValueError("Shadow/strict mode requires strict KNN candidate")
    if not torch.isfinite(legacy_g0).all() or not torch.isfinite(legacy_raw).all():
        raise FloatingPointError("Nonfinite G0 displacement")
    if strict_g0 is not None and not torch.isfinite(strict_g0).all():
        raise FloatingPointError("Nonfinite strict displacement")
    source_reject=~source_valid[legacy_indices.long()]
    target_reject=~target_valid[legacy_indices.long()]
    legacy_norm=torch.linalg.vector_norm(legacy_g0,dim=-1)
    gate_legacy=legacy_norm<gate_m
    removed=~keep
    shared_legacy=legacy_g0[keep]
    baseline_pass=gate_legacy[keep]
    diff=torch.empty((0,),dtype=legacy_g0.dtype,device=legacy_g0.device)
    strict_pass=baseline_pass
    gate_change=0
    if strict_g0 is not None:
        strict_pass=torch.linalg.vector_norm(strict_g0,dim=-1)<gate_m
        diff=torch.linalg.vector_norm(strict_g0-shared_legacy,dim=-1)
        gate_change=int(torch.count_nonzero(strict_pass!=baseline_pass).item())
    row={
        "keyframe_id":int(frame_id),"prior_keyframe_id":int(prior_keyframe_id),
        "mode":mode,"n_all_dynamic":int(source_valid.numel()),
        "n_source_true_fov":int(source_valid.sum().item()),
        "n_source_and_target_true_fov":int((source_valid&target_valid).sum().item()),
        "n_legacy_candidates":n,
        "n_strict_candidates":int(len(strict_idx)),
        "n_removed_candidates":int(removed.sum().item()),
        "n_removed_source_fov":int(source_reject.sum().item()),
        "n_removed_target_fov":int(target_reject.sum().item()),
        "n_removed_both_fov":int((source_reject&target_reject).sum().item()),
        "n_removed_that_pass_legacy_0p3m_gate":int((removed&gate_legacy).sum().item()),
        "n_legacy_gate_pass":int(gate_legacy.sum().item()),
        "n_common_gate_pass_legacy":int(baseline_pass.sum().item()),
        "n_common_gate_pass_strict":int(strict_pass.sum().item())
                                     if strict_g0 is not None else -1,
        "n_common_gate_changed":gate_change if strict_g0 is not None else -1,
        "shared_strict_vs_legacy_g0_mean_m":_mean(diff),
        "shared_strict_vs_legacy_g0_p95_m":_p95(diff),
        "removed_legacy_motion_mean_m":_mean(legacy_norm[removed]),
        "fraction_candidates_rejected":float(removed.float().mean().item())
                                             if n else float("nan"),
        "fraction_legacy_accepted_removed":float((removed&gate_legacy).sum().item()/
                                                   gate_legacy.sum().item())
                                      if int(gate_legacy.sum().item()) else float("nan"),
    }
    return row,keep,strict_idx


def append_comparison(folder,row,details=None):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    fname=folder/"v51a_per_keyframe.csv"
    new=not fname.exists() or fname.stat().st_size==0
    with fname.open("a",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if new:writer.writeheader()
        writer.writerow(row)
    if details:
        np.savez_compressed(folder/("v51a_%05d.npz"%int(row["keyframe_id"])),
                            **{k:(v.detach().cpu().numpy() if torch.is_tensor(v)
                                 else np.asarray(v)) for k,v in details.items()})
