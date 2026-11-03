"""V5.0: observation/visibility failure attribution for 3D Gaussian motion.

Read-only analysis of Flow4DGS-SLAM's *unchanged* G0 motion pipeline.
We explicitly distinguish missing depth / projection failure / flow failure
from *verified occlusion*. A missing measurement is NOT proof of occlusion.

Covers all pre-existing dynamic Gaussians (flow reason histogram) and the
subset that survives projection, depth filling and the 1m motion pre-gate.
The analysis does not change camera poses, deformation, masks or Mapping.
"""
import csv
from pathlib import Path
import numpy as np
import torch

REASONS = (
    "valid_nonzero_q",
    "nonfinite_projection",
    "projection_outside_image",
    "previous_depth_invalid",
    "forward_flow_invalid",
    "forward_correspondence_outside_image",
    "backward_flow_invalid_at_warp",
    "current_depth_invalid_at_warp",
    "q_near_zero_despite_valid_correspondence",
)
GROUPS = (
    "source_projection_invalid",
    "previous_depth_missing",
    "flow_correspondence_invalid",
    "target_observation_missing",
    "low_flow_confidence",
    "observed_reliable_proxy",
)
GROUP_CODES = {name:i for i,name in enumerate(GROUPS)}


def eligible_event(event_index, every):
    if not isinstance(every,int) or every < 1 or event_index < 1:
        raise ValueError("Event index and sampling interval must be positive")
    return (event_index-1) % every == 0


@torch.no_grad()
def classify_selected(reason, q, target_depth_observed, q_threshold=.3):
    """Disjoint failure-proxy groups for *surviving* Gaussian motions.

    Ordering: projection > previous missing depth > invalid flow/FB >
    target depth missing (from flow diagnostics OR original raw depth) >
    numeric flow confidence > observed proxy.

    Group 'observed_reliable_proxy' does NOT imply 3D ground-truth motion.
    """
    if reason.ndim!=1 or reason.shape!=q.shape or reason.shape!=target_depth_observed.shape:
        raise ValueError("Selected reason, reliability and observed depth must align")
    if not (0 <= q_threshold <= 1):
        raise ValueError("Confidence cutoff must be in [0,1]")
    if reason.numel() and ((reason < 0).any() or (reason >= len(REASONS)).any()):
        raise ValueError("Unknown reliability reason code")
    labels=torch.full((reason.numel(),),GROUP_CODES["observed_reliable_proxy"],
                      dtype=torch.int64,device=reason.device)
    labels[(reason==8)|(~torch.isfinite(q))|(q < q_threshold)] = GROUP_CODES["low_flow_confidence"]
    labels[(reason==7)|(~target_depth_observed.bool())] = GROUP_CODES["target_observation_missing"]
    labels[(reason==4)|(reason==5)|(reason==6)] = GROUP_CODES["flow_correspondence_invalid"]
    labels[reason==3] = GROUP_CODES["previous_depth_missing"]
    labels[(reason==1)|(reason==2)] = GROUP_CODES["source_projection_invalid"]
    return labels


def _frac(mask):
    return float(mask.float().mean().item()) if mask.numel() else float("nan")


def _mean(x):
    return float(x.float().mean().item()) if x.numel() else float("nan")


def _p95(x):
    return float(torch.quantile(x.float(),.95).item()) if x.numel() else float("nan")


@torch.no_grad()
def summarize_v50(*,frame_id,prior_keyframe_id,event_index,
                  all_reasons,selected_indices,selected_target_observed,
                  raw_disp,g0_disp,selected_q,stage_counts,q_threshold=.3,
                  update_gate_m=.3):
    """Compute paired stage / group summaries without changing tensors."""
    if not 0 < update_gate_m:
        raise ValueError("update gate must be positive")
    if all_reasons.ndim!=1 or selected_indices.ndim!=1:
        raise ValueError("Invalid Gaussian indexing")
    n=selected_indices.numel()
    if (raw_disp.shape!=(n,3) or g0_disp.shape!=(n,3)
            or selected_q.shape!=(n,) or selected_target_observed.shape!=(n,)):
        raise ValueError("Selected motion arrays differ in length")
    if n and (selected_indices.min()<0 or selected_indices.max()>=all_reasons.numel()):
        raise IndexError("Selected index outside dynamic Gaussian list")
    if n and selected_indices.unique().numel()!=n:
        raise ValueError("Repeated selected Gaussian index")
    if n and not (torch.isfinite(raw_disp).all() and torch.isfinite(g0_disp).all()):
        raise FloatingPointError("Nonfinite raw or smoothed Gaussian motion")
    reason=all_reasons[selected_indices].long()
    groups=classify_selected(reason,selected_q,selected_target_observed,q_threshold)
    raw_norm=torch.linalg.vector_norm(raw_disp,dim=-1)
    smooth_norm=torch.linalg.vector_norm(g0_disp,dim=-1)
    smooth_delta=torch.linalg.vector_norm(g0_disp-raw_disp,dim=-1)
    accepted=smooth_norm<update_gate_m

    row={
        "keyframe_id":int(frame_id),"prior_keyframe_id":int(prior_keyframe_id),
        "eligible_event_index":int(event_index),
        "n_all_dynamic_gaussians":int(all_reasons.numel()),
        "n_selected_movements":int(n),
        "fraction_selected_of_all_dynamic":float(n/all_reasons.numel()) if all_reasons.numel() else float("nan"),
        "n_actual_g0_update_gate_pass":int(accepted.sum().item()),
        "g0_update_gate_pass_fraction":_frac(accepted),
        "selected_original_target_depth_valid_fraction":_frac(selected_target_observed.bool()),
        "selected_raw_motion_mean_m":_mean(raw_norm),
        "selected_g0_motion_mean_m":_mean(smooth_norm),
        "selected_smoothing_correction_mean_m":_mean(smooth_delta),
        "selected_smoothing_correction_p95_m":_p95(smooth_delta),
        "selected_confidence_mean":_mean(selected_q),
    }
    for key,val in sorted(stage_counts.items()):
        if not isinstance(val,int) or val<0:
            raise ValueError("Invalid stage count "+str(key))
        row["stage_"+key]=val
    for code,name in enumerate(REASONS):
        count=int((all_reasons==code).sum().item())
        row["all_reason_"+name+"_n"]=count
        row["all_reason_"+name+"_fraction"]=(float(count/all_reasons.numel())
                                             if all_reasons.numel() else float("nan"))
    if sum(int((all_reasons==code).sum()) for code in range(len(REASONS)))!=all_reasons.numel():
        raise ValueError("Unclassified dynamic Gaussian reason")
    groups_rows=[]
    for label,group_name in enumerate(GROUPS):
        chosen=(groups==label)
        m=int(chosen.sum().item())
        groups_rows.append({
            "keyframe_id":int(frame_id),
            "prior_keyframe_id":int(prior_keyframe_id),
            "group_name":group_name,
            "n_selected_group":m,
            "group_fraction_of_selected":float(m/n) if n else float("nan"),
            "q_mean":_mean(selected_q[chosen]),
            "raw_motion_mean_m":_mean(raw_norm[chosen]),
            "raw_motion_p95_m":_p95(raw_norm[chosen]),
            "g0_motion_mean_m":_mean(smooth_norm[chosen]),
            "g0_motion_p95_m":_p95(smooth_norm[chosen]),
            "smoothing_correction_mean_m":_mean(smooth_delta[chosen]),
            "smoothing_correction_p95_m":_p95(smooth_delta[chosen]),
            "g0_update_gate_pass_fraction":_frac(accepted[chosen]),
            "target_original_depth_observed_fraction":_frac(selected_target_observed[chosen].bool()),
        })
    if sum(int(r["n_selected_group"]) for r in groups_rows)!=n:
        raise RuntimeError("Group partition does not sum to selected count")
    detail={
        "selected_gaussian_index":selected_indices.detach().cpu().numpy(),
        "reason_code":reason.detach().cpu().numpy(),
        "group_code":groups.detach().cpu().numpy(),
        "target_original_depth_observed":selected_target_observed.detach().cpu().numpy(),
        "flow_q":selected_q.detach().cpu().numpy(),
        "raw_disp_m":raw_disp.detach().cpu().numpy(),
        "g0_disp_m":g0_disp.detach().cpu().numpy(),
        "g0_update_gate_pass":accepted.detach().cpu().numpy(),
    }
    return row,groups_rows,detail


def append_v50(folder,frame_row,group_rows,detail=None):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    for fname,rows in (("v50_per_keyframe.csv",[frame_row]),
                       ("v50_per_group.csv",group_rows)):
        file=folder/fname
        new=not file.exists() or file.stat().st_size==0
        with file.open("a",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            if new:writer.writeheader()
            writer.writerows(rows)
    if detail is not None:
        np.savez_compressed(folder/("v50_%05d.npz"%int(frame_row["keyframe_id"])),
                            **detail)
