"""V5.2a controlled Visibility-Conditioned Historical Motion, SHADOW ONLY.

Fourth candidate 'strict_history' takes the original STRICT-G0 motion and
adds a bounded historical velocity step ONLY to Gaussian IDs which
(i) were in the original Legacy motion candidate set,
(ii) were excluded by Strict's source/warped target FOV,
(iii) existed before the oldest of 3 previous KFs,
(iv) have two nonzero, direction-consistent optimized offset increments
    on a common GaussianModel topology, with plausible speed.

Past optimized offsets are NOT ground-truth 3D velocities. Unique IDs are
not persistent physical tracking: the creation-KF check only reduces,
not eliminates, uncertainty due to Gaussian splits/densification.
No online update, no learned reliability weights or future-data tuning.
"""
import csv
from pathlib import Path

import torch

from utils.v51c_controlled_attribution import compose_controlled
from utils.v51b_dynamic_metrics import measure_dynamic_reconstruction
from utils.v51d_temporal_persistence import evaluate_future as eval_three_future

MODES=("hold","legacy","strict","strict_history")


def validate_history_cfg(*,min_step_m,max_speed_m_per_frame,
                         min_direction_cos,max_speed_ratio,
                         max_predicted_increment_m,history_frame_gap_cap):
    if (not 0<min_step_m<max_predicted_increment_m
            or max_speed_m_per_frame<=0
            or not -1<=min_direction_cos<=1
            or max_speed_ratio<1
            or not isinstance(history_frame_gap_cap,int)
            or history_frame_gap_cap<1):
        raise ValueError("V5.2a unsafe or invalid history threshold settings")


def _masks(model,n,device):
    dynamic=model.dygs.to(device=device,dtype=torch.bool)
    unique=torch.as_tensor(model.unique_kfIDs,device=device)
    if dynamic.ndim!=1 or unique.shape!=dynamic.shape or int(dynamic.sum())!=n:
        raise ValueError("Gaussian creation IDs must match dynamic Gaussian ordering")
    return unique[dynamic]


@torch.no_grad()
def predict_conditioned_history(*,model,source_uid,prior_uid,
        legacy_indices,legacy_g0,strict_indices,strict_g0,
        min_step_m=.005,max_speed_m_per_frame=.04,min_direction_cos=.5,
        max_speed_ratio=2.,max_predicted_increment_m=.10,
        history_frame_gap_cap=20):
    """Produce fourth offset tensor and auditable eligibility stages."""
    validate_history_cfg(min_step_m=min_step_m,
        max_speed_m_per_frame=max_speed_m_per_frame,
        min_direction_cos=min_direction_cos,max_speed_ratio=max_speed_ratio,
        max_predicted_increment_m=max_predicted_increment_m,
        history_frame_gap_cap=history_frame_gap_cap)
    if int(source_uid)<=int(prior_uid) or model.init_deform!="offset":
        raise ValueError("V5.2a requires source after prior and offset model")
    if prior_uid not in model.kf_list:
        raise ValueError("Prior keyframe not registered")
    pos=model.kf_list.index(prior_uid)
    old=model.get_delta_xyz(pos)
    n=int(model.dygs.sum().item())
    if old.shape!=(n,3):
        raise ValueError("Previous dynamic offsets are not aligned")
    offsets,_=compose_controlled(old,legacy_indices,legacy_g0,
                                 strict_indices,strict_g0)
    base=offsets["strict"]
    selected=torch.isin(legacy_indices.long(),strict_indices.long())
    rejected=legacy_indices.long()[~selected]
    if rejected.numel() and ((rejected.min()<0) or rejected.max()>=n):
        raise ValueError("Excluded Gaussian id outside dynamic domain")
    diag=dict(
      n_legacy_candidates=int(len(legacy_indices)),
      n_strict_candidates=int(len(strict_indices)),
      n_fov_excluded=int(len(rejected)),
      n_history_available=0,n_old_enough=0,n_nonzero_steps=0,
      n_direction_coherent=0,n_speed_plausible=0,
      n_history_predicted=0,n_history_clipped=0,
      predicted_increment_mean_m=float("nan"),
      predicted_increment_p95_m=float("nan"),
      oldest_history_uid=-1,
      prior_history_uid=-1,
      source_gap_frames=int(source_uid)-int(prior_uid))
    if rejected.numel()==0 or pos<2:
        return base.detach().clone(),diag
    k0,k1,k2=[int(model.kf_list[j]) for j in (pos-2,pos-1,pos)]
    dt0=k1-k0
    dt1=k2-k1
    dt_future=int(source_uid)-k2
    diag.update(oldest_history_uid=k0,prior_history_uid=k1)
    if min(dt0,dt1,dt_future)<=0 or max(dt0,dt1,dt_future)>history_frame_gap_cap:
        return base.detach().clone(),diag
    oldest=model.get_delta_xyz(pos-2)
    middle=model.get_delta_xyz(pos-1)
    if oldest.shape!=(n,3) or middle.shape!=(n,3):
        # Model topology can change; do not index across mismatched history.
        return base.detach().clone(),diag
    creation_ids=_masks(model,n,old.device)
    diag["n_history_available"]=int(len(rejected))
    old_enough=creation_ids[rejected] <= k0
    diag["n_old_enough"]=int(old_enough.sum().item())
    candidate_ids=rejected[old_enough]
    if not len(candidate_ids):
        return base.detach().clone(),diag
    recent=(old[candidate_ids]-middle[candidate_ids])/float(dt1)
    earlier=(middle[candidate_ids]-oldest[candidate_ids])/float(dt0)
    step_recent=torch.linalg.vector_norm(old[candidate_ids]-middle[candidate_ids],dim=-1)
    step_older=torch.linalg.vector_norm(middle[candidate_ids]-oldest[candidate_ids],dim=-1)
    speed_recent=torch.linalg.vector_norm(recent,dim=-1)
    speed_older=torch.linalg.vector_norm(earlier,dim=-1)
    finite=torch.isfinite(recent).all(-1)&torch.isfinite(earlier).all(-1)
    active=finite&(step_recent>=min_step_m)&(step_older>=min_step_m)
    diag["n_nonzero_steps"]=int(active.sum().item())
    if not bool(active.any()):
        return base.detach().clone(),diag
    cosine=(recent*earlier).sum(-1)/(speed_recent*speed_older).clamp_min(1e-9)
    ratio=speed_recent/(speed_older+1e-9)
    coherent=active&(cosine>=min_direction_cos)&(
        ratio>=1./max_speed_ratio)&(ratio<=max_speed_ratio)
    diag["n_direction_coherent"]=int(coherent.sum().item())
    plausible=coherent&(speed_recent<=max_speed_m_per_frame)
    diag["n_speed_plausible"]=int(plausible.sum().item())
    if not bool(plausible.any()):
        return base.detach().clone(),diag
    candidate_ids=candidate_ids[plausible]
    predicted=recent[plausible]*float(dt_future)
    raw_mag=torch.linalg.vector_norm(predicted,dim=-1)
    clipped=raw_mag>max_predicted_increment_m
    diag["n_history_clipped"]=int(clipped.sum().item())
    scales=(max_predicted_increment_m/raw_mag.clamp_min(1e-9)).clamp(max=1.)
    predicted=predicted*scales[:,None]
    result=base.detach().clone()
    # FOV-rejected candidates are disjoint from strict accepted candidates.
    result[candidate_ids] += predicted
    mag=torch.linalg.vector_norm(predicted,dim=-1)
    diag["n_history_predicted"]=int(len(candidate_ids))
    diag["predicted_increment_mean_m"]=float(mag.float().mean().item())
    diag["predicted_increment_p95_m"]=float(torch.quantile(mag.float(),.95).item())
    if not torch.equal(base,offsets["strict"]):
        raise RuntimeError("V5.2a modified strict baseline")
    return result,diag


@torch.no_grad()
def evaluate_history_row(*,model,viewpoint,offset,render_fn,pipeline,background,
                          phase,opacity_threshold,uncovered_penalty_m,boundary_width_px):
    pkg=render_fn(viewpoint,model,pipeline,background,
                  dynamic=False,dx=offset,ds=0,dr=0,uid=int(viewpoint.uid))
    return measure_dynamic_reconstruction(
       frame_idx=int(viewpoint.uid),frame_uid=int(viewpoint.uid),
       phase=phase,mode="strict_history",is_keyframe=True,
       render_rgb=pkg["render"],render_depth=pkg["depth"],
       render_opacity=pkg["opacity"],
       gt_rgb=viewpoint.original_image.to(offset.device),
       gt_depth=viewpoint.depth,
       static_mask=viewpoint.motion_mask.to(offset.device),
       opacity_threshold=opacity_threshold,
       uncovered_penalty_m=uncovered_penalty_m,
       boundary_width_px=boundary_width_px)


@torch.no_grad()
def extend_source(*,pending,source_viewpoint,history_offset,diagnostics,
                  render_fn,pipeline,background,opacity_threshold=.1,
                  uncovered_penalty_m=.5,boundary_width_px=5):
    if len(pending["offsets"])!=3 or len(pending["source_rows"])!=3:
        raise ValueError("V5.2a requires three V5.1d baseline source candidates")
    if history_offset.shape!=pending["offsets"]["strict"].shape:
        raise ValueError("V5.2a history/source topology mismatch")
    # Source and future rendering use frozen GaussianModel, so all four
    # candidates share geometry, fixed GT masks and pose.
    frozen=pending["model"]
    data=evaluate_history_row(
        model=frozen,viewpoint=source_viewpoint,offset=history_offset,
        render_fn=render_fn,pipeline=pipeline,background=background,
        phase="pre_mapping_fixed_map",
        opacity_threshold=opacity_threshold,
        uncovered_penalty_m=uncovered_penalty_m,
        boundary_width_px=boundary_width_px)
    for field in ("dynamic_mask_hash","depth_valid_mask_hash",
                  "dynamic_depth_n","dynamic_rgb_n"):
        if data[field]!=pending["source_rows"]["strict"][field]:
            raise ValueError("V5.2a source evaluation targets mismatch")
    # Match all original V5.1c source-row column keys to ensure
    # strict CSV schema equality across modes.
    history_row=dict(pending["source_rows"]["strict"])
    history_row.update(data)
    history_row["n_candidate_gate_pass"]=int(diagnostics["n_history_predicted"])
    history_row["n_actual_moved_from_hold"]=int(
        (history_offset!=pending["offsets"]["hold"]).any(-1).sum().item())
    pending["offsets"]["strict_history"]=history_offset.detach().clone()
    pending["source_rows"]["strict_history"]=history_row
    pending["history_diag"]=dict(diagnostics)
    return [dict(pending["source_rows"][mode],
                 source_keyframe_id=pending["source_uid"],
                 source_selection=pending["source_kind"],
                 **diagnostics) for mode in MODES]


@torch.no_grad()
def evaluate_four_future(*,pending,viewpoint,future_keyframe_order,render_fn,
                         pipeline,background,opacity_threshold=.1,
                         uncovered_penalty_m=.5,boundary_width_px=5,
                         min_dynamic_depth_pixels=256):
    if set(pending["offsets"])!=set(MODES):
        raise ValueError("V5.2a snapshot must contain all four frozen candidates")
    rows,status=eval_three_future(
       pending=pending,viewpoint=viewpoint,
       future_keyframe_order=future_keyframe_order,
       render_fn=render_fn,pipeline=pipeline,background=background,
       opacity_threshold=opacity_threshold,
       uncovered_penalty_m=uncovered_penalty_m,
       boundary_width_px=boundary_width_px,
       min_dynamic_depth_pixels=min_dynamic_depth_pixels)
    if not rows:
        return [],status
    metrics=evaluate_history_row(
        model=pending["model"],viewpoint=viewpoint,
        offset=pending["offsets"]["strict_history"],
        render_fn=render_fn,pipeline=pipeline,background=background,
        phase="future_frozen_map",
        opacity_threshold=opacity_threshold,
        uncovered_penalty_m=uncovered_penalty_m,
        boundary_width_px=boundary_width_px)
    src=pending["source_rows"]["strict_history"]
    row=dict(
       source_keyframe_id=pending["source_uid"],
       source_prior_keyframe_id=pending["prior_uid"],
       source_selection=pending["source_kind"],
       source_eligible_event=pending["event_index"],
       future_keyframe_id=int(viewpoint.uid),
       future_keyframe_order=int(future_keyframe_order),
       elapsed_image_frames=int(viewpoint.uid)-pending["source_uid"],
       temporal_model="frozen_source_offset_zero_order_hold",
       source_n_excluded=pending["n_excluded"],
       source_n_excluded_legacy_gate_accepted=
           pending["n_excluded_legacy_gate_accepted"],
       source_excluded_prev_change_mean_m=pending["excluded_prev_change_mean_m"],
       source_excluded_prev_change_gt_threshold_fraction=
           pending["excluded_prev_change_gt_threshold_fraction"],
       source_dynamic_depth_penalized_mae_m=src["dynamic_depth_penalized_mae_m"],
       source_dynamic_rgb_mae=src["dynamic_rgb_mae"],
       source_interior_depth_penalized_mae_m=src["interior_depth_penalized_mae_m"],
       source_boundary_depth_penalized_mae_m=src["boundary_depth_penalized_mae_m"],
       source_dynamic_depth_n=src["dynamic_depth_n"],
       **metrics)
    for field in ("dynamic_mask_hash","depth_valid_mask_hash",
                  "dynamic_depth_n","dynamic_rgb_n",
                  "interior_depth_n","boundary_depth_n"):
        if row[field]!=rows[0][field]:
            raise RuntimeError("V5.2a future evaluation regions differ")
    rows.append(row)
    return rows,"evaluated"


def _append(directory,name,rows):
    if len(rows)!=4 or [r["mode"] for r in rows]!=list(MODES):
        raise ValueError("Expected Hold/Legacy/Strict/Strict+History triplet + one")
    path=Path(directory)/name
    path.parent.mkdir(parents=True,exist_ok=True)
    exists=path.exists() and path.stat().st_size>0
    with path.open("a",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        if not exists:writer.writeheader()
        writer.writerows(rows)
    return path


def append_source_rows(directory,rows):
    return _append(directory,"v52a_source_per_keyframe.csv",rows)


def append_future_rows(directory,rows):
    return _append(directory,"v52a_future_per_horizon.csv",rows)
