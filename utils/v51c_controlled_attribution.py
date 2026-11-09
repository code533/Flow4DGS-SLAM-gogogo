"""V5.1c controlled read-only motion attribution on one frozen pre-update map.

For a sampled KEYFRAME before GaussianModel.update_delta_xyz:
- HOLD: inherited previous dynamic offsets; no new movement.
- LEGACY: apply the original G0 candidate and exact 0.3m gate.
- STRICT: remove source/warped-target FOV-invalid candidates, then rerun
  G0 KNN on surviving points and apply the exact same 0.3m gate.

The Gaussian map, camera, observation, render time and estimated mask
are identical for all three candidates. No map clone or model mutation.
This is an in-sample, pre-Mapping local counterfactual, NOT an unbiased
estimate of online benefit or physically correct motion.

Also log a conservative previous-increment 'staleness risk' proxy for
excluded Gaussians. The proxy is NOT a measurement of real object lag.
"""
import csv
from pathlib import Path

import torch

from utils.v43a_observation_consistency import compose_offsets
from utils.v51b_dynamic_metrics import measure_dynamic_reconstruction

CHOICES=("hold","legacy","strict")


def should_capture(event_number, every, cap, captured):
    if every<1 or cap<1 or event_number<1 or captured<0:
        raise ValueError("Invalid diagnostic sampling configuration")
    return captured<cap and (event_number-1)%every==0


@torch.no_grad()
def compose_controlled(prev_offset, legacy_indices, legacy_g0,
                       strict_indices, strict_g0, gate_m=.3):
    if prev_offset.ndim!=2 or prev_offset.shape[1]!=3:
        raise ValueError("Expected Nx3 previous offsets")
    if (legacy_indices.ndim!=1 or strict_indices.ndim!=1 or
            legacy_g0.shape!=(len(legacy_indices),3) or
            strict_g0.shape!=(len(strict_indices),3)):
        raise ValueError("Candidate index/displacement mismatch")
    if (prev_offset.device!=legacy_indices.device or
            prev_offset.device!=strict_indices.device or
            prev_offset.device!=legacy_g0.device or
            prev_offset.device!=strict_g0.device):
        raise ValueError("Different index/offset devices")
    hold=prev_offset.detach().clone()
    legacy,legacy_gate=compose_offsets(
        prev_offset,legacy_indices,legacy_g0,max_motion=gate_m)
    strict,strict_gate=compose_offsets(
        prev_offset,strict_indices,strict_g0,max_motion=gate_m)
    return {"hold":hold,"legacy":legacy,"strict":strict}, {
        "legacy_gate_pass_n":int(legacy_gate.sum().item()),
        "strict_gate_pass_n":int(strict_gate.sum().item())}


@torch.no_grad()
def previous_increment_proxy(model, prior_uid, excluded_indices,
                             history_min_m=.005):
    """Compare adjacent stored offset parameters, within current topology.

    The offsets may have been optimized by Mapping. Therefore a large
    previous increment is a 'motion/history activity' proxy, NOT velocity
    ground truth and NOT proof of present-day drift.
    """
    if history_min_m<0:
        raise ValueError("history_min_m must be >=0")
    if prior_uid not in model.kf_list:
        raise ValueError("Unknown prior keyframe")
    pos=model.kf_list.index(prior_uid)
    if pos<1:
        return dict(n_history_available=0,
                    excluded_prev_change_mean_m=float("nan"),
                    excluded_prev_change_gt_threshold_fraction=float("nan"),
                    excluded_prev_change_p95_m=float("nan"))
    a=model.get_delta_xyz(pos)
    b=model.get_delta_xyz(pos-1)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3:
        return dict(n_history_available=0,
                    excluded_prev_change_mean_m=float("nan"),
                    excluded_prev_change_gt_threshold_fraction=float("nan"),
                    excluded_prev_change_p95_m=float("nan"))
    if excluded_indices.numel() and (
          excluded_indices.min()<0 or excluded_indices.max()>=len(a)):
        raise IndexError("Excluded index outside offset table")
    vals=torch.linalg.vector_norm((a-b)[excluded_indices.long()],dim=1)
    if not vals.numel():
        return dict(n_history_available=0,
                    excluded_prev_change_mean_m=float("nan"),
                    excluded_prev_change_gt_threshold_fraction=float("nan"),
                    excluded_prev_change_p95_m=float("nan"))
    return dict(
        n_history_available=int(vals.numel()),
        excluded_prev_change_mean_m=float(vals.float().mean().item()),
        excluded_prev_change_gt_threshold_fraction=float(
            (vals>history_min_m).float().mean().item()),
        excluded_prev_change_p95_m=float(torch.quantile(vals.float(),.95).item()))


@torch.no_grad()
def evaluate_controlled(*,model,viewpoint,prior_uid,frame_id,event_index,
                        legacy_indices,legacy_g0,strict_indices,strict_g0,
                        render_fn,pipeline,background,opacity_threshold=.1,
                        uncovered_penalty_m=.5,boundary_width_px=5,
                        history_min_m=.005,gate_m=.3):
    """Render three candidate dynamic-offset tensors without state mutation."""
    if int(viewpoint.uid)!=int(frame_id):
        raise ValueError("Frame and viewpoint do not match")
    if model.init_deform!="offset":
        raise ValueError("Only offset-based dynamic models are supported")
    if prior_uid not in model.kf_list:
        raise ValueError("Prior keyframe not in Gaussian model")
    prev=model.get_delta_xyz(model.kf_list.index(prior_uid))
    if prev.shape!=(int(model.dygs.sum().item()),3):
        raise ValueError("Previous offsets and dynamic Gaussians mismatch")
    offsets,gates=compose_controlled(prev,legacy_indices,legacy_g0,
                                      strict_indices,strict_g0,gate_m=gate_m)
    keep=torch.isin(legacy_indices.long(),strict_indices.long())
    removed=legacy_indices[~keep]
    lag=previous_increment_proxy(model,prior_uid,removed,
                                 history_min_m=history_min_m)
    rows=[]
    for mode in CHOICES:
        pkg=render_fn(viewpoint,model,pipeline,background,
                      dynamic=False,dx=offsets[mode],ds=0,dr=0,uid=frame_id)
        metrics=measure_dynamic_reconstruction(
            frame_idx=int(frame_id),frame_uid=int(frame_id),
            phase="pre_mapping_fixed_map",mode=mode,
            is_keyframe=True,render_rgb=pkg["render"],
            render_depth=pkg["depth"],render_opacity=pkg["opacity"],
            gt_rgb=viewpoint.original_image.to(prev.device),
            gt_depth=viewpoint.depth,
            static_mask=viewpoint.motion_mask.to(prev.device),
            opacity_threshold=opacity_threshold,
            uncovered_penalty_m=uncovered_penalty_m,
            boundary_width_px=boundary_width_px)
        rows.append(dict(
            source_keyframe_id=int(prior_uid),
            keyframe_id=int(frame_id),
            eligible_event_index=int(event_index),
            n_dynamic_gaussians=int(len(prev)),
            n_legacy_candidates=int(len(legacy_indices)),
            n_strict_candidates=int(len(strict_indices)),
            n_excluded=int(len(removed)),
            n_excluded_legacy_gate_accepted=int((
                torch.linalg.vector_norm(legacy_g0[~keep],dim=1)<gate_m
              ).sum().item()),
            n_candidate_gate_pass=int(gates.get(mode+"_gate_pass_n",0)),
            n_actual_moved_from_hold=int(
                (offsets[mode]!=offsets["hold"]).any(-1).sum().item()),
            **lag,**metrics))
    if not torch.equal(prev,offsets["hold"]):
        raise RuntimeError("Pre-update offset mutated during V5.1c")
    return rows


def append_rows(directory,rows):
    if len(rows)!=3 or [r["mode"] for r in rows]!=list(CHOICES):
        raise ValueError("Expected hold/legacy/strict in this order")
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/"v51c_frozen_per_keyframe.csv"
    new=not path.exists() or path.stat().st_size==0
    with path.open("a",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        if new:writer.writeheader()
        writer.writerows(rows)
    return path
