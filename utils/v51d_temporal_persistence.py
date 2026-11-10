"""V5.1d: frozen-source motion persistence on later incoming keyframes.

Three source displacement states (HOLD / LEGACY / STRICT) are constructed
once using V5.1c exact update-gate semantics. The *same* pre-source-Mapping
GaussianModel snapshot and offsets are rendered at the 1st, 3rd and 5th
subsequent keyframe arrival. Actual RGB-D frame-ID gap is also recorded.

No flow re-estimation, motion extrapolation, online update, or map
optimization is applied to any of the hypotheses. This isolates the
short-horizon persistence of a frozen source state; it cannot establish
true Gaussian lag or motion ground truth. Backend has tracked cameras
only for keyframe messages, NOT for every intermediate video frame.
"""
import csv
from pathlib import Path

import torch

from utils.v51c_controlled_attribution import CHOICES, compose_controlled
from utils.v51b_dynamic_metrics import measure_dynamic_reconstruction

DEFAULT_HORIZONS = (1, 3, 5)


def validate_protocol(*,every_n_event,max_sources,horizons,max_frame_gap,
                      focused_source_frames,min_dynamic_depth_pixels,
                      opacity_threshold,uncovered_penalty_m,boundary_width_px):
    if not isinstance(every_n_event,int) or every_n_event<1:
        raise ValueError("V5.1d every_n_event must be >=1")
    if not isinstance(max_sources,int) or max_sources<1:
        raise ValueError("V5.1d max_sources must be >=1")
    if not isinstance(horizons,(list,tuple)) or not horizons:
        raise ValueError("V5.1d future_keyframe_orders must be a nonempty list")
    if any(type(x)!=int or x<1 for x in horizons) or tuple(horizons)!=tuple(sorted(set(horizons))):
        raise ValueError("V5.1d future_keyframe_orders must be positive, sorted and unique")
    if not isinstance(max_frame_gap,int) or max_frame_gap<max(horizons):
        raise ValueError("V5.1d max_frame_gap cannot be smaller than the horizon")
    if (not isinstance(focused_source_frames,(list,tuple)) or
        any(type(x)!=int or x<0 for x in focused_source_frames)):
        raise ValueError("V5.1d focused_source_frames must list nonnegative frame IDs")
    if not isinstance(min_dynamic_depth_pixels,int) or min_dynamic_depth_pixels<1:
        raise ValueError("V5.1d minimum dynamic depth pixels must be >=1")
    if not 0<=opacity_threshold<=1 or uncovered_penalty_m<=0 or boundary_width_px<1:
        raise ValueError("V5.1d invalid RGB-D evaluation settings")


def selection_kind(event_number,source_uid,*,every_n_event,focus_ids):
    """The focus list is fixed before new data; log missed/busy requested IDs."""
    if event_number<1:
        raise ValueError("Event index starts at one")
    periodic = (event_number-1)%every_n_event==0
    focused = int(source_uid) in focus_ids
    if periodic and focused:return "systematic_and_focused"
    if focused:return "focused"
    if periodic:return "systematic"
    return None


@torch.no_grad()
def capture_source(*,model,source_uid,prior_uid,event_index,source_kind,
                   legacy_indices,legacy_g0,strict_indices,strict_g0,
                   source_rows,clone_fn):
    if source_kind not in ("systematic","focused","systematic_and_focused"):
        raise ValueError("Invalid source selection stratum")
    if model.init_deform!="offset" or int(source_uid)<=int(prior_uid):
        raise ValueError("Source frame and deform model mismatch")
    if prior_uid not in model.kf_list:
        raise ValueError("Missing source predecessor keyframe")
    if len(source_rows)!=len(CHOICES) or {r["mode"] for r in source_rows}!=set(CHOICES):
        raise ValueError("Source-frame triplet must be complete")
    if any(int(row["frame_uid"])!=int(source_uid) for row in source_rows):
        raise ValueError("Source rendering rows refer to other frame")
    for key in ("dynamic_mask_hash","depth_valid_mask_hash","dynamic_depth_n","dynamic_rgb_n"):
        if len({row[key] for row in source_rows})!=1:
            raise ValueError("Source hypotheses did not share observations")
    prior = model.get_delta_xyz(model.kf_list.index(prior_uid))
    if prior.shape!=(int(model.dygs.sum().item()),3):
        raise ValueError("Dynamic offset topology mismatch at capture")
    offsets,_=compose_controlled(
        prior,legacy_indices,legacy_g0,strict_indices,strict_g0)
    frozen=clone_fn(model)
    if frozen is model:
        raise ValueError("Frozen Gaussian map must not alias live map")
    if int(frozen.dygs.sum().item())!=int(prior.shape[0]):
        raise ValueError("Cloned map topology differs from source")
    return dict(
        source_uid=int(source_uid),
        prior_uid=int(prior_uid),
        event_index=int(event_index),
        source_kind=source_kind,
        model=frozen,
        offsets={name:val.detach().clone() for name,val in offsets.items()},
        source_rows={row["mode"]:dict(row) for row in source_rows},
        n_excluded=int(source_rows[0]["n_excluded"]),
        n_excluded_legacy_gate_accepted=int(
            source_rows[0]["n_excluded_legacy_gate_accepted"]),
        excluded_prev_change_mean_m=float(
            source_rows[0]["excluded_prev_change_mean_m"]),
        excluded_prev_change_gt_threshold_fraction=float(
            source_rows[0]["excluded_prev_change_gt_threshold_fraction"]),
        seen_future_keyframes=0)


@torch.no_grad()
def evaluate_future(*,pending,viewpoint,future_keyframe_order,
                    render_fn,pipeline,background,
                    opacity_threshold=.1,uncovered_penalty_m=.5,
                    boundary_width_px=5,min_dynamic_depth_pixels=256):
    future_uid=int(viewpoint.uid)
    if future_uid<=pending["source_uid"] or future_keyframe_order<1:
        raise ValueError("V5.1d future must follow source")
    if future_keyframe_order!=pending["seen_future_keyframes"]:
        raise ValueError("V5.1d keyframe arrival counter mismatch")
    model=pending["model"]
    if model.init_deform!="offset":
        raise ValueError("Frozen model must use dynamic offsets")
    target_rows=[]
    for mode in CHOICES:
        displacement=pending["offsets"][mode]
        if displacement.shape!=(int(model.dygs.sum().item()),3):
            raise ValueError("Frozen snapshot offsets no longer match its Gaussian topology")
        rendered=render_fn(viewpoint,model,pipeline,background,
                           dynamic=False,dx=displacement,ds=0,dr=0,uid=future_uid)
        metrics=measure_dynamic_reconstruction(
            frame_idx=future_uid,frame_uid=future_uid,phase="future_frozen_map",
            mode=mode,is_keyframe=True,
            render_rgb=rendered["render"],render_depth=rendered["depth"],
            render_opacity=rendered["opacity"],
            gt_rgb=viewpoint.original_image.to(displacement.device),
            gt_depth=viewpoint.depth,
            static_mask=viewpoint.motion_mask.to(displacement.device),
            opacity_threshold=opacity_threshold,
            uncovered_penalty_m=uncovered_penalty_m,
            boundary_width_px=boundary_width_px)
        source=pending["source_rows"][mode]
        target_rows.append(dict(
            source_keyframe_id=pending["source_uid"],
            source_prior_keyframe_id=pending["prior_uid"],
            source_selection=pending["source_kind"],
            source_eligible_event=pending["event_index"],
            future_keyframe_id=future_uid,
            future_keyframe_order=int(future_keyframe_order),
            elapsed_image_frames=future_uid-pending["source_uid"],
            temporal_model="frozen_map_zero_order_hold",
            source_n_excluded=pending["n_excluded"],
            source_n_excluded_legacy_gate_accepted=
                pending["n_excluded_legacy_gate_accepted"],
            source_excluded_prev_change_mean_m=pending["excluded_prev_change_mean_m"],
            source_excluded_prev_change_gt_threshold_fraction=
                pending["excluded_prev_change_gt_threshold_fraction"],
            source_dynamic_depth_penalized_mae_m=source["dynamic_depth_penalized_mae_m"],
            source_dynamic_rgb_mae=source["dynamic_rgb_mae"],
            source_interior_depth_penalized_mae_m=
                source["interior_depth_penalized_mae_m"],
            source_boundary_depth_penalized_mae_m=
                source["boundary_depth_penalized_mae_m"],
            source_dynamic_depth_n=source["dynamic_depth_n"],
            **metrics))
    for key in ("dynamic_mask_hash","depth_valid_mask_hash",
                "dynamic_depth_n","dynamic_rgb_n","interior_depth_n","boundary_depth_n"):
        if len({row[key] for row in target_rows})!=1:
            raise RuntimeError("Future candidate observation masks were not held fixed")
    if target_rows[0]["dynamic_depth_n"]<min_dynamic_depth_pixels:
        return [],"too_few_future_dynamic_depth_pixels"
    return target_rows,"evaluated"


def _append_csv(filename, rows):
    if not rows:return
    filename=Path(filename)
    filename.parent.mkdir(parents=True,exist_ok=True)
    new=not filename.exists() or filename.stat().st_size==0
    with filename.open("a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        if new:writer.writeheader()
        writer.writerows(rows)


def append_future_rows(folder, rows):
    if len(rows)!=len(CHOICES) or [r["mode"] for r in rows]!=list(CHOICES):
        raise ValueError("Incomplete hold/legacy/strict future comparison")
    _append_csv(Path(folder)/"v51d_future_per_horizon.csv",rows)


def log_event(folder,*,source_uid,source_kind,event_index,action,
              future_uid=None,future_order=None,frame_gap=None,reason=""):
    _append_csv(Path(folder)/"v51d_event_log.csv",[dict(
        source_keyframe_id=int(source_uid),
        source_selection=str(source_kind),
        source_eligible_event=int(event_index),
        action=str(action),
        future_keyframe_id="" if future_uid is None else int(future_uid),
        future_keyframe_order="" if future_order is None else int(future_order),
        elapsed_image_frames="" if frame_gap is None else int(frame_gap),
        reason=str(reason))])
