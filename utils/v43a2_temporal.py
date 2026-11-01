"""V4.3a.2: frozen-state, next-keyframe zero-order-hold temporal evaluation.

Captures exactly ONE expensive GaussianModel snapshot before online G0
propagation/Mapping. At the next keyframe message (before modifying the map),
re-renders G0/Support/G4 hypotheses with the ORIGINAL frozen Gaussian
parameters, same candidate offsets and next keyframe's observed camera/RGB-D.

This is a short-horizon *zero-order-hold persistence* diagnostic, NOT
motion extrapolation/flow ground truth, and NOT independent training data
when the tracked camera has itself used earlier mapping information.
It never selects a candidate and never modifies online Gaussian state.
"""
import csv
from pathlib import Path
import torch

from utils.v43a_observation_consistency import evaluate_hypotheses, HYPOTHESES


def validate_future_settings(enabled, source_frames, max_gap):
    if not enabled:
        return
    if not isinstance(source_frames, (list, tuple)) or not source_frames:
        raise ValueError("V4.3a.2 temporal source_frames must be a nonempty list")
    if any(not isinstance(x, int) or x < 0 for x in source_frames):
        raise ValueError("source_frames must contain nonnegative integer frame IDs")
    if not isinstance(max_gap, int) or max_gap < 1:
        raise ValueError("max_frame_gap must be a positive integer")


@torch.no_grad()
def capture_frozen_source(model, *, source_uid, prior_uid, event_index,
                          prev_delta, indices, candidates, current_rows,
                          clone_fn):
    """Copy model ONCE. Caller must bound snapshot count / GPU memory."""
    if set(candidates) != set(HYPOTHESES) or len(current_rows) != 3:
        raise ValueError("Frozen source requires all three candidate renderings")
    if {r["hypothesis"] for r in current_rows} != set(HYPOTHESES):
        raise ValueError("Source render rows not paired")
    if prev_delta.shape[0] != int(model.dygs.sum().item()):
        raise ValueError("Previous Gaussian offset length mismatch")
    if indices.numel() and (indices.max() >= prev_delta.shape[0] or indices.min() < 0):
        raise ValueError("Snapshot index outside source model")
    with torch.no_grad():
        frozen = clone_fn(model)
    if frozen is model:
        raise ValueError("Frozen model clone must not alias live model")
    # The update optimizer is not used for rendering. Existing clone_obj
    # handles tensor clone; no writes to frozen model after this point.
    return {
        "source_uid":int(source_uid),
        "prior_uid":int(prior_uid),
        "event_index":int(event_index),
        "model":frozen,
        "prev_delta":prev_delta.detach().clone(),
        "indices":indices.detach().clone(),
        "candidates":{name:value.detach().clone() for name,value in candidates.items()},
        "source_rows":{r["hypothesis"]:dict(r) for r in current_rows},
    }


def _status(delta_rgb, delta_depth, rgb_tol, depth_tol):
    good_rgb = delta_rgb < -rgb_tol
    good_depth = delta_depth < -depth_tol
    bad_rgb = delta_rgb > rgb_tol
    bad_depth = delta_depth > depth_tol
    if good_rgb and good_depth:
        return "both_better"
    if bad_rgb and bad_depth:
        return "both_worse"
    if good_rgb and bad_depth:
        return "rgb_better_depth_worse"
    if bad_rgb and good_depth:
        return "depth_better_rgb_worse"
    return "tie_or_mixed"


@torch.no_grad()
def evaluate_frozen_future(pending, future_viewpoint, *,
                           render_fn, pipeline, background,
                           max_frame_gap=12, opacity_threshold=.1,
                           uncovered_penalty_m=.5,
                           rgb_tolerance=1e-5,
                           depth_tolerance_m=1e-4):
    """Evaluate exactly the next keyframe or reject it with explicit reason.

    Uses same snapshot, same accepted 0.3m gated offsets and GMM query time
    equal to future viewpoint.uid for all three methods. No flow at future
    time is computed; offset is held constant from source frame.
    """
    src = pending["source_uid"]
    future = int(future_viewpoint.uid)
    gap = future-src
    if gap <= 0:
        raise ValueError("Future UID must strictly follow source")
    if gap > max_frame_gap:
        raise ValueError("Temporal holdout gap {} exceeds {}".format(gap,max_frame_gap))
    scored,_ = evaluate_hypotheses(
        frame_id=future,pc=pending["model"],viewpoint=future_viewpoint,
        prior_keyframe_id=pending["prior_uid"],
        prev_delta=pending["prev_delta"],
        selected_indices=pending["indices"],
        candidates=pending["candidates"],
        render_fn=render_fn,pipeline=pipeline,background=background,
        opacity_threshold=opacity_threshold,
        uncovered_penalty_m=uncovered_penalty_m,
        save_maps=False)
    prior=pending["source_rows"]
    future_dict={r["hypothesis"]:r for r in scored}
    rows=[]
    for mode in HYPOTHESES[1:]:
        r0=prior["g0"]
        ri=prior[mode]
        f0=future_dict["g0"]
        fi=future_dict[mode]
        source_rgb_delta=ri["dynamic_rgb_mae"]-r0["dynamic_rgb_mae"]
        source_depth_delta=ri["dynamic_depth_penalized_mae_m"]-r0["dynamic_depth_penalized_mae_m"]
        future_rgb_delta=fi["dynamic_rgb_mae"]-f0["dynamic_rgb_mae"]
        future_depth_delta=fi["dynamic_depth_penalized_mae_m"]-f0["dynamic_depth_penalized_mae_m"]
        if min(r0["dynamic_rgb_n"],r0["dynamic_depth_n"],
               f0["dynamic_rgb_n"],f0["dynamic_depth_n"]) == 0:
            raise ValueError("Empty source/future dynamic observation region")
        rows.append(dict(
            source_keyframe_id=src,
            future_keyframe_id=future,
            frame_gap=gap,
            eligible_event_index=pending["event_index"],
            hypothesis=mode,
            model_state="frozen_before_source_mapping",
            temporal_model="zero_order_hold_offsets",
            source_dynamic_rgb_n=r0["dynamic_rgb_n"],
            source_dynamic_depth_n=r0["dynamic_depth_n"],
            future_dynamic_rgb_n=f0["dynamic_rgb_n"],
            future_dynamic_depth_n=f0["dynamic_depth_n"],
            source_dynamic_rgb_delta=source_rgb_delta,
            source_dynamic_depth_delta_m=source_depth_delta,
            future_dynamic_rgb_delta=future_rgb_delta,
            future_dynamic_depth_delta_m=future_depth_delta,
            source_dynamic_depth_coverage_delta=(
                ri["dynamic_depth_coverage"]-r0["dynamic_depth_coverage"]),
            future_dynamic_depth_coverage_delta=(
                fi["dynamic_depth_coverage"]-f0["dynamic_depth_coverage"]),
            source_outcome=_status(source_rgb_delta,source_depth_delta,
                                   rgb_tolerance,depth_tolerance_m),
            future_outcome=_status(future_rgb_delta,future_depth_delta,
                                   rgb_tolerance,depth_tolerance_m),
            rgb_sign_retained=(
                (source_rgb_delta < -rgb_tolerance) and
                (future_rgb_delta < -rgb_tolerance)),
            depth_sign_retained=(
                (source_depth_delta < -depth_tolerance_m) and
                (future_depth_delta < -depth_tolerance_m)),
            both_sign_retained=(
                (source_rgb_delta < -rgb_tolerance) and
                (source_depth_delta < -depth_tolerance_m) and
                (future_rgb_delta < -rgb_tolerance) and
                (future_depth_delta < -depth_tolerance_m)),
        ))
    return rows


def append_future_rows(folder, rows):
    if not rows:
        return
    path=Path(folder)
    path.mkdir(parents=True,exist_ok=True)
    csvfile=path/"v43a2_temporal.csv"
    create=not csvfile.exists() or csvfile.stat().st_size==0
    with csvfile.open("a",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        if create:writer.writeheader()
        writer.writerows(rows)
