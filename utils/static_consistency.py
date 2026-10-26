"""Experimental RGB-D evidence; independent of CUDA rasterization and SLAM state.

State labels are hypotheses, not calibrated motion probabilities. In particular,
UNKNOWN must never be interpreted as a negative dynamic label for mapping.
"""

from dataclasses import dataclass
import math

import torch


UNKNOWN, STATIC, DYNAMIC = 0, 1, 2


@dataclass(frozen=True)
class ConsistencyConfig:
    mode: str = "off"  # off | observe | tracking
    depth_definition: str = "normalized"  # preserve the original detector default
    warmup_iterations: int = 10
    opacity_min: float = 0.95
    depth_min: float = 0.01
    depth_abs_tolerance: float = 0.05  # metres; starting values, not tuned
    depth_rel_tolerance: float = 0.02
    rgb_tolerance: float = 0.15  # mean absolute error in [0, 1] RGB
    min_tracking_pixels: int = 256
    min_tracking_fraction: float = 0.25
    inconsistent_weight: float = 0.0  # 0 hard exclusion; 1 matched no-filter control
    save_tracking_decisions: bool = False
    save_every: int = 30  # 0 disables dense NPZ dumps, not JSONL summaries

    def __post_init__(self):
        if not math.isfinite(self.inconsistent_weight) or not 0 <= self.inconsistent_weight <= 1:
            raise ValueError("inconsistent_weight must be finite and in [0, 1]")
        if not isinstance(self.save_tracking_decisions, bool):
            raise ValueError("save_tracking_decisions must be boolean")
        if self.mode not in ("off", "observe", "tracking"):
            raise ValueError("StaticConsistency.mode must be off, observe or tracking")
        if self.depth_definition not in ("accumulated", "normalized"):
            raise ValueError("depth_definition must be accumulated or normalized")
        for name in ("warmup_iterations", "save_every", "min_tracking_pixels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        for name in ("depth_min", "depth_abs_tolerance", "rgb_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if not math.isfinite(self.depth_rel_tolerance) or self.depth_rel_tolerance < 0:
            raise ValueError("depth_rel_tolerance must be finite and nonnegative")
        if not 0 < self.opacity_min <= 1:
            raise ValueError("opacity_min must be in (0, 1]")
        if not 0 < self.min_tracking_fraction <= 1:
            raise ValueError("min_tracking_fraction must be in (0, 1]")


@torch.no_grad()
def compute_evidence(observed_rgb, observed_depth, rendered_rgb, accumulated_depth,
                     opacity, config, flow_dynamic=None, flow_valid=None):
    """Compute detached evidence on the input device.

The repository rasterizer returns sum(alpha * transmittance * z), NOT
normalized surface depth. The detector selects a definition without changing losses.
Validity uses normalized depth in both modes to keep paired support identical.
RGB must already use the same exposure correction as the tracking loss.
Flow inputs, if supplied, must be aligned to this frame and provided together.
"""
    if observed_depth.ndim != 2:
        raise ValueError("observed_depth must have shape H x W")
    shape = observed_depth.shape
    expected_rgb = (3,) + tuple(shape)
    if tuple(observed_rgb.shape) != expected_rgb or tuple(rendered_rgb.shape) != expected_rgb:
        raise ValueError("RGB inputs must have shape 3 x H x W")
    for value in (accumulated_depth, opacity):
        if tuple(value.shape) not in (tuple(shape), (1,) + tuple(shape)):
            raise ValueError("rendered depth and opacity must have shape H x W or 1 x H x W")
    if (flow_dynamic is None) != (flow_valid is None):
        raise ValueError("flow_dynamic and flow_valid must be supplied together")
    if flow_dynamic is not None and (flow_dynamic.shape != shape or flow_valid.shape != shape):
        raise ValueError("flow evidence must have shape H x W")

    alpha = opacity.reshape(shape)
    raw_depth = accumulated_depth.reshape(shape)
    predicted_depth = raw_depth / alpha.clamp_min(1e-6)
    observation_valid = torch.isfinite(observed_depth) & (observed_depth > config.depth_min)
    observation_valid &= torch.isfinite(observed_rgb).all(dim=0)
    map_valid = observation_valid & torch.isfinite(alpha) & (alpha >= config.opacity_min)
    map_valid &= (alpha <= 1.0001) & torch.isfinite(predicted_depth)
    map_valid &= (predicted_depth > config.depth_min) & torch.isfinite(rendered_rgb).all(dim=0)

    # Positive signed residual = current surface is in front of the map.
    selected_depth = raw_depth if config.depth_definition == "accumulated" else predicted_depth
    signed_depth = selected_depth - observed_depth
    tolerance = config.depth_abs_tolerance + config.depth_rel_tolerance * observed_depth
    depth_score = signed_depth.abs() / tolerance.clamp_min(1e-6)
    rgb_score = (observed_rgb - rendered_rgb).abs().mean(dim=0) / config.rgb_tolerance
    depth_bad = map_valid & (depth_score > 1)
    rgb_bad = map_valid & (rgb_score > 1)
    map_inconsistent = depth_bad | rgb_bad
    foreground = map_valid & (signed_depth > tolerance)
    revealed = map_valid & (signed_depth < -tolerance)

    state = torch.full_like(observed_depth, UNKNOWN, dtype=torch.uint8)
    static = map_valid & ~map_inconsistent
    dynamic = torch.zeros_like(map_valid)
    if flow_dynamic is not None:
        fv = flow_valid.bool() & observation_valid
        fd = flow_dynamic.bool() & fv
        static &= ~fd
        # Map disagreement alone is not motion. Flow/map conflicts stay unknown;
        # a farther observed surface may be newly revealed static background.
        dynamic = fd & (~map_valid | (map_inconsistent & ~revealed))
    state[static] = STATIC
    state[dynamic] = DYNAMIC

    nan = torch.full_like(observed_depth, float("nan"))
    return {
        "state": state,
        "map_valid": map_valid,
        "map_inconsistent": map_inconsistent,
        "foreground_candidate": foreground,
        "revealed_candidate": revealed,
        "depth_score": torch.where(map_valid, depth_score, nan),
        "rgb_score": torch.where(map_valid, rgb_score, nan),
        # Preserve raw diagnostics outside map_valid for offline validity audits.
        # Depths/residuals are in metres; opacity is dimensionless. Consumers
        # must apply validity masks before interpreting residuals.
        "observed_depth": observed_depth.detach(),
        "accumulated_depth": raw_depth.detach(),
        "render_opacity": alpha.detach(),
        "normalized_depth": predicted_depth.detach(),
        "depth_tolerance": tolerance.detach(),
        "observation_valid": observation_valid.detach(),
        # Positive means rendered depth is farther than the observation.
        # Include subthreshold residuals, not just detected anomalies.
        "signed_depth_accumulated": (raw_depth - observed_depth).detach(),
        "signed_depth_normalized": (predicted_depth - observed_depth).detach(),
    }


@torch.no_grad()
def select_tracking_mask(evidence, baseline_support, config):
    """Return an extra keep-mask or None if too little baseline support remains.

Unknown map coverage does not introduce an extra exclusion. This is deliberately
NOT a dynamic assignment mask and must not overwrite Camera.motion_mask.
"""
    keep = ~evidence["map_inconsistent"]
    before = int(baseline_support.count_nonzero().item())
    after = int((baseline_support.bool() & keep).count_nonzero().item())
    accepted = (after >= config.min_tracking_pixels and before > 0
                and after / before >= config.min_tracking_fraction)
    return (keep if accepted else None), {
        "baseline_support_pixels": before,
        "retained_support_pixels": after,
        "candidate_excluded_pixels": before - after,
        "candidate_excluded_fraction": (before - after) / before if before else None,
        "applied_retained_support_pixels": after if accepted else before,
        "applied_excluded_pixels": before - after if accepted else 0,
        "fallback": not accepted,
        "fallback_reason": (None if accepted else "empty_support" if before == 0
                            else "minimum_pixels" if after < config.min_tracking_pixels
                            else "minimum_fraction"),
        "mask_applied": accepted,
    }


@torch.no_grad()
def tracking_supports(observed_rgb, observed_depth, opacity, grad_mask,
                      motion_mask, rgb_threshold, frame_id):
    """Mirror RGB-D tracking loss masks before the extra consistency mask.

    RGB is opacity-weighted but has no opacity cutoff; depth has no gradient mask.
    Counts describe selected pixels, not gradient magnitude or pose information.
    """
    shape = observed_depth.shape
    baseline = torch.ones_like(observed_depth, dtype=torch.bool)
    if motion_mask is not None and frame_id > 0:
        baseline &= motion_mask.reshape(shape).bool()
    rgb = baseline & (observed_rgb.sum(dim=0) > rgb_threshold) & grad_mask.reshape(shape).bool()
    depth = baseline & (observed_depth > 0.01) & (observed_depth < 1000)
    depth &= opacity.reshape(shape) > 0.95
    return {"rgb": rgb, "depth": depth, "union": rgb | depth}


@torch.no_grad()
def tracking_support_statistics(evidence, supports, applied_keep):
    result = {}
    candidate_keep = ~evidence["map_inconsistent"]
    for name, support in supports.items():
        before = int(support.count_nonzero().item())
        candidate = int((support & candidate_keep).count_nonzero().item())
        after = int((support & applied_keep).count_nonzero().item()) if applied_keep is not None else before
        result.update({f"{name}_support_pixels": before,
                       f"{name}_candidate_retained_pixels": candidate,
                       f"{name}_applied_retained_pixels": after,
                       f"{name}_applied_excluded_fraction": (before-after)/before if before else None})
    return result


@torch.no_grad()
def save_tracking_decision(folder, frame, iteration, evidence, supports, applied_keep,
                           info, config, observed_rgb, rotation, translation, weights=None):
    """Save the decision BEFORE its first optimizer update; never change inputs.

    Masks are saved every decision frame. Dense evidence/RGB/pose use save_every.
    Support-based exclusions describe this instant, not all later iterations.
    """
    if not config.save_tracking_decisions:
        return
    import json
    from pathlib import Path
    import numpy as np

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    candidate_keep = ~evidence["map_inconsistent"]
    effective_keep = torch.ones_like(candidate_keep) if applied_keep is None else applied_keep
    tensors = {"candidate_keep_mask": candidate_keep,
               "applied_keep_mask": effective_keep,
               "map_valid": evidence["map_valid"]}
    for name, support in supports.items():
        tensors[name + "_support_mask"] = support
        tensors[name + "_candidate_excluded_mask"] = support & ~candidate_keep
        tensors[name + "_applied_excluded_mask"] = support & ~effective_keep
    dense = bool(config.save_every and frame % config.save_every == 0)
    if weights is not None:
        tensors["tracking_weights"] = weights
    if dense:
        tensors.update(evidence)
        tensors.update(observed_rgb=observed_rgb, world_to_camera_R=rotation,
                       world_to_camera_T=translation)
    arrays = {k: v.detach().cpu().numpy() for k, v in tensors.items()}
    arrays.update(frame=np.int64(frame), mask_iteration=np.int64(iteration),
                  dense_diagnostics=np.bool_(dense), mask_applied=np.bool_(info.get("mask_applied", applied_keep is not None)),
                  depth_definition=np.asarray(config.depth_definition))
    np.savez_compressed(folder / f"{frame:06d}.npz", **arrays)
    record = {**info, "frame": int(frame), "mask_iteration": int(iteration),
              "mode": config.mode, "depth_definition": config.depth_definition,
              "evaluation_pose": "before_mask_iteration_update",
              "dense_diagnostics": dense, "npz_file": f"{frame:06d}.npz"}
    with (folder / "summary.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


@torch.no_grad()
def select_tracking_weights(evidence, supports, config):
    """Use the same conservative hard-mask acceptance test for all weights.

    Float masks multiply both sides of the existing L1 residual, hence weight
    the loss linearly. No change to the original mean denominator is needed.
    """
    candidate, info = select_tracking_mask(evidence, supports["union"], config)
    accepted = candidate is not None
    weights = torch.ones_like(evidence["depth_score"])
    if accepted:
        weights[~candidate] = config.inconsistent_weight
    applied_keep = weights > 0
    info.update(tracking_support_statistics(evidence, supports, applied_keep))
    info.update(inconsistent_weight=config.inconsistent_weight,
                decision_accepted=accepted,
                mask_applied=accepted and config.inconsistent_weight < 1,
                applied_retained_support_pixels=int((supports["union"] & applied_keep).sum().item()),
                applied_excluded_pixels=int((supports["union"] & ~applied_keep).sum().item()))
    for name, support in supports.items():
        n = int(support.sum().item())
        info[name + "_downweighted_pixels"] = int((support & (weights < 1)).sum().item())
        info[name + "_weight_sum"] = float(weights[support].sum().item())
        info[name + "_mean_weight"] = float(weights[support].mean().item()) if n else None
    return weights, info
