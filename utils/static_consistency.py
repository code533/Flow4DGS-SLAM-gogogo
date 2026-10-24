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
    warmup_iterations: int = 10
    opacity_min: float = 0.95
    depth_min: float = 0.01
    depth_abs_tolerance: float = 0.05  # metres; starting values, not tuned
    depth_rel_tolerance: float = 0.02
    rgb_tolerance: float = 0.15  # mean absolute error in [0, 1] RGB
    min_tracking_pixels: int = 256
    min_tracking_fraction: float = 0.25
    save_every: int = 30  # 0 disables dense NPZ dumps, not JSONL summaries

    def __post_init__(self):
        if self.mode not in ("off", "observe", "tracking"):
            raise ValueError("StaticConsistency.mode must be off, observe or tracking")
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
normalized surface depth. Divide by accumulated opacity for this detector only.
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
    signed_depth = predicted_depth - observed_depth
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
        "mask_applied": accepted,
    }
