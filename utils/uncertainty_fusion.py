"""RGB-D static-map residual weighting for Flow4DGS-SLAM Tracking (V1).

Inspired by UP-SLAM's multimodal consistency cues; does NOT reproduce its
DINO representation or learned uncertainty model. The score is heuristic,
not a calibrated probability of motion.
Note: viewpoint.motion_mask == True denotes pixels considered STATIC.
"""
from typing import Dict

import torch
import torch.nn.functional as F


def _hw(x: torch.Tensor, name: str) -> torch.Tensor:
    """Convert [H,W] or [1,H,W] to [H,W]."""
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    raise ValueError("{} must have shape [H,W] or [1,H,W]; got {}".format(name, tuple(x.shape)))


def compute_rgbd_uncertainty(
    rendered_rgb: torch.Tensor,
    rendered_depth: torch.Tensor,
    rendered_opacity: torch.Tensor,
    observed_rgb: torch.Tensor,
    observed_depth: torch.Tensor,
    static_mask: torch.Tensor,
    cfg: Dict,
) -> Dict[str, torch.Tensor]:
    """Calculate detached soft weights for originally-static Tracking pixels.

    No dynamic classification is performed on unmapped pixels. Original hard
    motion-mask filtering remains in get_loss_tracking. The candidate map is
    diagnostic only and is NOT passed to dynamic 4D Mapping.
    """
    with torch.no_grad():
        if rendered_rgb.ndim != 3 or rendered_rgb.shape[0] != 3:
            raise ValueError("rendered_rgb must be [3,H,W]")
        if observed_rgb.shape != rendered_rgb.shape:
            raise ValueError("observed_rgb shape mismatch")
        H, W = rendered_rgb.shape[-2:]
        device, dtype = rendered_rgb.device, rendered_rgb.dtype
        pred = rendered_rgb.detach()
        gt = observed_rgb.detach().to(device=device, dtype=dtype)
        pd = _hw(rendered_depth.detach(), "rendered_depth").to(device=device, dtype=dtype)
        gd = _hw(observed_depth.detach(), "observed_depth").to(device=device, dtype=dtype)
        op = _hw(rendered_opacity.detach(), "rendered_opacity").to(device=device, dtype=dtype)
        static = _hw(static_mask.detach(), "static_mask").to(device=device, dtype=torch.bool)
        for name, x in (("rendered_depth", pd), ("observed_depth", gd),
                        ("rendered_opacity", op), ("static_mask", static)):
            if x.shape != (H, W):
                raise ValueError("{} spatial shape mismatch".format(name))

        opacity_min = float(cfg.get("opacity_min", 0.95))
        min_depth = float(cfg.get("min_depth", 0.01))
        max_depth = float(cfg.get("max_depth", 1000.0))
        rgb_scale = max(float(cfg.get("rgb_scale", 0.10)), 1e-6)
        depth_scale = max(float(cfg.get("depth_rel_scale", 0.10)), 1e-6)
        wc = float(cfg.get("rgb_coeff", 0.40))
        wd = float(cfg.get("depth_coeff", 0.60))
        thresh = float(cfg.get("threshold", 1.60))
        temp = max(float(cfg.get("temperature", 0.35)), 1e-5)
        min_weight = float(cfg.get("min_weight", 0.10))
        min_coverage = float(cfg.get("min_coverage", 0.08))
        binary_threshold = float(cfg.get("binary_threshold", 0.80))
        kernel = int(cfg.get("smoothing_kernel", 3))
        if not (wc >= 0 and wd >= 0 and wc + wd > 0):
            raise ValueError("rgb_coeff/depth_coeff must be nonnegative and nonzero in sum")
        if not (0 < min_weight <= 1 and 0 <= min_coverage <= 1):
            raise ValueError("min_weight or min_coverage outside allowed range")
        if kernel < 1 or kernel % 2 != 1:
            raise ValueError("smoothing_kernel must be positive and odd")

        depth_ok = (torch.isfinite(pd) & torch.isfinite(gd)
                    & (pd >= min_depth) & (gd >= min_depth)
                    & (pd < max_depth) & (gd < max_depth))
        rgb_ok = torch.isfinite(pred).all(dim=0) & torch.isfinite(gt).all(dim=0)
        covered = depth_ok & rgb_ok & torch.isfinite(op) & (op >= opacity_min)
        valid = covered & static

        rgb_err = torch.nan_to_num((gt - pred).abs().mean(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
        # Relative-depth heuristic mitigates disproportionate absolute errors at range.
        depth_err = torch.nan_to_num((gd - pd).abs() / gd.clamp_min(0.5),
                                     nan=0.0, posinf=0.0, neginf=0.0)
        score = ((wc * rgb_err / rgb_scale) + (wd * depth_err / depth_scale)) / (wc + wd)
        inconsistent = torch.sigmoid((score.clamp(0.0, 50.0) - thresh) / temp)

        if kernel > 1:
            # Normalize the pooled confidence by valid coverage, preventing
            # unreliable map regions from contaminating reliable neighbors.
            v = valid.float()[None, None]
            sums = F.avg_pool2d(inconsistent[None, None] * v, kernel, stride=1, padding=kernel//2)
            counts = F.avg_pool2d(v, kernel, stride=1, padding=kernel//2)
            inconsistent = (sums / counts.clamp_min(1e-6))[0, 0]

        inconsistent = torch.where(valid, inconsistent, torch.zeros_like(inconsistent))
        coverage = (valid.sum().float() / static.sum().clamp_min(1).float()).item()
        applied = coverage >= min_coverage
        weights = torch.ones((H, W), device=device, dtype=dtype)
        if applied:
            weights = torch.where(valid, (1.0 - inconsistent).clamp(min=min_weight), weights)
        else:
            inconsistent.zero_()

        candidates = valid & (inconsistent >= binary_threshold) if applied else torch.zeros_like(valid)
        return {
            "weights": weights.detach(),
            "uncertainty": inconsistent.detach(),
            "new_dynamic_candidate": candidates.detach(),
            "valid": valid.detach(),
            "coverage": coverage,
            "applied": bool(applied),
        }
