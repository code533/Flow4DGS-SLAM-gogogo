"""UP-SLAM-inspired tracking-time uncertainty mask from a *static* Gaussian map.

This is a lightweight, training-free RGB-D consistency detector, not a
reproduction of the complete UP-SLAM feature/MLP uncertainty model.

Contract:
    motion_mask == True: static / eligible for tracking.
    motion_mask == False: dynamic / excluded from tracking.

No RAFT flow is used to construct this mask. Optical flow in other parts
of Flow4DGS-SLAM remains unchanged in the layer-1 experiment.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def estimate_static_consistency_mask(
    viewpoint,
    render_pkg,
    prior_static_mask,
    cfg,
):
    """Return a conservative static mask and detached diagnostics.

    Rendered tensors are produced exclusively with static Gaussians
    (pc.dygs == False), at the current *initial* camera pose.
    Uncovered pixels are not automatically classified as dynamic.
    """
    rgb_gt = viewpoint.original_image.to(
        device=render_pkg["render"].device, dtype=torch.float32
    )
    rgb_pred = render_pkg["render"].detach().float()
    d_pred = render_pkg["depth"].detach().float().squeeze()
    alpha = render_pkg["opacity"].detach().float().squeeze()

    d_gt = torch.as_tensor(
        viewpoint.depth, device=d_pred.device, dtype=torch.float32
    ).squeeze()

    if rgb_gt.shape != rgb_pred.shape or d_gt.shape != d_pred.shape:
        raise ValueError(
            "Static uncertainty shape mismatch: "
            f"rgb {tuple(rgb_gt.shape)} vs {tuple(rgb_pred.shape)}, "
            f"depth {tuple(d_gt.shape)} vs {tuple(d_pred.shape)}"
        )

    rgb_err = (rgb_gt - rgb_pred).abs().mean(dim=0)
    depth_err = (d_gt - d_pred).abs()

    alpha_min = float(cfg.get("opacity_min", 0.95))
    depth_abs = float(cfg.get("depth_abs_threshold", 0.08))
    depth_rel = float(cfg.get("depth_rel_threshold", 0.04))
    rgb_threshold = float(cfg.get("rgb_threshold", 0.15))
    min_valid_ratio = float(cfg.get("min_valid_ratio", 0.05))
    min_valid_pixels = int(cfg.get("min_valid_pixels", 256))

    prior = prior_static_mask.to(device=d_gt.device, dtype=torch.bool)
    valid = (
        torch.isfinite(d_gt)
        & (d_gt > 0)
        & torch.isfinite(d_pred)
        & (d_pred > 0)
        & torch.isfinite(alpha)
        & (alpha >= alpha_min)
        & torch.isfinite(rgb_err)
        & torch.isfinite(depth_err)
        & prior
    )
    valid_count = int(valid.count_nonzero().item())
    n_pixels = d_gt.numel()

    # Map warm-up / unreliable render fallback. Do not erase the prior mask.
    if valid_count < max(min_valid_pixels, int(min_valid_ratio * n_pixels)):
        return prior_static_mask.clone().bool(), {
            "status": "fallback_low_coverage",
            "coverage": float(valid_count) / max(n_pixels, 1),
            "dynamic_frac": 0.0,
            "rgb_err": rgb_err,
            "depth_err": depth_err,
            "candidate": torch.zeros_like(valid),
            "valid": valid,
        }

    # Relative depth gate tolerates larger geometric error at greater depth.
    depth_outlier = depth_err > (depth_abs + depth_rel * d_gt)
    rgb_outlier = rgb_err > rgb_threshold

    # Conservative evidence fusion, to reduce false dynamic labels caused
    # by illumination, color/exposure drift, and imperfect static maps.
    fusion = cfg.get("fusion", "and")
    if fusion == "and":
        inconsistent = depth_outlier & rgb_outlier
    elif fusion == "or":
        inconsistent = depth_outlier | rgb_outlier
    elif fusion == "depth":
        inconsistent = depth_outlier
    else:
        raise ValueError(f"Unknown uncertainty fusion: {fusion}")

    candidate = inconsistent & valid

    kernel_size = int(cfg.get("median_kernel", 5))
    if kernel_size > 1:
        if kernel_size % 2 != 1:
            raise ValueError("median_kernel must be odd")
        # Majority filter prevents isolated false positives.
        count = F.avg_pool2d(
            candidate[None, None].float(), kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2
        )[0, 0]
        candidate = (count > 0.5) & valid

    max_dynamic_frac = float(cfg.get("max_dynamic_frac", 0.35))
    candidate_frac = float(candidate.count_nonzero().item()) / max(valid_count, 1)
    if candidate_frac > max_dynamic_frac:
        return prior_static_mask.clone().bool(), {
            "status": "fallback_excessive_rejection",
            "coverage": float(valid_count) / max(n_pixels, 1),
            "dynamic_frac": candidate_frac,
            "rgb_err": rgb_err,
            "depth_err": depth_err,
            "candidate": candidate,
            "valid": valid,
        }

    mask = prior & (~candidate)
    return mask.to(device=prior_static_mask.device), {
        "status": "accepted",
        "coverage": float(valid_count) / max(n_pixels, 1),
        "dynamic_frac": candidate_frac,
        "rgb_err": rgb_err,
        "depth_err": depth_err,
        "candidate": candidate,
        "valid": valid,
    }
