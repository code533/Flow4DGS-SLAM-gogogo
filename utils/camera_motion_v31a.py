"""V3.1a: read-only same-frame comparison of camera motion estimates.

No GT enters either estimator. Caller always uses the original pose for Tracking.
"""
import torch

from utils.pose_utils import SE3_exp, SO3_exp, so3_log


@torch.no_grad()
def candidate_initial_pose(T_previous, xi, max_translation, max_rotation, translation_mode="legacy"):
    """Mirror the original frontend's right-multiplied SE(3) and step constraints."""
    if T_previous.shape != (4, 4) or xi.shape != (6,):
        raise ValueError("Invalid pose/twist shape")
    if translation_mode not in ("legacy", "capped"):
        raise ValueError("Unexpected translation mode")
    if not bool(torch.isfinite(T_previous).all() and torch.isfinite(xi).all()):
        raise ValueError("Nonfinite candidate pose inputs")
    T_prev = T_previous.detach().clone()
    T_new = T_prev @ SE3_exp(xi.detach())
    delta_t = T_new[:3, 3] - T_prev[:3, 3]
    delta_R = T_new[:3, :3] @ T_prev[:3, :3].T
    tr = torch.clamp((torch.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0)
    angle = torch.acos(tr)
    mtrans = torch.as_tensor(max_translation, device=xi.device, dtype=xi.dtype)
    mrot = torch.as_tensor(max_rotation, device=xi.device, dtype=xi.dtype)
    s_t = mtrans / (torch.norm(delta_t) + 1e-10)
    s_r = mrot / (angle + 1e-10)
    s = torch.clamp(torch.minimum(s_t, s_r), max=1.0)
    if translation_mode == "capped":
        s_t = torch.clamp(s_t, max=1.0)
    t_step = s_t * delta_t
    w = so3_log(delta_R)
    new_rot = SO3_exp(s * w) @ T_prev[:3, :3]
    pose = T_new.clone()
    pose[:3, :3] = new_rot
    pose[:3, 3] = T_prev[:3, 3] + t_step
    return pose


@torch.no_grad()
def compare_flow_residuals(depth, flow_px, K, static_mask, xi_legacy, xi_geo, holdout_stride=4):
    """Evaluate both fits on the exact same dense static pixels and unsampled pixels.

    Holdout excludes the V3.1 stride grid: y or x not divisible by stride.
    The holdout is diagnostic only and never influences either estimate.
    """
    if depth.ndim != 2 or flow_px.shape != (2, *depth.shape) or K.shape != (3, 3):
        raise ValueError("Invalid geometry tensor shape")
    if static_mask.shape != depth.shape or xi_legacy.shape != (6,) or xi_geo.shape != (6,):
        raise ValueError("Invalid mask/twist shape")
    if int(holdout_stride) < 1:
        raise ValueError("holdout_stride must be >=1")
    from utils.camera_motion_v31 import interaction_rows

    valid = (static_mask.bool() & torch.isfinite(depth) & (depth > 0)
             & torch.isfinite(flow_px).all(0))
    y, x = torch.nonzero(valid, as_tuple=True)
    result = {"n_common": int(y.numel()), "n_holdout": 0}
    for name in ("legacy", "geo"):
        for subset in ("common", "holdout"):
            for metric in ("median_px", "p90_px", "mean_px", "rmse_px"):
                result[f"{name}_{subset}_{metric}"] = float("nan")
    if y.numel() == 0:
        return result
    L = interaction_rows(depth, K, y, x).to(torch.float64)
    f = flow_px[:, y, x].T.to(torch.float64)
    hold = ((y % int(holdout_stride) != 0) | (x % int(holdout_stride) != 0))
    result["n_holdout"] = int(hold.sum().item())
    for name, xi in (("legacy", xi_legacy), ("geo", xi_geo)):
        pred = torch.einsum("nij,j->ni", L, xi.to(torch.float64))
        residual = torch.linalg.vector_norm(f - pred, dim=1)
        for subset, arr in (("common", residual), ("holdout", residual[hold])):
            if arr.numel():
                result[f"{name}_{subset}_median_px"] = float(arr.median().item())
                result[f"{name}_{subset}_p90_px"] = float(torch.quantile(arr, .9).item())
                result[f"{name}_{subset}_mean_px"] = float(arr.mean().item())
                result[f"{name}_{subset}_rmse_px"] = float(torch.mean(arr.square()).sqrt().item())
    return result


@torch.no_grad()
def relative_pose_metrics(T_prev, T_next, GT_prev, GT_next):
    """Right-increment translation/rotation disagreement (not global ATE)."""
    matrices = (T_prev, T_next, GT_prev, GT_next)
    if any(m.shape != (4, 4) for m in matrices):
        raise ValueError("Expected 4x4 poses")
    Tprev, Tnext, Gprev, Gnext = (m.double() for m in matrices)
    if not all(bool(torch.isfinite(m).all()) for m in (Tprev, Tnext, Gprev, Gnext)):
        return float("nan"), float("nan")
    A = torch.linalg.solve(Tprev, Tnext)
    B = torch.linalg.solve(Gprev, Gnext)
    te = float(torch.linalg.vector_norm(A[:3, 3] - B[:3, 3]).item())
    r = A[:3, :3] @ B[:3, :3].T
    U, _, Vh = torch.linalg.svd(r)
    correction = torch.eye(3, device=r.device, dtype=r.dtype)
    correction[-1, -1] = torch.det(U @ Vh).sign()
    r = U @ correction @ Vh
    cosine = torch.clamp((torch.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    return te, float(torch.rad2deg(torch.acos(cosine)).item())
