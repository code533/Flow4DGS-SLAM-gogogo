"""V4.3a read-only, pre-optimization RGB-D hypothesis comparison.

Evaluate G0, G3-Support and G4 using the SAME Gaussian map, target camera,
RGB-D observation and prior keyframe offset. Render from temporary offset
TENSORS: never touch GaussianModel._delta_xyz, the optimizer or poses.

This is a hypothesis consistency diagnostic on the *selection frame*;
it is NOT independent reconstruction validation or online motion selection.
"""
import csv
from pathlib import Path

import numpy as np
import torch


HYPOTHESES = ("g0", "g3_support", "g4")


@torch.no_grad()
def compose_offsets(prev_delta, point_indices, displacement, max_motion=.3):
    """Exact update_delta_xyz semantics without writing model state.

    For norm(displacement) < max_motion, add displacement to previous
    keyframe offset; otherwise keep previous offset unchanged. The strict
    < and dynamic-Gaussian index domain match GaussianModel.update_delta_xyz.
    """
    if (prev_delta.ndim != 2 or prev_delta.shape[1] != 3
            or displacement.ndim != 2 or displacement.shape[1] != 3
            or point_indices.ndim != 1
            or displacement.shape[0] != point_indices.shape[0]
            or max_motion <= 0):
        raise ValueError("Malformed offsets/point indices or invalid motion gate")
    if point_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("Gaussian index tensor must be integer")
    if prev_delta.device != displacement.device or prev_delta.device != point_indices.device:
        raise ValueError("Offset and index device mismatch")
    if not torch.isfinite(prev_delta).all() or not torch.isfinite(displacement).all():
        raise FloatingPointError("Nonfinite Gaussian offset")
    if point_indices.numel() and ((point_indices.min() < 0) or (point_indices.max() >= len(prev_delta))):
        raise IndexError("Dynamic Gaussian index outside previous offset domain")
    if point_indices.unique().numel() != point_indices.numel():
        raise ValueError("Repeated Gaussian indices would make candidates ambiguous")
    passed = torch.linalg.vector_norm(displacement, dim=1) < max_motion
    result = prev_delta.detach().clone()
    result[point_indices[passed]] += displacement.detach()[passed]
    return result, passed


def _safe_mean(data, mask):
    count = int(mask.sum().item())
    if not count:
        return float("nan"), 0
    return float(data[mask].mean().item()), count


@torch.no_grad()
def score_rgbd(image, depth, opacity, gt_rgb, gt_depth, dynamic_region,
               *, opacity_threshold=.1, uncovered_penalty_m=.5):
    """Candidate-independent GT masks; report coverage separately.

    dynamic_region True means motion, never map confidence. RGB is assessed
    even at uncovered pixels. Depth fixed-mask MAE charges an explicitly
    configured penalty for no predicted depth/low opacity, so a hypothesis
    cannot win simply by hiding its geometry.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected candidate RGB [3,H,W]")
    h,w=image.shape[1:]
    if (gt_rgb.shape != image.shape or
            gt_depth.squeeze().shape != (h,w) or
            depth.squeeze().shape != (h,w) or
            opacity.squeeze().shape != (h,w) or
            dynamic_region.squeeze().shape != (h,w)):
        raise ValueError("RGB-D/mask shapes differ")
    if not 0 <= opacity_threshold <= 1 or uncovered_penalty_m <= 0:
        raise ValueError("Opacity and missing depth penalty invalid")

    rgb_observed = torch.isfinite(gt_rgb).all(0)
    gt_rgb = torch.nan_to_num(gt_rgb.detach().float(),nan=0.,posinf=0.,neginf=0.)
    image = torch.nan_to_num(image.detach().float(),nan=0.,posinf=0.,neginf=0.).clamp(0,1)
    depth = depth.detach().squeeze().float()
    opacity = opacity.detach().squeeze().float()
    gt_depth = gt_depth.detach().squeeze().float()
    dynamic = dynamic_region.detach().squeeze().bool()
    static = ~dynamic

    pixel_rgb = (image - gt_rgb).abs().mean(0)
    depth_obs = torch.isfinite(gt_depth) & (gt_depth > .01)
    predicted = (torch.isfinite(depth) & (depth > 0) &
                 torch.isfinite(opacity) & (opacity >= opacity_threshold))
    pix_depth = torch.where(predicted & depth_obs,
                            (torch.nan_to_num(depth,nan=0.,posinf=0.,neginf=0.) -
                             torch.nan_to_num(gt_depth,nan=0.,posinf=0.,neginf=0.)).abs(),
                            torch.full_like(depth,uncovered_penalty_m))
    out = {}
    for label,region in (("all",torch.ones_like(dynamic)),("dynamic",dynamic),("static",static)):
        rm=region & rgb_observed
        dm=region & depth_obs
        mean_rgb,n_rgb=_safe_mean(pixel_rgb,rm)
        depth_fixed,n_depth=_safe_mean(pix_depth,dm)
        depth_on_pred,_=_safe_mean(pix_depth, dm & predicted)
        coverage,_=_safe_mean(predicted.float(),dm)
        out.update({
            label+"_rgb_mae":mean_rgb,
            label+"_rgb_n":n_rgb,
            label+"_depth_penalized_mae_m":depth_fixed,
            label+"_depth_pred_valid_mae_m":depth_on_pred,
            label+"_depth_n":n_depth,
            label+"_depth_coverage":coverage,
        })
    return out, {
        "rgb_abs_mean":pixel_rgb.detach(),
        "depth_penalized_abs":pix_depth.detach(),
        "prediction_covered":predicted.detach(),
        "dynamic_region":dynamic.detach(),
        "gt_depth_valid":depth_obs.detach(),
        "gt_rgb_valid":rgb_observed.detach(),
    }


@torch.no_grad()
def evaluate_hypotheses(
        *, frame_id, pc, viewpoint, prior_keyframe_id, prev_delta,
        selected_indices, candidates, render_fn, pipeline, background,
        max_motion=.3, opacity_threshold=.1, uncovered_penalty_m=.5,
        save_maps=False):
    """Always render at the current target frame time, prior to Mapping.

    render_fn is injected for tests, normally
    gaussian_splatting.gaussian_renderer.render. All candidates share the
    exact target viewpoint object and same Gaussian model object.
    """
    if tuple(candidates) != HYPOTHESES and set(candidates) != set(HYPOTHESES):
        raise ValueError("Must supply exactly G0, G3-Support and G4")
    if pc.init_deform != "offset" or viewpoint.uid != frame_id:
        raise ValueError("Offset model and matching target camera required")
    if prev_delta.shape[0] != int(pc.dygs.sum().item()):
        raise ValueError("Prior offsets not aligned with dynamic Gaussian count")
    if len(pc._delta_xyz) < 1:
        raise ValueError("No prior dynamic offset")
    original = prev_delta.detach().clone()
    gt_rgb = viewpoint.original_image.to(device=prev_delta.device, dtype=prev_delta.dtype)
    gt_depth = torch.as_tensor(viewpoint.depth,device=prev_delta.device,dtype=prev_delta.dtype)
    # In Flow4DGS-SLAM motion_mask True denotes static pixels.
    dynamic_region = ~viewpoint.motion_mask.to(device=prev_delta.device).bool()
    rows=[]
    maps = {}
    for name in HYPOTHESES:
        offset,gate = compose_offsets(original,selected_indices,candidates[name],
                                       max_motion=max_motion)
        # render accepts all dynamic offsets (not a sparse selected subset).
        # Keep GMM time, pose, attributes, scaling/rotation and background
        # fixed across candidates. No optimization or parameter mutation.
        pkg=render_fn(
            viewpoint,pc,pipeline,background,dynamic=False,
            dx=offset,ds=0,dr=0,uid=frame_id)
        if pkg is None:
            raise RuntimeError("Renderer returned no image")
        measures,errmaps=score_rgbd(
            pkg["render"],pkg["depth"],pkg["opacity"],
            gt_rgb,gt_depth,dynamic_region,
            opacity_threshold=opacity_threshold,
            uncovered_penalty_m=uncovered_penalty_m)
        rows.append({
            "keyframe_id":int(frame_id),
            "prior_keyframe_id":int(prior_keyframe_id),
            "hypothesis":name,
            "n_dynamic_gaussians":int(original.shape[0]),
            "n_selected":int(selected_indices.numel()),
            "gate_pass_n":int(gate.sum().item()),
            "gate_rejected_n":int((~gate).sum().item()),
            "changed_from_prior_n":int((offset!=original).any(-1).sum().item()),
            "opacity_threshold":float(opacity_threshold),
            "uncovered_penalty_m":float(uncovered_penalty_m),
            **measures
        })
        if save_maps:
            maps[name] = {
                **{k:v.detach().cpu().numpy() for k,v in errmaps.items()},
                "rendered_rgb":pkg["render"].detach().cpu().numpy(),
                "rendered_depth":pkg["depth"].detach().cpu().numpy(),
                "rendered_opacity":pkg["opacity"].detach().cpu().numpy(),
            }
    # Read-only invariant: at least the provided offsets must be unchanged.
    if not torch.equal(prev_delta,original):
        raise RuntimeError("Prior Gaussian offsets mutated during evaluation")
    return rows,maps


def save_evaluation(folder,rows,maps=None):
    """Append one row per hypothesis; optional detailed, compressed images."""
    path=Path(folder)
    path.mkdir(parents=True,exist_ok=True)
    csvpath=path/"v43a_per_keyframe.csv"
    new=not csvpath.exists() or csvpath.stat().st_size==0
    with csvpath.open("a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        if new:writer.writeheader()
        writer.writerows(rows)
    if maps:
        fid=int(rows[0]["keyframe_id"])
        tensors={}
        for h,items in maps.items():
            for key,value in items.items():
                tensors[h+"__"+key]=np.asarray(value)
        np.savez_compressed(path/("v43a_%05d.npz"%fid),**tensors)
