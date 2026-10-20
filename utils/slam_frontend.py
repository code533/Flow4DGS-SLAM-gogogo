import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render, get_dynamic_mask, render_flow
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth, get_loss_network, pearson_loss
from utils.uncertainty_fusion import compute_rgbd_uncertainty  # UP-FLOW-RGBD-V1
from utils.flow_rgbd_fusion_v2 import compute_flow_rgbd_gate_v2, tracking_loss_v2  # UP-FLOW-RGBD-V2
from utils.flow_rgbd_fusion_v21 import compute_flow_rgbd_gate_v21  # UP-FLOW-RGBD-V21
from utils.flow_rgbd_fusion_v22 import compute_decoupled_weights, tracking_loss_decoupled  # UP-FLOW-RGBD-V22
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
import os
import matplotlib.pyplot as plt
import random
from utils.slam_backend import segment_motion_parametric, fit_twist_weighted, pixels_to_flow_units, flow_to_pixels
import torch.nn.functional as F
from utils.pose_utils import SE3_exp, scale_se3_step, SO3_exp, so3_log
from utils.camera_motion_v30 import translation_step, condition_number_normalized_jacobian, rigid_transform  # CIMD-V30


def flow_to_pixels(flow_norm, H, W, mode='grid'):
    if mode == 'pixel':
        return flow_norm
    if mode == 'grid':
        sx, sy = (W - 1) / 2.0, (H - 1) / 2.0
    elif mode == 'ratio':
        sx, sy = float(W), float(H)
    else:
        raise ValueError("flow_mode must be 'grid' | 'ratio' | 'pixel'")
    scale = torch.tensor([sx, sy], device=flow_norm.device, dtype=flow_norm.dtype)[:, None, None]
    return flow_norm * scale

def build_interaction_matrix(depth, K):
    H, W = depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    y, x = torch.meshgrid(torch.arange(H, device=depth.device, dtype=depth.dtype),
                          torch.arange(W, device=depth.device, dtype=depth.dtype), indexing='ij')
    u, v = x, y
    Z = depth.clamp_min(1e-6)
    du_dt = torch.stack([-fx/Z, torch.zeros_like(Z), (u-cx)/Z], dim=-1)
    dv_dt = torch.stack([ torch.zeros_like(Z), -fy/Z, (v-cy)/Z], dim=-1)
    du_dw = torch.stack([ (u-cx)*(v-cy)/fy, -(fx + (u-cx)*(u-cx)/fx), (v-cy) ], dim=-1)
    dv_dw = torch.stack([ (fy + (v-cy)*(v-cy)/fy), -(u-cx)*(v-cy)/fx, -(u-cx) ], dim=-1)
    L_top = torch.cat([du_dt, du_dw], dim=-1)
    L_bot = torch.cat([dv_dt, dv_dw], dim=-1)
    L = torch.stack([L_top, L_bot], dim=2).reshape(-1, 6)  # [2HW,6]
    return L

def predict_rigid_flow_px(depth, K, xi):
    H, W = depth.shape
    L = build_interaction_matrix(depth, K)                # [2HW,6]
    fhat = (L @ xi).reshape(H, W, 2).permute(2,0,1)       # [2,H,W] in pixels
    return fhat

def robust_median_residual(flow_meas_px, rigid_flow_px, valid_mask):
    """Median L2 residual over valid pixels."""
    resid = (flow_meas_px - rigid_flow_px).pow(2).sum(0).sqrt()
    rv = resid[valid_mask]
    if rv.numel() == 0:
        return torch.tensor(float('inf'), device=flow_meas_px.device)
    return rv.median()

def warp_mask_from_tminus1(flow_t_to_tm1, mask_tm1, mode='bilinear', padding_mode='zeros'):
    """
    Backward-warp mask at t-1 into frame t using flow from t -> t-1.

    Args:
        flow_t_to_tm1: Tensor of shape [H, W, 2] or [B, H, W, 2], pixel units (u,v).
                       u is x-displacement (cols), v is y-displacement (rows).
        mask_tm1:     Tensor of shape [H, W], [1, H, W], or [B, 1, H, W].
                      Values will be sampled from this (e.g., {0,1} motion mask at t-1).
        mode:         'bilinear' (default) or 'nearest' for sampling.
        padding_mode: 'zeros' | 'border' | 'reflection' (what to do when sampling outside).
    Returns:
        mask_t: Tensor of shape [H, W] (if inputs unbatched) or [B, 1, H, W] (if batched).
    """
    # --- Normalize to batched shapes ---
    batched = flow_t_to_tm1.dim() == 4
    if not batched:
        flow = flow_t_to_tm1.unsqueeze(0)       # [1,H,W,2]
    else:
        flow = flow_t_to_tm1                     # [B,H,W,2]

    if mask_tm1.dim() == 2:
        mask = mask_tm1.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif mask_tm1.dim() == 3:  # [1,H,W] -> [1,1,H,W]
        mask = mask_tm1.unsqueeze(0) if mask_tm1.size(0) != 1 else mask_tm1.unsqueeze(1)
        if mask.dim() != 4:
            mask = mask_tm1.unsqueeze(0)
    else:
        mask = mask_tm1  # assume [B,1,H,W]

    B, H, W = flow.size(0), flow.size(1), flow.size(2)
    device = flow.device
    dtype = flow.dtype

    # --- Build base grid in normalized coords [-1,1] ---
    # grid_sample expects grid[..., 0] = x in [-1,1], grid[..., 1] = y in [-1,1]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    base_x = (xx / (W - 1)) * 2 - 1  # [H,W]
    base_y = (yy / (H - 1)) * 2 - 1  # [H,W]
    base_grid = torch.stack([base_x, base_y], dim=-1).unsqueeze(0).expand(B, H, W, 2)  # [B,H,W,2]

    # --- Convert pixel flow to normalized offsets and add to base grid ---
    # flow[...,0] = u (x), flow[...,1] = v (y) in pixels
    norm_flow = torch.empty_like(flow)
    norm_flow[..., 0] = flow[..., 0] * (2.0 / max(W - 1, 1))
    norm_flow[..., 1] = flow[..., 1] * (2.0 / max(H - 1, 1))
    grid = base_grid + norm_flow  # sampling locations in frame t-1 for each pixel in frame t

    # Ensure mask has batch & channel
    if mask.dim() == 3:  # [B,H,W] -> [B,1,H,W]
        mask = mask.unsqueeze(1)
    if mask.size(0) != B:
        # If a single mask is provided but multiple flows, broadcast it
        if mask.size(0) == 1:
            mask = mask.expand(B, -1, -1, -1)
        else:
            raise ValueError("Batch size of mask and flow must match, or mask batch must be 1.")

    # --- Sample: this performs the backward warp ---
    mask_t = F.grid_sample(
        mask, grid, mode=mode, align_corners=True, padding_mode=padding_mode
    )  # [B,1,H,W]

    # Squeeze if inputs were unbatched
    if not batched:
        return mask_t[0, 0]  # [H,W]
    return mask_t  # [B,1,H,W]

def _skew(omega):
    wx, wy, wz = omega[...,0], omega[...,1], omega[...,2]
    O = torch.zeros((*omega.shape[:-1], 3, 3), device=omega.device, dtype=omega.dtype)
    O[..., 0,1] = -wz; O[..., 0,2] =  wy
    O[..., 1,0] =  wz; O[..., 1,2] = -wx
    O[..., 2,0] = -wy; O[..., 2,1] =  wx
    return O




def vis_render_process(gaussians, pipeline_params, background, viewpoint, cur_frame_idx, save_dir, out_dir="track", mask=None):
    with torch.no_grad():
        render_pkg = render(
            viewpoint, gaussians, pipeline_params, background, mask=mask)
        viz_im = torch.clip(render_pkg["render"].permute(1, 2, 0).detach().cpu(), 0, 1)
        
        fig, ax = plt.subplots(figsize=(8, 8))  # the size of the figure
        cax = ax.imshow(viz_im)
        ax.axis('off')
        # save the figure
        os.makedirs(save_dir, exist_ok=True)
        process_dir = os.path.join(save_dir, out_dir)
        os.makedirs(process_dir, exist_ok=True)
        save_path = os.path.join(process_dir, f"{cur_frame_idx}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        return
        

        

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        self.dynamic_model = config["model_params"].get("dynamic_model", False)
        self.insertion = config["model_params"].get("insertion", False)
        self.flow_mask = config["model_params"].get("flow_mask", False)
        self.flow_cam = config["model_params"].get("flow_cam", False)
        self.flow_cam_thres = config["model_params"].get("flow_cam_thres", 5.0)
        self.flow_cam_maxt = config["model_params"].get("flow_cam_maxt", 0.01)
        self.flow_cam_maxr = config["model_params"].get("flow_cam_maxr", 0.1)
        self.rgbd_uncertainty_cfg = config["model_params"].get("rgbd_uncertainty", {})  # UP-FLOW-RGBD-V1
        self.rgbd_v2_cfg = config["model_params"].get("flow_rgbd_v2", {})  # UP-FLOW-RGBD-V2
        self.rgbd_v21_cfg = config["model_params"].get("flow_rgbd_v21", {})  # UP-FLOW-RGBD-V21
        self.rgbd_v22_cfg = config["model_params"].get("flow_rgbd_v22", {})  # UP-FLOW-RGBD-V22
        self.flow_cam_v30_cfg = config["model_params"].get("flow_cam_v30", {})  # CIMD-V30

        self.dynamic_objects = 0

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0 
            return initial_depth.cpu().numpy()[0]
        
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0
        if self.dynamic_model:
            initial_depth = initial_depth.detach().clone()  
            if (not self.insertion) or init:
                initial_depth[0][~viewpoint.motion_mask.cpu().numpy()] = 0
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        print('SETTING GT POSE DURING INITIALIZATION')

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

            

    def tracking(self, cur_frame_idx, viewpoint, last_keyframe_idx):
        start_time = time.time()
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        
        viewpoint.update_RT(prev.R, prev.T)
        
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )
        
        pose_optimizer = torch.optim.Adam(opt_params, capturable=True)
        
        with torch.no_grad():
            dxyz, d_rot, d_scale = None, None, None
        
        output_dir = os.path.join(self.config["Results"]["save_dir"], "tracking")
        os.makedirs(output_dir, exist_ok=True)
        initial_motion_mask = viewpoint.motion_mask.clone()
        flow_residual_px_v2 = None  # V2 uses flow residual from the unmodified motion decomposition
        # CIMD-V30: diagnostics and optional translation clamp are isolated
        # from V1-V2.2 tracking-weight experiments.
        v30cfg = self.flow_cam_v30_cfg
        v30on = bool(v30cfg.get("enabled", False))
        if v30on and not (self.flow_mask and self.flow_cam):
            raise RuntimeError("CIMD-V30 requires both flow_mask and flow_cam enabled")
        v30mode = str(v30cfg.get("translation_mode", "legacy"))
        if v30on and v30mode not in ("legacy", "capped"):
            raise ValueError("CIMD-V30 translation_mode must be legacy or capped")
        v30_stride = max(1, int(v30cfg.get("debug_every", 10)))
        v30log = (v30on and bool(v30cfg.get("save_debug", False)) and
                  viewpoint.uid > 0 and viewpoint.uid % v30_stride == 0)
        v30data = None

        if self.flow_mask:
            if self.save_results:
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, mask=(self.gaussians.dygs==False))

            flow_back = viewpoint.generate_flow(viewpoint.original_image.cuda(), prev.original_image.cuda(), 
                                            tracking=True)
            
            depth_tensor = torch.from_numpy(viewpoint.depth).cuda().squeeze().type(torch.float32)
            H, W = depth_tensor.shape
            results = segment_motion_parametric(
                                    depth_tensor, 
                                    flow_back.permute(2,0,1), 
                                    viewpoint.intrinsic.cuda(),
                                    flow_mode='grid',
                                    ds=1,
                                    robust_iters=30,
                                    k_mad=self.flow_cam_thres,
                                    motion_mask=viewpoint.motion_mask.cuda()
                                )
            flow_mask = results['mask_bool']
            flow_residual_px_v2 = results['resid_px'].detach()  # UP-FLOW-RGBD-V2
            
            viewpoint.motion_mask = viewpoint.motion_mask & (~flow_mask)

            rigid_flow = results['rigid_flow_out'].permute(1,2,0)

            if self.save_results:
                value_np = results['resid_px'].detach().cpu().numpy()

                plt.figure(figsize=(6, 4))
                plt.imshow(value_np, cmap='viridis', origin='upper')
                plt.colorbar(label='Value')
                plt.xlabel('W')
                plt.ylabel('H')
                plt.title('Value Map Heatmap')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"tracking_{viewpoint.uid}_resid_heatmap.png"))


            if self.flow_cam:

                xi = results['xi']                                # [6], on CUDA/float32 already
                if v30log:
                    v30_xi_initial = xi.detach().clone()

                depth_valid = (depth_tensor > 0) & torch.isfinite(depth_tensor)

                static_prior = initial_motion_mask.cuda().bool() if 'initial_motion_mask' in locals() else torch.ones_like(depth_valid, dtype=torch.bool)

                static_from_residual = ~results['mask_bool']
                static_refit_mask = depth_valid & static_prior & static_from_residual  
                valid_extent = static_refit_mask.sum().item() / static_refit_mask.numel()
                static_inliers_ds = F.interpolate(static_refit_mask[None,None].float(), size=(H,W), mode='nearest').squeeze().bool()
                
                flow_px = flow_to_pixels(flow_back.permute(2,0,1), H, W, mode='grid')
                depth_valid = (depth_tensor > 0) & torch.isfinite(depth_tensor)
                depth_ds, depth_valid_ds, flow_px_ds, Kds = depth_tensor, depth_valid, flow_px, viewpoint.intrinsic.cuda()
                
                xi = fit_twist_weighted(depth_ds, flow_px_ds, Kds, static_inliers_ds, robust=True, iters=30)
                if v30log:
                    v30_xi_refined = xi.detach().clone()

                rigid_flow_px = predict_rigid_flow_px(depth_ds, Kds, xi)
                rigid_flow_out = pixels_to_flow_units(rigid_flow_px, H, W, mode='grid')

                rigid_flow_out[:, ~static_inliers_ds] = 0.0

                rigid_flow_out = rigid_flow_out.permute(1,2,0)
                
            
                T_rel = SE3_exp(xi)                        
                R_prev = prev.R.cuda().float()
                t_prev = prev.T.cuda().float()
                T_prev = torch.eye(4, device=R_prev.device, dtype=R_prev.dtype)
                T_prev[:3,:3] = R_prev
                T_prev[:3,  3] = t_prev

                T_new = T_prev @ T_rel                         
                if v30log:
                    v30_raw_T = T_new.detach().clone()

                T_curr = T_new

                max_translation = self.flow_cam_maxt * valid_extent  
                max_rot_deg = self.flow_cam_maxr * valid_extent      
                max_rot = torch.deg2rad(torch.tensor(max_rot_deg, device=T_curr.device))

                delta_t = T_curr[:3, 3] - T_prev[:3, 3]
                delta_R = T_curr[:3, :3] @ T_prev[:3, :3].T

                trace = torch.clamp((torch.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0)
                angle = torch.acos(trace)

                trans_norm = torch.norm(delta_t)
                s_t = max_translation / (trans_norm + 1e-10)

                s_r = max_rot / (angle + 1e-10)

                s = torch.clamp(torch.minimum(s_t, s_r), max=1.0)

                # CIMD-V30: preserve bitwise legacy behavior when disabled.
                if v30on:
                    delta_t_scaled, v30_ratio, v30_scale = translation_step(
                        delta_t, max_translation, mode=v30mode)
                else:
                    delta_t_scaled = s_t * delta_t

                w = so3_log(delta_R)
                delta_R_scaled = SO3_exp(s * w) 

                T_curr[:3, :3] = delta_R_scaled @ T_prev[:3, :3]
                T_curr[:3, 3]  = T_prev[:3, 3] + delta_t_scaled

                viewpoint.R = T_curr[:3,:3]
                viewpoint.T = T_curr[:3, 3]
                if v30log:
                    with torch.no_grad():
                        resid0 = results["resid_px"].detach()
                        resid1 = (flow_px - rigid_flow_px).pow(2).sum(dim=0).sqrt()
                        static_resid0 = resid0[static_refit_mask]
                        static_resid1 = resid1[static_refit_mask]
                        cond30, cond_n30 = condition_number_normalized_jacobian(
                            depth_tensor, viewpoint.intrinsic.cuda(), static_refit_mask,
                            stride=max(1, int(v30cfg.get("condition_stride", 8))))
                        def robust_median(t):
                            return float(t.median().item()) if t.numel() else float("nan")
                        v30data = dict(
                            xi_initial=v30_xi_initial.detach().cpu().numpy(),
                            xi_refined=v30_xi_refined.detach().cpu().numpy(),
                            T_previous=rigid_transform(prev.R, prev.T).detach().cpu().numpy(),
                            T_raw=v30_raw_T.cpu().numpy(),
                            T_tracking_initial=rigid_transform(viewpoint.R, viewpoint.T).cpu().numpy(),
                            translation_raw=float(trans_norm.item()),
                            translation_applied=float(delta_t_scaled.norm().item()),
                            translation_max=float(max_translation),
                            scale_translation_unclamped=float(s_t.item()),
                            scale_translation_applied=float(v30_scale.item()),
                            rotation_raw_rad=float(angle.item()),
                            rotation_applied_rad=float((s * w).norm().item()),
                            rotation_max_rad=float(max_rot.item()),
                            scale_rotation_joint=float(s.item()),
                            static_inliers=int(static_refit_mask.sum().item()),
                            static_fraction=float(valid_extent),
                            residual_first_median=robust_median(static_resid0),
                            residual_second_median=robust_median(static_resid1),
                            condition_normalized=float(cond30),
                            condition_sample_count=int(cond_n30),
                            translation_mode=v30mode,
                        )
                        if bool(v30cfg.get("save_residual_maps", False)):
                            v30data.update({
                                "residual_first_px": resid0.cpu().numpy().astype(np.float32),
                                "residual_second_px": resid1.cpu().numpy().astype(np.float32),
                                "static_inlier_mask": static_refit_mask.cpu().numpy(),
                            })



            # print('flow mask tracking time:', time.time() - start_time)
            if self.save_results:# and (cur_frame_idx-last_keyframe_idx==5):
                fig, axes = plt.subplots(1, 4, figsize=(15, 5))
                axes[0].imshow(viewpoint.original_image.permute(1, 2, 0).cpu().numpy())
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                axes[1].imshow(~initial_motion_mask.cpu().numpy(), cmap='gray')
                axes[1].set_title("Initial Motion Mask")
                axes[1].axis("off")

                axes[2].imshow(flow_mask.cpu().numpy(), cmap='gray')
                axes[2].set_title("Refined Flow-based Motion Mask")
                axes[2].axis("off")

                axes[3].imshow((~viewpoint.motion_mask).cpu().numpy(), cmap='gray')
                axes[3].set_title("Final Motion Mask")
                axes[3].axis("off")

                output_dir = os.path.join(self.config["Results"]["save_dir"], "tracking")
                os.makedirs(output_dir, exist_ok=True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'tracking_{viewpoint.uid}_flowmask.png'), dpi=150)
                plt.close()

        loss_tracking_init = 0.0
        rgbd_gate = None  # UP-FLOW-RGBD-V1; never written to viewpoint.motion_mask
        v2_on = bool(self.rgbd_v2_cfg.get("enabled", False))
        v21_on = bool(self.rgbd_v21_cfg.get("enabled", False))
        v22_on = bool(self.rgbd_v22_cfg.get("enabled", False))
        v1_on = bool(self.rgbd_uncertainty_cfg.get("enabled", False))
        if sum([v1_on, v2_on, v21_on, v22_on]) > 1:
            raise RuntimeError("Enable only one of RGBD V1, Flow-RGBD V2, V2.1, V2.2")
        if (v2_on or v21_on or v22_on) and self.monocular:
            raise RuntimeError("Flow-RGBD fusion requires RGB-D input")
        v2_gate = None
        v2_update_count = 0
        v2_last_iteration = -1
        v21_gate = None
        v21_update_count = 0
        v21_last_iteration = -1
        v22_gate = None
        v22_update_count = 0
        v22_last_iteration = -1

        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, mask=(self.gaussians.dygs==False),)
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            
            # UP-FLOW-RGBD-V1: one detached RGB-D reliability map per tracked frame.
            # The existing render contains only STATIC Gaussians. Keep the original
            # semantic/flow hard mask and camera-motion initialization unchanged.
            uc = self.rgbd_uncertainty_cfg
            if (uc.get("enabled", False) and viewpoint.uid > 0
                    and tracking_itr == int(uc.get("warmup_iters", 3))):
                with torch.no_grad():
                    pred_rgb = torch.exp(viewpoint.exposure_a.detach()) * image.detach() + viewpoint.exposure_b.detach()
                    depth_gt = torch.from_numpy(viewpoint.depth).to(device=depth.device, dtype=depth.dtype)
                    rgbd_gate = compute_rgbd_uncertainty(
                        pred_rgb, depth, opacity, viewpoint.original_image.to(image.device),
                        depth_gt, viewpoint.motion_mask.to(image.device), uc)
                    if uc.get("save_debug", False):
                        stride = max(int(uc.get("debug_every", 30)), 1)
                        if viewpoint.uid % stride == 0:
                            np.savez_compressed(
                                os.path.join(output_dir, "rgbd_uncertainty_{:05d}.npz".format(viewpoint.uid)),
                                uncertainty=rgbd_gate["uncertainty"].cpu().numpy(),
                                weights=rgbd_gate["weights"].cpu().numpy(),
                                valid=rgbd_gate["valid"].cpu().numpy(),
                                dynamic_candidate=rgbd_gate["new_dynamic_candidate"].cpu().numpy(),
                                original_static=viewpoint.motion_mask.cpu().numpy(),
                                coverage=rgbd_gate["coverage"],
                                applied=rgbd_gate["applied"],
                            )

            # UP-FLOW-RGBD-V2: alternate pose refinement and fixed reliability weights.
            # Do NOT overwrite viewpoint.motion_mask: Mapping still receives original mask.
            v2c = self.rgbd_v2_cfg
            warm = max(0, int(v2c.get("warmup_iters", 6)))
            interval = max(1, int(v2c.get("refresh_interval", 8)))
            if (v2_on and viewpoint.uid > 0 and tracking_itr >= warm
                    and (tracking_itr - warm) % interval == 0):
                with torch.no_grad():
                    corrected_rgb = (torch.exp(viewpoint.exposure_a.detach()) * image.detach()
                                     + viewpoint.exposure_b.detach())
                    observed_depth = torch.as_tensor(viewpoint.depth, device=depth.device, dtype=depth.dtype)
                    v2_gate = compute_flow_rgbd_gate_v2(
                        corrected_rgb, depth, opacity, viewpoint.original_image.to(image.device),
                        observed_depth, viewpoint.motion_mask.to(image.device), flow_residual_px_v2, v2c)
                v2_update_count += 1
                v2_last_iteration = tracking_itr

            # UP-FLOW-RGBD-V21: update both branch scores after short pose warmup.
            # Scores/weights stay fixed between refreshes and are detached.
            v21c = self.rgbd_v21_cfg
            warm21 = max(0, int(v21c.get("warmup_iters", 6)))
            interval21 = max(1, int(v21c.get("refresh_interval", 8)))
            if (v21_on and viewpoint.uid > 0 and tracking_itr >= warm21
                    and (tracking_itr - warm21) % interval21 == 0):
                with torch.no_grad():
                    corrected21 = (torch.exp(viewpoint.exposure_a.detach()) * image.detach()
                                   + viewpoint.exposure_b.detach())
                    v21_gate = compute_flow_rgbd_gate_v21(
                        corrected21, depth, opacity, viewpoint.original_image.to(image.device),
                        torch.as_tensor(viewpoint.depth, device=depth.device, dtype=depth.dtype),
                        viewpoint.motion_mask.to(image.device), flow_residual_px_v2, v21c)
                v21_update_count += 1
                v21_last_iteration = tracking_itr

            # UP-FLOW-RGBD-V22: extract detached V2.1 cues then split
            # RGB and Depth reliabilities. Camera decomposition, masks and
            # 4D Mapping are deliberately unchanged.
            v22c = self.rgbd_v22_cfg
            v22warm = max(int(v22c.get("warmup_iters", 6)), 0)
            v22interval = max(int(v22c.get("refresh_interval", 8)), 1)
            if (v22_on and viewpoint.uid > 0 and tracking_itr >= v22warm
                    and (tracking_itr - v22warm) % v22interval == 0):
                with torch.no_grad():
                    corrected22 = (torch.exp(viewpoint.exposure_a.detach()) * image.detach()
                                   + viewpoint.exposure_b.detach())
                    # We only need the V2.1 cue maps, not its shared weight.
                    cue_cfg = dict(self.rgbd_v21_cfg)
                    cue_cfg.update(v22c)
                    cue_cfg["diagnostic_only"] = True
                    cues22 = compute_flow_rgbd_gate_v21(
                        corrected22, depth, opacity,
                        viewpoint.original_image.to(image.device),
                        torch.as_tensor(viewpoint.depth, device=depth.device, dtype=depth.dtype),
                        viewpoint.motion_mask.to(image.device), flow_residual_px_v2, cue_cfg)
                    v22_gate = compute_decoupled_weights(cues22, v22c)
                    v22_gate["cue_reason"] = cues22["reason"]
                v22_update_count += 1
                v22_last_iteration = tracking_itr

            # remove the dynamic object at frame 0
            with torch.no_grad():
                if self.dynamic_model and self.gaussians.deform_init and False:
                    mask = viewpoint.reproject_mask(self.dataset, self.cameras[last_keyframe_idx])
                else:
                    mask = None


            # Original flow/semantic hard mask remains active in
            # get_loss_tracking(); optional float mask adds soft pixel weights.
            if rgbd_gate is not None and rgbd_gate["applied"]:
                weights = rgbd_gate["weights"]
                mask = weights if mask is None else (weights * mask.float())

            if v22_on and self.rgbd_v22_cfg.get("mode", "decoupled") == "all_ones" and not self.rgbd_v22_cfg.get("diagnostic_only", False):
                # Strict control experiment: custom loss, BOTH weights == 1
                unit22 = torch.ones_like(opacity[0]).detach()
                loss_tracking = tracking_loss_decoupled(
                    self.config, image, depth, opacity, viewpoint, unit22, unit22,
                    normalization=self.rgbd_v22_cfg.get("normalization", "mass_preserving"))
            elif v22_on and v22_gate is not None and v22_gate["applied"]:
                loss_tracking = tracking_loss_decoupled(
                    self.config, image, depth, opacity, viewpoint,
                    v22_gate["rgb_weights"], v22_gate["depth_weights"],
                    normalization=self.rgbd_v22_cfg.get("normalization", "mass_preserving"))
            elif v21_on and v21_gate is not None and v21_gate["applied"]:
                loss_tracking = tracking_loss_v2(
                    self.config, image, depth, opacity, viewpoint, v21_gate["weights"])
            elif v2_on and v2_gate is not None and v2_gate["applied"]:
                # Preserve the original loss scale when weights are soft.
                loss_tracking = tracking_loss_v2(
                    self.config, image, depth, opacity, viewpoint, v2_gate["weights"])
            else:
                loss_tracking = get_loss_tracking(
                    self.config,
                    image,
                    depth,
                    opacity,
                    viewpoint,
                    rm_dynamic=True,
                    mask=mask,
                    save_img=False,
                )

            if tracking_itr == 0:
                loss_tracking_init = loss_tracking.item()


            

            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                pose_optimizer.zero_grad()
                if self.gaussians.init_deform == 'mlp':
                    self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                converged = update_pose(viewpoint)

            if converged:
                break

        # Diagnostic files also record early convergence / skipped updates.
        # This is independent of Results.save_results.
        if v2_on and self.rgbd_v2_cfg.get("save_debug", False) and viewpoint.uid > 0:
            debug_stride = max(int(self.rgbd_v2_cfg.get("debug_every", 30)), 1)
            if viewpoint.uid % debug_stride == 0:
                if v2_gate is None:
                    sh = viewpoint.motion_mask.shape
                    diag = {
                        "weights": torch.ones(sh, device=image.device),
                        "joint_score": torch.zeros(sh, device=image.device),
                        "candidate": torch.zeros(sh, device=image.device, dtype=torch.bool),
                        "valid": torch.zeros(sh, device=image.device, dtype=torch.bool),
                        "coverage": 0.0,
                        "effective_ratio": 0.0,
                        "applied": False,
                        "reason": "no_refresh_before_convergence",
                    }
                else:
                    diag = v2_gate
                np.savez_compressed(
                    os.path.join(output_dir, "flow_rgbd_v2_{:05d}.npz".format(viewpoint.uid)),
                    weights=diag["weights"].detach().cpu().numpy(),
                    joint_score=diag["joint_score"].detach().cpu().numpy(),
                    candidate=diag["candidate"].detach().cpu().numpy(),
                    valid=diag["valid"].detach().cpu().numpy(),
                    original_static=viewpoint.motion_mask.detach().cpu().numpy(),
                    applied=diag["applied"], reason=diag["reason"],
                    coverage=diag["coverage"], effective_ratio=diag["effective_ratio"],
                    update_count=v2_update_count, last_update_iteration=v2_last_iteration,
                )
                print("[Flow-RGBD-V2] frame={} applied={} reason={} coverage={:.3f} "
                      "downweight={:.3f} updates={}".format(
                          viewpoint.uid, diag["applied"], diag["reason"],
                          diag["coverage"], diag["effective_ratio"], v2_update_count))


        # UP-FLOW-RGBD-V21: diagnostics are independent of Results.save_results.
        # Even frames converging before warmup write an explicit fallback record.
        if v21_on and self.rgbd_v21_cfg.get("save_debug", False) and viewpoint.uid > 0:
            every = max(1, int(self.rgbd_v21_cfg.get("debug_every", 30)))
            if viewpoint.uid % every == 0:
                diag21 = v21_gate
                if diag21 is None:
                    with torch.no_grad():
                        corrected21 = (torch.exp(viewpoint.exposure_a.detach()) * image.detach()
                                       + viewpoint.exposure_b.detach())
                        diag21 = compute_flow_rgbd_gate_v21(
                            corrected21, depth, opacity,
                            viewpoint.original_image.to(image.device),
                            torch.as_tensor(viewpoint.depth, device=depth.device, dtype=depth.dtype),
                            viewpoint.motion_mask.to(image.device), flow_residual_px_v2,
                            self.rgbd_v21_cfg)
                    # Snapshot ONLY. This diagnostic did not affect Tracking.
                    diag21["reason"] = "no_refresh_before_convergence"
                    diag21["applied"] = False
                    diag21["weights"] = torch.ones_like(diag21["weights"])
                    diag21["effective_ratio"] = 0.0
                image21 = torch.clamp(viewpoint.original_image.detach().cpu(), 0, 1)
                debug_fields = {k: diag21[k].detach().cpu().numpy() for k in (
                    "weights", "valid", "candidate_raw", "candidate",
                    "original_static", "p_flow", "p_rgb", "p_front",
                    "p_map", "score_product", "score_layered", "score_selected")}
                debug_fields.update({
                    "rgb_uint8": (image21.mul(255).byte().permute(1, 2, 0).numpy()),
                    "reason": diag21["reason"], "applied": diag21["applied"],
                    "mode": diag21["mode"], "coverage": diag21["coverage"],
                    "effective_ratio": diag21["effective_ratio"],
                    "count_flow_prior": diag21["count_flow_prior"],
                    "count_map_prior": diag21["count_map_prior"],
                    "count_both_priors": diag21["count_both_priors"],
                    "count_raw": diag21["count_raw"],
                    "count_selected": diag21["count_selected"],
                    "update_count": v21_update_count,
                    "last_update_iteration": v21_last_iteration,
                })
                np.savez_compressed(
                    os.path.join(output_dir, "flow_rgbd_v21_{:05d}.npz".format(viewpoint.uid)),
                    **debug_fields)
                print("[Flow-RGBD-V2.1] frame={} mode={} reason={} valid_cov={:.3f} "
                      "cues(flow/map/both)={}/{}/{} candidates(raw/selected)={}/{} "
                      "downweight={:.6f} updates={}".format(
                          viewpoint.uid, diag21["mode"], diag21["reason"],
                          diag21["coverage"], diag21["count_flow_prior"],
                          diag21["count_map_prior"], diag21["count_both_priors"],
                          diag21["count_raw"], diag21["count_selected"],
                          diag21["effective_ratio"], v21_update_count))


        # UP-FLOW-RGBD-V22: explicit split RGB/Depth diagnostics, including
        # frames that converged before the first refresh.
        if v22_on and self.rgbd_v22_cfg.get("save_debug", False) and viewpoint.uid > 0:
            every22 = max(int(self.rgbd_v22_cfg.get("debug_every", 30)), 1)
            if viewpoint.uid % every22 == 0:
                d22 = v22_gate
                if d22 is None:
                    sh22 = viewpoint.motion_mask.shape
                    zeros22 = torch.zeros(sh22, device=image.device)
                    ones22 = torch.ones_like(zeros22)
                    d22 = {
                        "rgb_weights": ones22, "depth_weights": ones22,
                        "rgb_score": zeros22, "depth_score": zeros22,
                        "rgb_candidate": zeros22.bool(), "depth_candidate": zeros22.bool(),
                        "valid": zeros22.bool(), "coverage": 0.0,
                        "rgb_raw_count": 0, "depth_raw_count": 0,
                        "rgb_selected_count": 0, "depth_selected_count": 0,
                        "rgb_changed_fraction": 0.0, "depth_changed_fraction": 0.0,
                        "applied": False, "reason": "no_refresh_before_convergence",
                        "mode": self.rgbd_v22_cfg.get("mode", "decoupled"),
                        "cue_reason": "not_computed",
                    }
                save22 = {name: d22[name].detach().cpu().numpy() for name in (
                    "rgb_weights", "depth_weights", "rgb_score", "depth_score",
                    "rgb_candidate", "depth_candidate", "valid")}
                save22.update({
                    "original_static": viewpoint.motion_mask.detach().cpu().numpy(),
                    "mode": d22["mode"], "reason": d22["reason"],
                    "cue_reason": d22.get("cue_reason", "not_computed"),
                    "applied": d22["applied"], "coverage": d22["coverage"],
                    "rgb_raw_count": d22["rgb_raw_count"],
                    "depth_raw_count": d22["depth_raw_count"],
                    "rgb_selected_count": d22["rgb_selected_count"],
                    "depth_selected_count": d22["depth_selected_count"],
                    "rgb_changed_fraction": d22["rgb_changed_fraction"],
                    "depth_changed_fraction": d22["depth_changed_fraction"],
                    "update_count": v22_update_count,
                    "last_update_iteration": v22_last_iteration,
                })
                np.savez_compressed(
                    os.path.join(output_dir, "flow_rgbd_v22_{:05d}.npz".format(viewpoint.uid)),
                    **save22)
                print("[Flow-RGBD-V2.2] frame={} mode={} reason={} rgb={}/{} "
                      "depth={}/{} changed={:.6f}/{:.6f} updates={}".format(
                          viewpoint.uid, d22["mode"], d22["reason"],
                          d22["rgb_selected_count"], d22["rgb_raw_count"],
                          d22["depth_selected_count"], d22["depth_raw_count"],
                          d22["rgb_changed_fraction"], d22["depth_changed_fraction"],
                          v22_update_count))

        # CIMD-V30: save *after* Tracking for initial/optimized pose comparison.
        # The GT transform below is diagnostic ONLY; no optimizer reads it.
        if v30log:
            if v30data is None:
                raise RuntimeError("CIMD-V30 debug enabled but motion estimator did not run")
            with torch.no_grad():
                v30data["T_tracking_optimized"] = rigid_transform(
                    viewpoint.R, viewpoint.T).cpu().numpy()
                if hasattr(viewpoint, "R_gt") and hasattr(viewpoint, "T_gt"):
                    v30data["T_gt"] = rigid_transform(
                        viewpoint.R_gt.to(viewpoint.R.device),
                        viewpoint.T_gt.to(viewpoint.T.device)).cpu().numpy()
                np.savez_compressed(
                    os.path.join(output_dir, "camera_motion_v30_{:05d}.npz".format(viewpoint.uid)),
                    **v30data)
                print("[CIMD-V3.0] frame={} mode={} raw_t={:.6f} applied_t={:.6f} "
                      "limit={:.6f} scale={:.3f} inliers={:.3f} "
                      "resid_med={:.3f}->{:.3f} cond={:.2e}".format(
                          viewpoint.uid, v30mode, v30data["translation_raw"],
                          v30data["translation_applied"], v30data["translation_max"],
                          v30data["scale_translation_applied"], v30data["static_fraction"],
                          v30data["residual_first_median"], v30data["residual_second_median"],
                          v30data["condition_normalized"]))

        self.median_depth = get_median_depth(depth, opacity)

        
        
        with torch.no_grad():
            render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot)
            
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))
            
            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap, add_new_gaussian=True, dynamic_render=False):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, add_new_gaussian, dynamic_render]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 1 == 0:
            torch.cuda.empty_cache()
    
            
    def run(self):
        # init
        cur_frame_idx = 0
        last_keyframe_idx = 0
        # projection_matrix for viewpoints
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        # 
        dynamic_render=False
        keyframe_list = [0]
        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        0,
                        final=True,
                        monocular=self.monocular,
                    )
                    if self.save_results:
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                if self.requested_keyframe > 0 and (cur_frame_idx - last_keyframe_idx) >= self.kf_interval:
                    time.sleep(0.01)
                    continue
                    
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )
                
                # Tracking
                start_time = time.time()
                render_pkg = self.tracking(cur_frame_idx, viewpoint, last_keyframe_idx)
                tracking_time = time.time() - start_time
                print('Tracking time: ', tracking_time)
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                if self.dynamic_model and self.gaussians.init_deform == 'mlp':
                    self.gaussians.deform.deform.reg_loss = 0.
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx-1)
                    cur_frame_idx += 1
                    print("skip")
                    continue

                last_keyframe_idx = self.current_window[0]
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                
                create_kf = ((cur_frame_idx - last_keyframe_idx) >= 5) or create_kf or cur_frame_idx == self.dystart
                
                
                
                if self.dataset.dynamic_objects > self.dynamic_objects and cur_frame_idx>0:
                    create_kf = True
                    new_object = True
                else:
                    new_object = False
                    
                if create_kf:
                    keyframe_list.append(cur_frame_idx)
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map, True, dynamic_render
                    )
                    
                    temp_log = ("create keyframe:", cur_frame_idx, 
                                "add new gaussian:", True, 
                                "point_ratio:", point_ratio, 
                                "dynamic_render:", dynamic_render)
                    
                    Log(tag="Frontend", *temp_log)
                    self.cameras[cur_frame_idx-1].clean_key()
                else:
                    self.cleanup(cur_frame_idx-1)
                cur_frame_idx += 1
                self.dynamic_objects = self.dataset.dynamic_objects
                if (
                    # self.save_results
                    # and self.save_trj
                    self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
