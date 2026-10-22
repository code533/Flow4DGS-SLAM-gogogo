"""V3.2b: Replay real Bonn RGB-D + RAFT without launching SLAM.

Reads the same TUM/Bonn timestamp associations and frontend RAFT checkpoint.
GT projects current depth into previous camera. Outputs per-frame diagnostics,
not camera poses. No frontend/backend state is modified.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from utils.camera_motion_v32b import (
    exact_backward_gt_flow, gt_camera_twist, jacobian_backward_flow,
    projected_depth_mask, flow_scores,
)


def load_rgbd(parser, index, cfg, calibration):
    """Mirror MonocularDataset RGB remapping and depth loading (not remapped)."""
    import cv2
    from PIL import Image
    K = np.array([[calibration["fx"],0.,calibration["cx"]],
                  [0.,calibration["fy"],calibration["cy"]],[0.,0.,1.]])
    rgb = np.asarray(Image.open(parser.color_paths[index]).convert("RGB"))
    if calibration.get("distorted", False):
        coeff=np.array([calibration["k1"],calibration["k2"],calibration["p1"],
                        calibration["p2"],calibration["k3"]])
        mx,my = cv2.initUndistortRectifyMap(
            K,coeff,np.eye(3),K,
            (int(calibration["width"]),int(calibration["height"])),cv2.CV_32FC1)
        rgb=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR)
    depth=np.asarray(Image.open(parser.depth_paths[index]),dtype=np.float32) / float(calibration["depth_scale"])
    pose=np.asarray(parser.poses[index],dtype=np.float64)
    if rgb.shape[:2]!=depth.shape:
        raise ValueError("RGB and depth sizes differ; project only aligned inputs")
    if (int(calibration["height"]),int(calibration["width"]))!=depth.shape:
        raise ValueError("Image dimensions differ from calibration")
    return rgb,depth,pose,K


def static_prior(parser, index, rgb, yolo, cfg):
    """Mirror the Bonn/TUM prior: mask file and optional YOLO class 0 + chair."""
    from PIL import Image
    h,w=rgb.shape[:2]
    dynamic=np.zeros((h,w),dtype=bool)
    if parser.mask_path is not None:
        mask_file=parser.mask_path[index]
        dynamic|=(np.asarray(Image.open(mask_file).convert("L"),dtype=float)/255.>0.01)
    if yolo is not None:
        import torch
        rgb_tensor=torch.from_numpy(np.ascontiguousarray(rgb)).to(device="cuda",dtype=torch.float32).permute(2,0,1)/255.
        classes=[0]
        if cfg["Dataset"].get("seg_chair",False):
            classes.append(56)
        for c in classes:
            results=yolo.predict(source=rgb_tensor.unsqueeze(0),classes=[c],
                                 save=False,stream=False,show=False,verbose=False,
                                 device="cuda")
            for result in results:
                masks=result.masks
                if masks is None:continue
                for obj_mask in masks.data:
                    arr=obj_mask.detach().cpu().numpy().astype(bool)
                    if arr.shape!=(h,w):
                        raise ValueError("YOLO mask shape != RGB; avoid silent resize")
                    dynamic|=arr
    return ~dynamic


def raft_backward(model, rgb_current, rgb_previous):
    """Same camera_utils.Camera.generate_flow RAFT order, iters and padding."""
    import torch
    from RAFT.utils.utils import InputPadder
    def tensor(rgb):
        x=np.ascontiguousarray(rgb)
        return torch.from_numpy(x).to(device="cuda",dtype=torch.float32).permute(2,0,1)[None]
    cur,prev=tensor(rgb_current),tensor(rgb_previous)
    padder=InputPadder(prev.shape)
    prev,cur=padder.pad(prev,cur)
    with torch.no_grad():
        _,flow=model(cur,prev,iters=20,test_mode=True)  # current -> previous
    return padder.unpad(flow[0]).permute(1,2,0).detach().cpu().numpy()


def prepare_raft(checkpoint):
    """Match Camera()'s DataParallel checkpoint loading, only once."""
    import torch
    from RAFT.raft import RAFT
    from utils.camera_utils import raft_param
    if not torch.cuda.is_available():raise RuntimeError("CUDA is needed to replay the original RAFT")
    args=raft_param()
    args.model=str(checkpoint)
    model=torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(str(checkpoint),map_location="cpu",weights_only=False))
    model=model.module.to("cuda").eval()
    return model


def original_mask_and_twist(depth, raft_px, K, static):
    """Optional original frontend parametric motion classification/second IRLS."""
    import torch
    from utils.slam_backend import segment_motion_parametric,fit_twist_weighted
    H,W=depth.shape
    d=torch.from_numpy(np.ascontiguousarray(depth)).to("cuda",dtype=torch.float32)
    f=torch.from_numpy(np.ascontiguousarray(raft_px)).to("cuda",dtype=torch.float32).permute(2,0,1)
    k=torch.from_numpy(np.ascontiguousarray(K)).to("cuda",dtype=torch.float32)
    static_t=torch.from_numpy(np.ascontiguousarray(static)).to("cuda")
    # Same normalization and 'grid' re-conversion as Camera.generate_flow -> segment_motion_parametric.
    scale=torch.tensor([2./W,2./H],device="cuda",dtype=f.dtype)[:,None,None]
    flow_grid=f*scale
    with torch.no_grad():
        fit=segment_motion_parametric(d,flow_grid,k,flow_mode="grid",ds=1,
                robust_iters=30,k_mad=5.0,motion_mask=static_t)
        static_after=static_t & (~fit["mask_bool"]) & (d>0) & torch.isfinite(d)
        flow_used=flow_grid*torch.tensor([(W-1)/2.,(H-1)/2.],device="cuda")[:,None,None]
        refit=fit_twist_weighted(d,flow_used,k,static_after,robust=True,iters=30)
    return fit["mask_bool"].cpu().numpy(), refit.detach().cpu().numpy()


def summarize_frame(frame_index, prior, rgb, prev_rgb, depth, prev_depth, curr_pose, prev_pose,
                    K, model, cfg, output_dir, min_depth, max_depth,
                    gt_depth_abs, gt_depth_rel, save_maps=False, skip_original_fit=False,
                    flow_min_px=.3):
    raw=raft_backward(model,rgb,prev_rgb)
    if raw.shape!=(*depth.shape,2):raise ValueError("RAFT output shape mismatch")
    H,W=depth.shape
    tracking=raw*np.array([(W-1)/W,(H-1)/H],dtype=np.float64)[None,None,:]
    gtflow,zprev,uvprev,valid=exact_backward_gt_flow(depth,K,curr_pose,prev_pose,min_depth,max_depth)
    occlusion=projected_depth_mask(prev_depth,uvprev,zprev,valid,
                                  abs_tol=gt_depth_abs,rel_tol=gt_depth_rel)
    xi_gt=gt_camera_twist(curr_pose,prev_pose)
    jplus=jacobian_backward_flow(depth,K,xi_gt)
    jminus=jacobian_backward_flow(depth,K,-xi_gt)
    masks={"projected":valid,
           "prior_static":valid&prior,
           "prior_rejected":valid&(~prior),
           "occlusion_static":occlusion&prior,
           "occlusion_rejected":occlusion&(~prior)}
    flow_targets={"raft_vs_gt":(raw,gtflow),
                  "raft_vs_negative_gt":(raw,-gtflow),
                  "tracking_flow_vs_gt":(tracking,gtflow),
                  "jac_positive_vs_gt":(jplus,gtflow),
                  "jac_negative_vs_gt":(jminus,gtflow),
                  "jac_positive_vs_raft":(jplus,raw),
                  "jac_negative_vs_raft":(jminus,raw)}
    row={"frame_id":frame_index,"valid_projected":int(valid.sum()),
         "depth_consistent":int(occlusion.sum()),
         "static_prior_count":int(prior.sum()),
         "rejected_prior_count":int((~prior).sum()),
         "gt_camera_twist_translation_norm":float(np.linalg.norm(xi_gt[:3])),
         "gt_camera_twist_rotation_deg":float(np.degrees(np.linalg.norm(xi_gt[3:])))}
    if not skip_original_fit:
        seg,xi=original_mask_and_twist(depth,raw,K,prior)
        masks["residual_static"]=valid&prior&(~seg)
        masks["residual_rejected"]=valid&prior&seg
        masks["occlusion_residual_static"]=occlusion&prior&(~seg)
        row["original_irls_xi_translation_norm"]=float(np.linalg.norm(xi[:3]))
        row["original_irls_xi_rotation_deg"]=float(np.degrees(np.linalg.norm(xi[3:])))
        flow_targets["irls_positive_vs_gt"]=(jacobian_backward_flow(depth,K,xi),gtflow)
        flow_targets["irls_negative_vs_gt"]=(jacobian_backward_flow(depth,K,-xi),gtflow)
        flow_targets["irls_positive_vs_raft"]=(jacobian_backward_flow(depth,K,xi),raw)
        flow_targets["irls_negative_vs_raft"]=(jacobian_backward_flow(depth,K,-xi),raw)
    for region,mask in masks.items():
        row[region+"_n"]=int(mask.sum())
        for test,(prediction,ref) in flow_targets.items():
            scores=flow_scores(prediction,ref,mask,min_magnitude_px=flow_min_px)
            for key,val in scores.items():
                row[f"{region}_{test}_{key}"]=val
    if save_maps:
        np.savez_compressed(output_dir/f"camera_motion_v32b_{frame_index:05d}.npz",
            frame_id=frame_index,raft_flow_px=raw.astype(np.float32),
            gt_flow_px=gtflow.astype(np.float32),
            jacobian_gt_positive_px=jplus.astype(np.float32),
            jacobian_gt_negative_px=jminus.astype(np.float32),
            prior_static=prior,projection_valid=valid,depth_consistent=occlusion,
            gt_twist=xi_gt,pose_previous_gt=prev_pose,pose_current_gt=curr_pose)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/rgbd/bonn/ballon2.yaml")
    p.add_argument("--inherit-from",default="configs/rgbd/bonn/config.yaml")
    p.add_argument("--start",type=int,default=1)
    p.add_argument("--step",type=int,default=10)
    p.add_argument("--count",type=int,default=40)
    p.add_argument("--yolo",type=int,choices=(0,1),default=1,
                   help="1 reproduces SLAM's default YOLO person mask; 0 uses dataset masks only")
    p.add_argument("--raft-checkpoint",default="pretrained/raft-things.pth")
    p.add_argument("--yolo-checkpoint",default="pretrained/yolov9e-seg.pt")
    p.add_argument("--output",default="results/v32b_real_flow")
    p.add_argument("--save-maps",action="store_true")
    p.add_argument("--skip-original-fit",action="store_true",help="Skip original flow residual segmentation/IRLS comparison")
    p.add_argument("--min-depth",type=float,default=0.1)
    p.add_argument("--max-depth",type=float,default=15.0)
    p.add_argument("--depth-abs-tol",type=float,default=0.06)
    p.add_argument("--depth-rel-tol",type=float,default=0.05)
    p.add_argument("--flow-direction-min-px",type=float,default=0.3)
    args=p.parse_args()
    if args.count<=0 or args.step<=0 or args.start<=0 or args.max_depth<=args.min_depth:
        p.error("count/step/start must be positive; max_depth must exceed min_depth")
    if args.depth_abs_tol<0 or args.depth_rel_tol<0 or args.flow_direction_min_px<0:
        p.error("depth tolerances and flow threshold must be nonnegative")
    from utils.config_utils import load_config
    cfg=load_config(args.config,inherit_from=args.inherit_from)
    if cfg["Dataset"]["type"]!="tum" or cfg["Dataset"]["sensor_type"]!="depth":
        p.error("V3.2b currently supports Bonn/TUM RGB-D only")
    from utils.dataset import TUMParser
    parser=TUMParser(cfg["Dataset"]["dataset_path"])
    if args.start>=len(parser.color_paths):
        p.error("start must be < the available number of frames")
    selected=list(range(args.start,min(len(parser.color_paths),args.start+args.step*args.count),args.step))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    model=prepare_raft(args.raft_checkpoint)
    yolo=None
    if args.yolo:
        from ultralytics import YOLO
        yolo=YOLO(args.yolo_checkpoint)
    calib=cfg["Dataset"]["Calibration"]
    rows=[];failures=[]
    for i in selected:
        try:
            rgb,depth,pose,K=load_rgbd(parser,i,cfg,calib)
            prev_rgb,prev_depth,prev_pose,_=load_rgbd(parser,i-1,cfg,calib)
            prior=static_prior(parser,i,rgb,yolo,cfg)
            row=summarize_frame(i,prior,rgb,prev_rgb,depth,prev_depth,
                pose,prev_pose,K,model,cfg,out,args.min_depth,args.max_depth,
                args.depth_abs_tol,args.depth_rel_tol,args.save_maps,
                args.skip_original_fit,args.flow_direction_min_px)
            rows.append(row)
            score=row.get("occlusion_static_raft_vs_gt_median_epe_px",float("nan"))
            print(f"frame={i:04d} projected={row['valid_projected']} static={row['static_prior_count']} "
                  f"consistent={row['depth_consistent']} static_RAFT_GT_median_EPE={score:.4f}px",flush=True)
        except (RuntimeError,ValueError,IndexError,np.linalg.LinAlgError) as exc:
            failures.append((i,str(exc)))
            print(f"frame={i:04d} FAILED: {exc}",flush=True)
    if rows:
        columns=sorted(set().union(*(r.keys() for r in rows)))
        with (out/"v32b_per_frame.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader();writer.writerows(rows)
        summary_keys=("occlusion_static_raft_vs_gt_mean_epe_px",
            "occlusion_static_raft_vs_gt_negative_cosine_fraction",
            "occlusion_static_jac_positive_vs_gt_mean_epe_px",
            "occlusion_static_jac_negative_vs_gt_mean_epe_px",
            "occlusion_static_irls_positive_vs_gt_mean_epe_px",
            "occlusion_static_irls_negative_vs_gt_mean_epe_px")
        print("Summary (per-frame means; do not pool empty regions):")
        for k in summary_keys:
            a=np.array([r.get(k,np.nan) for r in rows],dtype=float)
            a=a[np.isfinite(a)]
            if a.size:print(f" {k}: n={a.size}, median={np.median(a):.5f}, mean={np.mean(a):.5f}")
        print("Saved",out/"v32b_per_frame.csv")
    print(f"selected={len(selected)}, success={len(rows)}, failed={len(failures)}")
    if failures:
        with (out/"failures.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(("frame_id","error"));w.writerows(failures)
    return 0 if rows and not failures else 1


if __name__=="__main__":
    raise SystemExit(main())
