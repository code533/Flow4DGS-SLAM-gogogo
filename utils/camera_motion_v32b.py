"""V3.2b camera motion diagnostics: exact GT optical flow and interaction Jacobian.

NumPy/SciPy-only geometry. No SLAM/RAFT imports, no optimizer and no GT used
to modify any camera pose. Pixel coordinate convention: u right, v down.
"""
import numpy as np


def _pose(T):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Pose must be finite [4,4]")
    if not np.allclose(T[3], (0, 0, 0, 1), atol=1e-5):
        raise ValueError("Invalid homogeneous pose")
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=2e-3):
        raise ValueError("Rotation is not orthonormal")
    return T


def _intrinsics(K):
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid intrinsics")
    return K


def exact_backward_gt_flow(depth, K, T_current_w2c, T_previous_w2c,
                           min_depth=0.1, max_depth=15.0):
    """Warp current RGB-D pixels into previous image with GT w2c poses.

    Returns GT current->previous flow (H,W,2), Z in previous-camera frame,
    projected pixel coordinate (H,W,2), and valid mask. Invalid flows are nan.
    """
    d = np.asarray(depth, dtype=np.float64)
    K = _intrinsics(K)
    curr, prev = _pose(T_current_w2c), _pose(T_previous_w2c)
    if d.ndim != 2 or min_depth < 0 or max_depth <= min_depth:
        raise ValueError("Invalid depth/limits")
    h, w = d.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    z = d
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    zsafe = np.where(valid, z, 1.)
    xyz = np.stack(((xx-K[0,2])*zsafe/K[0,0],
                    (yy-K[1,2])*zsafe/K[1,1], zsafe), axis=0).reshape(3,-1)
    relative = prev @ np.linalg.inv(curr)  # x_prev = T_prev * T_cur^-1 * x_cur
    xyz_previous = (relative[:3,:3] @ xyz + relative[:3,3:4])
    zp = xyz_previous[2].reshape(h,w)
    visible = np.isfinite(zp) & (zp > 1e-6)
    zprojection = np.where(visible, zp, 1.)
    up = (K[0,0] * xyz_previous[0].reshape(h,w) / zprojection + K[0,2])
    vp = (K[1,1] * xyz_previous[1].reshape(h,w) / zprojection + K[1,2])
    uv_prev = np.stack((up, vp), axis=-1)
    inside = (up >= 0) & (up <= w-1) & (vp >= 0) & (vp <= h-1)
    valid &= visible & inside & np.isfinite(up) & np.isfinite(vp)
    flow = uv_prev - np.stack((xx,yy),axis=-1)
    flow[~valid] = np.nan
    uv_prev[~valid] = np.nan
    return flow, zp, uv_prev, valid


def gt_camera_twist(T_current_w2c, T_previous_w2c):
    """Log(Tcur @ inv(Tprev)); estimate camera-left positive twist in w2c.

    Matches pose_utils.SE3_exp's translation convention, not a naive
    rotation-vector + raw translation concatenation.
    """
    from scipy.spatial.transform import Rotation
    relative = _pose(T_current_w2c) @ np.linalg.inv(_pose(T_previous_w2c))
    rotation = Rotation.from_matrix(relative[:3,:3]).as_rotvec()
    theta = float(np.linalg.norm(rotation))
    x, y, z = rotation
    W = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]], dtype=np.float64)
    W2 = W @ W
    if theta < 1e-5:
        V = np.eye(3) + .5 * W + W2 / 6.
    else:
        V = np.eye(3) + (1.-np.cos(theta))/theta**2*W + (theta-np.sin(theta))/theta**3*W2
    return np.concatenate((np.linalg.solve(V,relative[:3,3]),rotation))


def jacobian_backward_flow(depth, K, xi):
    """Repository interaction Jacobian times +/-GT camera-left twist.

    This is *first-order* and need not agree exactly with finite-motion
    projective flow. Use exact_backward_gt_flow as the independent reference.
    """
    d = np.asarray(depth,dtype=np.float64)
    K = _intrinsics(K)
    xi = np.asarray(xi,dtype=np.float64)
    if d.ndim != 2 or xi.shape != (6,) or not np.isfinite(xi).all():
        raise ValueError("Invalid jacobian input")
    h,w = d.shape
    yy,xx = np.indices((h,w),dtype=np.float64)
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    u,v = xx-cx, yy-cy
    z = np.where(np.isfinite(d)&(d>0),d,np.nan)
    du = -fx/z * xi[0] + u/z * xi[2] + (u*v/fy)*xi[3] - (fx+u*u/fx)*xi[4] + v*xi[5]
    dv = -fy/z * xi[1] + v/z * xi[2] + (fy+v*v/fy)*xi[3] - (u*v/fx)*xi[4] - u*xi[5]
    return np.stack((du,dv),axis=-1)


def projected_depth_mask(previous_depth, projected_uv, projected_z,
                         valid, abs_tol=0.06, rel_tol=0.05):
    """Approximate occlusion gate from depth at previous pixel (nearest).

    A conservative diagnostic filter, not a ground-truth visibility mask.
    """
    d = np.asarray(previous_depth,dtype=np.float64)
    uv = np.asarray(projected_uv,dtype=np.float64)
    z = np.asarray(projected_z,dtype=np.float64)
    valid = np.asarray(valid,dtype=bool)
    h,w = d.shape
    if uv.shape != (h,w,2) or z.shape != (h,w) or valid.shape != (h,w):
        raise ValueError("Invalid projected depth shapes")
    if abs_tol < 0 or rel_tol < 0:raise ValueError("Tolerance must be >=0")
    u = np.rint(np.nan_to_num(uv[...,0],nan=-1)).astype(np.int64)
    v = np.rint(np.nan_to_num(uv[...,1],nan=-1)).astype(np.int64)
    inside = (u>=0)&(u<w)&(v>=0)&(v<h)
    mask = valid & inside
    observed = np.full((h,w),np.nan)
    observed[mask] = d[v[mask],u[mask]]
    return (mask & np.isfinite(observed)&(observed>0)&np.isfinite(z)
            & (np.abs(observed-z)<=np.maximum(abs_tol,rel_tol*np.abs(z))))


def remap_depth_to_rectified(depth, map_x, map_y):
    """Remap RGB-registered depth to the rectified RGB pixel grid.

    Bonn depth is registered to its corresponding raw RGB image. When RGB is
    undistorted, apply the same inverse rectification maps to depth using
    nearest-neighbor sampling so metric depths are not blended at edges.
    Invalid/out-of-image pixels become 0.
    """
    import cv2
    d = np.asarray(depth, dtype=np.float32)
    mx = np.asarray(map_x, dtype=np.float32)
    my = np.asarray(map_y, dtype=np.float32)
    if d.ndim != 2 or mx.ndim != 2 or my.shape != mx.shape:
        raise ValueError("Expected depth and two 2-D rectification maps")
    if mx.shape != d.shape:
        raise ValueError("Rectification maps and depth must share the output image size")
    return cv2.remap(d, mx, my, interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def flow_scores(observed, reference, mask, min_magnitude_px=.3):
    """EPE, direction and polarity on the given region, without NaN leakage.

    Direction angles use atan2(|2-D cross product|, dot product) rather than
    arccos(cosine), which loses precision when the vectors are almost parallel
    or antiparallel and can turn exact 0/180 degree cases into ~1e-6 degree
    errors.
    """
    observed=np.asarray(observed,dtype=np.float64)
    reference=np.asarray(reference,dtype=np.float64)
    mask=np.asarray(mask,dtype=bool)
    if observed.ndim!=3 or observed.shape[-1]!=2 or observed.shape!=reference.shape or mask.shape!=observed.shape[:2]:
        raise ValueError("Expected two [H,W,2] flows and [H,W] mask")
    good=mask & np.isfinite(observed).all(-1)&np.isfinite(reference).all(-1)
    out={'count':int(good.sum()),'mean_epe_px':float('nan'),
         'median_epe_px':float('nan'),'p90_epe_px':float('nan'),
         'rmse_epe_px':float('nan'),'valid_angles':0,
         'mean_direction_deg':float('nan'),'negative_cosine_fraction':float('nan'),
         'mean_cosine':float('nan')}
    if not good.any():return out
    a,b = observed[good],reference[good]
    error=np.linalg.norm(a-b,axis=1)
    out.update(mean_epe_px=float(error.mean()),median_epe_px=float(np.median(error)),
               p90_epe_px=float(np.percentile(error,90)),
               rmse_epe_px=float(np.sqrt(np.mean(error**2))))
    norm_a,norm_b=np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1)
    strong=(norm_a>=min_magnitude_px)&(norm_b>=min_magnitude_px)
    out['valid_angles']=int(strong.sum())
    if strong.any():
        aa,bb=a[strong],b[strong]
        dot=np.sum(aa*bb,axis=1)
        cosine=np.clip(dot/(norm_a[strong]*norm_b[strong]),-1.,1.)
        cross=np.abs(aa[:,0]*bb[:,1]-aa[:,1]*bb[:,0])
        angles=np.degrees(np.arctan2(cross,dot))
        out['mean_direction_deg']=float(angles.mean())
        out['negative_cosine_fraction']=float(np.mean(cosine<0))
        out['mean_cosine']=float(cosine.mean())
    return out
