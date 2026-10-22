"""V3.2a: offline SE(3) sign/composition consistency analysis on V3.1a NPZ.

Compares raw, *unclipped* poses only. GT is used only for retrospective
scoring. This script never alters tracking, maps or trajectory poses.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np

METHODS = {'legacy': 'xi_legacy_common', 'geo': 'xi_geo'}
CASES = ('right_pos', 'left_pos', 'right_neg', 'left_neg')
FRAMES = ('camera', 'right')


def hat(w):
    a, b, c = w
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]], dtype=np.float64)


def se3_exp(xi):
    """Match pose_utils.SE3_exp using float64, xi=[translation,rotation]."""
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape != (6,) or not np.isfinite(xi).all():
        raise ValueError('xi must be finite and have shape (6,)')
    v, w = xi[:3], xi[3:]
    a = np.linalg.norm(w)
    W = hat(w)
    W2 = W @ W
    if a < 1e-5:
        R = np.eye(3) + W + .5 * W2
        V = np.eye(3) + .5 * W + W2 / 6.
    else:
        R = np.eye(3) + (np.sin(a) / a) * W + ((1. - np.cos(a)) / a**2) * W2
        V = np.eye(3) + ((1. - np.cos(a)) / a**2) * W + ((a - np.sin(a)) / a**3) * W2
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = V @ v
    return out


def rigid(T):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Expected finite 4x4 pose')
    if not np.allclose(T[3], [0., 0., 0., 1.], atol=1e-5):
        raise ValueError('Invalid homogeneous pose last row')
    if not np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=5e-3):
        raise ValueError('Pose is not close to SO(3)')
    if np.linalg.det(T[:3,:3]) < 0.:
        raise ValueError('Pose rotation is reflected')
    return T


def candidates(prev, xi):
    prev = rigid(prev)
    E, Einv = se3_exp(xi), se3_exp(-np.asarray(xi, dtype=np.float64))
    return {'right_pos':prev @ E, 'left_pos':E @ prev,
            'right_neg':prev @ Einv, 'left_neg':Einv @ prev}


def angle_deg(R):
    # Project numerical roundoff onto SO(3) before taking the angle.
    U, _, Vh = np.linalg.svd(R)
    fix = np.diag([1., 1., np.linalg.det(U @ Vh)])
    Q = U @ fix @ Vh
    return float(np.degrees(np.arccos(np.clip((np.trace(Q)-1.)/2., -1., 1.))))


def relative(Tprev, Tnext, frame):
    if frame == 'camera':
        # World-to-camera: maps prev-camera points to current-camera points.
        return Tnext @ np.linalg.inv(Tprev)
    if frame == 'right':
        # Matches the historical V3.1a CSV's right-increment convention.
        return np.linalg.solve(Tprev, Tnext)
    raise ValueError('Unknown relative convention')


def score(prev, cand, gtprev, gtcur, convention='camera', min_gt_mm=3.):
    P, G = relative(prev, cand, convention), relative(gtprev, gtcur, convention)
    pt, gt = P[:3,3], G[:3,3]
    pn, gn = np.linalg.norm(pt), np.linalg.norm(gt)
    direction = float('nan')
    if gn >= min_gt_mm * .001 and pn > 1e-8:
        direction = float(np.degrees(np.arccos(np.clip(np.dot(pt,gt)/(pn*gn), -1., 1.))))
    return {'translation_m':float(np.linalg.norm(pt-gt)),
            'rotation_deg':angle_deg(P[:3,:3] @ G[:3,:3].T),
            'magnitude_m':float(abs(pn-gn)),
            'direction_deg':direction,
            'gt_translation_m':float(gn)}


def analytic_backward_flow(xy, depth, K, xi):
    """First-order current->previous optical flow from repository interaction matrix.

    xy: pixel coordinates Nx2; xi corresponds to camera-left positive motion
    under world-to-camera poses and current->previous flow, infinitesimally.
    """
    xy = np.asarray(xy, np.float64)
    z = np.asarray(depth, np.float64)
    K = np.asarray(K, np.float64)
    u, v = xy[:,0]-K[0,2], xy[:,1]-K[1,2]
    fx, fy = K[0,0], K[1,1]
    Lx = np.stack((-fx/z, np.zeros_like(z), u/z,
                   u*v/fy, -(fx+u*u/fx), v), axis=-1)
    Ly = np.stack((np.zeros_like(z), -fy/z, v/z,
                   fy+v*v/fy, -u*v/fx, -u), axis=-1)
    return np.stack((Lx @ xi, Ly @ xi), axis=1)


def reproject_backward_flow(xy, depth, K, prev, curr):
    """Exact ray reprojection current->previous, using world-to-camera poses."""
    xy = np.asarray(xy, np.float64)
    z = np.asarray(depth, np.float64)
    K = np.asarray(K, np.float64)
    Xcur = np.stack(((xy[:,0]-K[0,2])*z/K[0,0],
                     (xy[:,1]-K[1,2])*z/K[1,1], z, np.ones_like(z)), axis=1)
    Xprev = (prev @ np.linalg.inv(curr) @ Xcur.T).T
    uvprev = np.stack((K[0,0]*Xprev[:,0]/Xprev[:,2]+K[0,2],
                       K[1,1]*Xprev[:,1]/Xprev[:,2]+K[1,2]), axis=1)
    return uvprev-xy


def geometry_check():
    """Synthetic only: independently verify direction and side with nonidentity prev."""
    prev = se3_exp(np.array([.12,-.07,.08,.09,-.16,.11]))
    xi = np.array([.00015,-.00020,.00009,.00011,-.00008,.00013])
    K = np.array([[510.,0.,310.],[0.,512.,235.],[0.,0.,1.]])
    x,y = np.meshgrid(np.linspace(120,500,9),np.linspace(90,370,7))
    xy = np.stack((x.ravel(),y.ravel()),axis=1)
    z = 1.3 + .8*np.arange(len(xy))/len(xy)
    target = analytic_backward_flow(xy,z,K,xi)
    scores = {}
    for case, pose in candidates(prev,xi).items():
        pred = reproject_backward_flow(xy,z,K,prev,pose)
        scores[case] = float(np.sqrt(np.mean((target-pred)**2)))
    return scores


def analyze_frame(path, min_gt_mm=3.):
    with np.load(path, allow_pickle=False) as d:
        if 'status' not in d or str(d['status'].item())!='ok':
            raise ValueError('Shadow status is not ok')
        required = ('T_previous_est','T_previous_gt','T_current_gt',*METHODS.values())
        missing = [k for k in required if k not in d]
        if missing:raise ValueError('Missing fields: '+','.join(missing))
        prev, gprev, gcur = (rigid(d[k]) for k in required[:3])
        xis = {name:np.asarray(d[key],np.float64) for name,key in METHODS.items()}
        if 'xi_legacy_actual' in d and np.isfinite(d['xi_legacy_actual']).all():
            xis['legacy_actual']=np.asarray(d['xi_legacy_actual'],np.float64)
        id_ = int(d['frame_id'].item()) if 'frame_id' in d else int(path.stem.split('_')[-1])
        row = {'frame_id':id_, 'file':str(path), 'status':'ok'}
        for method, xi in xis.items():
            opts=candidates(prev,xi)
            row[method+'_left_right_pose_translation_separation_m'] = float(np.linalg.norm(
                opts['left_pos'][:3,3]-opts['right_pos'][:3,3]))
            row[method+'_left_right_pose_rotation_separation_deg'] = angle_deg(
                opts['left_pos'][:3,:3] @ opts['right_pos'][:3,:3].T)
            for case, T in opts.items():
                for conv in FRAMES:
                    result = score(prev,T,gprev,gcur,conv,min_gt_mm)
                    for key,value in result.items():
                        row[f'{method}_{case}_{conv}_{key}']=value
        if 'T_tracking_initial_actual' in d:
            T_actual=rigid(d['T_tracking_initial_actual'])
            for conv in FRAMES:
                result=score(prev,T_actual,gprev,gcur,conv,min_gt_mm)
                for k,v in result.items(): row[f'actual_clipped_{conv}_{k}']=v
        return row


def paired_stats(rows,method,conv,metric,tol=1e-9):
    valid=[]
    for r in rows:
        arr = np.array([r.get(f'{method}_{case}_{conv}_{metric}',np.nan)
                        for case in CASES],dtype=float)
        if np.isfinite(arr).all():valid.append(arr)
    if not valid:return None
    data=np.array(valid)
    rank=np.argmin(data,axis=1)
    counts={case:int(np.sum(rank==i)) for i,case in enumerate(CASES)}
    win_ties=int(np.sum((np.sort(data,axis=1)[:,1]-np.sort(data,axis=1)[:,0])<=tol))
    return {'n':len(data), 'median':{case:float(np.median(data[:,i])) for i,case in enumerate(CASES)},
            'mean':{case:float(np.mean(data[:,i])) for i,case in enumerate(CASES)},
            'best_counts':counts,'ties':win_ties}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result_root',help='One V3.1a experiment folder containing tracking/*.npz')
    p.add_argument('--csv',default='',help='Save per-frame comparison CSV')
    p.add_argument('--min-gt-motion-mm',type=float,default=3.,help='Filter nearly stationary direction angles')
    p.add_argument('--convention',choices=['camera','right','both'],default='both')
    p.add_argument('--geometry-check-only',action='store_true',help='Run independent synthetic reprojection sanity check')
    args=p.parse_args(argv)
    if args.min_gt_motion_mm<0:p.error('min GT motion threshold must be nonnegative')
    check=geometry_check()
    print('Synthetic current->previous flow consistency (smaller RMSE is better):')
    for case in CASES:print(' ',case,':',format(check[case],'.8f'),'px')
    if args.geometry_check_only:return 0
    root=Path(args.result_root)
    if root.is_file():p.error('Pass the original V3.1a NPZ folder, not the V3.1a/b CSV')
    files=sorted(root.rglob('camera_motion_v31a_*.npz')) if root.exists() else []
    if not files:
        print('No V3.1a NPZ files found. Use the original run folder with tracking/camera_motion_v31a_*.npz; prior CSVs lack xi.')
        return 2
    rows=[];fail=Counter();bad=[]
    for f in files:
        try:rows.append(analyze_frame(f,args.min_gt_motion_mm))
        except (ValueError,KeyError,IndexError,np.linalg.LinAlgError,FloatingPointError) as ex:
            fail[str(ex).split(':')[0]]+=1
            bad.append((f,str(ex)))
    print(f'Files={len(files)}, valid={len(rows)}, skipped={len(bad)} errors={dict(fail)}')
    for f,reason in bad[:5]:print(' Skip:',f.name,reason)
    if not rows:return 3
    convs=FRAMES if args.convention=='both' else (args.convention,)
    print('Note: camera=Tcur@inv(Tprev) for w2c; right=inv(Tprev)@Tcur (V3.1a historical metric).')
    print('Hypotheses are RAW, without the original motion clipping. GT is offline only.')
    for conv in convs:
        motions=[r.get('legacy_right_pos_'+conv+'_gt_translation_m',np.nan) for r in rows]
        print(f'{conv}: GT motion below {args.min_gt_motion_mm:g}mm:',sum(x<args.min_gt_motion_mm*.001 for x in motions))
        for method in ('legacy','geo','legacy_actual'):
            if not any(f'{method}_right_pos_{conv}_translation_m' in r for r in rows):continue
            for metric in ('translation_m','rotation_deg','magnitude_m','direction_deg'):
                stats=paired_stats(rows,method,conv,metric,tol=.0001 if metric=='translation_m' else .01 if metric=='rotation_deg' else 1e-9)
                if stats:
                    print(f'{conv} {method} {metric}: n={stats["n"]} median={stats["median"]} best_counts={stats["best_counts"]} best_near_ties={stats["ties"]}')
    if args.csv:
        out=Path(args.csv);out.parent.mkdir(parents=True,exist_ok=True)
        fields=sorted(set().union(*(r.keys() for r in rows)))
        with out.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        print('CSV:',out)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
