"""V3.1b offline pre/post-clipping paired camera-motion evaluation (NumPy only).

Reads V3.1a shadow .npz archives without changing SLAM, tracking or maps.
GT is used only for offline scoring, never to fit or create a pose candidate.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np

METHODS = ('legacy', 'geo')
MODES = ('raw', 'legacy_clip', 'capped_clip')
CONVENTIONS = ('right', 'camera')


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def se3_exp(xi):
    """Match repository pose_utils.SE3_exp in double precision."""
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape != (6,) or not np.isfinite(xi).all():
        raise ValueError('xi must be finite [6]')
    rho, theta = xi[:3], xi[3:]
    a = np.linalg.norm(theta)
    W = skew(theta)
    W2 = W @ W
    if a < 1e-5:
        R = np.eye(3) + W + .5 * W2
        V = np.eye(3) + .5 * W + W2 / 6.
    else:
        R = np.eye(3) + np.sin(a) / a * W + (1. - np.cos(a)) / a**2 * W2
        V = np.eye(3) + (1. - np.cos(a)) / a**2 * W + (a - np.sin(a)) / a**3 * W2
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def rigid(m):
    a = np.asarray(m, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError('Expected finite 4x4 pose')
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('Invalid homogeneous transform row')
    if abs(np.linalg.det(a[:3, :3]) - 1.) > 1e-2:
        raise ValueError('Invalid rotation determinant')
    return a


def rotation_angle(R):
    u, _, vh = np.linalg.svd(R)
    M = np.diag([1., 1., np.linalg.det(u @ vh)])
    r = u @ M @ vh
    cos = np.clip((np.trace(r) - 1.) / 2., -1., 1.)
    return float(np.degrees(np.arccos(cos)))


def step_length(prev, nxt):
    return float(np.linalg.norm(nxt[:3, 3] - prev[:3, 3]))


def infer_limit(prev, saved_legacy, saved_geo, raw_legacy, raw_geo, atol=5e-5):
    """Legacy always forces nonzero t step to max_translation (approximately)."""
    pairs = ((saved_legacy, raw_legacy), (saved_geo, raw_geo))
    limits = [step_length(prev, saved) for saved, raw in pairs
              if np.linalg.norm(raw[:3, 3] - prev[:3, 3]) > 1e-8]
    if not limits:
        return 0., 'both_raw_zero'
    if max(limits) - min(limits) > atol:
        raise ValueError('Inferred per-frame limits disagree between saved candidates')
    return float(np.median(limits)), 'inferred_from_saved_legacy_clip'


def capped_candidate(prev, raw, saved_legacy, limit):
    """Translation-only cap; original rotation clipping remains unchanged."""
    candidate = saved_legacy.copy()
    dt = raw[:3, 3] - prev[:3, 3]
    norm = np.linalg.norm(dt)
    ratio = min(1., limit / (norm + 1e-10))
    candidate[:3, 3] = prev[:3, 3] + ratio * dt
    return candidate


def relative_metrics(prev_est, next_est, gt_prev, gt_next, convention, min_gt_m):
    if convention == 'right':
        A = np.linalg.solve(prev_est, next_est)
        B = np.linalg.solve(gt_prev, gt_next)
    elif convention == 'camera':
        # For world-to-camera poses, current-camera <- previous-camera.
        A = next_est @ np.linalg.inv(prev_est)
        B = gt_next @ np.linalg.inv(gt_prev)
    else:
        raise ValueError('Unknown relative-motion convention')
    vec = A[:3, 3] - B[:3, 3]
    terr = float(np.linalg.norm(vec))
    rerr = rotation_angle(A[:3, :3] @ B[:3, :3].T)
    at, bt = A[:3, 3], B[:3, 3]
    an, bn = float(np.linalg.norm(at)), float(np.linalg.norm(bt))
    direction = float('nan')
    if bn >= min_gt_m and an >= 1e-8:
        direction = float(np.degrees(np.arccos(np.clip(np.dot(at, bt) / (an * bn), -1, 1))))
    return dict(translation_error_m=terr, rotation_error_deg=rerr,
                magnitude_error_m=abs(an - bn), direction_error_deg=direction,
                predicted_translation_m=an, gt_translation_m=bn)


def read_frame(path, min_gt_m=0.001, compare_tol_m=5e-5,
               compare_tol_deg=0.05):
    required = ('xi_legacy_common', 'xi_geo', 'T_previous_est',
                'T_candidate_legacy_common', 'T_candidate_geo',
                'T_previous_gt', 'T_current_gt')
    with np.load(path, allow_pickle=False) as z:
        status = str(z['status'].item()) if 'status' in z else 'missing_status'
        if status != 'ok':
            raise ValueError('Shadow status is '+status)
        missing = [k for k in required if k not in z]
        if missing:
            raise ValueError('Missing fields: '+', '.join(missing))
        frames = {k:rigid(z[k]) for k in required if k.startswith('T_')}
        xis = {k:np.asarray(z[k],dtype=np.float64) for k in ('xi_legacy_common','xi_geo')}
        frame_id = int(z['frame_id']) if 'frame_id' in z else int(Path(path).stem.rsplit('_',1)[-1])
    prev = frames['T_previous_est']
    saved = {'legacy':frames['T_candidate_legacy_common'], 'geo':frames['T_candidate_geo']}
    raw = {key:prev @ se3_exp(xis['xi_'+ ('legacy_common' if key=='legacy' else 'geo')]) for key in METHODS}
    limit, limit_source = infer_limit(prev, saved['legacy'], saved['geo'], raw['legacy'], raw['geo'])
    cap = {key:capped_candidate(prev, raw[key], saved[key], limit) for key in METHODS}
    row = dict(frame_id=frame_id, file=str(path), translation_limit_m=limit,
               translation_limit_source=limit_source)
    for key in METHODS:
        # Independent consistency check: reconstruct the legacy translation
        # from xi and inferred per-frame limit. Rotation still uses saved
        # candidate because V3.1a did not log the per-frame max_rotation.
        dr = raw[key][:3, 3] - prev[:3, 3]
        if np.linalg.norm(dr) > 1e-8:
            predicted = prev[:3, 3] + limit * dr / (np.linalg.norm(dr) + 1e-10)
            if np.linalg.norm(predicted - saved[key][:3, 3]) > compare_tol_m:
                raise ValueError('Saved legacy translation disagrees with xi and limit')
        row[key+'_raw_step_m'] = step_length(prev, raw[key])
        row[key+'_legacy_step_m'] = step_length(prev, saved[key])
        row[key+'_capped_step_m'] = step_length(prev, cap[key])
        row[key+'_legacy_amplifies'] = int(row[key+'_legacy_step_m'] > row[key+'_raw_step_m'] + 1e-7)
        row[key+'_capped_reduces'] = int(row[key+'_capped_step_m'] < row[key+'_raw_step_m'] - 1e-7)
        for mode, candidate in (('raw',raw[key]),('legacy_clip',saved[key]),('capped_clip',cap[key])):
            for conv in CONVENTIONS:
                result = relative_metrics(prev, candidate,
                                          frames['T_previous_gt'],frames['T_current_gt'],conv,min_gt_m)
                for name, value in result.items():
                    row[f'{key}_{mode}_{conv}_{name}'] = value
        # The original clip of a motion step must not rotate MORE than the raw step.
        r_raw = raw[key][:3,:3] @ prev[:3,:3].T
        r_saved = saved[key][:3,:3] @ prev[:3,:3].T
        raw_angle = rotation_angle(r_raw)
        clip_angle = rotation_angle(r_saved)
        row[key+'_raw_rotation_step_deg'] = raw_angle
        row[key+'_legacy_rotation_step_deg'] = clip_angle
        if clip_angle > raw_angle + compare_tol_deg:
            raise ValueError('Saved candidate rotated farther than raw estimate')
        if step_length(prev, cap[key]) > limit + compare_tol_m:
            raise ValueError('Capped translation exceeds its inferred limit')
    # Check only the common input, do not compare against xi_legacy_actual.
    return row


def paired_summary(rows, conventions=CONVENTIONS):
    output = []
    def paired(a,b,label):
        ab = [(float(row[a]), float(row[b])) for row in rows
              if a in row and b in row and np.isfinite(row[a]) and np.isfinite(row[b])]
        if not ab:
            return
        pair = np.asarray(ab)
        diff = pair[:,1] - pair[:,0]
        output.append(f'{label}: n={len(diff)} first_med={np.median(pair[:,0]):.6f} '
                      f'second_med={np.median(pair[:,1]):.6f} '
                      f'mean_delta={np.mean(diff):+.6f} second_better={(diff < -1e-12).mean():.1%}')
    for conv in conventions:
        for metric in ('translation_error_m','rotation_error_deg','magnitude_error_m','direction_error_deg'):
            for method in METHODS:
                prefix=f'{method}_'
                paired(prefix+'raw_'+conv+'_'+metric, prefix+'legacy_clip_'+conv+'_'+metric,
                       f'{method} raw -> legacy_clip [{conv}] {metric}')
                paired(prefix+'raw_'+conv+'_'+metric, prefix+'capped_clip_'+conv+'_'+metric,
                       f'{method} raw -> capped_clip [{conv}] {metric}')
            for mode in MODES:
                paired('legacy_'+mode+'_'+conv+'_'+metric, 'geo_'+mode+'_'+conv+'_'+metric,
                       f'legacy -> geo [{conv}] {mode} {metric}')
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result_root', help='V3.1a run folder containing tracking/*.npz')
    p.add_argument('--csv', default='', help='Save per-frame CSV')
    p.add_argument('--min-gt-motion-mm', type=float, default=1.,
                   help='Suppress translation direction angle for GT step below this many mm')
    p.add_argument('--convention', choices=('right','camera','both'), default='both')
    args=p.parse_args(argv)
    if args.min_gt_motion_mm<0:p.error('min GT motion threshold must be nonnegative')
    root=Path(args.result_root)
    if root.is_file():p.error('Pass NPZ result directory, not prior summary CSV')
    files=sorted(root.rglob('camera_motion_v31a_*.npz')) if root.exists() else []
    if not files:
        print('No camera_motion_v31a_*.npz files found. V31a per-frame CSV cannot reconstruct raw poses; point at the original run tracking directory.')
        return 2
    rows=[]; errors=Counter();bad=[]
    for f in files:
        try:
            rows.append(read_frame(f,min_gt_m=args.min_gt_motion_mm/1000))
        except (ValueError,KeyError,np.linalg.LinAlgError,FloatingPointError) as exc:
            errors[str(exc).split(':')[0]]+=1
            bad.append((f,str(exc)))
    print(f'Files={len(files)}, valid_pairs={len(rows)}, skipped={len(bad)} errors={dict(errors)}')
    for f,reason in bad[:5]:print('  Skipped',f.name,reason)
    if not rows:return 3
    conventions=CONVENTIONS if args.convention=='both' else (args.convention,)
    print('NOTE: right=inv(Tprev)@Tcur matches V3.1a CSV; camera=Tcur@inv(Tprev) for w2c. Verify GT pose conventions.')
    skipped_gt = sum(row['legacy_raw_right_gt_translation_m'] < args.min_gt_motion_mm/1000 for row in rows)
    print(f'Direction threshold: GT < {args.min_gt_motion_mm:g} mm excluded; {skipped_gt}/{len(rows)} frames below threshold')
    for method in METHODS:
        amp=sum(row[method+'_legacy_amplifies'] for row in rows)
        reduced=sum(row[method+'_capped_reduces'] for row in rows)
        print(f'{method} amplified_by_legacy={amp}/{len(rows)} capped_reductions={reduced}/{len(rows)}')
    for msg in paired_summary(rows,conventions):print(msg)
    if args.csv:
        destination=Path(args.csv);destination.parent.mkdir(parents=True,exist_ok=True)
        keys=sorted(set().union(*(r.keys() for r in rows)))
        with destination.open('w',newline='',encoding='utf-8') as fp:
            writer=csv.DictWriter(fp,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        print('Saved:',destination)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
