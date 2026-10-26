#!/usr/bin/env python3
"""Small, dataset-free report from tracking_decisions; no Torch/CUDA required."""
import argparse
import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import yaml
from visualize_consistency import split_depth_regions, plt


def percent(n, d):
    return 100 * n / d if d else None


def read_records(path):
    if not path.exists():
        raise ValueError(f'Missing log: {path}')
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r['frame'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate frame IDs: {path}')
    return {r['frame']: r for r in records}


def validate(a, row):
    required = ['frame', 'mask_iteration', 'dense_diagnostics', 'mask_applied',
                'depth_definition', 'candidate_keep_mask', 'applied_keep_mask', 'map_valid']
    required += [s + suffix for s in ['rgb', 'depth', 'union'] for suffix in
                 ['_support_mask', '_candidate_excluded_mask', '_applied_excluded_mask']]
    missing = set(required) - a.keys()
    if missing:
        raise ValueError(f'Missing fields: {sorted(missing)}')
    for k in ['frame', 'mask_iteration', 'dense_diagnostics', 'mask_applied', 'depth_definition']:
        if a[k].item() != row[k]:
            raise ValueError(f'NPZ/log disagree on {k}')
    shape = a['map_valid'].shape
    for k in required:
        if k.endswith('_mask') or k == 'map_valid':
            if a[k].shape != shape or a[k].dtype != bool:
                raise ValueError(f'Invalid mask {k}')
    candidate, applied = a['candidate_keep_mask'], a['applied_keep_mask']
    if np.any(~candidate & ~a['map_valid']):
        raise ValueError('Candidate excludes unknown map coverage')
    expected = candidate if bool(a['mask_applied']) else np.ones(shape, bool)
    if not np.array_equal(applied, expected):
        raise ValueError('Applied mask inconsistent with fallback')
    if not np.array_equal(a['union_support_mask'], a['rgb_support_mask'] | a['depth_support_mask']):
        raise ValueError('Support union mismatch')
    for s in ['rgb', 'depth', 'union']:
        support = a[s + '_support_mask']
        for label, keep in [('candidate', candidate), ('applied', applied)]:
            excluded = a[s + '_' + label + '_excluded_mask']
            if not np.array_equal(excluded, support & ~keep):
                raise ValueError(f'{s} {label} exclusion mismatch')
            if int((support & keep).sum()) != row[s + '_' + label + '_retained_pixels']:
                raise ValueError(f'{s} {label} count disagrees with log')
        if int(support.sum()) != row[s + '_support_pixels']:
            raise ValueError('Support count disagrees with log')


def counts(a, support, scope, frame, region):
    candidate = support & ~a['candidate_keep_mask']
    excluded = support & ~a['applied_keep_mask']
    n = int(support.sum())
    return dict(frame=frame, scope=scope, region=region, support_pixels=n,
                candidate_excluded_pixels=int(candidate.sum()),
                applied_excluded_pixels=int(excluded.sum()),
                retained_pixels=n-int(excluded.sum()),
                applied_excluded_percent=percent(int(excluded.sum()), n))


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True, type=Path, help='Run root with config.yml and tracking_decisions/')
    p.add_argument('--output', required=True, type=Path, help='New report directory; must not exist')
    p.add_argument('--plot-frames', nargs='+', type=int, default=[10, 50, 90])
    p.add_argument('--edge-radius', type=int, default=3)
    args = p.parse_args()
    if not 0 <= args.edge_radius <= 30:
        p.error('--edge-radius must be between 0 and 30')
    run = args.results.resolve(); folder = run / 'tracking_decisions'
    config = yaml.safe_load((run / 'config.yml').read_text())
    log = read_records(folder / 'summary.jsonl')
    post = read_records(run / 'static_consistency' / 'summary.jsonl')
    paths = sorted(folder.glob('[0-9]*.npz'))
    if not paths or {int(x.stem) for x in paths} != set(log):
        p.error('Decision NPZ frames must exactly match decision summary.jsonl')
    if not set(log) <= set(post):
        p.error('Some decisions lack completed tracking records')
    args.output.mkdir(parents=True, exist_ok=False)
    rows, regions, cells, warnings = [], [], [], []
    plotted, dense_frames = [], []
    for path in paths:
        frame = int(path.stem); record = log[frame]
        with np.load(path, allow_pickle=False) as f:
            a = {k: f[k] for k in f.files}
        validate(a, record)
        if str(a['depth_definition']) != config['StaticConsistency'].get('depth_definition', 'normalized'):
            raise ValueError('Depth convention disagrees with config')
        h, w = a['map_valid'].shape
        for scope in ['rgb', 'depth', 'union']:
            support = a[scope + '_support_mask']
            r = counts(a, support, scope, frame, 'all_pixels')
            r.update(mask_applied=bool(a['mask_applied']), fallback=record.get('fallback'),
                     mask_iteration=int(a['mask_iteration']),
                     tracking_iterations=post[frame]['tracking_iterations'],
                     dense_diagnostics=bool(a['dense_diagnostics']))
            rows.append(r)
            # Fixed 4x4 screen grid: distribution diagnostic, not pose observability.
            for y in range(4):
                for x in range(4):
                    m = np.zeros((h, w), bool)
                    m[y*h//4:(y+1)*h//4, x*w//4:(x+1)*w//4] = True
                    c = counts(a, support & m, scope, frame, f'cell_{y}_{x}')
                    cells.append(c)
        if bool(a['dense_diagnostics']):
            dense_frames.append(frame)
            z = a['observed_depth']
            masks, valid, _ = split_depth_regions(z, config['StaticConsistency'].get('depth_min', .01), .05, .02, args.edge_radius)
            masks['invalid_observation'] = ~valid
            for scope in ['rgb', 'depth', 'union']:
                for name, mask in masks.items():
                    regions.append(counts(a, a[scope+'_support_mask'] & mask, scope, frame, name))
            if frame in args.plot_frames:
                rgb = np.clip(a['observed_rgb'].transpose(1, 2, 0), 0, 1)
                fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
                axes[0, 0].imshow(rgb); axes[0, 0].set_title(f'Frame {frame}: decision RGB')
                for ax, scope in [(axes[0, 1], 'rgb'), (axes[0, 2], 'depth')]:
                    image = rgb.copy(); m = a[scope+'_applied_excluded_mask']
                    image[m] = .3*image[m] + .7*np.array([1, .15, 0])
                    ax.imshow(image); ax.set_title(scope.upper()+' actual exclusions (red)')
                union = a['union_support_mask']
                image = rgb.copy(); retained = union & a['applied_keep_mask']
                image[retained] = .3*image[retained] + .7*np.array([0, 1, .2])
                axes[1, 0].imshow(image); axes[1, 0].set_title('Union retained (green)')
                for ax, key, title in [(axes[1, 1], 'signed_depth_normalized', 'Normalized residual'),
                                       (axes[1, 2], 'signed_depth_accumulated', 'Accumulated residual')]:
                    im = ax.imshow(np.ma.masked_where(~union | ~a['map_valid'], a[key]), cmap='coolwarm', vmin=-.2, vmax=.2)
                    ax.set_title(title+' (m)'); fig.colorbar(im, ax=ax, extend='both')
                for ax in axes.flat: ax.axis('off')
                fig.savefig(args.output/f'{frame:06d}_decision.png', dpi=110); plt.close(fig)
                plotted.append(frame)
    missing = sorted(set(args.plot_frames) - set(plotted))
    if missing: warnings.append(f'No dense snapshot for requested plot frames: {missing}')
    summaries = []
    for scope in ['rgb', 'depth', 'union']:
        group = [r for r in rows if r['scope'] == scope]
        total = {k: sum(r[k] for r in group) for k in ['support_pixels', 'candidate_excluded_pixels', 'applied_excluded_pixels', 'retained_pixels']}
        summaries.append(dict(scope=scope, **total, applied_excluded_percent=percent(total['applied_excluded_pixels'], total['support_pixels'])))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for scope in ['rgb', 'depth', 'union']:
        g = [r for r in rows if r['scope'] == scope]
        axes[0].plot([r['frame'] for r in g], [r['applied_excluded_percent'] if r['applied_excluded_percent'] is not None else np.nan for r in g], label=scope)
    axes[0].set(ylabel='Actual exclusions / support (%)'); axes[0].legend()
    g = [r for r in rows if r['scope'] == 'union']
    axes[1].plot([r['frame'] for r in g], [r['tracking_iterations'] for r in g])
    axes[1].set(xlabel='Frame', ylabel='Tracking iterations')
    for ax in axes: ax.grid(alpha=.25)
    fig.savefig(args.output/'decision_trends.png', dpi=120); plt.close(fig)
    for name, data in [('frame_statistics', rows), ('summary_statistics', summaries), ('dense_region_statistics', regions), ('grid_statistics', cells)]:
        write_csv(args.output/(name+'.csv'), data)
    stats_paths = sorted(run.rglob('stats_final.json'))
    ate = [{'source': str(path.relative_to(run)), 'statistics': json.loads(path.read_text())} for path in stats_paths]
    report = dict(run=str(run), seed=config.get('experiment_seed'), config=config,
                  frames=len(log), frame_ids=sorted(log), dense_frames=dense_frames,
                  mask_applied_frames=sum(bool(r['mask_applied']) for r in log.values()),
                  fallback_frames=sum(bool(r.get('fallback')) for r in log.values()),
                  mean_tracking_iterations=float(np.mean([post[f]['tracking_iterations'] for f in log])),
                  pooled_support_statistics=summaries, ate_files=ate, warnings=warnings,
                  edge_radius=args.edge_radius, edge_abs=.05, edge_rel=.02)
    if not ate: report['warnings'].append('No stats_final.json found; include ATE separately.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.output/'README.txt').write_text(
        'Decision-time diagnostics only; no SLAM rerun, dataset or Torch required.\n'
        'Exclusions are relative to each loss support BEFORE the first masked update.\n'
        'RGB and depth overlap; union is not their sum. Counts pool repeated pixel observations.\n'
        'Grid rows divide the image into 4x4 cells. Region rows exist ONLY for dense frames.\n'
        'Near-invalid, near-edge, interior and invalid-observation partition each support.\n'
        'RGB support can include invalid sensor depth. Interior does not imply static truth.\n'
        'Blank CSV percentages mean zero support, not zero error.\n'
        'Residual plots share +/-0.2 m; positive means rendered depth farther.\n'
        'No per-frame ATE/RPE or causal claims are inferred from masks.\n'
        'No raw NPZ or source images are included in the report archive.\n')
    archive = args.output/'analysis_report.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(args.output.iterdir()):
            if f != archive and f.is_file(): z.write(f, f.name)
    print('Send this compact report:', archive.resolve())
    for warning in report['warnings']: print('Warning:', warning)


if __name__ == '__main__':
    main()
