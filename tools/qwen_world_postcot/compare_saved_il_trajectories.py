"""CPU-only comparison of saved single-candidate requests against one raw GT.

This computes deployment errors, not the models' heterogeneous training losses.
No model forward, scoring, checkpoint selection, or parameter update is performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import pickle
import re
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def errors(prediction, target):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != (8, 3) or y.shape != (8, 3) or not np.isfinite([p, y]).all():
        raise ValueError('Nonfinite or non-8x3 physical trajectory')
    xy = np.abs(p[:, :2] - y[:, :2])
    heading = np.abs((p[:, 2] - y[:, 2] + np.pi) % (2*np.pi) - np.pi)
    distance = np.linalg.norm(p[:, :2] - y[:, :2], axis=-1)
    return dict(x_L1_m=float(xy[:, 0].mean()), y_L1_m=float(xy[:, 1].mean()),
                xy_coordinate_L1_m=float(xy.mean()), ADE_m=float(distance.mean()),
                FDE_m=float(distance[-1]), heading_MAE_rad=float(heading.mean()),
                heading_MAE_deg=float(np.rad2deg(heading.mean())))


def old_gt(raw):
    poses = np.asarray(raw['glo_status']['global_poses'], dtype=np.float64)[:12]
    if poses.shape != (12, 3):
        raise ValueError('Missing processed GT poses')
    current = poses[3]; c, s = np.cos(current[2]), np.sin(current[2])
    xy = (poses[4:, :2] - current[:2]) @ np.array([[c, -s], [s, c]])
    heading = (poses[4:, 2] - current[2] + np.pi) % (2*np.pi) - np.pi
    return np.column_stack([xy, heading])


def load_model(spec, tokens, records, valid_time, workers):
    root = Path(spec['root']); kind = spec['kind']
    identity_path = root / ('identity.json' if kind != 'dual_grid_pt' else 'manifest.json')
    identity = json.loads(identity_path.read_text())
    if kind == 'foresight_npz':
        identity_sha = fingerprint(identity)
        actual = {p.stem for p in (root/'predictions').glob('*.npz')}
        prefix = root/'predictions'
    elif kind == 'single_scene_json':
        body = dict(identity); identity_sha = body.pop('sha256')
        if fingerprint(body) != identity_sha:
            raise ValueError('Changed evaluation identity')
        actual = {p.stem for p in (root/'scenes').glob('*.json')}
        prefix = root/'scenes'
    elif kind == 'dual_grid_pt':
        import torch
        identity_sha = identity['sha256']
        actual = {p.stem for p in (root/'scenes').glob('*/*.pt')}
        prefix = root/'scenes'
    else:
        raise ValueError('Unknown saved-request format')
    if actual != set(tokens):
        raise ValueError(f"Incomplete/mismatched population: {spec['name']}")

    def one(token):
        record = records[token]
        if kind == 'single_scene_json':
            path = prefix/(token+'.json'); raw = path.read_bytes(); row = json.loads(raw)
            seal = row.pop('sha256')
            if seal != fingerprint(row) or row['identity'] != identity_sha or row['status'] != 'OK':
                raise ValueError('Invalid/changed scene JSON')
            prediction = row['requested_trajectory']; feedback = row['feedback']
            if not np.array_equal(np.asarray(prediction), np.asarray(feedback['requested_trajectory'])):
                raise ValueError('Request differs from scored request')
            if feedback['status'] != 'OK' or feedback['protocol'] != 'navsim_v1.1_independent_reference_candidate':
                raise ValueError('Wrong scoring protocol/status')
            score = feedback['metrics']['score']; data_sha = hashlib.sha256(raw).hexdigest()
        elif kind == 'foresight_npz':
            path = prefix/(token+'.npz'); row = json.loads(path.with_suffix('.json').read_text())
            data_sha = sha(path)
            if row['identity_sha256'] != identity_sha or row['status'] != 'ok' or row['proposal_sha256'] != data_sha:
                raise ValueError('Invalid/changed NPZ prediction')
            with np.load(path, allow_pickle=False) as data:
                prediction = data['trajectory']
            score = None
        else:
            path = prefix/token[:2]/(token+'.pt'); seal = json.loads(path.with_suffix('.json').read_text())
            data_sha = sha(path)
            if seal['payload_sha256'] != data_sha or seal['identity_sha256'] != identity_sha:
                raise ValueError('Invalid/changed PT payload')
            row = torch.load(path, map_location='cpu', weights_only=False)
            if row['identity_sha256'] != identity_sha:
                raise ValueError('PT identity differs')
            feedback = row['records'][spec['mode']]
            if feedback['status'] != 'OK' or feedback['protocol'] != 'navsim_v1.1_independent_reference_candidate':
                raise ValueError('Wrong scoring protocol/status')
            prediction = feedback['requested_trajectory']; score = feedback['metrics']['score']
        if row['token'] != token or row['log'] != record['log']:
            raise ValueError('Token/log differs from raw GT')
        return dict(token=token, log=record['log'], physical_log=re.sub(r'_\d+_\d+$', '', record['log']),
                    timestamp_valid=bool(valid_time[token].all()), **errors(prediction, record['requested_GT']),
                    score=score, prediction_sha256=data_sha)

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, tokens))
    # Keep every official scene, including timestamp-gap annotations. Also show
    # a fixed timestamp-valid sensitivity population, identical for all models.
    keys = list(errors(np.zeros((8, 3)), np.zeros((8, 3))))
    results = {}
    for group, selected in [('all', rows), ('timestamp_valid', [r for r in rows if r['timestamp_valid']])]:
        results[group] = dict(scenes=len(selected), **{k: float(np.mean([r[k] for r in selected])) for k in keys})
    results.update(name=spec['name'], identity_sha256=identity_sha, seed=spec['seed'],
                   grid=spec['grid'], checkpoint=spec['checkpoint'], worker_wall_seconds=time.monotonic()-start,
                   physical_log_count=len({r['physical_log'] for r in rows}),
                   saved_predictions_sha256=fingerprint([(r['token'], r['prediction_sha256']) for r in rows]))
    summary_path = Path(spec['summary']); summary = json.loads(summary_path.read_text())
    if kind == 'dual_grid_pt':
        reference_score = summary['modes'][spec['mode']]['mean']['score']*100
    elif kind == 'foresight_npz':
        reference_score = summary['PDMS']*100
    else:
        reference_score = summary['PDMS']
    scores = [r['score'] for r in rows]
    if all(s is not None for s in scores) and abs(np.mean(scores)*100-reference_score) > 1e-8:
        raise ValueError('Re-read scores differ from sealed summary')
    results.update(PDMS=reference_score, score_summary_sha256=sha(summary_path))
    if 'ego_summary' in spec:
        ego = json.loads(Path(spec['ego_summary']).read_text())['groups']['all']
        for actual_key, original_key in [('ADE_m', 'ADE'), ('FDE_m', 'FDE'), ('heading_MAE_rad', 'yaw_MAE_rad')]:
            if abs(results['all'][actual_key]-ego[original_key]) > 1e-5:
                raise ValueError('Raw GT and original label-side fit disagree')
        results['matches_original_ego_summary_within_1e-5'] = True
    return results, rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--plan', required=True); p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args(); output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(args.plan).read_text()); raw_root = Path(plan['raw_gt'])
    manifest = json.loads((raw_root/'manifest.json').read_text())
    seal = json.loads((raw_root/'COMPLETE').read_text())
    if seal['manifest_sha256'] != sha(raw_root/'manifest.json') or manifest['records_sha256'] != sha(raw_root/'records.pkl'):
        raise ValueError('Unsealed/changed raw GT')
    records = pickle.load(open(raw_root/'records.pkl', 'rb')); tokens = sorted(manifest['tokens'])
    if len(tokens) != 12146 or set(tokens) != set(records):
        raise ValueError('Expected the complete official Navtest population')
    valid_time = {t: np.abs(np.cumsum(records[t]['raw_future_dt_seconds'])-np.arange(1, 9)*.5)<.06 for t in tokens}
    # Independent check of the old DDP label coordinate transform on ALL scenes.
    old_root = Path(plan['old_processed_gt']); largest = 0.; label_digest = hashlib.sha256()
    for token in tokens:
        path = old_root/(token+'.pkl'); raw = path.read_bytes(); old = pickle.loads(raw)
        if old['token'] != token or old['log'] != records[token]['log']:
            raise ValueError('Old/raw label population mismatch')
        label_digest.update(raw)
        largest = max(largest, float(np.abs(old_gt(old)-np.asarray(records[token]['requested_GT'])).max()))
    if largest > 1e-4:
        raise ValueError('Old DDP and raw Scene GT coordinate definitions differ')
    result = dict(schema='cross_il_saved_request_fit_v1', scenes=len(tokens),
                  token_sha256=fingerprint(tokens), raw_gt_manifest_sha256=sha(raw_root/'manifest.json'),
                  old_processed_gt_joined_sha256=label_digest.hexdigest(), max_GT_coordinate_difference=largest,
                  timestamp_gap_scenes=sum(not v.all() for v in valid_time.values()),
                  definition='Final requested trajectories vs raw Scene GT, nominal 0.5..4s, rear axle x-forward/y-left; no simulated states.',
                  note='Deployment L1/ADE, NOT the native training objective; historical seed/grid/input/architecture differences remain.',
                  code_sha256=sha(__file__), plan_sha256=sha(args.plan), models=[])
    for spec in plan['models']:
        print('Analyzing', spec['name'], flush=True)
        summary, rows = load_model(spec, tokens, records, valid_time, args.workers)
        with (output/(spec['name']+'_scenes.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        result['models'].append(summary)
        print(json.dumps(summary), flush=True)
    (output/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    fields = ['name', 'checkpoint', 'seed', 'grid', 'PDMS'] + list(result['models'][0]['all'])
    with (output/'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for model in result['models']:
            writer.writerow({**{k:model[k] for k in fields if k not in model['all']}, **model['all']})
    (output/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE', analysis_sha256=sha(output/'analysis.json')))+'\n')


if __name__ == '__main__':
    main()
