"""Prepare causal history inputs and a separate label-only teacher image index.

Default smoke scope is at most64 scenes. Registered larger preparation requires
an explicit budget file and never starts teacher/GPU work. Existing student
indices/order/partition are retained; scores and GT are not opened.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from starVLA.dataloader.st_world_history import (
    VIEWS, build_history_record, file_sha256, image_reference, resolve_camera_path)
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import atomic_json, file_lock, validate_target_manifest


class TrustedNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module.replace('numpy._core', 'numpy.core'), name)


def build_target_scene(frames, token, log, split, sensor_roots, *, times_s=(0., .5, 1., 2., 4.), tolerance_s=.05):
    """Label-side future RGB lookup; never called from a model input function."""
    if not times_s or times_s[0] != 0 or list(sorted(set(times_s))) != list(times_s) or any(t < 0 for t in times_s):
        raise ValueError('Target times must start at0 and be positive/increasing afterwards')
    if not 0 <= tolerance_s <= .05:
        raise ValueError('Timestamp tolerance must be <=50ms')
    locations = {f['token']: i for i, f in enumerate(frames)}
    if len(locations) != len(frames):
        raise ValueError('Duplicate raw token')
    times = np.array([int(f['timestamp']) for f in frames], dtype=np.int64)
    if not (np.diff(times) > 0).all():
        raise ValueError('Nonmonotonic raw log')
    t0 = int(times[locations[token]])
    rows = []
    for t in times_s:
        wanted = t0 + round(t * 1e6)
        distance = np.abs(times - wanted)
        eligible = np.flatnonzero((distance <= round(tolerance_s * 1e6)) & ((times == t0) if t == 0 else (times > t0)))
        frame = None
        if len(eligible):
            nearest = eligible[distance[eligible] == distance[eligible].min()]
            if len(nearest) != 1:
                raise ValueError('Ambiguous teacher timestamp')
            frame = frames[int(nearest[0])]
        refs = []
        for name in VIEWS:
            cam = frame.get('cams', {}).get(name) if frame else None
            path = resolve_camera_path(cam['data_path'], sensor_roots) if cam else None
            refs.append(image_reference(path) if path is not None else None)
        if t == 0 and any(ref is None for ref in refs):
            raise ValueError('Current image missing from registered population')
        rows.append({'requested_s': float(t), 'timestamp_us': int(frame['timestamp']) if frame else None,
                     'frame_token': frame['token'] if frame else None, 'views': refs})
    return {'token': token, 'log': log, 'split': split, 'timestamp_us': t0, 'frames': rows}


def stable_selection(index, max_scenes, seed):
    # One scene per log first, then further scenes; no score-dependent selection.
    groups = defaultdict(list)
    key = lambda value: identity_hash({'seed': seed, 'value': value})
    for row in index:
        groups[row['log']].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: key(r['token']))
    logs = sorted(groups, key=key)
    selected = []
    depth = 0
    while len(selected) < min(max_scenes, len(index)):
        for log in logs:
            if depth < len(groups[log]):
                selected.append(groups[log][depth])
                if len(selected) == min(max_scenes, len(index)):
                    return selected
        depth += 1
    return selected


def pin_source_records(student, selected):
    """Pin supervised payload bytes without decoding GT or using its values."""
    student=Path(student)
    rows=[]
    for original in selected:
        row=dict(original)
        for key,relative in [('current_record_sha256','current/'+row['token']+'.json'),
                             ('ego_record_sha256','ego/'+row['token']+'.pt')]:
            actual=file_sha256(student/relative)
            if key in row and row[key]!=actual:
                raise ValueError('Previously pinned source record changed: '+relative)
            row[key]=actual
        rows.append(row)
    return rows


def upgrade_selection(args):
    """Explicit new manifest version; never rewrites old smoke evidence."""
    old_path=Path(args.pin_selection).resolve();out=Path(args.output).resolve()
    if out==old_path.parent:
        raise ValueError('Selection pin upgrade requires a new output namespace')
    source=Path(args.student_root)
    ident=json.loads((source/'identity.json').read_text());index=json.loads((source/'index.json').read_text())
    old=json.loads(old_path.read_text())
    if (args.max_scenes<1 or len(old['scenes'])>args.max_scenes or
            old['schema']!='st_fixed_scene_list_v1' or old['split']!=args.split or ident['split']!=args.split or
            old['source_index_hash']!=identity_hash(index) or ident['index_sha256']!=old['source_index_hash']):
        raise ValueError('Pinned selection source/role/budget mismatch')
    lookup={r['token']:r['log'] for r in index}
    if any(lookup.get(r['token'])!=r['log'] for r in old['scenes']):
        raise ValueError('Selection outside original source population')
    # Existing history receipts already pin current JSON; retain that binding.
    hp=old_path.parent/'HISTORY_MANIFEST.json'
    pins={}
    if hp.exists():
        history=json.loads(hp.read_text());complete=json.loads((hp.parent/'COMPLETE.json').read_text())
        if (complete['history_manifest_sha256']!=file_sha256(hp) or
                history['identity']!=identity_hash({k:v for k,v in history.items() if k!='identity'})):
            raise ValueError('Legacy history receipt mismatch')
        pins={r['token']:r['current_record_sha256'] for r in history['scenes']}
    rows=[dict(r,**({'current_record_sha256':pins[r['token']]} if r['token'] in pins else {})) for r in old['scenes']]
    rows=pin_source_records(source,rows)
    value={**old,'scenes':rows,'source_student_identity':ident['identity'],
           'ego_identity_sha256':file_sha256(source/'ego_identity.json'),
           'pin_upgrade':{'source_manifest':str(old_path),'source_sha256':file_sha256(old_path),
                          'history_manifest_sha256':file_sha256(hp) if hp.exists() else None,
                          'writer_sha256':file_sha256(__file__)}}
    with file_lock(out/'PIN_SELECTION.lock',blocking=False):
        atomic_json(out/'SELECTED_SCENES.json',value,immutable=True)
        result={'status':'COMPLETE','schema':'st_selection_content_pin_upgrade_v1',
                'manifest_sha256':file_sha256(out/'SELECTED_SCENES.json'),'scenes':len(rows),
                'source_manifest_sha256':file_sha256(old_path),'old_evidence_modified':False,
                'gt_values_decoded':False}
        atomic_json(out/'COMPLETE.json',result,immutable=True)
    return result


def run(args):
    if args.pin_selection:
        return upgrade_selection(args)
    if not args.raw_log_root or not args.sensor_root or not args.partition:
        raise ValueError('History preparation requires raw-log-root, sensor-root and partition')
    if args.max_scenes < 1 or not 0 <= args.tolerance <= .05:
        raise ValueError('Positive explicit sample budget and tolerance<=.05 required')
    if args.mode == 'smoke' and args.max_scenes > 64:
        raise ValueError('Smoke preparation is bounded to64 scenes')
    if args.mode == 'registered':
        if not args.registration:
            raise ValueError('Larger preparation requires a predeclared registration')
        budget = json.loads(Path(args.registration).read_text())
        if (budget['phase'] != 'data_preparation' or budget['split'] != args.split or
                budget['max_scenes'] < args.max_scenes):
            raise ValueError('Preparation outside explicit registration budget')
    student = Path(args.student_root).resolve()
    identity = json.loads((student / 'identity.json').read_text())
    index = json.loads((student / 'index.json').read_text())
    if (identity['schema'] != 'ddpolicy_current_cameras_v1' or identity['split'] != args.split or
            identity['index_sha256'] != identity_hash(index) or
            identity['identity'] != identity_hash({k:v for k,v in identity.items() if k != 'identity'})):
        raise ValueError('Current student source/index/role mismatch')
    partition = json.loads(Path(args.partition).read_text())
    if file_sha256(args.partition) != identity['partition_sha256']:
        raise ValueError('Original train/dev partition changed')
    if [r['token'] for r in index] != partition[args.split + '_tokens']:
        raise ValueError('Student scene population changed')
    if {partition['token_logs'][t] for t in partition['train_tokens']} & {partition['token_logs'][t] for t in partition['dev_tokens']}:
        raise ValueError('Train/dev raw log overlap')
    if args.scene_list:
        selected = json.loads(Path(args.scene_list).read_text())
        if isinstance(selected, dict):
            selected = selected['scenes']
        by_token = {r['token']: r for r in index}
        if len(selected) > args.max_scenes or len({r['token'] for r in selected}) != len(selected):
            raise ValueError('Duplicate/overbudget explicit scene list')
        for row in selected:
            if row['token'] not in by_token or row['log'] != by_token[row['token']]['log']:
                raise ValueError('Explicit scene list outside locked role')
        selected = [by_token[r['token']] for r in selected]
    else:
        selected = stable_selection(index, args.max_scenes, args.seed)
    selected = pin_source_records(student,selected)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), 'student_root': str(student), 'source_identity': identity['identity'],
              'partition_sha256': file_sha256(args.partition), 'selected': selected,
              'writer_sha256': file_sha256(__file__)}
    config.pop('resume')
    config['identity'] = identity_hash(config)
    with file_lock(out / 'PREPARE.lock', blocking=False):
        if (out / 'resolved_config.json').exists() and not args.resume:
            raise ValueError('Existing preparation requires --resume')
        atomic_json(out / 'resolved_config.json', config, immutable=True)
        atomic_json(out / 'SELECTED_SCENES.json', {'schema': 'st_fixed_scene_list_v1', 'split': args.split,
                    'seed': args.seed, 'source_index_hash': identity['index_sha256'], 'scenes': selected}, immutable=True)
        grouped = defaultdict(list)
        for row in selected:
            grouped[row['log']].append(row['token'])
        history_rows, targets, sources = {}, {}, []
        for log, tokens in grouped.items():
            if Path(log).name != log:
                raise ValueError('Invalid raw log identifier')
            raw = Path(args.raw_log_root) / (log + '.pkl')
            digest = file_sha256(raw)
            with raw.open('rb') as f:
                frames = TrustedNumpyUnpickler(f).load()
            sources.append({'log': log, 'path': str(raw), 'sha256': digest})
            for token in tokens:
                current_path = student / 'current' / (token + '.json')
                current = json.loads(current_path.read_text())
                if current['identity'] != identity['identity']:
                    raise ValueError('Foreign current record')
                past = build_history_record(frames, token, log, current,
                    [args.sensor_root, *args.fallback_sensor_root], raw_log_sha256=digest, tolerance_s=args.tolerance)
                rel = 'history/' + token + '.json'
                atomic_json(out / rel, past, immutable=True)
                history_rows[token] = {'token': token, 'log': log, 'record': rel, 'sha256': file_sha256(out / rel),
                                       'current_record_sha256': file_sha256(current_path)}
                # The future lookup remains in a separate artifact, never in history.
                targets[token] = build_target_scene(frames, token, log, args.split,
                    [args.sensor_root, *args.fallback_sensor_root], times_s=args.target_times, tolerance_s=args.tolerance)
                if any(Path(v['path']).resolve() != Path(p).resolve() for v,p in zip(targets[token]['frames'][0]['views'],current['image_paths'])):
                    raise ValueError('Original current view order/path changed')
        common = {'split': args.split, 'source_student_identity': identity['identity'],
                  'source_index_sha256': identity['index_sha256'], 'partition_sha256': identity['partition_sha256'],
                  'selection_identity': identity_hash(selected), 'view_names': list(VIEWS),
                  'source_logs': sources, 'timestamp_tolerance_s': args.tolerance}
        history = {'schema': 'st_history_manifest_v1', **common,
                   'scenes': [history_rows[r['token']] for r in selected]}
        history['identity'] = identity_hash(history)
        teacher = {'schema': 'st_teacher_requests_v1', **common, 'times_s': args.target_times,
                   'scenes': [targets[r['token']] for r in selected]}
        teacher['identity'] = identity_hash(teacher)
        validate_target_manifest(teacher)
        atomic_json(out / 'HISTORY_MANIFEST.json', history, immutable=True)
        atomic_json(out / 'TEACHER_REQUESTS.json', teacher, immutable=True)
        done = {'status': 'COMPLETE', 'scenes': len(selected), 'logs': len(grouped),
                'history_manifest_sha256': file_sha256(out / 'HISTORY_MANIFEST.json'),
                'teacher_requests_sha256': file_sha256(out / 'TEACHER_REQUESTS.json'),
                'history_identity': history['identity'], 'teacher_requests_identity': teacher['identity'],
                'mode': args.mode, 'original_gt_or_scores_read': False}
        atomic_json(out / 'COMPLETE.json', done, immutable=True)
        return done


def parser():
    p = argparse.ArgumentParser(__doc__)
    for key in ('student-root', 'output'):
        p.add_argument('--' + key, required=True)
    for key in ('raw-log-root','sensor-root','partition'):
        p.add_argument('--' + key)
    p.add_argument('--pin-selection',help='Explicitly upgrade an old selected manifest into a NEW output directory with current/ego content pins')
    p.add_argument('--fallback-sensor-root', action='append', default=[])
    p.add_argument('--split', choices=['train', 'dev'], required=True)
    p.add_argument('--mode', choices=['smoke', 'registered'], default='smoke')
    p.add_argument('--registration')
    p.add_argument('--max-scenes', type=int, required=True)
    p.add_argument('--scene-list')
    p.add_argument('--seed', type=int, default=20261017)
    p.add_argument('--tolerance', type=float, default=.05)
    p.add_argument('--target-times', nargs='+', type=float, default=[0., .5, 1., 2., 4.])
    p.add_argument('--resume', action='store_true')
    return p


def main():
    args = parser().parse_args()
    try:
        print(json.dumps(run(args), sort_keys=True))
    except Exception as exc:
        print(json.dumps({'status': 'FAILED', 'error': repr(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
