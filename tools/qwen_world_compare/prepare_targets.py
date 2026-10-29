"""Prepare sealed current-linked future track targets; never an actor input."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
from PIL import Image

from starVLA.rl.qwen_world_compare.track_targets import (
    SCHEMA, FUTURE_TIMES_S, TIMESTAMP_TOLERANCE_US, LIDAR_IDENTITY_TOLERANCE,
    audit_annotation_frame, build_scene_tracks, file_sha, official_coordinate_source)
from starVLA.rl.qwen_world_postcot.contracts import fingerprint


SOURCE = None
RAW = None
DESTINATION = None


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_log(argument):
    log, tokens = argument
    source_path = RAW / (log + '.pkl')
    with source_path.open('rb') as stream:
        frames = pickle.load(stream)
    by_token = {str(frame['token']): index for index, frame in enumerate(frames)}
    if len(by_token) != len(frames):
        raise ValueError('Repeated raw frame tokens in ' + log)
    audits = [audit_annotation_frame(frame) for frame in frames]
    audit = {key: sum(row[key] for row in audits) if key in ('frames', 'boxes')
             else max((row[key] for row in audits), default=0.) for key in audits[0]}
    # All current source frames in a log use the same native camera resolution;
    # old preparation used this same first-scene size contract.
    row = SOURCE[tokens[0]]
    sizes = {}
    for camera, path in row['images'].items():
        with Image.open(path) as image:
            sizes[camera] = image.size
    records = {}
    for position, token in enumerate(tokens):
        current = frames[by_token[token]]
        if int(current['timestamp']) != int(SOURCE[token]['timestamp']):
            raise ValueError('Raw and sealed current timestamps differ: ' + token)
        records[token] = build_scene_tracks(frames, by_token[token], sizes,
            SOURCE[token]['objects'], coordinates_audited=True,
            verify_current_builder=(position == 0))
    shard = 'shards/' + log + '.pkl'
    path = DESTINATION / shard
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('wb') as stream:
        pickle.dump(records, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return shard, dict(tokens=tokens, count=len(tokens), sha256=file_sha(path),
                       bytes=path.stat().st_size, raw_log_sha256=file_sha(source_path),
                       valid_track_horizon_count=sum(int(r['future_valid'].sum()) for r in records.values()),
                       possible_track_horizon_count=sum(r['current_count'] * 4 for r in records.values()),
                       duplicate_future_matches=sum(r['duplicate_future_track_matches'] for r in records.values()),
                       valid_future_frame_count=sum(int(r['future_time_valid'].sum()) for r in records.values()),
                       audit=audit)


def prepare_role(source, role, raw, destination, workers, limit):
    global SOURCE, RAW, DESTINATION
    original = source / role
    original_manifest = json.loads((original / 'manifest.json').read_text())
    original_complete = json.loads((original / 'COMPLETE').read_text())
    if (original_manifest['cohort'] != role or
            original_complete['manifest_sha256'] != file_sha(original / 'manifest.json') or
            original_manifest['records_sha256'] != file_sha(original / 'records.pkl')):
        raise ValueError('Source data must be the sealed current-input population')
    tokens = original_manifest['tokens']
    if limit:
        tokens = [tokens[int(index)] for index in np.linspace(0, len(tokens) - 1,
                  min(limit, len(tokens)), dtype=np.int64)]
    out = destination / role
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE').exists():
        old = json.loads((out / 'manifest.json').read_text())
        if old['tokens'] != tokens or old['source_manifest_sha256'] != file_sha(original / 'manifest.json'):
            raise ValueError('Existing complete future tracks have another identity')
        return dict(role=role, status='REUSED', count=len(tokens), root=str(out))
    with (original / 'records.pkl').open('rb') as stream:
        SOURCE = pickle.load(stream)
    RAW, DESTINATION = raw, out
    grouped = {}
    for token in tokens:
        grouped.setdefault(SOURCE[token]['log'], []).append(token)
    shards = {}
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('fork')) as pool:
        for shard, record in pool.map(build_log, grouped.items()):
            shards[shard] = record
            atomic(out / 'progress.json', dict(role=role, completed_logs=len(shards),
                total_logs=len(grouped), completed_scenes=sum(r['count'] for r in shards.values()),
                total_scenes=len(tokens), at=time.time()))
    index = {token: shard for shard, record in shards.items() for token in record['tokens']}
    if set(index) != set(tokens):
        raise ValueError('Future track population is incomplete')
    audit_keys = next(iter(shards.values()))['audit']
    audit = {key: sum(row['audit'][key] for row in shards.values()) if key in ('frames', 'boxes')
             else max(row['audit'][key] for row in shards.values()) for key in audit_keys}
    audit.update(status='PASS', logs=len(shards), selected_scenes=len(tokens),
        official_source=official_coordinate_source(),
        official_planar_rule='ego-local annotation xy/yaw -> global using rear-axle SE2; future -> fixed current rear-axle SE2',
        lidar_identity_tolerance=LIDAR_IDENTITY_TOLERANCE,
        current_object_source='tools.qwen_world_postcot.prepare_training_data.current_objects',
        current_selection_preserved=True, future_selection_uses_current_track_token_only=True,
        full_current_frustum_recomputed_first_scene_each_log=True,
        remaining_current_selections='sealed original current_objects; exact current annotation geometry/class/track linking',
        no_per_horizon_rematching=True)
    atomic(out / 'coordinate_audit.json', audit)
    manifest = dict(schema=SCHEMA, cohort=role, count=len(tokens), tokens=tokens,
        source_manifest_sha256=file_sha(original / 'manifest.json'),
        source_records_sha256=original_manifest['records_sha256'],
        source_role_partition_sha256=original_manifest['role_partition_sha256'],
        tokens_sha256=fingerprint(tokens), future_times_s=FUTURE_TIMES_S,
        future_shape=[32, 4, 6], future_channels=['x', 'y', 'length', 'width', 'sin_yaw', 'cos_yaw'],
        current_shape='N<=32 x8 ego_xyz_lwh_sinyaw_cosyaw',
        timestamp_tolerance_us=TIMESTAMP_TOLERANCE_US,
        missing_timestamp=-1, missing_track_state='zero storage with future_valid=False; not a zero target',
        coordinate_audit_sha256=file_sha(out / 'coordinate_audit.json'),
        source_files={str(path): file_sha(path) for path in [
            Path(__file__), Path(__file__).parents[2] / 'starVLA/rl/qwen_world_compare/track_targets.py',
            Path(__file__).parents[1] / 'qwen_world_postcot/prepare_training_data.py']},
        shards=shards, token_index=index, workers=workers, qualification=bool(limit),
        total_valid_track_horizons=sum(r['valid_track_horizon_count'] for r in shards.values()),
        total_possible_track_horizons=sum(r['possible_track_horizon_count'] for r in shards.values()),
        bytes=sum(r['bytes'] for r in shards.values()))
    atomic(out / 'manifest.json', manifest)
    atomic(out / 'COMPLETE', dict(status='COMPLETE', manifest_sha256=file_sha(out / 'manifest.json'),
                                 count=len(tokens), at=time.time()))
    SOURCE = None
    return dict(role=role, status='COMPLETE', count=len(tokens), root=str(out),
                bytes=manifest['bytes'], valid_track_horizons=manifest['total_valid_track_horizons'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-data', required=True)
    parser.add_argument('--raw-logs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--roles', nargs='+', choices=['TRAIN', 'DEV'], default=['TRAIN', 'DEV'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='Qualification-only evenly spaced scenes; never formal TRAIN')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'owner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic(output / 'OWNER.json', dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
        start_unix=time.time(), args=vars(args), GPU_used=False))
    try:
        source, raw = Path(args.source_data), Path(args.raw_logs)
        partition = json.loads((source / 'DATA_PARTITION.json').read_text())
        populations = {name: set(partition[name.lower() + '_physical_logs']) for name in ('TRAIN', 'DEV', 'NAVTEST')}
        if any(populations[a] & populations[b] for a, b in [('TRAIN', 'DEV'), ('TRAIN', 'NAVTEST'), ('DEV', 'NAVTEST')]):
            raise ValueError('Physical log overlap in source partition')
        results = [prepare_role(source, role, raw, output, args.workers, args.limit) for role in args.roles]
        atomic(output / 'COMPLETE.json', dict(status='COMPLETE', results=results,
            source_partition_sha256=file_sha(source / 'DATA_PARTITION.json'),
            physical_log_disjoint=True, qualification=bool(args.limit), GPU_used=False, at=time.time()))
        print(json.dumps(results), flush=True)
    except BaseException:
        atomic(output / 'FAILURE.json', dict(at=time.time(), error=traceback.format_exc(), GPU_used=False))
        raise


if __name__ == '__main__':
    main()

