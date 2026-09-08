"""Read-only full-data preflight. Calls the frozen production target/preprocess code."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.agents.EpisodeDrive.drivevla_features import DriveVLAFeatureBuilder
from navsim.agents.EpisodeDrive.planreg_v2 import CACHE_SCHEMA, LONG_TARGET_VERSION
from navsim.agents.EpisodeDrive.planreg_v2.targets import V2TrajectoryTargetBuilder
from navsim.agents.EpisodeDrive.planreg_v2.data import reject_cached_representations
from navsim.agents.EpisodeDrive.planreg_v2.normalizers import measured_statistics, TrajectoryNormalizer
from navsim.agents.EpisodeDrive.planreg_v2.motion import interval_selection
from navsim.agents.EpisodeDrive.planreg_v2.runtime import source_fingerprint
from navsim.agents.EpisodeDrive.planreg_v2.backbone import V2_SYSTEM_PROMPT
from navsim.agents.EpisodeDrive.layers.world_model.future_image_io import decode_path_tensor
from navsim.agents.EpisodeDrive.utils.internvl_preprocess import tile_metadata_from_image_size
from navsim.agents.EpisodeDrive.utils.internvl_tokenize import build_internvl_model_inputs
from navsim.agents.EpisodeDrive.utils.utils import build_drivevla_questions

SCAN_SCHEMA = 'planreg_v2p2_full_input_scan_v1'
PREPROCESS_FILES = (
    'navsim/agents/EpisodeDrive/planreg_v2/data.py',
    'navsim/agents/EpisodeDrive/planreg_v2/targets.py',
    'navsim/agents/EpisodeDrive/planreg_v2/motion.py',
    'navsim/agents/EpisodeDrive/planreg_v2/normalizers.py',
    'navsim/agents/EpisodeDrive/utils/internvl_preprocess.py',
    'navsim/agents/EpisodeDrive/utils/internvl_tokenize.py',
    'navsim/agents/EpisodeDrive/utils/utils.py',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def signature(path):
    s = Path(path).stat()
    if s.st_size <= 0: raise ValueError('Empty input file: ' + str(path))
    return [s.st_size, s.st_mtime_ns]


def preprocessing_identity():
    return {p: sha(ROOT / p) for p in PREPROCESS_FILES}


_tokenizer = None


def init_worker(vlm):
    global _tokenizer
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    _tokenizer = AutoTokenizer.from_pretrained(vlm, trust_remote_code=True, use_fast=False)


def token_length(feature, tile_count):
    question = build_drivevla_questions(feature['history_trajectory'], feature['high_command_one_hot'])[0]
    tokens = build_internvl_model_inputs(_tokenizer, [question], [tile_count], V2_SYSTEM_PROMPT)
    return int(tokens['attention_mask'].sum())


def inspect_log(job):
    log, records, raw_root, sensors = job
    raw_path = Path(raw_root) / (log + '.pkl')
    inventory = {str(raw_path): signature(raw_path)}
    with raw_path.open('rb') as f: raw = pickle.load(f)
    by_token = {frame['token']: i for i, frame in enumerate(raw)}
    builder = V2TrajectoryTargetBuilder('ego')
    feature_builder = DriveVLAFeatureBuilder(cache_hidden_state=False)
    sensor = SensorConfig(cam_f0=[3,4,6,11], cam_l0=[], cam_l1=[], cam_l2=[], cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[], lidar_pc=[])
    images = {}
    dimensions, tiles, lengths, reasons = Counter(), Counter(), Counter(), Counter()
    counts = dict(gt=np.zeros(8, np.int64), future=np.zeros(3, np.int64), motion=np.zeros(3, np.int64), long=0)
    statistics, extrema, errors, content = [], [], [], hashlib.sha256()

    def check_image(path):
        if path not in images:
            before = signature(path)
            with Image.open(path) as im:
                im.load()  # Detect truncated/corrupt pixels, not just a readable JPEG header.
                size = im.size
            if signature(path) != before: raise ValueError('Image changed during scan: ' + path)
            inventory[path] = before
            images[path] = size
        return images[path]

    for record in records:
        token = record['token']
        try:
            i = by_token[token]
            if i < 3: raise ValueError('Missing current numeric history')
            frames = raw[i-3:i+11]
            if any(f['log_name'] != log for f in frames): raise ValueError('Cross-log raw window')
            path = record['cache_path']; inventory[path] = signature(path)
            data = torch.load(path, map_location='cpu', weights_only=False)
            if data.get('schema') != CACHE_SCHEMA or data.get('long_target_version') != LONG_TARGET_VERSION:
                raise ValueError('Stale per-scene cache version')
            reject_cached_representations(data)
            target, feature = data['targets'], data['features']
            if target.get('long_target_version') != LONG_TARGET_VERSION: raise ValueError('Old long target')
            scene = Scene.from_scene_dict_list(frames, Path(sensors), 4, len(frames)-4, sensor, load_image_path=True)
            expected = builder.compute_targets(scene)
            for key, value in expected.items():
                same = torch.equal(value, target[key]) if torch.is_tensor(value) else value == target[key]
                if not same: raise ValueError('Cached supervision differs from raw production target: ' + key)
            reference_feature = feature_builder.compute_features(scene.get_agent_input())
            for key, value in reference_feature.items():
                # The production input-cache builder replaces the legacy byte
                # encoding with its fixed-size path helper. Compare its decoded
                # path below, not the two intentionally different tensor formats.
                if key in ('image_path_tensor', 'image_path_length'):
                    continue
                if torch.is_tensor(value) and not torch.equal(value, feature[key]):
                    raise ValueError('Current feature differs from raw scene: ' + key)
            current = decode_path_tensor(feature['image_path_tensor'], feature['image_path_length'])
            if current != str(scene.frames[3].cameras.cam_f0.image): raise ValueError('Current image/token mismatch')
            size = check_image(current)
            metadata = tile_metadata_from_image_size(*size)
            if feature.get('tile_metadata') is not None and not torch.equal(torch.as_tensor(metadata), feature['tile_metadata']):
                raise ValueError('Cached tile metadata mismatch')
            count = len(metadata); length = token_length(feature, count)
            dimensions['%dx%d' % size] += 1; tiles[str(count)] += 1; lengths[str(length)] += 1
            extrema.append(dict(token=token, log=log, image=current, size=list(size), tiles=count, prefix_tokens=length))
            for h, offset in enumerate((1,3,8)):
                if bool(target['future_valid_mask'][h]):
                    future = decode_path_tensor(target['future_image_paths'][h], target['future_image_path_lengths'][h])
                    if future != str(scene.frames[3+offset].cameras.cam_f0.image): raise ValueError('Future path/offset mismatch')
                    check_image(future)
                else:
                    cause = 'logged_window_short' if len(scene.frames) <= 3+offset else 'timestamp_or_logged_camera_unavailable'
                    reasons['future_%s:%s' % (offset, cause)] += 1
            _, covered = interval_selection(target['motion_timestamps'][None], target['motion_valid'][None])
            counts['gt'] += target['trajectory_valid'].numpy().astype(np.int64)
            counts['future'] += target['future_valid_mask'].numpy().astype(np.int64)
            counts['motion'] += covered[0].numpy().astype(np.int64)
            counts['long'] += int(target['trajectory_long_valid'])
            if not target['trajectory_long_valid']:
                reasons['long:' + ('logged_window_short' if len(frames) < 14 else 'timestamp_or_pose_contract')] += 1
            if not target['trajectory_valid'].all(): reasons['gt:logged_time_or_missing_point'] += 1
            if not covered.all(): reasons['motion:incomplete_1_2_5_interval'] += 1
            metric_path = record['metric_cache_path']; inventory[metric_path] = signature(metric_path)
            with lzma.open(metric_path, 'rb') as f: metric = pickle.load(f)
            # Older production caches lack log_name/timepoint; use their stored file_path and ego time.
            if Path(metric.file_path).parent.name != token or Path(metric.file_path).parents[2].name != log:
                raise ValueError('Metric cache identifies a different scene')
            if metric.ego_state.time_point.time_us != raw[i]['timestamp']: raise ValueError('Metric/current timestamp mismatch')
            for key in ('observation', 'centerline', 'drivable_area_map', 'pdm_progress'):
                if not hasattr(metric, key): raise ValueError('Incomplete production metric cache: ' + key)
            statistics.append((token, target['trajectory'].numpy(), target['trajectory_valid'].numpy()))
            content.update(token.encode()); content.update(bytes.fromhex(sha(path)))
        except Exception as e:
            errors.append(dict(token=token, log=log, error=type(e).__name__ + ': ' + str(e)))
    extrema = sorted(extrema, key=lambda r: (r['tiles'], r['prefix_tokens']), reverse=True)[:4]
    return dict(log=log, checked=len(records), errors=errors, inventory=inventory, dimensions=dict(dimensions),
        tiles=dict(tiles), lengths=dict(lengths), reasons=dict(reasons), statistics=statistics, extrema=extrema,
        counts={k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in counts.items()},
        cache_content_sha256=content.hexdigest())


def scan(args):
    start = time.time(); output = Path(args.output)
    if output.exists(): raise FileExistsError('Immutable scan output already exists: ' + str(output))
    manifest = json.loads(Path(args.manifest).read_text())
    if (manifest.get('schema'), manifest.get('long_target_version'), manifest.get('split'), manifest.get('smoke')) != (CACHE_SCHEMA, LONG_TARGET_VERSION, 'trainval_final_fit', False):
        raise ValueError('Full new-version final-fit manifest required')
    authorized = json.loads(Path(args.tokens).read_text()); records = manifest['records']
    actual = [r['token'] for r in records]
    if len(actual) != 103288 or len(set(actual)) != 103288 or set(actual) != set(authorized): raise ValueError('Exact authorized 103288 unique token set required')
    metric = {Path(r['file_name']).parent.name:r['file_name'] for r in csv.DictReader(open(args.metric_metadata))}
    groups = {}
    for r in records:
        if metric.get(r['token']) != r['metric_cache_path']: raise ValueError('Metric metadata mapping mismatch')
        groups.setdefault(r['log'], []).append(r)
    jobs = [(log, group, args.logs, args.sensors) for log,group in sorted(groups.items())]
    inventory, stats, errors, extrema, log_hashes = {}, [], [], [], {}
    totals = {k:Counter() for k in ('dimensions','tiles','lengths','reasons')}
    counts = dict(gt=np.zeros(8,np.int64),future=np.zeros(3,np.int64),motion=np.zeros(3,np.int64),long=0)
    checked = 0
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(args.vlm,)) as pool:
        for result in pool.map(inspect_log, jobs):
            checked += result['checked']; inventory.update(result['inventory']); stats.extend(result['statistics'])
            errors.extend(result['errors']); extrema.extend(result['extrema']); log_hashes[result['log']] = result['cache_content_sha256']
            for key in totals: totals[key].update(result[key])
            for key in counts: counts[key] += result['counts'][key]
            if len(log_hashes)%20 == 0:
                print(json.dumps(dict(checked=checked, logs=len(log_hashes), errors=len(errors), elapsed_seconds=time.time()-start)), flush=True)
    n = len(records); matches = False
    if not errors:
        measured = measured_statistics(stats, 'trainval_final_fit', 'navsim-raw-log:'+manifest['token_sha256'])
        saved = TrajectoryNormalizer.load(args.normalizer)
        matches = all(torch.equal(getattr(measured,k),getattr(saved,k)) for k in ('mean','raw_std','std')) and measured.metadata == saved.metadata
        if not matches: errors.append(dict(error='Full original-GT statistics differ from final-source recomputation'))
    inventory[str(Path(args.manifest).resolve())] = signature(args.manifest)
    inventory[str(Path(args.normalizer).resolve())] = signature(args.normalizer)
    inventory_path = output.with_suffix('.files.json.gz')
    with gzip.open(inventory_path,'wt') as f: json.dump(inventory,f,separators=(',',':'))
    ranges = dict(image_dimensions=dict(totals['dimensions']),tile_count_histogram=dict(totals['tiles']),
        prefix_token_histogram=dict(totals['lengths']),max_tiles=max(map(int,totals['tiles']),default=0),
        min_prefix_tokens=min(map(int,totals['lengths']),default=0),max_prefix_tokens=max(map(int,totals['lengths']),default=0))
    ranges['max_scene_memory_tokens'] = ranges['max_tiles']*16
    ranges['max_llm_tokens_with_queries'] = ranges['max_prefix_tokens']+16
    report=dict(schema=SCAN_SCHEMA,status='PASS' if not errors else 'FAIL',audited_count=checked,
        manifest_path=str(Path(args.manifest).resolve()),manifest_sha256=sha(args.manifest),normalizer_sha256=sha(args.normalizer),
        token_sha256=manifest['token_sha256'],raw_gt_sha256=manifest['raw_gt_sha256'],statistics_contract=manifest['statistics_contract'],
        raw_gt_statistics_recomputed_equal=matches,preprocessing=preprocessing_identity(),source_fingerprint_sha256=source_fingerprint()['sha256'],
        scan_code_sha256=sha(__file__),input_ranges=ranges,valid_counts={k:np.asarray(v).tolist() for k,v in counts.items()},
        valid_rates={k:(np.asarray(v)/n).tolist() for k,v in counts.items()},missing_reasons=dict(totals['reasons']),
        errors=errors,log_count=len(groups),cache_content_per_log_sha256=log_hashes,
        largest_inputs=sorted(extrema,key=lambda r:(r['tiles'],r['prefix_tokens']),reverse=True)[:16],
        file_inventory_path=str(inventory_path.resolve()),file_inventory_sha256=sha(inventory_path),file_count=len(inventory),
        image_validation='Every unique referenced current/valid-future image decoded with PIL.load; no VLM forward',
        metric_validation='Every production metric cache decompressed/unpickled, scene/time and required fields checked',
        elapsed_seconds=time.time()-start)
    output.write_text(json.dumps(report,indent=2)); print(json.dumps({k:report[k] for k in ('status','audited_count','valid_rates','input_ranges','elapsed_seconds')},indent=2))
    if errors: raise RuntimeError('Data errors block formal start; inspect '+str(output))


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('manifest','normalizer','tokens','metric-metadata','logs','sensors','vlm','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=8)
    scan(p.parse_args())
