"""Assemble existing sealed current observations; never fabricate or filter scenes."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re

POPULATION = Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json')
PARTITION = Path('/mnt/project/qwen-world-postcot-20261025/data/DATA_PARTITION.json')
SOURCE_ROOT = Path('/mnt/project/qwen-sourceflow-native-20261018/data')
HARD = Path('/mnt/project/simwam-source-process-20261018/source_manifest.json')
CACHE_ROOTS = [Path('/mnt/project/flow-safe-improvement-20261013/metric_cache_v1'),
               Path('/mnt/project/constrained-flow-noise-20261014/metric_cache_v1'),
               Path('/mnt/project/DriveDreamer-Policy-paired/runs/paired_dev_pdms_v1/metric_cache')]
SCHEMA = 'world_rl_research_scene_stream_v1'

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)

def physical(log):
    return re.sub(r'_\d+_\d+$', '', log)

def sealed(body):
    return dict(body, sha256=fingerprint(body))

def read_sealed(path):
    value = json.loads(Path(path).read_text()); body = dict(value)
    if body.pop('sha256', None) != fingerprint(body):
        raise ValueError('Unsealed or modified research manifest: '+str(path))
    return value

def _raw_records(root, cohort, population_sha):
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads((root/'COMPLETE').read_text())
    if (manifest['cohort'] != cohort or manifest['population_sha256'] != population_sha
            or complete['manifest_sha256'] != file_sha(manifest_path)
            or manifest['records_sha256'] != file_sha(root/'records.pkl')):
        raise ValueError('Raw current-observation source changed: '+str(root))
    with (root/'records.pkl').open('rb') as stream:
        records = pickle.load(stream)
    if set(records) != set(manifest['tokens']):
        raise ValueError('Raw record population incomplete')
    return manifest, records

def write_stream_manifest(path, *, data_root, tokens, name, mode='uniform', seed=42,
                          stage_exposures=None, stage_registry=None):
    """Bind an explicitly named legal sampling population to current observations."""
    data_root = Path(data_root).resolve()
    data = json.loads((data_root/'manifest.json').read_text())
    if len(tokens) != len(set(tokens)) or not tokens or not set(tokens) <= set(data['tokens']):
        raise ValueError('Sampling tokens must be a nonempty unique subset of NAVTRAIN data')
    if mode not in ('uniform', 'adaptive'):
        raise ValueError('Unknown stream mode')
    if mode == 'adaptive' and (not stage_exposures or not stage_registry):
        raise ValueError('Adaptive stream requires fixed stage size and append-only registry')
    body = dict(schema=SCHEMA, name=name, data_root=str(data_root),
                data_manifest_sha256=file_sha(data_root/'manifest.json'),
                population_manifest=str(POPULATION), population_sha256=file_sha(POPULATION),
                tokens=list(tokens), count=len(tokens), mode=mode, seed=int(seed),
                stage_exposures=stage_exposures, stage_registry=str(Path(stage_registry).resolve()) if stage_registry else None,
                observation='current L/F/R images, ego and navigation only',
                selection_population='NAVTRAIN only; NAVTEST tokens and physical logs forbidden',
                old_DEV_is_train_diagnostic=bool(set(tokens)&set(data['old_DEV_tokens'])))
    atomic_json(path, sealed(body))
    return read_sealed(path)

def build_population(output, *, verify_files=True, reuse_asset_audit=None):
    """Seal all 103,288 native NAVTRAIN records using existing source records."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    pop = json.loads(POPULATION.read_text()); roles = json.loads(PARTITION.read_text())
    tokens = pop['train_tokens']+pop['dev_tokens']
    if len(tokens) != len(set(tokens)) or set(tokens)&set(pop['navtest_tokens']):
        raise ValueError('NAVTRAIN/NAVTEST token overlap')
    population_sha = file_sha(POPULATION)
    sources = []; records = {}
    for src, cohort in [('TRAIN_v3', 'TRAIN'), ('DEV_v2', 'DEV')]:
        root = SOURCE_ROOT/src
        manifest, source = _raw_records(root, cohort, population_sha)
        sources.append(dict(root=str(root), manifest_sha256=file_sha(root/'manifest.json'),
                            records_sha256=manifest['records_sha256'], count=len(source)))
        if set(records)&set(source):
            raise ValueError('Duplicate source current observations')
        records.update(source)
    if set(records) != set(tokens):
        raise ValueError('Full NAVTRAIN current record membership differs from source population')
    test_tokens = set(pop['navtest_tokens'])
    test_logs = {pop['token_logs'][t] for t in test_tokens}
    test_physical = {physical(log) for log in test_logs}
    train_logs = {pop['token_logs'][t] for t in tokens}
    train_physical = {physical(log) for log in train_logs}
    if train_logs&test_logs or train_physical&test_physical:
        raise ValueError('NAVTRAIN/NAVTEST log overlap')
    missing_images = []; missing_caches = []; cache_counts = Counter(); reuse = None
    inspect_tokens = tokens
    if reuse_asset_audit is not None:
        previous = read_sealed(reuse_asset_audit)
        previous_missing = previous['missing_metric_caches']
        if (not verify_files or previous['schema'] != 'world_rl_NAVTRAIN_population_audit_v1'
                or previous['population_sha256'] != population_sha
                or previous['source_records'] != sources
                or previous['full_NAVTRAIN_count'] != len(tokens)
                or previous['file_existence_checked'] is not True
                or previous['missing_current_images']
                or len(previous_missing) != len(set(previous_missing))
                or not set(previous_missing) <= set(tokens)
                or sum(previous['cache_root_counts'].values()) + len(previous_missing) != len(tokens)):
            raise ValueError('Asset audit reuse must identify the exact fully checked immutable source population')
        cache_counts.update(previous['cache_root_counts'])
        inspect_tokens = previous_missing
        reuse = dict(path=str(Path(reuse_asset_audit).resolve()), sha256=file_sha(reuse_asset_audit),
                     prior_successful_image_checks=len(tokens)*3,
                     prior_successful_cache_checks=len(tokens)-len(previous_missing),
                     newly_checked_scenes=len(inspect_tokens),
                     inventory_is_not_a_new_payload_hash_or_future_existence_guarantee=True)
    if verify_files:
        for token in inspect_tokens:
            row = records[token]
            for camera in ('cam_l0', 'cam_f0', 'cam_r0'):
                path = row['images'].get(camera)
                if not path or not Path(path).is_file():
                    missing_images.append(dict(token=token, camera=camera, path=path))
            cache = next((root/row['log']/'unknown'/token/'metric_cache.pkl'
                          for root in CACHE_ROOTS
                          if (root/row['log']/'unknown'/token/'metric_cache.pkl').is_file()), None)
            if cache is None:
                missing_caches.append(token)
            else:
                cache_counts[str(cache.parents[3])] += 1
    audit = dict(schema='world_rl_NAVTRAIN_population_audit_v1', population_sha256=population_sha,
                 completed_at_utc=datetime.now(timezone.utc).isoformat(),asset_audit_reuse=reuse,
                 full_NAVTRAIN_count=len(tokens), original_native_TRAIN_count=len(pop['train_tokens']),
                 old_DEV_count=len(pop['dev_tokens']), world_TRAIN_count=len(roles['train_tokens']),
                 embargoed_native_TRAIN_count=len(roles['embargoed_train_tokens']),
                 NAVTEST_count=len(test_tokens), full_NAVTRAIN_segment_logs=len(train_logs),
                 full_NAVTRAIN_physical_logs=len(train_physical), NAVTEST_physical_logs=len(test_physical),
                 token_overlap=0, segment_log_overlap=0, physical_log_overlap=0,
                 file_existence_checked=verify_files, missing_current_images=missing_images,
                 missing_metric_caches=missing_caches, cache_root_counts=dict(cache_counts),
                 old_DEV_role_for_full103k='TRAIN_DIAGNOSTIC_NOT_HELDOUT',
                 source_records=sources, no_new_teacher_targets=True, no_new_model_forward=True)
    atomic_json(output/'POPULATION_AUDIT.json', sealed(audit))
    if missing_images or missing_caches:
        raise FileNotFoundError('Full NAVTRAIN has missing assets; no scene was silently excluded; inspect POPULATION_AUDIT.json')
    # The complete original raw records are small (~95MB), unlike World IL
    # teacher/annotation records. observation() only exposes current safe fields.
    record_path = output/'records.pkl'
    if record_path.exists():
        raise FileExistsError(record_path)
    tmp = output/'records.pkl.building'
    with tmp.open('wb') as stream:
        pickle.dump({t:records[t] for t in tokens}, stream, protocol=5)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, record_path)
    manifest = dict(schema='world_rl_research_current_observations_v1', cohort='NAVTRAIN',
                    count=len(tokens), tokens=tokens, old_DEV_tokens=pop['dev_tokens'],
                    source_population_path=str(POPULATION), population_sha256=population_sha,
                    token_logs={t:pop['token_logs'][t] for t in tokens},
                    records_file='records.pkl', records_sha256=file_sha(record_path),
                    current_camera_order=['cam_l0','cam_f0','cam_r0'], source_records=sources,
                    canonical_metric_cache_roots=[str(root) for root in CACHE_ROOTS],
                    current_only_actor_inputs=True, future_targets_are_not_actor_inputs=True,
                    NAVTEST_tokens_excluded=True, NAVTEST_physical_logs_excluded=True)
    atomic_json(output/'manifest.json', manifest)
    atomic_json(output/'COMPLETE', dict(status='COMPLETE', manifest_sha256=file_sha(output/'manifest.json')))
    full_set = set(tokens)
    hard = json.loads(HARD.read_text())['tokens']
    # RL-unseen split, selected from physical log hashes before inspecting scores.
    # Full103k intentionally includes it, so it is held out only for the named
    # FULL_MINUS_RL_DEV arm. No claim that IL never saw these scenes is made.
    heldout_logs = sorted(train_physical, key=lambda log:fingerprint({'RL_DEV_seed':42030,'physical_log':log}))[:8]
    heldout = [t for t in tokens if physical(pop['token_logs'][t]) in heldout_logs]
    heldout_set = set(heldout)
    world_train_set = set(roles['train_tokens'])
    pools = {'FULL_103288':tokens, 'NATIVE_TRAIN_101592':pop['train_tokens'],
             'WORLD_TRAIN_90890':roles['train_tokens'],
             'WORLD_HARD_15171':[t for t in hard if t in world_train_set],
             'FULL_HARD':[t for t in hard if t in full_set],
             'FULL_MINUS_RL_DEV':[t for t in tokens if t not in heldout_set]}
    for name, subset in pools.items():
        write_stream_manifest(output.parent/'manifests'/f'{name}.json',data_root=output,tokens=subset,name=name)
    heldout_receipt = dict(schema='world_rl_stage_diagnostic_split_v1',selection='hash-ranked whole physical logs before score inspection',
                           physical_logs=heldout_logs,tokens=heldout,count=len(heldout),
                           RL_unseen_for=['FULL_MINUS_RL_DEV'],in_sample_for=['FULL_103288'],
                           not_SFT_unseen=True,population_sha256=population_sha)
    atomic_json(output.parent/'manifests'/'RL_DEV_PHYSICAL.json',sealed(heldout_receipt))
    return audit
