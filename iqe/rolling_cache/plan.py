"""Windows follow the original consumed cursor, independently of resident data.

An eviction changes storage residency, never the sampler. A resumed consumer
requests windows from its checkpoint again; evicted values can be rebuilt.
"""
from pathlib import Path
from functools import lru_cache
import math
import torch
from ..contracts import require
from ..io import atomic_json, digest, file_hash, read_json
from ..training.execution import EpochPlan
from ..vision_cache.plan import load_plan


def seal(value):
    return {**value, 'content_hash': digest(value)}


def checked(path):
    path = Path(path)
    stat = path.stat()
    return _checked_cached(str(path.resolve()), (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


@lru_cache(maxsize=128)
def _checked_cached(path, stamp):
    value = read_json(path)
    require(value['content_hash'] == digest({k: v for k, v in value.items() if k != 'content_hash'}),
            'rolling artifact checksum mismatch')
    return value


def prepare_session(full_plan, checkpoint, root, *, window_updates=64, lookahead=3,
                    capacity_gib=512, reserve_gib=1700, reuse_roots=(), qualification):
    require(1 <= window_updates <= 200 and 2 <= lookahead <= 8, 'bounded lookahead required')
    full = load_plan(full_plan)
    require(full['scope'] == 'full' and full['requested_scenes'] == full['fit_count'] and 'phase' not in full,
            'immutable full population plan required; residency is separate')
    receipt = read_json(str(checkpoint) + '.COMPLETE.json')
    require(file_hash(checkpoint) == receipt['checksum'], 'parent checkpoint checksum')
    state = torch.load(checkpoint, mmap=True, map_location='cpu', weights_only=False)
    require(state['optimizer_step'] == receipt['optimizer_step'] and state['dependencies'] == receipt['dependencies'],
            'parent receipt/state mismatch')
    require(state['kind'] == 'iqe_query_base' and not state.get('terminal', False), 'nonterminal Query Base required')
    fit = state['fit_scene_ids']
    require(digest(fit) == full['fit_sha256'] and fit == [r['scene_id'] for r in full['rows']], 'fit identity/order changed')
    require(state['framework_contract_hash'] == full['framework_contract_hash'] and
            state['dependencies']['original_targets'] == full['data'], 'framework or input changed')
    from ..vision_cache.identity import checkpoint_visual_hash
    from ..vision_cache.qualification import validate_cache_qualification
    require(checkpoint_visual_hash(state) == full['visual_state_sha256'], 'frozen visual changed')
    cert = validate_cache_qualification(qualification, fit_hash=full['fit_sha256'], for_build=True)
    record_bytes = cert['representative_record_max_bytes']
    cfg = state['dependencies']['config']
    require(cfg['global_batch_size'] == full['global_batch'] == 32 and
            cfg['micro_batch_size'] == full['microbatch'] == 4 and state['seed'] == full['seed'], 'batch/seed changed')
    # One previous window protects reads already in flight; one atomic shard per
    # permitted builder and 5% record overhead are reserved independently.
    maximum = math.ceil((lookahead + 1) * window_updates * 32 * record_bytes * 1.05) + 16 * 1024**3
    require(maximum <= capacity_gib * 1024**3, 'pool too small for declared lookahead, retained window and temporary shards')
    from ..vision_cache.storage import CacheReader
    sources = []
    for path in reuse_roots:
        reader = CacheReader(path)
        require(reader.complete['identity_hash'] == cert['visual_identity_hash'] and
                set(reader.allowed) <= set(fit), 'foreign reusable cache')
        require(all(r['role'] == 'incremental_fit' for r in reader.index['access'].values()), 'reuse role leak')
        sources.append({'root': str(Path(path).resolve()), 'index_sha256': reader.complete['index_sha256']})
        reader.close()
    value = seal({'schema': 'iqe_remote_rolling_session_v1',
        'full_plan': str(Path(full_plan).resolve()), 'full_plan_sha256': file_hash(full_plan),
        'parent_checkpoint': str(Path(checkpoint).resolve()), 'parent_sha256': receipt['checksum'],
        'parent_step': state['optimizer_step'], 'epoch': state['epoch'], 'offset': state['offset'],
        'fit_sha256': full['fit_sha256'], 'seed': state['seed'], 'world_size': state['dependencies']['world_size'],
        'window_updates': window_updates, 'lookahead': lookahead, 'retained_previous_windows': 1,
        'max_optimizer_steps': cfg['max_optimizer_steps'], 'capacity_bytes': int(capacity_gib * 1024**3),
        'reserve_bytes': max(int(reserve_gib * 1024**3), full['checkpoint_retention_budget']['parent_bytes'] *
            full['checkpoint_retention_budget']['remaining_count'] + (180+200)*1024**3),
        'worst_case_pool_bytes': maximum, 'record_max_bytes': record_bytes,
        'qualification': str(Path(qualification).resolve()), 'qualification_sha256': file_hash(qualification),
        'reuse_sources': sources, 'training_order_changed': False, 'base_visual_compute_allowed': False,
        'training_visual_source': 'remote_rolling_fp32', 'inference_visual_source': 'online_fp32',
        'visual_numeric_contract': 'legacy_fp32', 'full_cache_complete': False})
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'SESSION.json', value, immutable=True)
    return value


def load_session(root):
    session = checked(Path(root) / 'SESSION.json')
    require(session['schema'] == 'iqe_remote_rolling_session_v1', 'foreign rolling session')
    require(file_hash(session['full_plan']) == session['full_plan_sha256'], 'full population plan changed')
    require(file_hash(session['qualification']) == session['qualification_sha256'], 'FP32 evidence changed')
    return session, load_plan(session['full_plan'])


def window_sequence(session, full, number, make_batches):
    require(type(number) is int and number >= 0, 'invalid window number')
    offset = number * session['window_updates']
    remaining = session['max_optimizer_steps'] - session['parent_step'] - offset
    require(remaining > 0, 'window beyond original schedule')
    per_epoch = math.ceil(full['fit_count'] / full['global_batch'])
    extra_epoch, within = divmod(session['offset'] + offset, per_epoch)
    plan = EpochPlan(full['fit_count'], full['global_batch'], session['seed'], 0, 1, make_batches)
    rows = full['rows']; count = min(session['window_updates'], remaining)
    return [{'epoch': a.epoch, 'offset': a.offset, 'scene_ids': [rows[i]['scene_id'] for i in a.entries]}
            for a in plan.iter_from(session['epoch'] + extra_epoch, within, count)]


def window_plan(session, full, number, make_batches):
    batches = window_sequence(session, full, number, make_batches)
    by_id = {r['scene_id']: r for r in full['rows']}
    ids = list(dict.fromkeys(s for b in batches for s in b['scene_ids']))
    return seal({'schema': 'iqe_remote_rolling_window_v1', 'session_hash': session['content_hash'],
        'number': number, 'first_step': session['parent_step'] + number * session['window_updates'] + 1,
        'last_step': session['parent_step'] + number * session['window_updates'] + len(batches),
        'batches': batches, 'rows': [by_id[s] for s in ids]})


def floor_window(session, prepared_step):
    require(session['parent_step'] <= prepared_step <= session['max_optimizer_steps'], 'invalid advisory step')
    return (prepared_step - session['parent_step']) // session['window_updates']
