"""Complete immutable windows, read leases, and owned-file-only eviction.

Permanent cache roots are references, never copied/deleted by this pool. New
payloads use the existing qualified FP32 writer. A window is usable only when
every requested observation has a checked source. Incomplete windows wait.
"""
from contextlib import contextmanager
from collections import OrderedDict
from pathlib import Path
import fcntl
import math
import os
import time
import uuid
import threading
from ..contracts import require
from ..io import atomic_json, digest, file_hash, lock, read_json
from ..vision_cache.storage import CacheReader
from .plan import checked, seal, floor_window, window_plan


@contextmanager
def lease(path, exclusive=False, nonblocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) |
                    (fcntl.LOCK_NB if nonblocking else 0))
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


class Pool:
    def __init__(self, root, session, full, make_batches):
        self.root, self.session, self.full = Path(root), session, full
        self.make_batches = make_batches
        self.sources = OrderedDict()
        for src in session['reuse_sources']:
            r = CacheReader(src['root'], max_open_shards=4)
            require(r.complete['index_sha256'] == src['index_sha256'], 'permanent cache changed')
            self.sources[src['root']] = r

    def folder(self, number):
        require(type(number) is int and number >= 0, 'invalid window')
        return self.root / 'windows' / f'{number:06d}'

    def plan(self, number):
        value = window_plan(self.session, self.full, number, self.make_batches)
        atomic_json(self.folder(number) / 'PLAN.json', value, immutable=True)
        return value

    def start_attempt(self, step, *, attempt=None):
        """Advisory preparation only; checkpoints remain the sole resume cursor."""
        floor_window(self.session, step)
        value = seal({'session_hash': self.session['content_hash'], 'attempt': attempt or uuid.uuid4().hex,
                      'checkpoint_step': step, 'world_size': self.session['world_size']})
        atomic_json(self.root / 'REQUEST.json', value)
        return value

    def progress(self, request, rank, prepared_step):
        require(0 <= rank < request['world_size'] and prepared_step >= request['checkpoint_step'], 'invalid reader progress')
        require(checked(self.root / 'REQUEST.json') == request, 'superseded consumer attempt')
        floor_window(self.session, prepared_step)
        path = self.root / 'progress' / request['attempt'] / f'rank_{rank}.json'
        if path.exists(): require(checked(path)['prepared_step'] <= prepared_step, 'preparation cursor reversed within attempt')
        atomic_json(path, seal({'request_hash': request['content_hash'], 'rank': rank,
            'prepared_step': prepared_step, 'NOT_A_TRAINING_RESUME_CURSOR': True}))

    def desired(self):
        request = checked(self.root / 'REQUEST.json')
        require(request['session_hash'] == self.session['content_hash'], 'foreign request')
        steps = []
        for rank in range(request['world_size']):
            path = self.root / 'progress' / request['attempt'] / f'rank_{rank}.json'
            row = checked(path) if path.exists() else None
            require(row is None or row['request_hash'] == request['content_hash'], 'foreign reader progress')
            steps.append(row['prepared_step'] if row else request['checkpoint_step'])
        first = floor_window(self.session, min(steps))
        total = math.ceil((self.session['max_optimizer_steps'] - self.session['parent_step']) /
                          self.session['window_updates'])
        return list(range(first, min(first + self.session['lookahead'], total))), request

    def ready(self, number):
        path = self.folder(number) / 'READY.json'
        if not path.exists(): return None
        value = checked(path)
        require(value['session_hash'] == self.session['content_hash'] and value['number'] == number,
                'foreign window readiness')
        require(value['plan_hash'] == checked(self.folder(number) / 'PLAN.json')['content_hash'], 'stale window plan')
        return value

    def split_sources(self, plan):
        refs, missing = {}, []
        for row in plan['rows']:
            sid = row['scene_id']; chosen = None
            for root, reader in self.sources.items():
                if sid in reader.allowed:
                    old = reader.access(sid)
                    require(old['observation_hash'] == row['observation_hash'] and
                            old['current_sha256'] == row['current_sha256'] and old['role'] == 'incremental_fit',
                            'reused observation/data role changed')
                    chosen = root; break
            if chosen is None: missing.append(row)
            else: refs[sid] = chosen
        return refs, missing

    def publish(self, plan, refs, owned_root=None):
        number = plan['number']; folder = self.folder(number)
        require(set(refs) == {r['scene_id'] for r in plan['rows']}, 'incomplete window coverage')
        sources = {}
        by_id = {r['scene_id']: r for r in plan['rows']}
        for source in sorted(set(refs.values())):
            reader = CacheReader(source, max_open_shards=1)
            try:
                cert = read_json(self.session['qualification'])
                require(reader.complete['identity_hash'] == cert['visual_identity_hash'], 'foreign FP32 identity')
                for sid, path in refs.items():
                    if path == source:
                        row = reader.access(sid)
                        require(row['observation_hash'] == by_id[sid]['observation_hash'] and
                                row['current_sha256'] == by_id[sid]['current_sha256'] and
                                row['role'] == 'incremental_fit', 'wrong window record')
                sources[source] = reader.complete['index_sha256']
            finally: reader.close()
        if owned_root:
            require(Path(owned_root).resolve().parent == (folder / 'data').resolve(), 'foreign owned path')
        value = seal({'status': 'WINDOW_COMPLETE', 'session_hash': self.session['content_hash'],
            'number': number, 'plan_hash': plan['content_hash'], 'refs': refs, 'sources': sources,
            'owned_root': str(owned_root) if owned_root else None,
            'scenes': len(refs), 'full_training_cache_complete': False})
        with lease(folder / 'READ.lock', exclusive=True):
            if (folder / 'READY.json').exists(): require(checked(folder / 'READY.json') == value, 'window conflict')
            atomic_json(folder / 'READY.json', value, immutable=True)
        return value

    def evict(self, number):
        """Only derived .bin files; retain metadata and source evidence forever."""
        folder = self.folder(number)
        with lease(folder / 'BUILD.lock', exclusive=True, nonblocking=True), \
             lease(folder / 'READ.lock', exclusive=True, nonblocking=True):
            desired, request = self.desired()
            first = desired[0] if desired else floor_window(self.session, self.session['max_optimizer_steps'])
            keep = set(range(max(0, first - self.session['retained_previous_windows']),
                             min(first + self.session['lookahead'], math.ceil((self.session['max_optimizer_steps']-
                                 self.session['parent_step'])/self.session['window_updates']))))
            require(number not in keep, 'cannot evict current/lookahead/retained window')
            # Rename readiness first. A crash during removal resumes from the
            # durable eviction intent. Readers never see a partial ready window.
            ready_path = folder / 'READY.json'
            intent_path = folder / 'EVICTING.json'
            if ready_path.exists():
                value = checked(ready_path)
                atomic_json(intent_path, seal({'ready': value, 'request_hash': request['content_hash']}))
                ready_path.unlink()
            elif intent_path.exists(): value = checked(intent_path)['ready']
            else: return {'status': 'NOT_RESIDENT', 'number': number}
            removed = 0
            if value['owned_root']:
                owned = Path(value['owned_root'])
                require(owned.resolve().parent == (folder / 'data').resolve() and not owned.is_symlink(),
                        'eviction outside owned rolling files refused')
                index = read_json(owned / 'INDEX.json')
                require(file_hash(owned / 'INDEX.json') == value['sources'][str(owned)], 'eviction identity changed')
                for name, meta in index['shards'].items():
                    require(Path(name).name == name and name.endswith('.bin'), 'invalid eviction path')
                    path = owned / name
                    if path.exists():
                        require(not path.is_symlink() and path.stat().st_size == meta['bytes'], 'unexpected owned shard')
                        removed += path.stat().st_size; path.unlink()
                building = folder / 'BUILDING.json'
                if building.exists():
                    require(checked(building)['root'] == str(owned), 'unexpected partial generation during eviction')
                    building.unlink()
            result = seal({'status': 'EVICTED', 'number': number, 'ready_hash': value['content_hash'],
                           'removed_bytes': removed, 'permanent_sources_untouched': True})
            atomic_json(folder / ('EVICTED_' + value['content_hash'] + '.json'), result)
            if intent_path.exists(): intent_path.unlink()
            return result

    def resident_owned_bytes(self):
        # Bounded pool directory only, never walk the permanent dataset/caches.
        return sum(p.stat().st_size for p in (self.root / 'windows').glob('*/data/*/*')
                   if p.is_file() and (p.suffix in {'.bin', '.partial'}))

    def close(self):
        for r in self.sources.values(): r.close()


def build_window(pool, number, *, model_loader, rate_mib_s, reserve_gib, max_gib, hold_file,
                 deadline=None, worker=0, workers=1):
    """Run on a remote producer, not a Base rank. No optimizer or target IO."""
    from .placement import producer_placement
    from ..vision_cache.plan import capacity_check
    from ..vision_cache.builder import build_cache
    from ..vision_cache.cli import hold_check
    producer_placement(); hold_check(hold_file)
    folder = pool.folder(number)
    require(0 <= worker < workers <= 8, 'invalid bounded producer partition')
    # Multiple frozen producers may fill disjoint native shards of one window.
    # GC needs an exclusive lease, so it cannot remove any in-progress window.
    with lease(folder / 'BUILD.lock', exclusive=False, nonblocking=True):
        wanted, _ = pool.desired()
        require(number in wanted, 'producer may only build requested upcoming windows')
        old = pool.ready(number)
        if old is not None:
            partial = folder / 'BUILDING.json'
            with lease(folder / 'GENERATION.lock', exclusive=True):
                if partial.exists():
                    require(checked(partial)['root'] == old['owned_root'], 'window commit/partial generation mismatch')
                    partial.unlink()
            return {**old, 'reused': True}
        plan = pool.plan(number); refs, missing = pool.split_sources(plan)
        require(not (folder / 'EVICTING.json').exists(), 'finish interrupted eviction before rebuilding')
        if not missing: return pool.publish(plan, refs)
        estimate = math.ceil(len(missing) * pool.session['record_max_bytes'] * 1.05)
        require(estimate <= max_gib * 1024**3, 'explicit producer write budget too small')
        require(pool.resident_owned_bytes() + estimate + 2*1024**3 <= pool.session['capacity_bytes'], 'POOL_CAPACITY_WAIT')
        reserve = max(reserve_gib * 1024**3, pool.session['reserve_bytes'])
        cap = capacity_check(pool.root, estimate, reserve_bytes=reserve, temporary_bytes=2*1024**3)
        atomic_json(folder / 'CAPACITY.json', cap)
        require(cap['sufficient'], 'SHARED_STORAGE_RESERVE_WAIT')
        # Resume a partial generation; never resurrect deleted payload bytes by
        # trusting its historical COMPLETE marker.
        partial = folder / 'BUILDING.json'
        with lease(folder / 'GENERATION.lock', exclusive=True):
            if partial.exists(): generation = checked(partial)['root']
            else:
                generation = str((folder / 'data' / uuid.uuid4().hex).resolve())
                atomic_json(partial, seal({'root': generation, 'plan_hash': plan['content_hash']}))
            require(checked(partial)['plan_hash'] == plan['content_hash'], 'stale window generation')
        native = {k:v for k,v in pool.full.items() if k not in {'plan_hash','rows'}}
        native.update(rows=missing, requested_scenes=len(missing), rolling_window_hash=plan['content_hash'])
        native['plan_hash'] = digest(native)
        native_path = Path(generation) / 'PLAN.json'
        atomic_json(native_path, native, immutable=True)
        model, contract = model_loader(native)
        if deadline is not None and time.monotonic() >= deadline: return {'status':'OPERATOR_PAUSED','reason':'time budget before build'}
        control = Path(generation) / 'CONTROL.json'
        atomic_json(control, {'pause':False,'rate_mib_s_per_worker':rate_mib_s})
        timer = threading.Timer(max(.001, deadline-time.monotonic()),
            lambda: atomic_json(control, {'pause':True,'reason':'rolling producer time budget'})) if deadline else None
        if timer: timer.start()
        try:
            result = build_cache(model, contract, native_path, generation, worker=worker, workers=workers,
                max_shards=math.ceil(math.ceil(len(missing)/native['records_per_shard'])/workers), max_gib=max_gib,
                reserve_gib=reserve / 1024**3, rate_mib_s=rate_mib_s,
                qualification=pool.session['qualification'], hold_file=hold_file)
        finally:
            if timer: timer.cancel(); timer.join()
        if result['status'] != 'COMPLETE': return result
        refs.update({r['scene_id']: generation for r in missing})
        value = pool.publish(plan, refs, generation)
        partial.unlink(missing_ok=True)
        return value
