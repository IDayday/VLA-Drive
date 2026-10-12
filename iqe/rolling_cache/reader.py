"""Cache-only reader with bounded waiting; never computes a missing payload."""
from collections import OrderedDict
from pathlib import Path
import threading
import time
from ..contracts import require
from ..io import digest
from ..training.execution import EpochPlan
from ..vision_cache.storage import CacheReader
from .pool import lease
from .plan import floor_window


class RollingReader:
    def __init__(self, pool, request, *, rank, epoch, offset, step, identity=None,
                 wait_seconds=900, max_open_shards=4):
        require(0 < wait_seconds <= 1800 and 1 <= max_open_shards <= 16, 'reader budget')
        self.pool, self.request, self.rank = pool, request, rank
        self.epoch, self.offset, self.step = epoch, offset, step
        self.identity, self.wait_seconds, self.max_open = identity, wait_seconds, max_open_shards
        self.readers = OrderedDict(); self.mutex = threading.RLock()
        self.users = {}; self.condition = threading.Condition(self.mutex)
        self.allowed = {r['scene_id'] for r in pool.full['rows']}
        # Stable access identity, independent of residency/generation/rank. Never
        # masquerades as a COMPLETE full-cache manifest.
        self.complete = {'index_sha256': pool.session['content_hash'], 'status': 'ROLLING_ACCESS_ONLY'}
        self.stats = {'reads': 0, 'bytes': 0, 'wait_seconds': 0., 'miss_waits': 0,
                      'visual_computations': 0, 'image_decodes': 0, 'max_reader_roots': 0}
        self.expected = EpochPlan(len(self.allowed), pool.full['global_batch'], pool.session['seed'],
                                  rank, request['world_size'], pool.make_batches)
        pool.progress(request, rank, step)

    def _source(self, sid, *, read):
        require(sid in self.allowed, 'scene outside locked fit list')
        started = time.monotonic(); deadline = started + self.wait_seconds
        waited = False
        while True:
            first = floor_window(self.pool.session, self.step)
            numbers = range(max(0, first - 1), first + self.pool.session['lookahead'])
            for number in numbers:
                folder = self.pool.folder(number)
                if not (folder / 'READY.json').exists(): continue
                with lease(folder / 'READ.lock'):
                    ready = self.pool.ready(number)
                    if ready is None or sid not in ready['refs']: continue
                    path = ready['refs'][sid]
                    with self.mutex:
                        reader = self.readers.pop(path, None)
                        limit = len(self.pool.sources) + self.pool.session['lookahead'] + 1
                        if reader is None:
                            while len(self.readers) >= limit:
                                available = next((p for p in self.readers if not self.users.get(p, 0)), None)
                                if available is not None:
                                    self.readers.pop(available).close(); break
                                self.condition.wait(timeout=.25)
                            reader = CacheReader(path, identity=self.identity, max_open_shards=self.max_open)
                        require(reader.complete['index_sha256'] == ready['sources'][path], 'source checksum changed')
                        self.readers[path] = reader
                        self.users[path] = self.users.get(path, 0) + 1
                        self.stats['max_reader_roots'] = max(self.stats['max_reader_roots'], len(self.readers))
                    try:
                        row = dict(reader.access(sid))
                        if read:
                            value = reader.read(sid)
                            with self.mutex:
                                self.stats['reads'] += 1
                                self.stats['bytes'] += reader.index['records'][row['key']]['length']
                        else: value = row
                    finally:
                        with self.mutex:
                            # Never retain descriptors to rolling shards after
                            # releasing their read lease: unlink must actually
                            # release capacity. Permanent sources keep their LRU.
                            if path == ready['owned_root']: reader.close()
                            self.users[path] -= 1; self.condition.notify_all()
                    with self.mutex:
                        if waited: self.stats['wait_seconds'] += time.monotonic() - started
                    return value
            if not waited:
                with self.mutex: self.stats['miss_waits'] += 1
                waited = True
            if time.monotonic() >= deadline:
                raise TimeoutError(f'REMOTE_CACHE_NOT_READY: scene={sid} window={first}; no online/zero fallback')
            time.sleep(min(.25, max(0, deadline - time.monotonic())))

    def access(self, sid): return self._source(sid, read=False)

    def read(self, sid): return self._source(sid, read=True)

    def prepared(self, scene_ids):
        """Validate original rank order, then publish disposable prefetch progress.

        This is deliberately NOT saved in model/optimizer checkpoints. A crash
        discards it and initializes from the restored training cursor.
        """
        address = self.expected.at(self.epoch, self.offset)
        fit = self.pool.full['rows']
        require(scene_ids == [fit[i]['scene_id'] for i in address.mine], 'prefetch changed original rank sample order')
        old_window = floor_window(self.pool.session, self.step)
        self.step += 1; self.epoch, self.offset = address.epoch, address.offset + 1
        if old_window != floor_window(self.pool.session, self.step):
            self.pool.progress(self.request, self.rank, self.step)

    def close(self):
        with self.mutex:
            for reader in self.readers.values(): reader.close()
            self.readers.clear()


def validate_rolling_certificate(path, runtime_identity, session):
    from .plan import checked
    from ..io import read_json, file_hash
    value = checked(path)
    require(value['schema'] == 'iqe_rolling_runtime_qualification_v1' and value['status'] == 'PASS',
            'rolling runtime requires its own real qualification')
    require(value['runtime_identity'] == runtime_identity and value['session_hash'] == session['content_hash'],
            'rolling runtime/lineage qualification mismatch')
    evidence = {}
    for name, item in value['evidence'].items():
        require(file_hash(item['path']) == item['sha256'], 'rolling evidence changed')
        evidence[name] = read_json(item['path'])
        require(evidence[name]['status'] == 'PASS', 'failed rolling qualification')
    require({'numerical', 'resume', 'io'} <= evidence.keys(), 'missing real rolling evidence')
    require(evidence['numerical']['full_state_rng_cursor_equal'] and
            evidence['numerical']['no_images_or_visual_on_hit'] and
            evidence['numerical']['deepstack_count'] == 3, 'rolling training equivalence not established')
    require(evidence['resume']['evicted_window_rebuild_pass'] and
            evidence['resume']['next_sample_ids_equal'] and evidence['resume']['full_state_equal'],
            'rolling resume not established')
    require(evidence['io']['shared_storage'] and evidence['io']['steady_updates'] >= 100,
            'rolling supply/consumption throughput unmeasured')
    return value
