"""Bounded, collection-local CPU scoring overlap for fixed-draw D policies.

This is an opt-in execution helper. It never computes credit or invokes an
optimizer and does not add anything to the recipe/scientific identity. Futures
stay outside native material. The synchronous collector remains the default.
"""
from collections import deque
from dataclasses import dataclass
import copy
import time

from .batch_runtime import collect_scene, finalize_scene_score


FIXED_DRAW_ARMS = frozenset(('D0', 'D5', 'D3_FPO', 'D3_AWM'))
EXECUTION_PROTOCOL = 'd46_fixed_draw_async_scoring_collection_v1'


@dataclass
class _Pending:
    future: object
    scene: object = None


class AsyncScoringCollector:
    """One actor behavior version, one fresh collection, at most four Futures.

    ``collect`` runs original GPU work and same-scene MC draws synchronously;
    only scorer waiting is deferred. ``finish`` is the sole publication point.
    Use as a context manager so exceptions also drain/cancel every score job.
    The caller must use its existing COLLECTING -> global error consensus ->
    credit/finalize transaction; never install a partial scene list in a batch.
    """

    def __init__(self, actor, service, cfg, epoch, version, *, max_pending=4, cost=None):
        if cfg['arm'] not in FIXED_DRAW_ARMS:
            raise ValueError('Async scoring requires a fixed-draw D0/D5/FPO/AWM arm')
        if type(max_pending) is not int or not 1 <= max_pending <= 4:
            raise ValueError('max_pending must be an integer in [1,4]')
        if not callable(getattr(service, 'submit', None)):
            raise TypeError('RewardService.submit is required')
        self.actor, self.service = actor, service
        self.cfg, self.epoch, self.version = copy.deepcopy(cfg), epoch, copy.deepcopy(version)
        self.max_pending, self.cost = max_pending, cost
        self._pending, self._results = deque(), []
        self._finished = self._failed = False
        self.max_observed_pending = 0

    @property
    def inflight(self):
        # Includes finished but not yet validated Futures: none may enter a CP.
        return len(self._pending)

    def assert_drained(self):
        if self._pending:
            raise RuntimeError('Cannot checkpoint with async scorer Futures in flight')

    def _submit(self, request):
        if len(self._pending) >= self.max_pending:
            raise RuntimeError('Async scorer pending bound exceeded')
        item = _Pending(self.service.submit(request))
        self._pending.append(item)
        self.max_observed_pending = max(self.max_observed_pending, len(self._pending))

    def _resolve_first(self):
        item = self._pending[0]
        tick = time.monotonic()
        try:
            records = item.future.result()
        finally:
            waited = time.monotonic() - tick
            if self.cost is not None:
                self.cost(scoring_wait_worker_seconds=waited)
        if item.scene is None:
            raise RuntimeError('GPU scene preparation did not complete')
        result = finalize_scene_score(item.scene, records, seconds_scoring=waited)
        self._pending.popleft()
        self._results.append(result)

    def collect(self, descriptor, token, log, generator, *, metadata=None):
        """Prepare one scene; return nothing until the entire collection succeeds.

        Place the existing virtual-lane ``before_scene``/``after_scene`` hooks
        around this method. FPO/AWM MC draws and old energy are complete before
        it returns, exactly before the next scene's generator/context call.
        """
        if self._finished or self._failed:
            raise RuntimeError('Async collection is already closed')
        try:
            if len(self._pending) == self.max_pending:
                self._resolve_first()
            scene = collect_scene(self.actor, descriptor, token, log, self.service,
                generator, self.cfg, self.epoch, self.version, cost=self.cost,
                score_submit=self._submit)
            if metadata:
                if set(metadata) & set(scene) or set(metadata) & {'records', 'factors', 'seconds_scoring'}:
                    raise ValueError('Scene metadata cannot replace prepared/scored material')
                scene.update(metadata)
            self._pending[-1].scene = scene
        except BaseException:
            self.abort()
            raise

    def finish(self):
        """Validate all official responses, then return original scene order once."""
        if self._finished or self._failed:
            raise RuntimeError('Async collection is already closed')
        try:
            while self._pending:
                self._resolve_first()
            self.assert_drained()
            self._finished = True
            results, self._results = self._results, []
            return results
        except BaseException:
            self.abort()
            raise

    def abort(self):
        """Cancel not-started scoring and join started work; discard all scenes."""
        self._failed = True
        for item in self._pending:
            item.future.cancel()
        while self._pending:
            item = self._pending.popleft()
            try:
                item.future.result()
            except BaseException:
                # Keep the initiating error; no partial material is published.
                pass
        self._results.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._finished:
            self.abort()
        self.assert_drained()


def collect_scenes(actor, requests, service, generator, cfg, epoch, version, *,
                   max_pending=4, cost=None, before_scene=None, after_scene=None):
    """Collect ordered ``{descriptor, token, log, metadata?}`` requests atomically.

    Hooks receive ``(generator, index)`` and preserve the existing world16
    virtual-lane convention. No hook, materialization or RNG work is moved into
    a scorer thread. This helper is only for a single existing collection.
    """
    with AsyncScoringCollector(actor, service, cfg, epoch, version,
                              max_pending=max_pending, cost=cost) as collector:
        for index, request in enumerate(requests):
            if before_scene is not None:
                before_scene(generator, index)
            collector.collect(request['descriptor'], request['token'], request['log'],
                              generator, metadata=request.get('metadata'))
            if after_scene is not None:
                after_scene(generator, index)
        return collector.finish()
