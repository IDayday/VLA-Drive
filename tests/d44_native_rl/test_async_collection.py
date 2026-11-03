"""Real collector, deterministic CPU actor and delayed synthetic scorer only."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import threading
import time

import numpy as np
import pytest
import torch

from starVLA.rl.d44_native_rl.async_collection import AsyncScoringCollector, collect_scenes
from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
from starVLA.rl.d44_native_rl.recovery import material_digest
from starVLA.rl.d44_native_rl.safety_credit import FACTORS, normalize, raw_credit


class SyntheticActor:
    device = torch.device('cpu')

    def __init__(self, group=16, events=None):
        self.group, self.events = group, events if events is not None else []
        self.action_expert = self
        self.contexts = []

    def sample_action_chain(self, descriptor, steps, *, generator):
        self.events.append(('generate', descriptor))
        chain = torch.randn(self.group, 11, 8, 3, generator=generator) * .02
        times, deltas = torch.tensor([.3, .2, .1]), torch.tensor([.1, .1, .1])
        means, std = self.transition_terms(descriptor, chain, times, deltas)
        self.sampling_terms = [(means[:, i].clone(), s.clone()) for i, s in enumerate(std)]
        return chain, times, deltas, torch.tensor([7, 8, 9])

    def transition_terms(self, descriptor, chain, times, deltas, use_ref=False):
        return chain[:, 7:10] + (.001 if use_ref else 0.), torch.tensor([.03, .02, .01])

    @contextmanager
    def _context_call(self, descriptor):
        self.contexts.append(descriptor)
        yield descriptor

    def _prepare(self, condition, use_ref):
        return float(condition)

    def denoise_x0(self, state, sigma, prefix):
        return state.float() * .3 + sigma[:, None, None] * .01 + prefix * .001


class SyntheticService:
    """Scoring consumes no policy/global RNG and returns original request order."""
    def __init__(self, *, events=None, fail=None, invalid=None, block_first=False):
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.events = events if events is not None else []
        self.fail, self.invalid, self.block_first = fail, invalid, block_first
        self.released = threading.Event()
        self.started = threading.Event()
        self.futures = []
        self.submissions = []
        self.requests = []

    def _score(self, request):
        token, log, requested = request
        if self.block_first and token == '0':
            self.started.set()
            if not self.released.wait(3):
                raise RuntimeError('test scorer release timed out')
        time.sleep(.002 if int(token) % 2 else .008)
        if token == self.fail:
            raise RuntimeError('synthetic scorer failure')
        rows = []
        for index, trajectory in enumerate(requested):
            metrics = dict(zip(FACTORS, (float(index % 5 != 0), 1., 1.,
                                       float(abs(trajectory[0, 0]) % 1), 1., 1.)))
            metrics['score'] = metrics[FACTORS[0]] * metrics[FACTORS[3]]
            rows.append(dict(status='BAD' if token == self.invalid else 'OK', token=token,
                             log=log, metrics=metrics,
                             requested_trajectory=np.asarray(trajectory).copy()))
        self.events.append(('scored', int(token)))
        return rows

    def submit(self, request):
        self.submissions.append(request[0])
        self.requests.append(copy.deepcopy(request))
        future = self.pool.submit(self._score, request)
        self.futures.append(future)
        return future

    def score(self, request):
        return self.submit(request).result()

    def close(self):
        self.released.set()
        self.pool.shutdown(wait=True, cancel_futures=True)


def requests(n=8):
    return [dict(descriptor=i, token=str(i), log='log' + str(i),
                 metadata=dict(sampling_source='uniform', padding=False,
                               beta=1., expect_fresh=True)) for i in range(n)]


def counts(cost):
    return {key: value for key, value in cost.items() if not key.endswith('_seconds')}


def test_all_fixed_draw_arms_match_real_sync_material_credit_counters_and_rng(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    for arm in ('D0', 'D5', 'D3_FPO', 'D3_AWM'):
        cfg = dict(arm=arm, group=16, mc_samples=4)
        version = dict(run='same-science', update=12, collection=7)
        initial_rng = torch.get_rng_state().clone()
        outputs, end_states, totals, scorer_requests = [], [], [], []
        for asynchronous in (False, True):
            actor, service, generator, cost = SyntheticActor(), SyntheticService(), torch.Generator().manual_seed(419), Counter()
            try:
                if asynchronous:
                    scene_rows = collect_scenes(actor, requests(), service, generator, cfg, .25, version,
                                                cost=lambda **x: cost.update(x))
                else:
                    scene_rows = []
                    for request in requests():
                        row = collect_scene(actor, request['descriptor'], request['token'], request['log'],
                                            service, generator, cfg, .25, version, cost=lambda **x: cost.update(x))
                        row.update(request['metadata'])
                        scene_rows.append(row)
                raw, _ = raw_credit(torch.stack([row['factors'] for row in scene_rows]))
                advantages, _ = normalize(raw)
                for row, advantage in zip(scene_rows, advantages):
                    row['advantages'] = advantage
                outputs.append(scene_rows)
                end_states.append(generator.get_state())
                totals.append(counts(cost))
                scorer_requests.append(service.requests)
                assert [r['token'] for r in scene_rows] == [str(i) for i in range(8)]
                assert actor.contexts == (list(range(8)) if arm in ('D3_FPO', 'D3_AWM') else [])
                assert torch.equal(torch.get_rng_state(), initial_rng)
            finally:
                service.close()
        assert material_digest(dict(scenes=outputs[0])) == material_digest(dict(scenes=outputs[1]))
        assert torch.equal(end_states[0], end_states[1])
        assert totals[0] == totals[1] == dict(candidate_requests=128, physical_candidates=128,
                                             scorer_candidate_attempts=128)
        for first, second in zip(scorer_requests[0], scorer_requests[1]):
            assert first[:2] == second[:2]
            np.testing.assert_array_equal(first[2], second[2])


def test_pending_bound_checkpoint_barrier_and_true_generation_overlap(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    events = []
    actor, service = SyntheticActor(events=events), SyntheticService(events=events, block_first=True)
    generator = torch.Generator().manual_seed(21)
    try:
        with AsyncScoringCollector(actor, service, dict(arm='D3_FPO', group=16, mc_samples=4),
                                   .2, dict(update=2, collection=1)) as collector:
            for request in requests(4):
                assert collector.collect(request['descriptor'], request['token'], request['log'], generator) is None
            assert service.started.wait(1)
            assert ('generate', 3) in events and ('scored', 0) not in events
            assert collector.inflight == collector.max_observed_pending == 4
            with pytest.raises(RuntimeError, match='checkpoint'):
                collector.assert_drained()
            service.released.set()
            for request in requests(8)[4:]:
                collector.collect(request['descriptor'], request['token'], request['log'], generator)
                assert collector.inflight <= 4
            result = collector.finish()
            assert [r['token'] for r in result] == [str(i) for i in range(8)]
            assert collector.max_observed_pending == 4
            collector.assert_drained()
            assert all(f.done() for f in service.futures)
            with pytest.raises(RuntimeError, match='closed'):
                collector.finish()
    finally:
        service.close()


def test_failed_or_invalid_future_never_publishes_partial_batch_or_steps_adam(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    for failure in ('fail', 'invalid'):
        service = SyntheticService(**{failure: '2'})
        weight = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.AdamW([weight], lr=.01)
        batch = dict(scenes=[], uncommitted=True)
        collector = AsyncScoringCollector(SyntheticActor(), service, dict(arm='D0', group=16),
                                          .2, dict(update=0, collection=1))
        try:
            with pytest.raises((RuntimeError, ValueError), match='scorer'):
                with collector:
                    for request in requests():
                        collector.collect(request['descriptor'], request['token'], request['log'],
                                          torch.Generator().manual_seed(int(request['token'])))
                    batch['scenes'] = collector.finish()
                    weight.square().backward()
                    optimizer.step()
            assert batch == dict(scenes=[], uncommitted=True)
            assert weight.item() == 1. and optimizer.state_dict()['state'] == {}
            assert collector.inflight == 0 and all(f.done() for f in service.futures)
            with pytest.raises(RuntimeError, match='closed'):
                collector.finish()
        finally:
            service.close()


def test_conditional_arms_rejected_before_rng_or_submit_and_scene_hooks_keep_order(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    service, generator, actor = SyntheticService(), torch.Generator().manual_seed(73), SyntheticActor()
    before_rng = generator.get_state().clone()
    try:
        for arm in ('D1', 'D2', 'D6'):
            with pytest.raises(ValueError, match='fixed-draw'):
                collect_scenes(actor, requests(), service, generator, dict(arm=arm, group=16), 0., {})
        for maximum in (0, 5, True):
            with pytest.raises(ValueError, match='max_pending'):
                AsyncScoringCollector(actor, service, dict(arm='D0', group=16), 0., {}, max_pending=maximum)
        assert torch.equal(before_rng, generator.get_state()) and not service.submissions
        hooks = []
        row_states = []
        def before(gen, index):
            hooks.append(('before', index))
            gen.manual_seed(600 + index)
        def after(gen, index):
            hooks.append(('after', index))
            row_states.append(gen.get_state().clone())
        collect_scenes(actor, requests(4), service, generator,
                       dict(arm='D3_FPO', group=16, mc_samples=4), 0., {},
                       before_scene=before, after_scene=after)
        assert hooks == [(kind, index) for index in range(4) for kind in ('before', 'after')]
        for index, expected in enumerate(row_states):
            gen = torch.Generator().manual_seed(600 + index)
            collect_scene(SyntheticActor(), index, str(index), 'log', service, gen,
                          dict(arm='D3_FPO', group=16, mc_samples=4), 0., {})
            assert torch.equal(gen.get_state(), expected)
    finally:
        service.close()
