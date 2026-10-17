"""Independent bounded-provider counterexamples; no real teacher or GPU work."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from tests.st_world.test_ondemand_teacher import (
    Executor, fixture, provider, z_for_key,
)
from starVLA.dataloader.st_world_targets import payload_key


def rng_state():
    return random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()


def unchanged(before):
    py, np_before, cpu = before
    np_after = np.random.get_state()
    assert random.getstate() == py
    assert np_before[0] == np_after[0]
    assert np.array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert torch.equal(torch.get_rng_state(), cpu)


def test_review_exception_restores_gaussian_and_cpu_rng_caches(tmp_path):
    c, m = fixture(tmp_path)

    class Failing(Executor):
        def compute_batch(self, *args, **kwargs):
            random.gauss(0, 1)
            np.random.normal(size=3)
            torch.randn(7)
            raise RuntimeError('failure after consuming every CPU RNG')

    random.seed(13)
    np.random.seed(17)
    torch.manual_seed(19)
    random.gauss(0, 1)
    np.random.normal()
    p = provider(c, m, executor=Failing(c))
    before = rng_state()
    with pytest.raises(RuntimeError, match='consuming every CPU RNG'):
        p.read_scene('one')
    unchanged(before)
    assert p.stats['failures'] == 1 and p.stats['computed_images'] == 0


def test_review_partial_batch_failure_retries_same_four_future_slots(tmp_path):
    c, m = fixture(tmp_path)

    class SecondBatchFailure(Executor):
        failed = False

        def compute_batch(self, *args, **kwargs):
            if len(self.calls) == 1 and not self.failed:
                self.failed = True
                random.random()
                np.random.rand()
                torch.rand(1)
                raise TimeoutError('second batch unavailable')
            return super().compute_batch(*args, **kwargs)

    p = provider(c, m, executor=SecondBatchFailure(c))
    before = rng_state()
    with pytest.raises(TimeoutError, match='second batch'):
        p.read_scene('one')
    unchanged(before)
    # Two valid records may remain hot, but no partial scene was returned.
    assert p.stats['computed_images'] == 2
    result = p.read_scene('one')
    unchanged(before)
    assert result['future_times_s'] == [.5, 1., 2., 4.]
    assert result['future_dense_features'].shape == (4, 3, 2, 2, 2)
    assert result['future_valid'].all()
    assert p.stats['computed_images'] == 5
    for at, frame in enumerate(m['scenes'][0]['frames'][1:]):
        for view, ref in enumerate(frame['views']):
            assert torch.equal(result['future_dense_features'][at, view],
                               z_for_key(payload_key(ref, c)))


def test_review_scene_budget_rejects_before_allocation_or_executor(tmp_path, monkeypatch):
    c, m = fixture(tmp_path)
    p = provider(c, m, max_scene_bytes=1)

    def forbidden(*args, **kwargs):
        raise AssertionError('allocation before budget check')

    monkeypatch.setattr(torch, 'zeros', forbidden)
    with pytest.raises(ValueError, match='output tensor RAM budget'):
        p.read_scene('one')
    assert not p.executor.calls


def test_review_tiny_resident_budget_never_keeps_batch_storage(tmp_path):
    c, m = fixture(tmp_path)
    p = provider(c, m, max_cached_bytes=36)  # one raw record: 8 FP32 + 4 bool
    p.read_scene('one')
    assert len(p._resident) == 1
    z, mask = next(iter(p._resident.values()))
    assert z.untyped_storage().nbytes() == 32
    assert mask.untyped_storage().nbytes() == 4
    assert p.memory_report()['resident_bytes'] == 36
    assert p.memory_report()['peak_resident_bytes'] == 36
    p2 = provider(c, m, max_cached_bytes=1)
    result = p2.read_scene('one')
    assert result['future_valid'].all() and not p2._resident


def tiny_executor(tmp_path, monkeypatch, *, mutate_during_forward=False):
    from tools.st_world_policy.ondemand_teacher import FrozenDINOExecutor
    from tools.st_world_policy import cache_teacher
    from starVLA.model.modules.foresight import dinov3_target_encoder as dino
    c, m = fixture(tmp_path)
    monkeypatch.setattr(cache_teacher, 'make_contract', lambda *a, **k: (c, {}, {}))

    class Tiny(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.register_buffer('seen', torch.zeros(1))
            self.patch_size = (16, 16)
            self.feature_dim = 2
            self.prefix = 5
            self.calls = 0
            self.eval()

        def forward(self, pixels):
            self.calls += 1
            torch.rand(2)
            random.gauss(0, 1)
            np.random.normal()
            if mutate_during_forward:
                self.seen.add_(1)
            return pixels.new_ones((len(pixels), 2, 2, 2))

    monkeypatch.setattr(dino, 'DINOv3TargetEncoder', Tiny)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    e = FrozenDINOExecutor(tmp_path, contract=c, variant='SP2', device='cpu', max_batch_images=2)
    ref = m['scenes'][0]['frames'][0]['views'][0]
    return e, c, {'key': payload_key(ref, c), 'image': ref}


def test_review_encoder_buffer_mutation_is_rejected_and_rng_restored(tmp_path, monkeypatch):
    e, c, record = tiny_executor(tmp_path, monkeypatch, mutate_during_forward=True)
    before = rng_state()
    with pytest.raises(ValueError, match='inference mutated frozen state'):
        e.compute_batch((record,), contract=c, execution_identity=e.execution_identity)
    unchanged(before)


def test_review_changed_image_rejected_before_forward_and_rng_preserved(tmp_path, monkeypatch):
    e, c, record = tiny_executor(tmp_path, monkeypatch)
    from pathlib import Path
    Path(record['image']['path']).write_bytes(b'changed bytes')
    before = rng_state()
    with pytest.raises(ValueError, match='source image changed'):
        e.compute_batch((record,), contract=c, execution_identity=e.execution_identity)
    assert e.model.calls == 0
    unchanged(before)


def test_review_backend_change_is_not_silently_adopted(tmp_path, monkeypatch):
    e, c, record = tiny_executor(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', not e.execution['cudnn_deterministic'])
    with pytest.raises(ValueError, match='backend changed'):
        e.compute_batch((record,), contract=c, execution_identity=e.execution_identity)
    assert e.model.calls == 0
