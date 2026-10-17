"""Independent review tests; synthetic seams, not real checkpoint evidence."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from starVLA.dataloader.st_world_schema import ModelInputs
from starVLA.model.framework.st_world_model import SpaceTimeStudent
from tools.st_world_policy import spatial_probe as p


def labels(n=2):
    d = torch.full((n, 3, 18, 24), 10., dtype=torch.float32)
    small = torch.zeros_like(d)
    small[..., 0, 0] = 1
    return {'depth': d, 'depth_valid': torch.ones_like(d, dtype=torch.bool),
            'small_vehicle': small, 'small_valid': torch.ones_like(d, dtype=torch.bool)}


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('tokens', [144, 288])
def test_actual_strip_and_encode_methods_keep_W_without_readout_or_training_forward(tokens):
    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
            self.qwen_vl_interface = SimpleNamespace(model=SimpleNamespace(
                model=SimpleNamespace(visual=torch.nn.Identity())))
            self.calls = 0

        def encode_current(self, observation):
            self.calls += 1
            # ModelInputs preserves the legal scene token. Original encode_current
            # performs its further image/state/lang whitelist before the VLM.
            assert all(set(x) == {'image', 'state', 'lang', 'token'} for x in observation)
            return {'W': self.value.expand(len(observation), tokens, 8)}

        def forward(self, *a, **kw):
            raise AssertionError('Training forward forbidden')

    # Avoid weights/model construction; exercise the real production strip and encode methods.
    model = SpaceTimeStudent.__new__(SpaceTimeStudent)
    torch.nn.Module.__init__(model)
    model.base = Parent()
    model.st_config = SimpleNamespace(history=False)
    model.sparse_readout = torch.nn.Identity()
    model._deployment_only = False
    model.strip_auxiliary_heads()
    model.eval().requires_grad_(False)
    assert not hasattr(model, 'sparse_readout') and model._deployment_only
    obs = ModelInputs({'image': [object()] * 3, 'state': [[0., 0., 0., 0.]], 'lang': '', 'token': 'synthetic'})
    w = p.encode_world(model, [obs], tokens=tokens, hidden=8)
    assert model.base.calls == 1 and w.shape == (1, tokens, 8)
    assert not w.requires_grad and torch.equal(w[0, 0], model.base.value)


def test_masked_task_means_use_full_two_scene_counts_and_exact_gradient():
    pred = torch.zeros((2, 4, 2), requires_grad=True)
    target = {'depth': torch.tensor([[1., float('nan'), float('nan'), float('nan')], [2., 3., 4., 5.]]),
              'depth_valid': torch.tensor([[True, False, False, False], [True, True, True, True]]),
              'small_vehicle': torch.tensor([[1., 0., 0., 0.], [1., float('nan'), float('nan'), float('nan')]]),
              'small_valid': torch.tensor([[True, True, True, True], [True, False, False, False]])}
    stats = {'log_mean': 0., 'log_std': 1., 'pos_weight': 3.}
    loss, _, counts = p.task_loss(pred, target, stats)
    assert counts == {'depth': 5, 'small': 5}
    expected = torch.nn.functional.smooth_l1_loss(pred[..., 0][target['depth_valid']],
        target['depth'][target['depth_valid']].log(), beta=.1, reduction='mean')
    expected = expected + torch.nn.functional.binary_cross_entropy_with_logits(
        pred[..., 1][target['small_valid']], target['small_vehicle'][target['small_valid']],
        pos_weight=torch.tensor(3.), reduction='mean')
    assert torch.equal(loss, expected)
    assert torch.equal(torch.autograd.grad(loss, pred, retain_graph=True)[0], torch.autograd.grad(expected, pred)[0])
    invalid = pred.detach().clone(); invalid[0, 1, 0] = float('nan')
    with pytest.raises(ValueError, match='prediction'):
        p.task_loss(invalid, target, stats)


def test_terminal_state_resume_consumes_no_batch_and_changes_no_train_state(tmp_path, monkeypatch):
    y = labels(); world = torch.arange(2*144*8, dtype=torch.float32).reshape(2, 144, 8) / 100
    masks = torch.ones(2, 144, dtype=torch.bool)
    rows = {s: [dict(token='s0', log='l0'), dict(token='s1', log='l1')] for s in ('train', 'dev')}
    data = {s: (world.clone(), masks.clone(), copy.deepcopy(y)) for s in rows}
    monkeypatch.setattr(p, 'load_fit_data', lambda _: ({'identity': 'fixture', 'interpretation': 'synthetic'}, rows, data, {}))
    monkeypatch.setattr(p, 'source_runtime', lambda *_: None)
    clock = [0.]
    monkeypatch.setattr(p.time, 'monotonic', lambda: clock[0])
    real_update = p.update_probe
    def one(*args):
        answer = real_update(*args); clock[0] += 1.; return answer
    monkeypatch.setattr(p, 'update_probe', one)
    reg = p.seal(dict(schema='st_spatial_probe_fit_v1', protocol={}, caches={'train': {}, 'dev': {}},
        arm='S-P0', source_sha='fixture', environment={}, device='cpu', max_seconds=1., max_output_bytes=64<<20))
    assert p.fit(reg, tmp_path)['completed'] == 1
    receipt = json.loads((tmp_path/'STATE.json').read_text())
    file = tmp_path / receipt['file']
    terminal = torch.load(file, weights_only=False)
    # Explicit fixture terminal metadata; not evidence of 300 actual training updates.
    terminal['stream']['completed'] = 300
    terminal['exposure'] = [copy.deepcopy(terminal['exposure'][0]) for _ in range(300)]
    torch.save(terminal, file)
    receipt.update(completed=300, sha256=p.file_sha256(file))
    (tmp_path/'STATE.json').write_text(json.dumps(receipt))
    before = copy.deepcopy(terminal)
    def forbidden(*args):
        raise AssertionError('Terminal resume must not draw a batch or update optimizer')
    monkeypatch.setattr(p, 'update_probe', forbidden)
    monkeypatch.setattr(p.PatchStream, 'next', forbidden)
    monkeypatch.setattr(p, 'evaluate', lambda *args, **kwargs: {'scope': 'synthetic evaluation seam'})
    clock[0] = 0.
    result = p.fit(reg, tmp_path, resume=True)
    assert result['status'] == 'COMPLETE' and result['completed'] == 300
    after_ref = json.loads((tmp_path/'STATE.json').read_text())
    after = torch.load(tmp_path/after_ref['file'], weights_only=False)
    same(before, after)


def test_encode_rejects_in_place_buffer_change():
    class Changes(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.register_buffer('counter', torch.zeros(1))
        def encode_inputs(self, inputs):
            self.counter.add_(1)
            return {'W': torch.ones(1, 144, 8)}
    model = Changes().eval()
    obs = ModelInputs({'image': [object()] * 3, 'state': [[0., 0., 0., 0.]], 'lang': '', 'token': 'synthetic'})
    with pytest.raises(ValueError, match='buffer changed'):
        p.encode_world(model, [obs], tokens=144, hidden=8)


def test_metadata_only_runtime_does_not_query_or_activate_cuda(monkeypatch):
    """Regression: planning a future CUDA extraction must work without a GPU."""
    source = 'a' * 40
    monkeypatch.setattr(p.subprocess, 'check_output', lambda argv, **kwargs:
        b'' if argv[1] == 'status' else source + '\n')
    # The binding retains the full expected runtime.  Only device/cuDNN checks
    # are deferred by metadata-only planning; static runtime identity is real.
    expected = p.runtime_environment(metadata_only=True)
    expected['base']['cudnn'] = 12345
    def forbidden(*args, **kwargs):
        raise AssertionError('Metadata-only planning queried or activated CUDA')
    monkeypatch.setattr(p.torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(p.torch.cuda, 'set_device', forbidden)
    monkeypatch.setattr(p.torch.cuda, 'init', forbidden)
    monkeypatch.setattr(p.torch.backends.cudnn, 'version', forbidden)
    p.source_runtime(source, expected, 'cuda:0', configure_device=False)
