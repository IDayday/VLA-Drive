from copy import deepcopy
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from iqe.contracts import TrajectoryContract
from iqe.internal_baseline.scorer import initialization_rng
from iqe.robust_scorer.model import RobustScorer
from iqe.robust_scorer.objectives import terms, denominators
from iqe.specialist_study.objectives import StudySampler
from iqe.training.trainer import train, slice_batch
from iqe.io import read_json


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def contract():
    return TrajectoryContract(8, .5, 'normalized_xy_sincos', 'ego_relative', 'rear_axle',
                              'radian', (10., .3, 0., 0.), (8., 2., 1., 1.), 'test')


def model(kind='mean_bce', dropout=0.):
    with initialization_rng(7):
        return RobustScorer(8, 8, contract(), objective=kind, dropout=dropout)


def batch(n=4, k=4):
    rng = torch.Generator().manual_seed(29)
    context = {'world': torch.randn(n, 3, 8, generator=rng, requires_grad=True),
               'action': torch.randn(n, 4, 8, generator=rng, requires_grad=True),
               'current_ego': torch.randn(n, 4, generator=rng, requires_grad=True),
               'world_valid': torch.ones(n, 3, dtype=torch.bool),
               'action_valid': torch.ones(n, 4, dtype=torch.bool)}
    return {'context': context, 'trajectories': torch.randn(n, k, 8, 3, generator=rng, requires_grad=True),
            'scores': torch.rand(n, k, generator=rng), 'valid': torch.ones(n, k, dtype=torch.bool),
            'components': torch.rand(n, k, 6, generator=rng), 'component_valid': torch.ones(n, k, 6, dtype=torch.bool),
            'natural': torch.arange(n) < n//2, 'comparison': torch.arange(n) >= n//2}


def total(pred, b, kind):
    ts = terms(pred, b, kind)
    return sum(t.numerator/t.denominator.clamp_min(1) for t in ts.values())


def test_bce_learns_mean_not_huber_outlier_clipped_location():
    truth = torch.tensor([1.]*9+[0.])
    logit = torch.tensor(np.log(9.), requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logit.expand_as(truth), truth)
    loss.backward()
    assert abs(logit.grad.item()) < 1e-7
    p = torch.tensor(.9, requires_grad=True)
    F.smooth_l1_loss(p.expand_as(truth), truth, beta=.1).backward()
    assert p.grad.item() < -.7  # Huber moves away from the true mean toward 1.


@pytest.mark.parametrize('kind', ['mean_bce', 'zero_hurdle'])
def test_candidate_independence_context_detach_and_gradient(kind):
    m = model(kind).eval()
    b = batch()
    p = m(b['context'], b['trajectories'])
    perm = torch.tensor([2, 0, 3, 1])
    other = m(b['context'], b['trajectories'][:, perm])
    assert torch.allclose(p['values'][:, perm], other['values'], atol=1e-7)
    one = m(b['context'], b['trajectories'][:, :1])
    assert torch.allclose(p['values'][:, :1], one['values'], atol=2e-7)
    duplicated = m(b['context'], b['trajectories'][:, [0, 0]])
    assert torch.equal(duplicated['values'][:, 0], duplicated['values'][:, 1])
    total(p, b, kind).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert b['context']['world'].grad is None and b['trajectories'].grad is None


def test_hurdle_no_false_conditional_gradient_for_zero_outcomes():
    m = model('zero_hurdle')
    b = batch()
    b['scores'].zero_()
    b['natural'].fill_(True)
    b['comparison'].fill_(False)
    pred = m(b['context'], b['trajectories'])
    t = terms(pred, b, 'zero_hurdle')
    assert t['value'].numerator.item() == 0
    t['nonzero'].numerator.backward()
    assert m.value_head.weight.grad is None
    assert m.nonzero_head.weight.grad.norm() > 0


@pytest.mark.parametrize('kind', ['mean_bce', 'zero_hurdle'])
def test_microbatch_global_denominators_and_missing_labels(kind):
    first, second = model(kind), model(kind)
    b = batch(6)
    b['valid'][0, 2:] = False
    b['component_valid'][1, :, :] = False
    b['components'][1, :, :] = float('nan')
    b['scores'][4, :2] = .5  # a tie in the comparison stream
    d = denominators(b, kind)
    total(first(b['context'], b['trajectories']), b, kind).backward()
    counts = {k: 0. for k in d}
    for start in (0, 2, 4):
        micro = slice_batch(b, start, start+2)
        t = terms(second(micro['context'], micro['trajectories']), micro, kind)
        sum(v.numerator/d[k].clamp_min(1) for k, v in t.items()).backward()
        for k, v in t.items():
            counts[k] += v.denominator
    assert all(counts[k] == d[k] for k in d)
    for a, c in zip(first.parameters(), second.parameters()):
        assert torch.allclose(a.grad, c.grad, atol=2e-6, rtol=2e-5)


def test_common_initialization_and_rng_preservation():
    torch.manual_seed(2)
    before = torch.get_rng_state().clone()
    a, b = model('mean_bce'), model('zero_hurdle', .1)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(t, b.state_dict()[k]) for k, t in a.state_dict().items())


def test_empty_denominators_are_finite_connected_zeros():
    b = batch()
    b['valid'].fill_(False)
    b['component_valid'].fill_(False)
    m = model('zero_hurdle')
    loss = total(m(b['context'], b['trajectories']), b, 'zero_hurdle')
    assert loss.item() == 0
    loss.backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in m.parameters())


def test_nonfinite_legal_output_rejected():
    b = batch()
    p = model()(b['context'], b['trajectories'])
    p['value_logits'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        terms(p, b, 'mean_bce')


def test_dropout_two_stream_phase_and_terminal_exact_resume(tmp_path):
    # Real optimizer, dropout and native trainer, including the phase boundary.
    ids = [str(i) for i in range(8)]
    scores = np.tile([.95, .1, .8, 0.], (8, 1))
    def sampler():
        return StudySampler(ids, scores, np.ones_like(scores, bool), seed=3, bank_hash='unit', phase_steps=2)
    data = batch(8)
    def fetch(entries, device):
        idx = torch.tensor([int(e['scene_id']) for e in entries])
        b = {k: ({n: x[idx] for n, x in v.items()} if isinstance(v, dict) else v[idx]) for k, v in data.items()}
        b['natural'] = torch.tensor([e['natural'] for e in entries])
        b['comparison'] = torch.tensor([e['comparison'] for e in entries])
        return b
    cfg = dict(lr=1e-4, weight_decay=.01, global_batch_size=128, micro_batch_size=32,
               max_optimizer_steps=4, warmup_steps=1, lr_min_ratio=1., schedule='warmup_constant',
               grad_clip_norm=1., checkpoint_every_steps=2, validation_every_steps=2, early_stop_patience=5)
    deps = {'test': 'real_cpu_dropout_resume'}
    def run(m, s, path, **kw):
        return train(m, lambda m, b: terms(m(b['context'], b['trajectories']), b, 'mean_bce'),
                     fetch, s, cfg, deps, path, seed=51, denominator_function=lambda b: denominators(b, 'mean_bce'), **kw)
    full, sf = model(dropout=.1), sampler()
    hf = run(full, sf, tmp_path/'full')
    partial, sp = model(dropout=.1), sampler()
    run(partial, sp, tmp_path/'resumed', stop_after=2)
    resumed, sr = model(dropout=.1), sampler()
    hr = run(resumed, sr, tmp_path/'resumed', resume=str(tmp_path/'resumed/step_000002.pt'))
    assert [x['sample_ids'] for x in hf[2:]] == [x['sample_ids'] for x in hr]
    assert [x['loss'] for x in hf[2:]] == [x['loss'] for x in hr]
    assert all(torch.equal(p, resumed.state_dict()[k]) for k, p in full.state_dict().items())
    assert sf.state_dict() == sr.state_dict()
    final = read_json(tmp_path/'resumed/result.json')
    assert final['status'] == 'COMPLETE'
    before = deepcopy(sr.state_dict())
    assert run(resumed, sr, tmp_path/'resumed', resume=final['checkpoint']) == []
    assert before == sr.state_dict()
