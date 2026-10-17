import copy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from tools.st_world_policy.calibration_measurement import (
    OBJECTIVES, measure, model_identity, preserved_rng, propose_weights, validate_groups)
from tools.st_world_policy.calibrate_temporal_scale import RUNTIME_ENVIRONMENT, pinned, validate, wall_limit


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.language = nn.Parameter(torch.tensor([1., 2.]))
        self.world = nn.Parameter(torch.tensor([3.]))
        self.head = nn.Parameter(torch.tensor([7.]))
        self.register_buffer('counter', torch.zeros(()), persistent=False)


GROUPS = {'last_language_block': ['language'], 'world_queries': ['world']}
COUNTS = {'current': 4, 'future_a': 4, 'future_b': 4, 'delta': 4, 'ego_scenes': 4}


def closure(model, multiplier=1., dropout=False):
    def forward():
        x = model.language.sum() + model.world.sum()
        if dropout:
            x = x * torch.rand(())
        return {'losses': {name: x * scale * multiplier for name, scale in
                zip(OBJECTIVES, [0.5, 1., 4., 2.])}}
    return forward


def test_analytic_gradients_cosines_and_fixed_rule_not_total_gradient():
    model = Tiny()
    before = model_identity(model)
    r = measure(model, [closure(model)], groups=GROUPS, counts=COUNTS, generators={}, max_microbatches=1)
    assert r['losses'] == dict(zip(OBJECTIVES, [3., 6., 24., 12.]))
    g = r['gradient_groups']['declared_shared_union']
    assert g['norms']['current'] == pytest.approx(3**.5)
    assert g['norms']['future'] == pytest.approx(4 * 3**.5)
    assert all(c == pytest.approx(1.) for c in g['cosines'].values())
    assert r['proposal']['weights'] == {'current': 1., 'future': .25, 'delta': .5}
    assert model_identity(model) == before
    assert all(p.grad is None for p in model.parameters())


def test_microbatches_sum_already_global_normalized_contributions():
    a, b = Tiny(), Tiny()
    whole = measure(a, [closure(a)], groups=GROUPS, counts=COUNTS, generators={}, max_microbatches=1)
    split = measure(b, [closure(b, .25), closure(b, .75)], groups=GROUPS, counts=COUNTS,
                    generators={}, max_microbatches=2)
    assert whole['losses'] == split['losses']
    assert whole['gradient_groups'] == split['gradient_groups']
    assert whole['proposal'] == split['proposal']


def test_same_graph_dropout_only_once_for_all_tasks_and_rng_restored():
    model = Tiny()
    state = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(44)
    original = generator.get_state().clone()
    calls = []
    def forward():
        calls.append(1)
        torch.rand(3, generator=generator)
        random.random(); np.random.rand()
        return closure(model, dropout=True)()
    r = measure(model, [forward], groups=GROUPS, counts=COUNTS, generators={'g': generator}, max_microbatches=1)
    assert calls == [1]
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(original, generator.get_state())
    assert r['proposal']['weights']['future'] == pytest.approx(.25)


def test_preserve_all_rng_on_failure():
    random.seed(88); np.random.seed(88); torch.manual_seed(88)
    expected = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    with pytest.raises(RuntimeError):
        with preserved_rng({}):
            random.random(); np.random.rand(); torch.rand(2)
            raise RuntimeError('stop')
    assert random.getstate() == expected[0]
    assert np.array_equal(np.random.get_state()[1], expected[1][1])
    assert torch.equal(torch.get_rng_state(), expected[2])


@pytest.mark.parametrize('change', ['param', 'buffer', 'mode'])
def test_no_update_contract_detects_mutation(change):
    model = Tiny()
    def forward():
        if change == 'param':
            with torch.no_grad(): model.head.add_(1)
        elif change == 'buffer': model.counter.add_(1)
        else: model.eval()
        return closure(model)()
    with pytest.raises(RuntimeError, match='changed model'):
        measure(model, [forward], groups=GROUPS, counts=COUNTS, generators={}, max_microbatches=1)


@pytest.mark.parametrize('kind', ['zero', 'missing', 'nonfinite', 'missing_task'])
def test_signal_failures_never_invent_coefficients(kind):
    model = Tiny()
    def forward():
        result = closure(model)()
        if kind == 'zero': result['losses']['delta'] = model.world.sum() * 0
        elif kind == 'missing': result['losses']['delta'] = model.head.sum()
        elif kind == 'nonfinite': result['losses']['delta'] = model.world.sum() * float('nan')
        else: del result['losses']['delta']
        return result
    if kind in ('nonfinite', 'missing_task'):
        with pytest.raises(ValueError):
            measure(model, [forward], groups=GROUPS, counts=COUNTS, generators={}, max_microbatches=1)
    else:
        r = measure(model, [forward], groups=GROUPS, counts=COUNTS, generators={}, max_microbatches=1)
        assert r['proposal']['status'] == 'UNCALIBRATABLE'
        assert r['proposal']['weights'] is None
        if kind == 'missing':
            assert r['gradient_groups']['world_queries']['missing_gradient_parameters']['delta'] == ['world']


def test_zero_valid_counts_not_loss_denominator_fallback():
    model = Tiny()
    counts = {**COUNTS, 'delta': 0}
    r = measure(model, [closure(model)], groups=GROUPS, counts=counts, generators={}, max_microbatches=1)
    assert r['proposal']['status'] == 'UNCALIBRATABLE'


def test_existing_gradients_and_unknown_frozen_overlap_groups_rejected():
    model = Tiny()
    model.language.grad = torch.zeros_like(model.language)
    with pytest.raises(ValueError, match='existing gradients'):
        measure(model, [closure(model)], groups=GROUPS, counts=COUNTS, generators={}, max_microbatches=1)
    assert model.language.grad is not None
    for groups in ({**GROUPS, 'world_queries': ['language']},
                   {**GROUPS, 'world_queries': ['missing']},
                   {**GROUPS, 'world_queries': []}):
        with pytest.raises(ValueError): validate_groups(model, groups)
    model.world.requires_grad_(False)
    with pytest.raises(ValueError, match='frozen'): validate_groups(model, GROUPS)


def test_budget_exhaustion_before_extra_forward_and_rng_preserved():
    model = Tiny(); calls = []
    def forward():
        calls.append(1); return closure(model)()
    with pytest.raises(ValueError, match='budget'):
        measure(model, [forward, forward], groups=GROUPS, counts=COUNTS, generators={}, max_microbatches=1)
    assert len(calls) == 1


def test_conflicting_gradients_have_negative_cosine():
    model = Tiny()
    def forward():
        x = model.language.sum() + model.world.sum()
        return {'losses': {'ego_fm': x, 'current': x, 'future': 20-x, 'delta': 30-2*x}}
    r = measure(model, [forward], groups=GROUPS, counts=COUNTS, generators={}, max_microbatches=1)
    assert r['gradient_groups']['declared_shared_union']['cosines']['current:future'] == pytest.approx(-1.)


def test_bf16_accumulation_is_reported_not_silently_fp32():
    model = Tiny().bfloat16()
    r = measure(model, [closure(model, 0.1)] * 3, groups=GROUPS, counts=COUNTS, generators={}, max_microbatches=3)
    assert r['parameter_subset']['world_queries'][0]['dtype'] == 'torch.bfloat16'
    expected = torch.zeros((), dtype=torch.bfloat16)
    for _ in range(3): expected += torch.tensor(.1, dtype=torch.bfloat16)
    assert r['gradient_groups']['world_queries']['norms']['current'] == float(expected)


def test_pinned_wrong_hash_rejected(tmp_path):
    p = tmp_path / 'a.json'; p.write_text('{}')
    with pytest.raises(ValueError, match='checksum'): pinned({'path': str(p), 'sha256': '0'*64})


def test_missing_common_initialization_and_unknown_config_rejected():
    with pytest.raises(ValueError): validate({'schema': 'bogus'})
    cfg = dict(schema='st_temporal_scale_calibration_v1', purpose='SCIENTIFIC_CALIBRATION',
               source={'kind': 'common_s0', 'binding': {}, 'initialization_reference': 'unregistered'}, training_config={}, data_manifest={},
               teacher_manifest={}, teacher_requests={}, batch={}, parameter_groups='', output_root='',
               runtime_environment=RUNTIME_ENVIRONMENT)
    with pytest.raises(ValueError, match='registered'): validate(cfg)


@pytest.mark.parametrize('scope', ['SMOKE', 'PROFILE'])
def test_diagnostic_learned_native_weights_rejected_before_payload(tmp_path, scope):
    from tools.st_world_policy.calibrate_temporal_scale import SCHEMA
    from starVLA.model.modules.vehicle_joint.initialization import file_sha256
    def save(name, value):
        p = tmp_path / name; p.write_text(json.dumps(value))
        return {'path': str(p), 'sha256': file_sha256(p)}
    parent = save('parent.json', {'execution': {'scope': scope}})
    reg = save('registration.json', {'artifacts': {'config': parent}, 'weights_usable_for_formal_init': False})
    receipt = save('COMPLETE.json', {'state': {'completed': 4}})
    cfg = dict(schema=SCHEMA, purpose='SCIENTIFIC_CALIBRATION',
               source={'kind': 'native_st', 'registration': reg, 'tag': 'step4', 'receipt': receipt},
               training_config={}, data_manifest={}, teacher_manifest={}, teacher_requests={},
               batch={}, parameter_groups='', output_root='', runtime_environment=RUNTIME_ENVIRONMENT)
    with pytest.raises(ValueError, match='SMOKE/PROFILE'):
        validate(cfg)


def test_distributed_state_is_not_silently_averaged(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    model = Tiny()
    with pytest.raises(ValueError, match='single process'):
        measure(model, [closure(model)], groups=GROUPS, counts=COUNTS, generators={}, max_microbatches=1)


def test_true_temporal_losses_preserve_masks_delta_and_teacher_detach():
    from starVLA.model.modules.st_world.losses import temporal_losses, optimizer_batch_counts
    model = Tiny()
    teacher = torch.zeros(2, 1, 1, 1, requires_grad=True)
    current_valid = torch.tensor([[[True]], [[False]]])
    future_valid = torch.tensor([[[[True]], [[True]]], [[[False]], [[True]]]])
    counts = {k: int(v) for k, v in optimizer_batch_counts(1, current_valid=current_valid,
                          future_valid=future_valid, delta=True, distributed=False).items()}
    counts['ego_scenes'] = 2
    def forward():
        x = model.language.sum() + model.world.sum()
        current = x.expand(2, 1, 1, 1)
        future = torch.stack((current, current * 2), 1)
        terms, _ = temporal_losses({k: v for k, v in counts.items() if k != 'ego_scenes'},
            current_prediction=current, current_target=teacher, current_valid=current_valid,
            future_prediction=future, future_target=teacher[:, None].expand_as(future),
            future_valid=future_valid, delta=True)
        return {'losses': {'ego_fm': x, **terms}}
    r = measure(model, [forward], groups=GROUPS, counts=counts, generators={}, max_microbatches=1)
    assert r['losses']['current'] == 36.
    assert r['losses']['future'] == 90.
    assert r['losses']['delta'] == 36.
    assert teacher.grad is None
    assert r['gradient_groups']['world_queries']['norms']['current'] == 12.
    assert r['gradient_groups']['world_queries']['norms']['future'] == 30.
    assert r['gradient_groups']['world_queries']['norms']['delta'] == 12.


def test_mixed_valid_microbatches_keep_original_batch_denominators():
    from starVLA.model.modules.st_world.losses import temporal_losses
    masks = torch.tensor([[[True]], [[False]], [[True]]])
    masks_future = torch.stack((masks, masks), 1)
    counts = {'current': 2, 'future_a': 2, 'future_b': 2, 'delta': 2, 'ego_scenes': 3}
    def build(model, start, stop):
        def forward():
            x = model.language.sum() + model.world.sum()
            p = x.expand(stop-start, 1, 1, 1)
            f = torch.stack((p, p*2), 1)
            terms, _ = temporal_losses({k:v for k,v in counts.items() if k != 'ego_scenes'},
                current_prediction=p, current_target=torch.zeros_like(p), current_valid=masks[start:stop],
                future_prediction=f, future_target=torch.zeros_like(f), future_valid=masks_future[start:stop], delta=True)
            return {'losses': {'ego_fm': x*(stop-start)/3, **terms}}
        return forward
    a,b = Tiny(),Tiny()
    whole = measure(a,[build(a,0,3)],groups=GROUPS,counts=counts,generators={},max_microbatches=1)
    split = measure(b,[build(b,i,i+1) for i in range(3)],groups=GROUPS,counts=counts,generators={},max_microbatches=3)
    assert whole['losses'] == split['losses']
    assert whole['gradient_groups'] == split['gradient_groups']
