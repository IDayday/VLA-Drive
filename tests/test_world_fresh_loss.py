import copy
import json
from pathlib import Path
import pytest
import torch

from starVLA.rl.qwen_world_postcot.contracts import fingerprint, file_sha
from starVLA.rl.qwen_world_postcot.fresh_loss import objective, student_initialization_digest, select_survivors, verify_selection
from starVLA.rl.qwen_world_postcot.loss_designs import LossDesign
from starVLA.rl.qwen_world_postcot.coordinate_objective import CoordinateObjective, tolerance_weights


@pytest.mark.parametrize('kind', ['PHYSICAL', 'TIME_RESIDUAL', 'DELTA', 'HUBER'])
def test_fresh_objective_matches_existing_value_and_gradient(kind):
    torch.manual_seed(42)
    x = torch.randn(2, 8, 3, requires_grad=True); y = torch.randn_like(x)
    valid = torch.ones(2, 8, dtype=torch.bool); valid[0, 3] = False
    options = dict(scale=.9)
    if kind == 'PHYSICAL':
        options['weights'] = tolerance_weights(); reference = CoordinateObjective(**options)
        config = dict(kind=kind, **options)
    else:
        if kind == 'TIME_RESIDUAL': options['horizon_weights'] = [1.2]*4+[.8]*4
        if kind == 'DELTA': options['delta_coefficient'] = .1
        reference = LossDesign(kind, **options); config = dict(kind='LOSS_DESIGN', loss_kind=kind, **options)
    loss, _ = objective(config)(x, y, valid); expected, _ = reference(x, y, valid)
    assert torch.equal(loss, expected)
    assert torch.equal(torch.autograd.grad(loss, x)[0], torch.autograd.grad(expected, x)[0])


def test_initial_student_digest_excludes_only_separately_verified_qwen():
    model = torch.nn.Module(); model.qwen = torch.nn.Linear(3, 3); model.actor = torch.nn.Linear(3, 3)
    before = student_initialization_digest(model)
    with torch.no_grad(): model.qwen.weight.add_(1)
    assert before == student_initialization_digest(model)
    with torch.no_grad(): model.actor.weight.add_(1)
    assert before != student_initialization_digest(model)


def test_selection_requires_all_full_test_and_removes_clear_inferior_only():
    arms = ('PHYSICAL', 'TIME_RESIDUAL', 'DELTA', 'HUBER')
    rows = {a: dict(PDMS=80-i*.1, scenes=12146, invalid_scenes=0, delta_points=-i*.1,
                   paired_CI_points=[-.5, .5]) for i, a in enumerate(arms)}
    assert select_survivors(rows, {}) == ['PHYSICAL', 'TIME_RESIDUAL']
    rows['PHYSICAL'].update(delta_points=-.6, paired_CI_points=[-1., -.1])
    assert select_survivors(rows, {}) == ['TIME_RESIDUAL', 'DELTA']
    rows['HUBER']['scenes'] = 12145
    with pytest.raises(ValueError): select_survivors(rows, {})


def test_full_continuation_requires_selected_sealed_same_lineage(tmp_path):
    cp = tmp_path/'checkpoint'; cp.mkdir(); (cp/'files.json').write_text('{}')
    (cp/'COMPLETE').write_text(json.dumps(dict(updates=10000)))
    (cp/'identity.json').write_text(json.dumps(dict(fresh_loss_experiment=dict(registration_sha256='reg', arm='HUBER'))))
    selection = dict(status='LOCKED', registration_sha256='reg', selected=['HUBER'],
        parents=dict(HUBER=dict(checkpoint=str(cp), files_sha256=file_sha(cp/'files.json'))))
    selection['sha256'] = fingerprint(selection); path = tmp_path/'selection.json'; path.write_text(json.dumps(selection))
    reg = dict(sha256='reg', pilot_updates=10000)
    verify_selection(path, reg, 'HUBER', cp)
    with pytest.raises(ValueError): verify_selection(path, reg, 'DELTA', cp)
    (cp/'files.json').write_text('{"changed": true}')
    with pytest.raises(ValueError): verify_selection(path, reg, 'HUBER', cp)
