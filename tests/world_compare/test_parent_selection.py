"""Fixed full-window own-parent selection; no Test-derived override."""
from pathlib import Path

import pytest

from starVLA.rl.qwen_world_compare import parent_selection as selection
from starVLA.rl.qwen_world_compare.contracts import fingerprint


def record(step, score=None):
    return dict(updates=step, checkpoint=None if step == 0 else f'/run/checkpoints/update{step:06}',
        model_sha256='IL_native', training_identity_sha256='IL_identity',
        deployment_model_sha256=f'model{step}', PDMS=80. if score is None else score,
        evaluation=f'/run/DEV/{step}', evaluation_identity_sha256=f'eval{step}',
        summary_sha256=f'summary{step}', protocol_sha256='canonical_v1.1',
        token_sha256='DEV1696', data_sha256='DEV_data', numerics={'BF16':True})


def fake_IL(path, **_):
    step = int(Path(path).name.removeprefix('update'))
    return dict(path=str(path), model_sha256='IL_native',
        seal=dict(updates=step, files_sha256='IL_files'),
        identity=dict(schema='qwen_world_compare_IL_v1', architecture_id='A', qualification=False))


@pytest.fixture
def IL_runtime(monkeypatch):
    identity = fake_IL('/run/checkpoints/update100000')['identity']
    monkeypatch.setattr(selection, 'inspect_IL_checkpoint', fake_IL)
    def candidate(row, architecture, stage):
        data = record(row['updates'], row.get('PDMS'))
        data['training_identity_sha256'] = fingerprint(identity)
        return data
    monkeypatch.setattr(selection, '_candidate', candidate)
    return candidate


def test_full_IL_window_exact_tie_chooses_earliest_and_freezes_receipt(tmp_path, IL_runtime):
    candidates = [dict(updates=step, PDMS=90. if step in (15000, 50000) else 80.)
                  for step in selection.IL_MILESTONES]
    path = tmp_path/'selection.json'
    data = selection.select_il_parent(candidates, '/run/checkpoints/update100000', path)
    assert data['selected']['updates'] == 15000
    assert data['Navtest_used'] is False
    assert selection.validate_selection(path) == data
    assert selection.select_il_parent(candidates, '/run/checkpoints/update100000', path) == data
    changed = [dict(row, PDMS=92. if row['updates'] == 100000 else row['PDMS']) for row in candidates]
    with pytest.raises(FileExistsError):
        selection.select_il_parent(changed, '/run/checkpoints/update100000', path)


def test_missing_duplicate_window_and_premature_endpoint_rejected(IL_runtime):
    rows = [dict(updates=step) for step in selection.IL_MILESTONES]
    with pytest.raises(ValueError, match='twenty'):
        selection.select_il_parent(rows[:-1], '/run/checkpoints/update100000')
    with pytest.raises(ValueError, match='twenty'):
        selection.select_il_parent(rows+[rows[-1]], '/run/checkpoints/update100000')
    with pytest.raises(ValueError, match='complete100k'):
        selection.select_il_parent(rows, '/run/checkpoints/update095000')


def test_resealed_wrong_selected_parent_and_Test_selection_are_rejected(IL_runtime):
    rows = [dict(updates=step, PDMS=90. if step == 20000 else 80.) for step in selection.IL_MILESTONES]
    data = selection.select_il_parent(rows, '/run/checkpoints/update100000')
    data['selected'] = data['evaluated_candidates'][-1]
    data.pop('sha256');data['sha256'] = fingerprint(data)
    with pytest.raises(ValueError, match='tie-earliest'):
        selection.validate_selection(data)
    data['selected'] = data['evaluated_candidates'][3]
    data['Navtest_used'] = True
    data.pop('sha256');data['sha256'] = fingerprint(data)
    with pytest.raises(ValueError, match='DEV-only'):
        selection.validate_selection(data)


def test_RL_selection_includes_zero_and_own_IL_parent(monkeypatch, IL_runtime):
    il = selection.select_il_parent([dict(updates=u, PDMS=90. if u == 15000 else 80.)
        for u in selection.IL_MILESTONES], '/run/checkpoints/update100000')
    identity = dict(schema='qwen_world_compare_RL_v1', architecture_id='A', purpose='formal',
        IL_parent={'model_sha256':'IL_native'}, sha256='RL_identity')
    monkeypatch.setattr(selection, 'inspect_RL_checkpoint', lambda path, **kwargs:
        dict(path=str(path), identity=identity, seal={'updates':50000, 'files_sha256':'RL_files'}))
    def candidate(row, architecture, stage):
        data = record(row['updates'], row.get('PDMS'))
        data['training_identity_sha256'] = ('IL_identity' if row['updates'] == 0 else 'RL_identity')
        return data
    monkeypatch.setattr(selection, '_candidate', candidate)
    rows = [dict(updates=u, PDMS=91. if u in (0, 20000) else 89.) for u in selection.RL_MILESTONES]
    result = selection.select_rl_policy(rows, '/run/RL/update050000', il)
    assert result['selected']['updates'] == 0  # No forced RL winner if IL ties it.
    assert selection.validate_selection(result, stage='RL') == result
    with pytest.raises(ValueError, match='every registered'):
        selection.select_rl_policy(rows[1:], '/run/RL/update050000', il)


def test_formal_navtest_roles_bind_exact_DEV_observed_model(IL_runtime):
    data = selection.select_il_parent([dict(updates=u, PDMS=90. if u == 15000 else 80.)
        for u in selection.IL_MILESTONES], '/run/checkpoints/update100000')
    export = dict(architecture_id='A', model_sha256='model15000', IL_updates=15000,
                  IL_parent={'model_sha256':'IL_native'})
    assert selection.validate_evaluation_selection(data, 'il_selected', export)['Navtest_used_for_selection'] is False
    with pytest.raises(ValueError, match='frozen DEV-observed'):
        selection.validate_evaluation_selection(data, 'il_endpoint', export)
    with pytest.raises(ValueError, match='frozen DEV-observed'):
        selection.validate_evaluation_selection(data, 'il_selected', dict(export, architecture_id='B'))


def test_candidate_path_cannot_substitute_Navtest_score(monkeypatch):
    from tools.qwen_world_compare import evaluate
    monkeypatch.setattr(evaluate, 'summarize', lambda _:dict(PDMS=99., invalid_scenes=0))
    monkeypatch.setattr(evaluate, 'read_evaluation_identity', lambda _:
        dict(cohort='NAVTEST', qualification=False, seed=2101, architecture_id='A', count=12146,
             inference='single_candidate_cleanx_K10_shift5'))
    with pytest.raises(ValueError, match='complete formal DEV'):
        selection._candidate(dict(updates=100000, evaluation='/test', checkpoint='/native'), 'A', 'IL')
