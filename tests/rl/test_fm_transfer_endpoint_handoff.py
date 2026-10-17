import copy
import json
from pathlib import Path

import pytest

from scripts.analysis.fm_transfer import endpoint_handoff as h
from starVLA.rl.flow_grpo.transactions import atomic_json


def make(tmp_path):
    return h.plan(tmp_path, tmp_path / 'controller-source', ['host-a', 'host-b', 'host-c'], 'host-b')


def test_plan_keeps_registered_budget_source_and_own_parent(tmp_path):
    p = make(tmp_path)
    assert h.validate_plan(p) == p
    assert p['worker_source'].endswith('225b9af660cb0210208e2c46fc4216d0f151275b')
    assert p['screen_dispatch_source'] == p['controller_source']
    assert p['screen_analysis_source'] == p['controller_source']
    assert p['schema'] == 'fm_transfer_B_endpoint_handoff_v3'
    assert p['SFT_updates'] == 100000 and p['RL_updates'] == 20000
    ranks = []
    for i in range(3):
        cmd = p['stages'][f'screen_{i}']
        ranks.extend(map(int, cmd[cmd.index('--ranks') + 1].split(',')))
        assert cmd[cmd.index('--shards') + 1] == '24'
        assert 'update100000' in cmd[cmd.index('--sft-checkpoint') + 1]
    assert ranks == list(range(24))
    assert '--resume' not in p['stages']['formal_RL']
    assert '--qualification-root' in p['stages']['formal_RL']
    assert not any('navtest' in str(c).lower() for c in p['stages'].values())


@pytest.mark.parametrize('field', ['RL_updates', 'SFT_updates', 'worker_source', 'training_seed',
                                 'screen_dispatch_source', 'screen_analysis_source'])
def test_plan_tampering_cannot_change_science(tmp_path, field):
    p = make(tmp_path); p[field] = 'changed'
    with pytest.raises(ValueError): h.validate_plan(p)


def test_endpoint_absent_performs_no_dispatch(tmp_path, monkeypatch):
    p = make(tmp_path)
    monkeypatch.setattr(h, 'launch', lambda *args: pytest.fail('Premature dispatch'))
    assert h.tick(p, tmp_path / 'control', 'local')['status'] == 'WAITING_FOR_COMPLETE_B100K_SFT'


def test_registered_endpoint_and_actual_terminal_progress_required(tmp_path, monkeypatch):
    p = make(tmp_path); endpoint = Path(p['endpoint']); endpoint.mkdir(parents=True)
    (endpoint / 'COMPLETE').write_text('{}')
    parent = {'config': {'variant': h.VARIANT, 'max_updates': 100000, 'seed': 42}}
    monkeypatch.setattr(h, 'completed_sft', lambda path: (parent, {}))
    atomic_json(endpoint.parent.parent / 'progress.json', dict(status='RUNNING', updates=100000))
    assert not h.endpoint_ready(p)
    atomic_json(endpoint.parent.parent / 'progress.json', dict(status='COMPLETE', updates=100000))
    assert h.endpoint_ready(p)
    parent['config']['max_updates'] = 25000
    with pytest.raises(ValueError): h.endpoint_ready(p)


def states(p):
    return {name:'NOT_LAUNCHED' for name in p['stages']}


def test_stage_dependencies_never_substitute_interim_model_or_skip_gate(tmp_path):
    s = states(make(tmp_path))
    assert h.next_actions(s, False, False) == ['sft_summary']
    s['sft_summary'] = 'RUNNING'
    assert h.next_actions(s, True, True) == []
    s['sft_summary'] = 'COMPLETE'
    assert h.next_actions(s, False, False) == ['screen_0']
    s['screen_0'] = 'RUNNING'
    assert h.next_actions(s, False, False) == []  # Do not race the screen dispatch flock.
    s['screen_0'] = 'COMPLETE'
    assert h.next_actions(s, False, False) == ['screen_1']
    for i in range(3): s[f'screen_{i}'] = 'COMPLETE'
    assert h.next_actions(s, False, True) == []  # Dispatch completed != score workers completed.
    assert h.next_actions(s, True, False) == ['hard_manifest']
    s['hard_manifest'] = 'COMPLETE'
    assert h.next_actions(s, True, False) == ['screen_summary']
    s['screen_summary'] = 'COMPLETE'
    assert h.next_actions(s, True, False) == []  # Separate full own-SFT DEV baseline is required.
    assert h.next_actions(s, True, True) == ['endpoint_gate']
    s['endpoint_gate'] = 'RUNNING'
    assert h.next_actions(s, True, True) == []
    s['endpoint_gate'] = 'COMPLETE'
    assert h.next_actions(s, True, True) == ['formal_RL']
    s['formal_RL'] = 'COMPLETE'
    assert h.next_actions(s, True, True) == []


def test_no_implicit_stage_retry_or_stale_pid_inference(tmp_path):
    d = tmp_path / 'stages/endpoint_gate'; d.mkdir(parents=True)
    assert h.stage_state(tmp_path, 'sft_summary') == 'NOT_LAUNCHED'
    with pytest.raises(ValueError): h.stage_state(tmp_path, 'endpoint_gate')
    intent = dict(host='host-a', attempt='attempt', plan_sha256='plan')
    atomic_json(d / 'intent.json', intent)
    with pytest.raises(RuntimeError, match='Ambiguous'): h.stage_state(tmp_path, 'endpoint_gate')
    atomic_json(d / 'launch.json', dict(pid=1, start_ticks=2))
    assert h.stage_state(tmp_path, 'endpoint_gate', lambda *args: [True]) == 'RUNNING'
    with pytest.raises(RuntimeError, match='exited'):
        h.stage_state(tmp_path, 'endpoint_gate', lambda *args: [False])
    def unavailable(*args): raise TimeoutError('SSH observation unavailable')
    with pytest.raises(TimeoutError): h.stage_state(tmp_path, 'endpoint_gate', unavailable)
    atomic_json(d / 'status.json', dict(**intent, state='FAILED', exit_code=1))
    with pytest.raises(RuntimeError, match='failed'): h.stage_state(tmp_path, 'endpoint_gate')
    atomic_json(d / 'status.json', dict(**intent, state='COMPLETE', exit_code=0))
    assert h.stage_state(tmp_path, 'endpoint_gate') == 'COMPLETE'


def test_capacity_rejects_unknown_occupancy_and_changed_gpu_identity():
    a = dict(gpus=[dict(uuid=str(i), used=0, free=80000) for i in range(8)], owners=[])
    assert h.capacity(a, copy.deepcopy(a), 36000)
    b = copy.deepcopy(a); b['gpus'][0]['free'] = 35999
    assert not h.capacity(a, b, 36000)
    b = copy.deepcopy(a); b['gpus'][0]['used'] = 800
    assert not h.capacity(a, b, 36000)
    b['owners'] = [dict(uuid='0', visible=False)]
    assert not h.capacity(a, b, 36000)
    b['owners'][0]['visible'] = True
    assert h.capacity(a, b, 36000)  # Authorized sharing with a visible real owner is allowed.
    b['gpus'][0]['uuid'] = '7'
    with pytest.raises(ValueError): h.capacity(a, b, 36000)


def test_screen_dispatch_success_does_not_mask_dead_partial_workers(tmp_path):
    p = make(tmp_path); screen = Path(p['screen']); dest = screen / 'dispatches'; dest.mkdir(parents=True)
    for rank in range(24):
        atomic_json(dest / f'rank{rank}.json', dict(rank=rank, host='host-a', pid=rank+1,
            start_ticks=2, source=p['worker_source'], global_shards=24))
    occupied, done = h.screen_state(p, lambda host, rows, module: [True] * len(rows))
    assert occupied == {'host-a'} and not done
    with pytest.raises(RuntimeError, match='failed'):
        h.screen_state(p, lambda host, rows, module: [False] * len(rows))
    for rank in range(24): atomic_json(screen / f'rank{rank}.json', dict(status='ASSIGNED_COMPLETE'))
    assert h.screen_state(p, lambda host, rows, module: [False] * len(rows)) == (set(), True)
    atomic_json(dest / 'duplicate.json', json.loads((dest / 'rank0.json').read_text()))
    with pytest.raises(ValueError, match='Repeated'): h.screen_state(p)


def test_baseline_must_be_full_own_endpoint_and_both_grids(tmp_path):
    p = make(tmp_path)
    assert not h.baseline_ready(p)
    (tmp_path / 'analysis').mkdir()
    for cohort in ('DEV', 'TRAIN_DIAGNOSTIC'):
        name = h.PREFIX + '_' + cohort
        (tmp_path / 'evaluations' / name).mkdir(parents=True)
        meta = dict(qualification_only=False, new_SFT=dict(updates=100000), stage='SFT',
                    seed=2101, modes=['E_NATIVE','E_SHIFT5'])
        atomic_json(tmp_path / 'evaluations' / name / 'manifest.json', meta)
        atomic_json(tmp_path / 'analysis' / (name + '.json'), dict(status='COMPLETE_POPULATION', identity=meta))
    assert h.baseline_ready(p)
    name = h.PREFIX + '_DEV'; meta['new_SFT']['updates'] = 25000
    atomic_json(tmp_path / 'evaluations' / name / 'manifest.json', meta)
    atomic_json(tmp_path / 'analysis' / (name + '.json'), dict(status='COMPLETE_POPULATION', identity=meta))
    with pytest.raises(ValueError, match='interim'): h.baseline_ready(p)


def test_stage_executor_rejects_unfinished_sft_without_launching_child(tmp_path, monkeypatch):
    p = make(tmp_path); control = tmp_path / 'control'; directory = control / 'stages/sft_summary'
    directory.mkdir(parents=True)
    atomic_json(control / 'plan.json', p)
    atomic_json(directory / 'intent.json', dict(host='local', attempt='attempt',
        plan_sha256=p['sha256'], command=p['stages']['sft_summary']))
    monkeypatch.setattr(h, 'require_dispatch', lambda: 'local')
    monkeypatch.setattr(h.subprocess, 'Popen', lambda *a, **k: pytest.fail('Premature child'))
    with pytest.raises(ValueError, match='unfinished'): h.stage_worker(control, 'sft_summary')
    status = h.read(directory / 'status.json')
    assert status['state'] == 'FAILED' and 'child' not in status
    with pytest.raises(ValueError, match='already been attempted'): h.stage_worker(control, 'sft_summary')


@pytest.mark.parametrize('exit_code', [0, 7])
def test_stage_executor_publishes_physical_attempt_and_never_retries(tmp_path, monkeypatch, exit_code):
    import os
    p = make(tmp_path); control = tmp_path / 'control'; directory = control / 'stages/sft_summary'
    directory.mkdir(parents=True); worker = tmp_path / 'worker'; worker.mkdir()
    # Mock the external entry point only; execute the real ownership/publication code.
    p['worker_source'] = str(worker)
    atomic_json(worker / 'SOURCE_COMPLETE.json', dict(commit='worker', git_archive_extraction_complete=True))
    atomic_json(control / 'plan.json', p)
    atomic_json(directory / 'intent.json', dict(host='local', attempt='attempt',
        plan_sha256=p['sha256'], command=p['stages']['sft_summary']))
    monkeypatch.setattr(h, 'validate_plan', lambda value: value)
    monkeypatch.setattr(h, 'require_dispatch', lambda: 'local')
    monkeypatch.setattr(h, 'endpoint_ready', lambda value: True)
    calls = []
    class Child:
        pid = os.getpid()
        def wait(self): return exit_code
    def spawn(command, **kwargs):
        calls.append(command)
        assert command == p['stages']['sft_summary']
        assert kwargs['cwd'] == worker and kwargs['env']['PYTHONPATH'] == str(worker)
        return Child()
    monkeypatch.setattr(h.subprocess, 'Popen', spawn)
    h.stage_worker(control, 'sft_summary')
    status = h.read(directory / 'status.json')
    assert status['exit_code'] == exit_code
    assert status['state'] == ('COMPLETE' if exit_code == 0 else 'FAILED')
    assert status['end_unix'] >= status['start_unix'] and status['child']['pid'] == os.getpid()
    with pytest.raises(ValueError, match='already been attempted'): h.stage_worker(control, 'sft_summary')
    assert len(calls) == 1


@pytest.mark.parametrize('name', ['sft_summary', 'screen_0', 'screen_1', 'screen_2',
                                'hard_manifest', 'screen_summary', 'endpoint_gate', 'formal_RL'])
def test_stage_uses_compatible_dispatcher_but_keeps_numerical_worker_source(tmp_path, monkeypatch, name):
    import os
    p = make(tmp_path)
    worker = tmp_path / 'worker'
    dispatcher = Path(p['screen_dispatch_source'])
    for source in (worker, dispatcher):
        source.mkdir()
        atomic_json(source / 'SOURCE_COMPLETE.json',
                    dict(commit=source.name, git_archive_extraction_complete=True))
    p['worker_source'] = str(worker)
    control = tmp_path / 'control'; directory = control / 'stages' / name
    directory.mkdir(parents=True)
    atomic_json(control / 'plan.json', p)
    atomic_json(directory / 'intent.json', dict(host='local', attempt='attempt',
        plan_sha256=p['sha256'], command=p['stages'][name]))
    monkeypatch.setattr(h, 'validate_plan', lambda value: value)
    monkeypatch.setattr(h, 'require_dispatch', lambda: 'local')
    monkeypatch.setattr(h, 'endpoint_ready', lambda value: True)
    expected = dispatcher if name in (*h.SCREEN_DISPATCH_STAGES, 'screen_summary') else worker
    class Child:
        pid = os.getpid()
        def wait(self): return 0
    def spawn(command, **kwargs):
        assert kwargs['cwd'] == expected
        assert kwargs['env']['PYTHONPATH'] == str(expected)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        if name in h.SCREEN_DISPATCH_STAGES:
            assert '--ranks' in command
            assert command[command.index('--source') + 1] == str(h.SOURCE)
        return Child()
    monkeypatch.setattr(h.subprocess, 'Popen', spawn)
    h.stage_worker(control, name)
    status = h.read(directory / 'status.json')
    assert status['state'] == 'COMPLETE'
    assert status['execution_source'] == str(expected)
    assert status['numerical_worker_source'] == str(worker)


def entrypoint_fixture(tmp_path):
    p = make(tmp_path)
    p['worker_source'] = str(tmp_path / 'worker')
    for name, command in p['stages'].items():
        source = h.stage_source(p, name)
        source.mkdir(exist_ok=True)
        atomic_json(source / 'SOURCE_COMPLETE.json',
                    dict(commit=source.name, git_archive_extraction_complete=True))
        entry = source.joinpath(*command[2].split('.')).with_suffix('.py')
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text('# CLI fixture; no numerical execution\n')
    return p


def test_preflight_checks_all_stages_in_their_actual_sources(tmp_path):
    p = entrypoint_fixture(tmp_path)
    h.validate_entrypoints(p)
    entry = Path(p['screen_analysis_source']) / 'scripts/analysis/fm_transfer/analyze_screen.py'
    entry.unlink()
    with pytest.raises(ValueError, match='Missing endpoint entry point: screen_summary'):
        h.validate_entrypoints(p)


def test_original_worker_snapshot_cannot_silently_supply_later_analyzer(tmp_path):
    p = entrypoint_fixture(tmp_path)
    # Reproduce the old effective routing: the numerical snapshot has no analyzer.
    p['screen_analysis_source'] = p['worker_source']
    with pytest.raises(ValueError, match='Missing endpoint entry point: screen_summary'):
        h.validate_entrypoints(p)


def test_source_resolution_never_falls_back_for_an_unknown_stage(tmp_path):
    with pytest.raises(ValueError, match='Unknown registered'):
        h.stage_source(make(tmp_path), 'unregistered_extra_training')
