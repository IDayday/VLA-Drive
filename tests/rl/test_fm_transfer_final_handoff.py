import copy
import json
from pathlib import Path
import subprocess

import pytest

from scripts.analysis.fm_transfer import final_handoff as handoff
from starVLA.rl.flow_grpo.transactions import atomic_json


def snapshot(count=8, prefix='p', *, free=70000, cpu=128, load=0):
    return dict(gpus={i: dict(uuid=f'{prefix}{i}', used_MiB=0, free_MiB=free) for i in range(count)},
                owners=[], CPU_affinity=cpu, CPU_capacity=cpu, load=[load, load, load])


def test_no_endpoint_gate_means_no_inventory_lock_or_launch(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('No final activity before all registered endpoint evidence exists')
    monkeypatch.setattr(handoff, 'stage', forbidden)
    monkeypatch.setattr(handoff, 'probe', forbidden)
    result = handoff.tick(dict(root=str(tmp_path)), tmp_path/'control')
    assert result['status'] == 'WAITING_FOR_BOTH_ENDPOINTS_AND_DEV'
    assert not result['Navtest_opened']
    assert any('update100000' in p for p in result['missing'])
    assert not (tmp_path/'final_navtest/LOCK.json').exists()


@pytest.mark.parametrize('n', [0, 1, 3, 8, 17, 32, 64])
def test_all_available_unique_gpus_map_to_fixed_models_and_exact_scene_partitions(n):
    devices = [dict(host=f'h{i//8}', gpu=i % 8, GPU_uuid=f'u{i}') for i in range(n)]
    plans = handoff.allocate(list(handoff.dispatch.MODELS), devices)
    assert len(plans) == min(4, n)
    allocated = [d['GPU_uuid'] for p in plans.values() for d in p['assignments']]
    assert allocated == [d['GPU_uuid'] for d in devices]
    for model, p in plans.items():
        assert model in handoff.dispatch.MODELS
        assert [d['rank'] for d in p['assignments']] == list(range(p['shards']))
        population = [x for rank in range(p['shards']) for x in range(rank, 12146, p['shards'])]
        assert sorted(population) == list(range(12146))
    with pytest.raises(ValueError, match='twice'):
        handoff.allocate(list(handoff.dispatch.MODELS), [dict(GPU_uuid='same')] * 2)


def test_busy_first_host_partial_capacity_and_aliases_do_not_hide_later_devices():
    busy = snapshot(free=1000)
    a = snapshot(3, 'a'); b = copy.deepcopy(a)
    b['gpus'][1]['used_MiB'] = 20000  # No visible owner: exclude.
    b['gpus'][2]['free_MiB'] = 1000
    other = snapshot(2, 'b')
    alias = snapshot(1, 'a')
    devices, reasons = handoff.available_devices({'busy': (busy, busy), 'a': (a, b),
                                                  'alias': (alias, alias), 'other': (other, other)})
    assert [d['GPU_uuid'] for d in devices] == ['a0', 'b0', 'b1']
    assert reasons['alias:0'] == 'aliased_or_reserved_UUID'
    assert 'a:1' in reasons and 'a:2' in reasons


def test_visible_shared_owner_cpu_capacity_reservations_and_inventory_changes():
    first = snapshot(3, cpu=2, load=1)
    first['gpus'][0]['used_MiB'] = 16000
    first['owners'] = [dict(GPU_uuid='p0', visible=True)]
    devices, reasons = handoff.available_devices({'h': (first, copy.deepcopy(first))})
    assert [d['gpu'] for d in devices] == [0]
    assert reasons['h:1'] == 'CPU_headroom'
    devices, _ = handoff.available_devices({'h': (first, copy.deepcopy(first))}, reserved={'p0'})
    assert [d['gpu'] for d in devices] == [1]
    second = copy.deepcopy(first); second['gpus'][1]['uuid'] = 'changed'
    with pytest.raises(ValueError, match='identity changed'):
        handoff.available_devices({'h': (first, second)})


@pytest.mark.parametrize('status,code', [('RUNNING', None), ('FAILED', 1), ('COMPLETE', 1)])
def test_failed_or_ambiguous_command_cannot_be_automatically_replayed(tmp_path, monkeypatch, status, code):
    d = tmp_path/'stages/lock'; d.mkdir(parents=True)
    cmd = ['python', 'existing-entry']
    atomic_json(d/'intent.json', dict(command=cmd, host=None, source=str(handoff.SOURCE)))
    atomic_json(d/'result.json', dict(status=status, exit_code=code, command=cmd, host=None))
    monkeypatch.setattr(handoff.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No repeated command'))
    with pytest.raises(RuntimeError, match='Unresolved'):
        handoff.stage(tmp_path, 'lock', cmd)


def test_completed_stage_reuse_requires_unchanged_command(tmp_path, monkeypatch):
    d = tmp_path/'stages/lock'; d.mkdir(parents=True)
    cmd = ['python', 'existing-entry']
    atomic_json(d/'intent.json', dict(command=cmd, host=None, source=str(handoff.SOURCE)))
    result = dict(status='COMPLETE', exit_code=0, command=cmd, host=None)
    atomic_json(d/'result.json', result)
    monkeypatch.setattr(handoff.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No repeated command'))
    assert handoff.stage(tmp_path, 'lock', cmd) == result
    with pytest.raises(RuntimeError):
        handoff.stage(tmp_path, 'lock', cmd + ['changed'])


def final_workers(tmp_path):
    model = 'A_DDP_TIME_RL'; output = tmp_path/'evaluations'/(model+'_final_NAVTEST')
    (output/'dispatches').mkdir(parents=True)
    manifest = handoff.dispatch.publish(output/'manifest.json', dict(schema='test'))
    plan = dict(shards=2, assignments=[])
    for rank in range(2):
        assignment = dict(host='host', rank=rank, gpu=rank, GPU_uuid=f'u{rank}')
        plan['assignments'].append(assignment)
        handoff.dispatch.publish(output/'dispatches'/f'rank{rank}.json', dict(**assignment,
            source=str(handoff.EVALUATION_SOURCE), output=str(output), model=model, shards=2, state='STARTED'))
    return model, output, plan, manifest


def test_worker_completion_requires_actual_exit_and_matched_identity(tmp_path):
    model, output, plan, manifest = final_workers(tmp_path)
    atomic_json(output/'rank0.json', dict(status='ASSIGNED_COMPLETE', identity_sha256=manifest['sha256']))
    assert handoff.model_state(tmp_path, model, plan, lambda r: True)['status'] == 'RUNNING'
    with pytest.raises(RuntimeError, match='failed'):
        handoff.model_state(tmp_path, model, plan, lambda r: False)
    atomic_json(output/'rank1.json', dict(status='ASSIGNED_COMPLETE', identity_sha256=manifest['sha256']))
    assert handoff.model_state(tmp_path, model, plan, lambda r: r['rank'] == 1)['reserved'] == ['u1']
    assert handoff.model_state(tmp_path, model, plan, lambda r: False)['status'] == 'WORKERS_EXITED_COMPLETE'
    atomic_json(output/'rank1.json', dict(status='ASSIGNED_COMPLETE', identity_sha256='wrong'))
    with pytest.raises(ValueError, match='Foreign'):
        handoff.model_state(tmp_path, model, plan, lambda r: False)


def test_partial_launch_no_receipt_and_observation_timeout_require_inspection(tmp_path):
    with pytest.raises(RuntimeError, match='no worker receipt'):
        handoff.model_state(tmp_path, 'A_DDP_TIME_RL', dict(shards=1), lambda r: False)
    model, output, plan, manifest = final_workers(tmp_path)
    def unknown(row):
        raise TimeoutError('SSH timeout is not a dead worker')
    with pytest.raises(TimeoutError):
        handoff.model_state(tmp_path, model, plan, unknown)
    (output/'dispatches/rank1.json').unlink()
    with pytest.raises(RuntimeError, match='Partial'):
        handoff.model_state(tmp_path, model, plan, lambda r: False)


def summary_fixture():
    model = 'A_DDP_TIME_RL'
    chosen = dict(weight_sha256='weight', stage='RL', new_SFT=dict(parent='own'), export_receipt_sha256='receipt')
    selection = dict(models={model: chosen}, population_identity='pop', sampler_sha256='sampler',
                     transfer_evaluation_sha256='transfer', sha256='lock')
    manifest = dict(cohort='NAVTEST', seed=2101, modes=['E_NATIVE', 'E_SHIFT5'], qualification_only=False,
        final_lock={k: v for k, v in selection.items() if k != 'sha256'},
        population_identity='pop', source_sha256='sampler', transfer_evaluation_sha256='transfer', **chosen)
    result = dict(status='COMPLETE_POPULATION', scenes=12146, logs=136, identity=manifest)
    return result, manifest, selection, model


def test_complete_summary_is_bound_to_fixed_population_and_own_endpoint():
    result, manifest, selection, model = summary_fixture()
    handoff.validate_summary(result, manifest, selection, model)
    result['scenes'] = 12145
    with pytest.raises(ValueError, match='locked population'):
        handoff.validate_summary(result, manifest, selection, model)


@pytest.mark.parametrize('key,value', [('cohort', 'DEV'), ('seed', 2102), ('modes', ['E_SHIFT5']),
    ('qualification_only', True), ('weight_sha256', 'wrong'), ('stage', 'SFT'),
    ('new_SFT', dict(parent='other')), ('source_sha256', 'new-code'), ('final_lock', {})])
def test_other_protocol_or_checkpoint_is_not_final_completion(key, value):
    result, manifest, selection, model = summary_fixture()
    manifest[key] = value
    with pytest.raises(ValueError, match='locked population'):
        handoff.validate_summary(result, manifest, selection, model)


def test_ready_tick_reuses_dispatcher_and_skips_unreachable_capacity(tmp_path, monkeypatch):
    control = tmp_path/'control'; control.mkdir()
    lock = tmp_path/'final_navtest/LOCK.json'; lock.parent.mkdir()
    selection = dict(status='LOCKED_BEFORE_FINAL_NAVTEST', models={m: {} for m in handoff.dispatch.MODELS})
    handoff.dispatch.publish(lock, selection)
    monkeypatch.setattr(handoff, 'readiness', lambda root: dict(missing=[]))
    monkeypatch.setattr(handoff, 'worker_allowed', lambda host: None)
    monkeypatch.setattr(handoff.time, 'sleep', lambda t: None)
    calls = []
    monkeypatch.setattr(handoff, 'stage', lambda control, name, command, **kw: calls.append((name, command, kw)))
    def probe(host):
        if host == 'unreachable':
            raise subprocess.TimeoutExpired('ssh', 10)
        return snapshot(8, prefix=host)
    monkeypatch.setattr(handoff, 'probe', probe)
    result = handoff.tick(dict(root=str(tmp_path), hosts=['unreachable', 'h1', 'h2'],
                              modes=['E_NATIVE', 'E_SHIFT5']), control)
    assert result['status'] == 'FINAL_PARTITIONS_STARTED'
    assert sum(p['shards'] for p in result['planned'].values()) == 16
    assert calls[0][0] == 'lock'
    for _, cmd, kw in calls[1:]:
        assert kw['host'] in ['h1', 'h2']
        assert cmd[2] == 'scripts.analysis.fm_transfer.dispatch_final_evaluation'
        assert cmd[cmd.index('--source')+1] == str(handoff.EVALUATION_SOURCE)
        assert '--retry' not in cmd and '--max-scenes' not in cmd
    inventory = json.loads(next(control.glob('inventory_*.json')).read_text())
    assert inventory['excluded_hosts']['unreachable']['capacity_not_inferred']


@pytest.mark.parametrize('fault', [None, 'partial_population', 'wrong_reference', 'paired_gain', 'safe_population'])
def test_final_completion_requires_four_populations_and_two_exact_parent_pairs(tmp_path, monkeypatch, fault):
    control = tmp_path/'control'; control.mkdir()
    analysis = tmp_path/'analysis'; analysis.mkdir()
    models = handoff.dispatch.MODELS
    selection = dict(status='LOCKED_BEFORE_FINAL_NAVTEST', population_identity='population',
                     sampler_sha256='sampler', transfer_evaluation_sha256='transfer', models={})
    for model in models:
        variant, stage_name = model.rsplit('_', 1)
        selection['models'][model] = dict(weight_sha256=model+'weight', stage=stage_name,
            new_SFT=dict(parent=variant), export_receipt_sha256=model+'export')
    lock = tmp_path/'final_navtest/LOCK.json'; lock.parent.mkdir()
    selected = handoff.dispatch.publish(lock, selection)
    raw = {}
    for model in models:
        output = tmp_path/'evaluations'/(model+'_final_NAVTEST'); output.mkdir(parents=True)
        manifest = handoff.dispatch.publish(output/'manifest.json', dict(cohort='NAVTEST', seed=2101,
            modes=['E_NATIVE', 'E_SHIFT5'], qualification_only=False, final_lock=selection,
            population_identity='population', source_sha256='sampler', transfer_evaluation_sha256='transfer',
            **selection['models'][model]))
        raw[model] = dict(status='COMPLETE_POPULATION', scenes=12146, logs=136, identity=manifest,
            modes={mode: dict(mean=dict(score=.91 if model.endswith('_RL') else .9),
                             fully_safe_scenes=12000) for mode in ['E_NATIVE', 'E_SHIFT5']})
        atomic_json(analysis/(output.name+'.json'), raw[model])
    for variant in ['A_DDP_TIME', 'B_SOURCE_ACTION_DIT']:
        paired = copy.deepcopy(raw[variant+'_RL'])
        paired['reference'] = str(tmp_path/'evaluations'/(variant+'_SFT_final_NAVTEST'))
        for mode in ['E_NATIVE', 'E_SHIFT5']:
            paired['modes'][mode]['paired_comparison'] = dict(
                PDMS_log_cluster_bootstrap=dict(mean=.01), reference_safe_population=12000)
        if variant == 'B_SOURCE_ACTION_DIT':
            if fault == 'partial_population': paired['scenes'] = 12145
            if fault == 'wrong_reference': paired['reference'] += '_other'
            if fault == 'paired_gain': paired['modes']['E_SHIFT5']['paired_comparison']['PDMS_log_cluster_bootstrap']['mean'] = .02
            if fault == 'safe_population': paired['modes']['E_NATIVE']['paired_comparison']['reference_safe_population'] = 11999
        atomic_json(analysis/(variant+'_RL_final_NAVTEST_paired_own_SFT.json'), paired)
    atomic_json(control/'routes.json', {model: {} for model in models})
    monkeypatch.setattr(handoff, 'readiness', lambda root: dict(missing=[]))
    monkeypatch.setattr(handoff, 'model_state', lambda *a: dict(status='WORKERS_EXITED_COMPLETE', reserved=[]))
    monkeypatch.setattr(handoff, 'probe', lambda *a: pytest.fail('Completed workers need no new resource dispatch'))
    calls = []
    monkeypatch.setattr(handoff, 'stage', lambda control, name, command, **kw: calls.append(name))
    value = dict(root=str(tmp_path), hosts=[], modes=['E_NATIVE', 'E_SHIFT5'])
    if fault:
        with pytest.raises(ValueError):
            handoff.tick(value, control)
    else:
        result = handoff.tick(value, control)
        assert result['status'] == 'ALL_FINAL_POPULATIONS_AND_PAIRS_COMPLETE'
        assert len(result['analyses']) == 4 and len(result['paired']) == 2
        assert result['final_lock_sha256'] == handoff.file_sha(lock)
        assert set(calls) == {'lock', *('analyze_'+m for m in models),
                              'paired_A_DDP_TIME', 'paired_B_SOURCE_ACTION_DIT'}
