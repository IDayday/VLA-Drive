"""Independent small-metadata/timing accounting tests; no model or GPU calls."""
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

OWNER = Path('/mnt/project/VLA-Drive-st-temporal-t0-timing240-20261018/reports/st_world_policy/20261018/temporal_T0_timing240_preparation_v1')
RUNTIME = Path('/mnt/project/VLA-Drive-st-temporal-profile-runtime-v1-20261018')


def read_pin(pin):
    path = Path(pin['path'])
    assert path.stat().st_size < 128 << 10
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin['sha256']
    return json.loads(raw)


def test_actual_parent_and_child_only_allowed_recipe_changes():
    child = read_pin(dict(path=str(OWNER/'prepared_inputs/INPUTS.json'),
                         sha256='f644c9c46be5e5918db909446674f2e3b01d443314b45e53c8fced765718dc2a'))
    parent = read_pin(child['parent_capacity_inputs'])
    old = read_pin(parent['pins']['config'])
    new = read_pin(child['pins']['config'])
    expected = copy.deepcopy(old)
    expected['execution']['stop_after'] = 240
    expected['runtime'].update(output_root=child['output_root']+'/runs', profile_mode='timing', save_every=240)
    expected['data']['teacher_provider'] = new['data']['teacher_provider']
    assert expected == new
    assert new['optimizer'] == old['optimizer'] and new['space_time'] == old['space_time']
    assert new['execution']['max_updates'] == 20000 and new['optimizer']['warmup'] == 1000
    assert 'continuation' not in new and '--resume' not in child['native_argv']
    provider = read_pin(child['pins']['teacher_manifest'])
    assert provider['temporal_profile_admission']['consumption'] == child['consumption']
    consumption = child['consumption']
    assert consumption['initial_completed_updates'] == consumption['initial_epoch'] == consumption['initial_offset'] == 0
    assert consumption['cumulative_optimizer_updates'] == 240
    assert consumption['gradient_accumulation_steps'] == 8
    assert consumption['cumulative_scene_exposures'] == 240*32 == sum(consumption['per_rank_scene_exposures'])
    for name in ('base_binding', 'data_manifest', 'target_contract'):
        assert child['pins'][name] == parent['pins'][name]


def profile_functions():
    source = (RUNTIME/'tools/st_world_policy/profile_pipeline.py').read_text()
    wanted = {'TimingPlan', 'summarize_ranks'}
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    assert len(nodes) == 2
    namespace = dict(dataclass=dataclass, np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'original_native_timing_functions', 'exec'), namespace)
    return namespace


def records():
    result = []
    for rank in range(4):
        updates = []
        for step in range(1, 241):
            start = (step-1)*1_000_000_000+rank*1_000_000
            checkpoint = .5 if step == 240 else 0.
            duration = 1+rank*.001+checkpoint
            updates.append(dict(step=step, start_ns=start, end_ns=start+round(duration*1e9),
                seconds=duration, local_scenes=8, checkpoint_seconds=checkpoint,
                cpu_start=dict(self_seconds=step-1, waited_children_seconds=0),
                cpu_end=dict(self_seconds=step, waited_children_seconds=0),
                peak_allocated_bytes=100+rank, peak_reserved_bytes=200+rank))
        result.append(dict(rank=rank, world_size=4, identity={'global_batch':32}, hostname='fixture',
                           gpu_uuids=['fixture0','fixture1','fixture2','fixture3'],
                           eligible_for_throughput=True, mode='timing', updates=updates))
    return result


def test_native_two_windows_include_slowest_rank_and_final_checkpoint():
    ns = profile_functions()
    plan = ns['TimingPlan']()
    assert plan.updates == 240 and plan.steady_steps(0) == list(range(21,121))
    assert plan.steady_steps(1) == list(range(141,241))
    result = ns['summarize_ranks'](records())
    first, second = result['windows']
    assert first['scenes'] == second['scenes'] == 3200
    assert first['normal_updates'] == 100 and second['normal_updates'] == 99
    assert first['normal_update_mean'] == pytest.approx(1.003)
    assert second['checkpoint_events'] == 1 and second['checkpoint_max_rank_seconds'] == .5
    assert first['complete_wall_seconds'] == pytest.approx(100.006)
    assert second['complete_wall_seconds'] == pytest.approx(100.506)
    assert result['status'] == 'MEASURED_NOT_AN_APPROVAL'


@pytest.mark.parametrize('kind', ['missing_rank', 'missing_step', 'wrong_batch', 'trace'])
def test_native_summary_refuses_incomplete_or_incomparable_runs(kind):
    ns = profile_functions()
    rows = records()
    if kind == 'missing_rank': rows.pop()
    if kind == 'missing_step': rows[2]['updates'].pop()
    if kind == 'wrong_batch': rows[2]['updates'][20]['local_scenes'] = 7
    if kind == 'trace': rows[1]['mode'] = 'trace'
    with pytest.raises(ValueError):
        ns['summarize_ranks'](rows)
