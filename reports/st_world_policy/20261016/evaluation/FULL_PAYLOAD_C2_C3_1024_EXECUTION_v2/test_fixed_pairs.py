"""Small CPU fixtures only: no production CLI, payload, model or scoring call."""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('fixed_pairs_adapter', Path(__file__).with_name('run_fixed_pairs.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_exact_existing_commands_and_monitor_sources():
    core, _, _ = m.dependencies()
    commands, budget = m.fixed_commands(core)
    assert [c['comparison'] for c in commands] == ['C2_minus_C0', 'C3_minus_C0']
    assert budget['statistics_unchanged']['bootstrap_seed'] == 20261017
    assert budget['statistics_unchanged']['bootstrap_replicates'] == 2000
    assert callable(m.rss_monitor(core).process_tree_rss)


@pytest.mark.parametrize('change', ['source_sha', 'sequential_only', 'commands'])
def test_command_envelope_rejects_changes(change):
    core, _, _ = m.dependencies()
    original = core.pinned
    document = copy.deepcopy(original(m.COMMANDS))
    document[change] = {'source_sha': 'wrong', 'sequential_only': False, 'commands': []}[change]
    stub = SimpleNamespace(pinned=lambda pin, **kw: document if pin == m.COMMANDS else original(pin, **kw))
    with pytest.raises(ValueError): m.fixed_commands(stub)


def test_two_waits_charge_absorbed_children_once():
    core, _, _ = m.dependencies()
    meter = m.ReadMeter(2000)
    assert meter.parent({'rchar': 150}) == 1850
    first = core.terminal_read_accounting({'rchar': 150}, {'rchar': 170}, {'rchar': 470}, 300, 1850, final_child_available=True)
    assert meter.reaped(core, first, {'rchar': 480}) == 330
    assert meter.total == 480  # not 780: child already absorbed by parent
    assert meter.parent({'rchar': 500}) == 1500
    second = core.terminal_read_accounting({'rchar': 500}, {'rchar': 540}, {'rchar': 940}, 400, 1500, final_child_available=True)
    assert meter.reaped(core, second, {'rchar': 950}) == 450
    assert meter.total == 950
    with pytest.raises(ValueError): meter.parent({'rchar': 949})
    with pytest.raises(ValueError): meter.parent({'rchar': 2001})


def test_incomplete_wait_accounting_consumes_remaining_allowance():
    core, _, _ = m.dependencies()
    meter = m.ReadMeter(1000); meter.parent({'rchar': 100})
    row = core.terminal_read_accounting({'rchar': 100}, {'rchar': 110}, {'rchar': 120}, None, 900, final_child_available=False)
    assert row['status'] == 'INCOMPLETE_CONSERVATIVE'
    meter.reaped(core, row, {'rchar': 120})
    assert meter.total == 1000


def test_completed_prefix_cannot_obtain_new_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'CONTROL', tmp_path)
    commands = [{'comparison': k} for k in ('C2_minus_C0', 'C3_minus_C0')]
    for command, spent in zip(commands, (400, 900)):
        attempt = tmp_path / command['comparison']; attempt.mkdir()
        (attempt / 'RESULT.json').write_text(json.dumps({'status': 'COMPLETE', 'cumulative_whole_process_rchar_bytes': spent}))
    with pytest.raises(ValueError, match='Consumed session'): m.session_state(commands)


def test_output_budget_includes_both_control_receipts():
    assert m.ANALYSIS_LIMIT + m.CONTROL_RESERVE + m.ADMISSION_RESERVE == 32 * m.MiB


def test_interrupted_session_cannot_spend_fresh_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'CONTROL', tmp_path)
    commands = [{'comparison': k} for k in ('C2_minus_C0', 'C3_minus_C0')]
    assert m.session_state(commands) is False
    (tmp_path / 'SESSION_STARTED.json').write_text('{}')
    with pytest.raises(ValueError, match='Consumed session'): m.session_state(commands)
    for c in commands:
        (tmp_path / c['comparison']).mkdir()
        (tmp_path / c['comparison'] / 'RESULT.json').write_text('{}')
    (tmp_path / 'SESSION_TERMINAL.json').write_text(json.dumps({'status': 'COMPLETE', 'pairs': [{}, {}]}))
    with pytest.raises(ValueError, match='Consumed session'): m.session_state(commands)


def test_serialized_bytes_checked_before_write_and_partial_retained(tmp_path):
    root = tmp_path / 'out'; root.mkdir()
    guard = m.OutputGuard(root, [], cap=10)
    with guard.writers():
        with (root / 'PAIRED_SCENES.csv').open('x', encoding='utf8') as stream:
            stream.write('中')
            with pytest.raises(ValueError): stream.write('中中中')
    assert guard.written == 3
    assert (root / 'PAIRED_SCENES.csv').read_bytes() == '中'.encode()
    with guard.writers(), pytest.raises(ValueError): (root / 'ANALYSIS.json').open('w')
    with guard.writers(), pytest.raises(ValueError): (tmp_path / 'foreign.json').open('x')


def test_json_temp_hardlink_not_double_charged(tmp_path):
    root = tmp_path / 'out'; root.mkdir()
    guard = m.OutputGuard(root, [], cap=64)
    temporary = root / ('ANALYSIS.json.tmp.' + 'a' * 32)
    with guard.writers(), temporary.open('x') as stream:
        json.dump({'x': '\n"'}, stream, ensure_ascii=True)
    os.link(temporary, root / 'ANALYSIS.json')
    assert m.output_size(root) == guard.written == temporary.stat().st_size
    temporary.unlink()


@pytest.mark.parametrize('path', ['/fake/model.pt', '/fake/image.jpg', '/fake/teacher.npy', '/dev/nvidia0'])
def test_unregistered_payload_denied_without_opening(path, tmp_path):
    guard = m.OutputGuard(tmp_path, [])
    with pytest.raises(ValueError): guard.audit('open', (path, 'r', os.O_RDONLY))


def test_exact_payload_allowlist_and_stdout_cap(tmp_path):
    path = tmp_path / 'RESULTS.pt'
    m.OutputGuard(tmp_path, [path]).audit('open', (str(path), 'r', os.O_RDONLY))
    stream = m.SmallText(); stream.write('x' * (64 * 1024))
    with pytest.raises(ValueError): stream.write('x')
    assert len(stream.getvalue()) == 64 * 1024


def result_fixture(tmp_path):
    root = tmp_path / 'out'; root.mkdir()
    command = {'comparison': 'C2_minus_C0', 'analysis_plan': {'sha256': 'plan'}, 'argv': []}
    for key in ('output', 'control-score', 'control-export', 'treatment-score', 'treatment-export'):
        command['argv'] += ['--' + key, str(root if key == 'output' else tmp_path / key)]
    fields = ['scene_id', 'source_log', 'control_raw_progress_m', 'treatment_mean_executed_speed_mps', 'new_stop_proxy']
    scenes = [{'token': str(i), 'log': 'log' + str(i % 16)} for i in range(1696)]
    with (root / 'PAIRED_SCENES.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows(dict(zip(fields, (r['token'], r['log'], 0, 0, 0))) for r in scenes)
    pins = {'population': {'scenes': scenes}}
    budget = {'statistics_unchanged': {'bootstrap_seed': 20261017}, 'analysis_source': {'sha256': 'source'},
              'population': {'key': 'population', 'sha256': 'population'}, 'runs': {}}
    result = {'schema': 'st_world_paired_analysis_v1', 'status': 'COMPLETE', 'scope': 'FORMAL',
              'comparison': 'C2_minus_C0', 'stage': 'DIAGNOSTIC_1024',
              'validity': {'requested': 1696, 'paired_valid': 1696, 'invalid': 0, 'unresolved': 0},
              'plan_sha256': 'plan', 'population_sha256': 'population',
              'statistics_definition': budget['statistics_unchanged'], 'analysis_source_sha256': 'source',
              'paired_scenes_csv_sha256': m.file_hash(root / 'PAIRED_SCENES.csv')}
    for side, arm in [('control', 'C0'), ('treatment', 'C2')]:
        budget['runs'][arm] = {k: {'key': arm + k} for k in ('score_identity', 'export_identity', 'score_complete')}
        pins[arm + 'score_identity'] = {'identity': arm + 'score'}
        pins[arm + 'export_identity'] = {'identity': arm + 'export', 'checkpoint': {'tag': arm + '1024'}}
        pins[arm + 'score_complete'] = {'scenes_csv_sha256': arm + 'csv'}
        result[side] = {'score_identity': arm + 'score', 'export_identity': arm + 'export',
                        'checkpoint': {'tag': arm + '1024'}, 'scenes_csv_sha256': arm + 'csv',
                        'score_root': m.option(command, side + '-score'), 'export_root': m.option(command, side + '-export')}
    (root / 'ANALYSIS.json').write_text(json.dumps(result))
    return command, SimpleNamespace(pinned=lambda pin: pins[pin['key']]), budget, root


def test_existing_output_validation_only_uses_small_fixtures(tmp_path):
    command, core, budget, root = result_fixture(tmp_path)
    result = m.complete_output(command, core, budget)
    assert result['output_bytes'] < 128 * 1024
    record = json.loads((root / 'ANALYSIS.json').read_text()); record['stage'] = 'DIAGNOSTIC_512'
    (root / 'ANALYSIS.json').write_text(json.dumps(record))
    with pytest.raises(ValueError): m.complete_output(command, core, budget)


def test_failed_attempt_retained_and_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'CONTROL', tmp_path)
    monkeypatch.setattr(m, 'complete_output', lambda *a: None)
    calls = []
    def monitor(*a, **kw):
        calls.append(kw)
        return {'returncode': 2, 'failure': None, 'read_accounting': {}, 'child_accounting': {'cli_result': {'exit_code': 2}}}
    core = SimpleNamespace(monitor=monitor, charge_after_reap=lambda *_: 100)
    observer = SimpleNamespace(proc_io=lambda: {'rchar': 100})
    command = {'comparison': 'C2_minus_C0', 'env': {'CUDA_VISIBLE_DEVICES': ''}}
    with (tmp_path / 'lock').open('x') as lock:
        with pytest.raises(ValueError): m.run_pair(command, {}, core, None, observer, None, lock, m.ReadMeter())
        result = tmp_path / 'C2_minus_C0/RESULT.json'; original = result.read_bytes()
        assert json.loads(original)['status'] == 'FAILED_NO_AUTO_RETRY'
        assert len(calls) == 1 and calls[0]['budget']['read_bytes'] == m.TOTAL_READ - 100
        with pytest.raises(ValueError): m.run_pair(command, {}, core, None, observer, None, lock, m.ReadMeter())
        assert len(calls) == 1 and result.read_bytes() == original


def test_missing_previously_complete_output_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'CONTROL', tmp_path)
    monkeypatch.setattr(m, 'complete_output', lambda *a: None)
    attempt = tmp_path / 'C2_minus_C0'; attempt.mkdir()
    (attempt / 'RESULT.json').write_text(json.dumps({'status': 'COMPLETE', 'analysis': {}}))
    with pytest.raises(ValueError, match='missing or changed'):
        m.run_pair({'comparison': 'C2_minus_C0'}, {}, None, None, None, None, None, m.ReadMeter())


@pytest.mark.parametrize('exit_code', [0, 3])
def test_actual_reviewed_monitor_tiny_cpu_child_only(tmp_path, exit_code):
    core, supervisor, observer = m.dependencies()
    core.BOOT_IO = observer.proc_io(); core.START = time.monotonic()
    script = """import json, pathlib, sys
def sample(): return {k:int(v) for k,v in (x.split(':') for x in pathlib.Path('/proc/self/io').read_text().splitlines())}
before=sample(); after=sample()
print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':{'exit_code':int(sys.argv[1])},'counters':{'proc_self_io_baseline':before,'proc_self_io_final':after,'proc_self_io_delta':{k:after[k]-before[k] for k in after}}}))
raise SystemExit(int(sys.argv[1]))
"""
    with (tmp_path / 'fixture.lock').open('x') as lock:
        result = core.monitor([sys.executable, '-c', script, str(exit_code)], runtime=str(tmp_path),
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, fd=lock.fileno(), root=tmp_path,
            s=supervisor, m=SimpleNamespace(process_tree_rss=lambda _: 0), obs=observer,
            budget={**m.PAIR_LIMITS, 'wall_seconds': 10, 'read_bytes': 64 * m.MiB})
    assert result['returncode'] == exit_code
    assert result['failure'] is None
    assert result['read_accounting']['status'] == 'COMPLETE'
    assert result['cleanup']['live_owned_remaining'] == 0
    assert result['charged_owned_child_rchar_bytes'] > 0
