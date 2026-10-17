"""CPU-only review fixtures. Never opens campaign raw data, ledger or GPUs."""
import ast
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
PREP = HERE.parent / 'full_causal_history_preparation_v1'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

p = load('independent_history_pilot', PREP / 'run_pilot_once.py')
core_path = HERE.parent / 'full_future_metadata_preparation_v1/run_build_once.py'
core = load('independent_history_core', core_path)
helper = HERE.parent / 'spatial_probe_actual_preparation_v1'
sys.path.insert(0, str(helper))
import supervise_once as supervisor
import observed_cli as observed
from tools.st_world_policy import metadata_storage, cache_storage_budget as budget


def test_production_cpu_route_is_explicit_6_7():
    # The actual launch adapter calls this parameter; no hard-coded 0/1 route.
    tree = ast.parse((PREP / 'run_pilot_once.py').read_text())
    monitor = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == 'monitor']
    assert len(monitor) == 1
    assert ast.literal_eval(next(k.value for k in monitor[0].keywords
                                 if k.arg == 'cpu_affinity')) == (6, 7)


def test_owned_real_cpu8_9_child_accounting_and_exit(tmp_path):
    # Review authorization is CPU8/9. Production parameterization is checked above.
    code = """import os,json
before={k:int(v) for k,v in (s.split(':') for s in open('/proc/self/io'))}
after={k:int(v) for k,v in (s.split(':') for s in open('/proc/self/io'))}
print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':{'affinity':sorted(os.sched_getaffinity(0))},'counters':{'proc_self_io_baseline':before,'proc_self_io_final':after,'proc_self_io_delta':{k:after[k]-before[k] for k in after}}}))
"""
    checks = []
    with (tmp_path / 'owned').open('w') as file:
        result = core.monitor([sys.executable, '-c', code], runtime=str(tmp_path),
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), fd=file.fileno(),
            root=tmp_path, s=supervisor, m=metadata_storage, obs=observed,
            budget={'wall_seconds': 10, 'read_bytes': 64 << 20, 'rss_bytes': 4 << 30},
            cpu_affinity=(8, 9), storage_guard=lambda: checks.append(True),
            lease_valid_until=time.time() + 60)
    assert result['returncode'] == 0 and result['failure'] is None
    assert result['child_accounting']['cli_result']['affinity'] == [8, 9]
    assert result['read_accounting']['status'] == 'COMPLETE'
    assert result['cleanup']['live_owned_remaining'] == 0 and checks


def test_wait_boundary_charge_does_not_double_count_child():
    accounting = core.terminal_read_accounting({'rchar': 10}, {'rchar': 30},
        {'rchar': 150}, 100, 1000, final_child_available=True)
    assert accounting['status'] == 'COMPLETE'
    assert accounting['charged_parent_child_rchar_bytes'] == 140
    assert core.charge_after_reap(accounting, {'rchar': 160}) == 150


def test_expired_actual_start_rejects_before_spawn(tmp_path):
    with patch.object(core.subprocess, 'Popen', side_effect=AssertionError('spawned')):
        with pytest.raises(ValueError, match='actual-start lease'):
            core.monitor([], runtime=str(tmp_path), env={}, fd=0, root=tmp_path,
                s=supervisor, m=metadata_storage, obs=observed,
                budget={'wall_seconds': 900}, cpu_affinity=(8, 9),
                lease_valid_until=time.time() + 1)


def test_reproduce_start_marker_wait_is_outside_monitor(tmp_path):
    """Characterize the defect, not a passing finite-launch qualification.

    The exact production StorageBudget._locked and publish_guarded are used
    against only a temporary synthetic ledger. No actual ledger is touched.
    """
    lock_path = tmp_path / 'CLAIMS.lock'
    policy_path = tmp_path / 'policy.json'
    policy_path.write_text('{}')
    fixture_budget = SimpleNamespace(policy_path=policy_path, state_dir=tmp_path,
        state_path=tmp_path / 'state.json', lock_path=lock_path, core={},
        _read_policy=lambda: ({}, {}, {}), _check_cap=lambda state: None)
    ctx = multiprocessing.get_context('fork')
    started, finished = ctx.Event(), ctx.Event()

    @contextmanager
    def fixture_writing(**_):
        started.set()
        with budget.StorageBudget._locked(fixture_budget):
            yield SimpleNamespace(limits={'max_disk_bytes': 1 << 20},
                check=lambda **kwargs: {'charged_disk_bytes': 0})

    def child():
        core.publish_guarded(SimpleNamespace(metadata_writing=fixture_writing),
            budget, {}, tmp_path / 'START.json', {'fixture': True})
        finished.set()

    with lock_path.open('a+') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        process = ctx.Process(target=child)
        process.start()
        try:
            assert started.wait(2)
            time.sleep(.3)
            assert process.is_alive() and not finished.is_set()
            assert not (tmp_path / 'START.json').exists()
            # No timeout or monitor exists at this blocking flock boundary.
            fcntl.flock(held, fcntl.LOCK_UN)
            process.join(2)
            assert process.exitcode == 0 and finished.is_set()
        finally:
            if process.is_alive():
                process.terminate(); process.join(2)
    main = next(n for n in ast.parse((PREP / 'run_pilot_once.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = sorted([n for n in ast.walk(main) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr in ('publish_guarded', 'monitor')],
        key=lambda n: n.lineno)
    assert [c.func.attr for c in calls] == ['publish_guarded', 'monitor', 'publish_guarded']


def test_reproduce_pre_core_reads_are_excluded(tmp_path):
    # core.BOOT_IO starts when controls imports core, after parent setup reads.
    source = tmp_path / 'control.bin'; source.write_bytes(b'x' * (1 << 20))
    before = observed.proc_io()['rchar']
    source.read_bytes()
    later = load('late_loaded_counter_example', core_path)
    assert later.BOOT_IO['rchar'] - before >= 1 << 20
    value = later.terminal_read_accounting(later.BOOT_IO,
        {'rchar': later.BOOT_IO['rchar'] + 10},
        {'rchar': later.BOOT_IO['rchar'] + 110}, 100, 1 << 30,
        final_child_available=True)
    assert value['charged_parent_child_rchar_bytes'] == 110
    assert value['charged_parent_child_rchar_bytes'] < 1 << 20


def test_reproduce_monitor_exception_leaves_no_failure_receipt(tmp_path):
    markers = []
    fake_core = SimpleNamespace(
        publish_guarded=lambda m,b,g,path,value: markers.append((Path(path).name, value)),
        monitor=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('fixture monitor error')))
    start = tmp_path / 'CONTROL_HISTORY_STARTED.json'; start.write_text('{}')
    admission = {'valid_until_unix': time.time() + 2000}
    with patch.object(p, 'ROOT', tmp_path), patch.object(p.os, 'sched_setaffinity'), \
         patch.object(p, 'controls', return_value=(admission, {}, {}, None, None,
                      None, None, fake_core, lambda: None)), \
         patch.object(sys, 'argv', ['review', '--admission', '/fixture',
                                   '--admission-sha256', '0'*64]):
        with pytest.raises(RuntimeError, match='fixture monitor error'):
            p.main()
    assert [name for name, _ in markers] == ['CONTROL_HISTORY_STARTED.json']

