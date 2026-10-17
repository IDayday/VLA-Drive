"""Lineage/budget/routing fixtures only; never dispatch the real analysis."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('c3_retry_delta', Path(__file__).with_name('run_c3_once.py'))
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def setup():
    module = r.original(); core, supervisor, observer = module.dependencies()
    return module, core, supervisor, observer


def test_actual_small_pins_and_original_second_command_only():
    module, core, _, _ = setup()
    selected, _ = r.prepare(module, core)
    original, _ = module.fixed_commands(core)
    assert selected == original[1]
    assert selected['comparison'] == 'C3_minus_C0'
    assert module.CORE['sha256'] == '8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644'
    assert r.PRIOR_CHARGE + r.NEW_ALLOWANCE == 2177518395


@pytest.mark.parametrize('case', ['old_cost', 'session_status', 'c3_status', 'cleanup', 'c2_complete', 'returncode'])
def test_wrong_lineage_rejected(case):
    module, core, _, _ = setup()
    rows = {k: core.pinned(pin) for k, pin in r.PRIOR.items()}
    if case == 'old_cost': rows['session']['conservative_spent_bytes'] -= 1
    if case == 'session_status': rows['session']['status'] = 'COMPLETE'
    if case == 'c3_status': rows['C3']['status'] = 'COMPLETE'
    if case == 'cleanup': rows['C3']['resources']['cleanup']['live_owned_remaining'] = 1
    if case == 'c2_complete': rows['C2']['status'] = 'FAILED_NO_AUTO_RETRY'
    if case == 'returncode': rows['C3']['resources']['returncode'] = 0
    by_path = {pin['path']: rows[k] for k,pin in r.PRIOR.items()}
    original_pinned = core.pinned
    core.pinned = lambda pin, **kw: copy.deepcopy(by_path[pin['path']]) if pin['path'] in by_path else original_pinned(pin, **kw)
    with pytest.raises(ValueError): r.prepare(module, core)


def test_changed_wrapper_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'changed.py'; path.write_text('# not reviewed\n')
    monkeypatch.setattr(r, 'WRAPPER', {'path': str(path), 'sha256': r.WRAPPER['sha256']})
    with pytest.raises(ValueError, match='changed'): r.original()


def test_existing_c3_output_rejected(tmp_path):
    module, core, _, _ = setup()
    original_option = module.option
    module.option = lambda c,k: str(tmp_path) if k == 'output' else original_option(c,k)
    with pytest.raises(ValueError, match='must be absent'): r.prepare(module, core)


def test_old_inherited_lock_blocks_new_work_without_signals(tmp_path, monkeypatch):
    module, _, _, _ = setup()
    monkeypatch.setattr(r, 'OLD', tmp_path)
    path = tmp_path / 'RUN.lock'; path.write_text('')
    with path.open('rb') as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            with r.old_session_inactive(module): pytest.fail('An old holder must block admission')
    with r.old_session_inactive(module): pass
    assert path.read_bytes() == b''


@pytest.mark.parametrize('fail', [False, True])
def test_one_call_routes_only_c3_and_preserves_old_cost(tmp_path, monkeypatch, fail):
    module, core, supervisor, _ = setup()
    old = tmp_path / 'old'; old.mkdir(); (old / 'RUN.lock').touch()
    new = tmp_path / 'new'
    monkeypatch.setattr(r, 'OLD', old); monkeypatch.setattr(r, 'NEW', new)
    monkeypatch.setattr(r, 'original', lambda: module)
    observer = SimpleNamespace(proc_io=lambda: {'rchar': 120})
    monkeypatch.setattr(module, 'dependencies', lambda: (core, supervisor, observer))
    monkeypatch.setattr(module, 'rss_monitor', lambda *_: None)
    monkeypatch.setattr(r.subprocess, 'check_output', lambda argv, **kw: module.SOURCE if 'rev-parse' in argv else '')
    calls=[]
    def fake_pair(command, budget, core_, supervisor_, observer_, metadata, lock, meter):
        assert command['comparison'] == 'C3_minus_C0'
        assert module.CONTROL == new and meter.total == 0
        assert (new / 'SESSION_STARTED.json').exists()
        calls.append(command)
        if fail: raise RuntimeError('fixture failure, no worker created')
        meter.parent({'rchar': 100})
        return {'status': 'COMPLETE', 'fixture_only': True}
    monkeypatch.setattr(module, 'run_pair', fake_pair)
    if fail:
        with pytest.raises(RuntimeError, match='fixture failure'): r.run()
    else: assert r.run() == 0
    row=json.loads((new/'SESSION_TERMINAL.json').read_text())
    assert row['prior_campaign_read_charge_bytes'] == 1103776571
    assert row['declared_campaign_read_charge_bytes'] == r.PRIOR_CHARGE + (r.NEW_ALLOWANCE if fail else 120)
    assert row['automatic_retry'] is False and row['C2_rerun'] is False
    with pytest.raises(ValueError, match='Consumed session'): r.run()
    assert len(calls) == 1
    assert (old/'RUN.lock').read_bytes() == b''


def test_shared_output_and_session_controls_fit_original_32mib():
    module, _, _, _ = setup()
    assert module.ANALYSIS_LIMIT + module.CONTROL_RESERVE + module.ADMISSION_RESERVE == 32 << 20
    assert (32 << 10) + 2 * (16 << 10) <= module.ADMISSION_RESERVE
