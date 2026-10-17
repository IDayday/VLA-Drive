"""Independent small-control fixtures; no payload, GPU or real analysis process."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE = Path(__file__).resolve().parents[2] / '20261016/evaluation/FULL_PAYLOAD_C3_1024_RETRY_PREPARATION_v1/run_c3_once.py'
spec = importlib.util.spec_from_file_location('independent_c3_delta', SOURCE)
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)

@pytest.mark.parametrize('mutation', ['unexpected_child_accounting', 'c2_survivor', 'wrong_c3_charge'])
def test_additional_lineage_fail_closed(mutation, monkeypatch):
    m = r.original(); core, _, _ = m.dependencies()
    rows = {k: core.pinned(pin) for k, pin in r.PRIOR.items()}
    if mutation == 'unexpected_child_accounting': rows['C3']['resources']['child_accounting'] = {}
    elif mutation == 'c2_survivor': rows['C2']['resources']['cleanup']['live_owned_remaining'] = 1
    else: rows['C3']['charged_cumulative_rchar_bytes'] -= 1
    values = {pin['path']: rows[k] for k, pin in r.PRIOR.items()}
    real = core.pinned
    monkeypatch.setattr(core, 'pinned', lambda pin, **kw: copy.deepcopy(values[pin['path']]) if pin['path'] in values else real(pin, **kw))
    with pytest.raises(ValueError): r.prepare(m, core)


def test_borrowed_run_pair_dispatches_only_original_child_c3(tmp_path, monkeypatch):
    m = r.original(); core, supervisor, _ = m.dependencies()
    command, budget = r.prepare(m, core)
    monkeypatch.setattr(m, 'CONTROL', tmp_path)
    seen = []
    answers = iter([None, {'path': '/fixture/ANALYSIS.json', 'sha256': 'fixture', 'output_bytes': 10}])
    monkeypatch.setattr(m, 'complete_output', lambda *a: next(answers))
    monkeypatch.setattr(core, 'charge_after_reap', lambda *a: 100)
    proc = iter([{'rchar': 100}, {'rchar': 200}, {'rchar': 200}])
    observer = SimpleNamespace(proc_io=lambda: next(proc))
    def fake_monitor(argv, **kw):
        seen.append((argv, kw))
        return {'returncode': 0, 'failure': None, 'read_accounting': {}, 'child_accounting': {'cli_result': {'exit_code': 0}}}
    monkeypatch.setattr(core, 'monitor', fake_monitor)
    meter = m.ReadMeter()
    with (tmp_path / 'fixture.lock').open('w') as lock:
        out = m.run_pair(command, budget, core, supervisor, observer, None, lock, meter)
    assert out['status'] == 'COMPLETE' and len(seen) == 1
    argv, kw = seen[0]
    assert argv == [m.PYTHON, r.WRAPPER['path'], '--child', 'C3']
    assert kw['root'] == tmp_path / 'C3_minus_C0'
    assert kw['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert kw['budget']['read_bytes'] == (1 << 30) - 100
    assert kw['budget']['rss_bytes'] == 2 << 30
    assert kw['budget']['wall_seconds'] == 1800 and kw['budget']['cleanup_seconds'] == 40
    assert meter.total == 200


def test_new_allowance_stops_without_erasing_old_charge():
    m = r.original(); meter = m.ReadMeter()
    assert meter.parent({'rchar': 1 << 30}) == 0
    with pytest.raises(ValueError, match='budget exhausted'): meter.parent({'rchar': (1 << 30) + 1})
    assert r.PRIOR_CHARGE == 1103776571
    assert r.PRIOR_CHARGE + meter.total == 2177518396
