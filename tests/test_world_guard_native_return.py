"""The resource return path saves a boundary, never admits new work."""
import ast
import copy
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from starVLA.rl.qwen_world_safety_exploration import contracts
from starVLA.rl.qwen_world_safety_exploration.guard_return import (
    native_dispatch, boundary_signature, stopped_checkpoint)


def test_admitted_bound_job_stops_before_revoked_permission_but_startup_cannot(monkeypatch):
    calls = []
    due = False
    def require(**kwargs):
        calls.append(kwargs)
        if due:
            raise PermissionError('new dispatch revoked')
        return 'permanent-qualified-host'
    monkeypatch.setattr(contracts, 'require_dispatch', require)
    monkeypatch.setattr(contracts, 'boundary_stop_due', lambda host: due)
    stops = []
    runtime = SimpleNamespace(request_stop=lambda: stops.append('stop'))
    before = random.getstate()
    assert native_dispatch(runtime) == 'permanent-qualified-host'
    assert calls == [dict(host=None, admit_long_job=True)]
    runtime.training_state = dict(updates=14501)
    runtime.training_identity = {'sha256': 'same-native-run'}
    due = True
    assert native_dispatch(runtime) == 'permanent-qualified-host'
    assert len(calls) == 1 and stops == ['stop']
    with pytest.raises(PermissionError):
        native_dispatch(SimpleNamespace(request_stop=lambda: stops.append('forbidden')))
    assert calls[-1] == dict(host=None, admit_long_job=True)
    assert stops == ['stop'] and random.getstate() == before
    checks = iter([False, True])
    monkeypatch.setattr(contracts, 'boundary_stop_due', lambda host: next(checks))
    assert native_dispatch(runtime) == 'permanent-qualified-host'
    assert stops == ['stop', 'stop'] and random.getstate() == before


def test_actual_trainer_top_stop_reuses_ready_and_saves_pending_without_work():
    # Execute the actual loop's prefix; the sentinel stands for collect/Adam.
    path = Path(__file__).parents[1]/'tools/qwen_world_safety_exploration/train.py'
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.While)
                and isinstance(n.test, ast.Compare))
    assert isinstance(loop.body[0], ast.Expr) and loop.body[0].value.func.id == 'dispatch'
    assert isinstance(loop.body[1], ast.If) and loop.body[1].test.func.id == 'stop_requested'
    sentinel = ast.parse("raise RuntimeError('TRAINING_CONTINUES')").body
    probe_loop = copy.deepcopy(loop)
    probe_loop.body = copy.deepcopy(loop.body[:2]) + sentinel
    source = ast.parse('def probe(state,rollout,last_saved,last_saved_signature,save_boundary):\n pass\n')
    source.body[0].body = [probe_loop, ast.parse('return last_saved,last_saved_signature').body[0]]
    stop = True
    dispatch_calls = []
    ns = dict(cfg={'max_steps': 50000}, device='cpu', rank=1,
              dispatch=lambda: dispatch_calls.append('dispatch'),
              stop_requested=lambda device: stop, stopped_checkpoint=stopped_checkpoint)
    exec(compile(ast.fix_missing_locations(source), str(path), 'exec'), ns)
    ready = dict(updates=14500,next_inner=0,collections=3625,scene_cursor=29000,phase='READY')
    saved = []
    save = lambda: saved.append('save') or 'new-exact-inner-checkpoint'
    cp, key = ns['probe'](ready,None,'existing-complete',boundary_signature(ready,None),save)
    assert cp == 'existing-complete' and not saved
    pending = dict(ready,updates=14501,next_inner=1,collections=3626,scene_cursor=29008,phase='INNER_COMMITTED')
    rollout = {'chain_identity': 'same-pending-G8-chain'}
    cp,key = ns['probe'](pending,rollout,'existing-complete',boundary_signature(ready,None),save)
    assert saved == ['save'] and cp == 'new-exact-inner-checkpoint'
    assert key == boundary_signature(pending,rollout)
    stop = False
    with pytest.raises(RuntimeError, match='TRAINING_CONTINUES'):
        ns['probe'](pending,rollout,cp,key,save)
    assert saved == ['save'] and len(dispatch_calls) == 3
