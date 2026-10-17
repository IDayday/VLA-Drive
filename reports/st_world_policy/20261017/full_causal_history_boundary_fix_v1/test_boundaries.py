import fcntl
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import patch
import pytest

HERE=Path(__file__).parent
path=HERE.parent/'full_causal_history_preparation_v1/run_pilot_once.py'
spec=importlib.util.spec_from_file_location('history_boundary',path);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from tools.st_world_policy import cache_storage_budget as b

def test_actual_locked_nonblocking_rejects_without_probe(tmp_path):
    policy=tmp_path/'policy';policy.write_text('{}');lock=tmp_path/'CLAIMS.lock'
    budget=SimpleNamespace(policy_path=policy,state_dir=tmp_path,state_path=tmp_path/'CLAIMS.json',lock_path=lock,core={},_read_policy=lambda: ({},{},{}),_check_cap=lambda _:None)
    with lock.open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX)
        start=time.monotonic()
        with p.nonblocking_original_ledger(b),pytest.raises(BlockingIOError):
            with b.StorageBudget._locked(budget):raise AssertionError('admitted while busy')
        assert time.monotonic()-start<1
    with p.nonblocking_original_ledger(b):
        with b.StorageBudget._locked(budget):pass


def test_supervisor_lock_held_and_contention_fast(tmp_path):
    with p.control_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with p.control_lock(tmp_path):raise AssertionError('second supervisor')


def test_start_and_result_busy_preserve_bounded_terminal(tmp_path):
    for fail_phase in ('START_PUBLICATION','RESULT_PUBLICATION','OWNED_MONITOR'):
        root=tmp_path/fail_phase;root.mkdir();control=root/'control';control.mkdir()
        observed=dict(returncode=0,failure=None,child_accounting={'cli_result':{'artifact':{'status':'COMPLETE'}}},read_accounting={},cleanup={'live_owned_remaining':0})
        calls=[]
        def publish(m,b,g,path,value):
            calls.append(path.name)
            if (fail_phase=='START_PUBLICATION' and len(calls)==1) or (fail_phase=='RESULT_PUBLICATION' and len(calls)==2):raise BlockingIOError('busy fixture')
        def monitor(*a,**kw):
            if fail_phase=='OWNED_MONITOR':raise RuntimeError('owned fixture failed')
            return observed
        core=SimpleNamespace(publish_guarded=publish,monitor=monitor,charge_after_reap=lambda *a:42)
        obs=SimpleNamespace(proc_io=lambda:{'rchar':50})
        args=SimpleNamespace(admission='/fixture',admission_sha256='a'*64)
        with patch.object(p,'ROOT',root),patch.object(p,'CONTROL_ROOT',control),p.whole_invocation_alarm(time.monotonic()+10):
            assert p.run_parent(args,{'valid_until_unix':time.time()+1000},{},{},None,obs,None,b,core,lambda:None,work_deadline=time.monotonic()+9)==2
        receipt=json.loads((control/'RESULT.json').read_text())
        assert receipt['status']=='FAILED_NO_AUTO_RETRY' and receipt['phase']==fail_phase
        assert receipt['charged_rchar_bytes']==p.LIMITS['read_bytes']
        if fail_phase=='OWNED_MONITOR':assert receipt['cleanup']['status']=='UNKNOWN_MONITOR_EXCEPTION'
        assert len((control/'RESULT.json').read_bytes())<4096
        with pytest.raises(FileExistsError):p.control_once(control,'RESULT.json',{'changed':True})


def test_entry_baseline_precedes_controls_and_core_import():
    import ast
    tree=ast.parse(path.read_text())
    baseline=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ENTRY_IO' for t in n.targets))
    controls=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controls')
    assert baseline<controls.lineno
    assigns=[n for n in ast.walk(controls) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='BOOT_IO' for t in n.targets)]
    assert len(assigns)==1 and isinstance(assigns[0].value,ast.Call) and assigns[0].value.args[0].id=='ENTRY_IO'


def test_terminal_receipt_has_no_original_ledger_dependency(tmp_path):
    with patch.object(b.StorageBudget,'_locked',side_effect=AssertionError('ledger touched')):
        p.control_once(tmp_path,'RESULT.json',{'status':'FAILED_NO_AUTO_RETRY'})
    assert json.loads((tmp_path/'RESULT.json').read_text())['status']=='FAILED_NO_AUTO_RETRY'


def test_deadline_and_reserved_control_cap(tmp_path):
    with pytest.raises(TimeoutError):p.deadline_check(time.monotonic()-1)
    (tmp_path/'RESULT.json.tmp').write_bytes(b'x'*p.CONTROL_CAP)
    with pytest.raises(ValueError):p.control_once(tmp_path,'RESULT.json',{'failure':1})


def test_whole_wrapper_alarm_is_real_and_restores_handler():
    import signal
    prior=signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError,match='Whole wrapper wall'):
        with p.whole_invocation_alarm(time.monotonic()+.02):time.sleep(.1)
    assert signal.getsignal(signal.SIGALRM)==prior and signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
