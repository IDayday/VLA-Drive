"""Small CPU controls/owned-child fixtures only; no real extraction or retry."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'spatial_probe_actual_preparation_v1'))
import prepare_bindings as b
import supervise_once as s


def test_writer_cannot_replace_during_strict_read_and_freshness(tmp_path, monkeypatch):
    path=tmp_path/'COUNTERS.json'; b.write_control(path, {'counter':1}, replace=True)
    original_mtime=path.stat().st_mtime
    original_fstat=os.fstat; original_lock=b.control_lock
    ready=threading.Event(); acquired=threading.Event(); errors=[]; calls=[0]
    @contextmanager
    def writer_lock(directory):
        ready.set()
        with original_lock(directory):
            acquired.set(); yield
    def writer():
        try: b.write_control(path, {'counter':2}, replace=True)
        except BaseException as exc: errors.append(repr(exc))
    thread=threading.Thread(target=writer)
    def competing_fstat(fd):
        value=original_fstat(fd)
        if threading.current_thread() is threading.main_thread():
            calls[0]+=1
            if calls[0]==2:
                thread.start(); assert ready.wait(1)
                assert not acquired.is_set()
        return value
    with monkeypatch.context() as m:
        m.setattr(b, 'control_lock', writer_lock)
        m.setattr(b.os, 'fstat', competing_fstat)
        value, modified=s.read_counter_snapshot(path)
    thread.join(3)
    assert not thread.is_alive() and not errors
    assert value=={'counter':1} and modified==original_mtime
    assert b.read(path)[0]=={'counter':2}


def test_lock_timeout_is_original_two_seconds_and_fail_closed(tmp_path):
    ready=threading.Event(); release=threading.Event(); errors=[]
    def holder():
        try:
            with b.control_lock(tmp_path): ready.set(); release.wait(4)
        except BaseException as exc: errors.append(repr(exc))
    thread=threading.Thread(target=holder); thread.start(); assert ready.wait(1)
    observed=[]; start=time.monotonic()
    try:
        with pytest.raises(TimeoutError, match='Bounded control write lock'):
            s.read_counter_snapshot(tmp_path/'COUNTERS.json', lambda op,exc:observed.append((op,type(exc).__name__)))
    finally:
        release.set(); thread.join(3)
    assert 1.9 <= time.monotonic()-start < 4
    assert observed==[('counter_lock','TimeoutError')] and not errors


@pytest.mark.parametrize('bad', ['invalid_json','oversize','symlink'])
def test_bad_snapshot_still_rejected(tmp_path,bad):
    path=tmp_path/'COUNTERS.json'
    if bad=='invalid_json': path.write_text('{broken')
    elif bad=='oversize': path.write_bytes(b'x'*65537)
    else:
        target=tmp_path/'actual.json';target.write_text('{}');path.symlink_to(target)
    with pytest.raises((ValueError,json.JSONDecodeError)):
        s.read_counter_snapshot(path)


def run_child(tmp_path,rss_reader):
    return s.supervise([sys.executable,'-c','import time;time.sleep(5)'],cwd=tmp_path,
        env=dict(os.environ),cpus=sorted(os.sched_getaffinity(0))[:2],descriptors=(),
        attempt=tmp_path,wall=3,rss=1<<30,rss_reader=rss_reader)


def test_bad_counter_records_real_error_and_owned_cleanup(tmp_path):
    (tmp_path/'COUNTERS.json').write_text('{broken')
    result=run_child(tmp_path,lambda pid:0)
    error=result['first_monitor_error']
    assert result['limit_reason']=='MONITOR_UNREADABLE'
    assert error['operation']=='counter_read' and error['exception_class']=='JSONDecodeError'
    assert error['time_unix']>0 and error['elapsed_seconds']>=0
    assert len(error['message'])<=1024
    assert result['process_cleanup']['live_owned_remaining']==0


def test_first_rss_error_bounded_and_not_overwritten_by_final_counter(tmp_path):
    (tmp_path/'COUNTERS.json').write_text('{broken')
    def failed_rss(pid): raise OSError('r'*5000)
    result=run_child(tmp_path,failed_rss)
    error=result['first_monitor_error']
    assert error['operation']=='rss' and error['exception_class']=='OSError'
    assert error['message']=='r'*1024
    assert len(json.dumps(error))<2048
    assert result['limit_reason']=='MONITOR_UNREADABLE'
    assert result['process_cleanup']['live_owned_remaining']==0


def test_final_snapshot_uses_same_locked_path_and_captures_failure(tmp_path,monkeypatch):
    calls=[]
    def snapshots(path,on_error):
        calls.append(str(path))
        if len(calls)>1:
            exc=TimeoutError('fixture final snapshot lock timeout')
            on_error('counter_lock',exc); raise exc
        return None,None
    monkeypatch.setattr(s,'read_counter_snapshot',snapshots)
    result=s.supervise([sys.executable,'-c','import time;time.sleep(.05)'],cwd=tmp_path,
        env=dict(os.environ),cpus=sorted(os.sched_getaffinity(0))[:2],descriptors=(),
        attempt=tmp_path,wall=3,rss=1<<30,rss_reader=lambda pid:0)
    assert len(calls)>=2 and result['limit_reason']=='MONITOR_UNREADABLE'
    assert result['first_monitor_error']['operation']=='counter_lock'
    assert result['first_monitor_error']['exception_class']=='TimeoutError'


def test_fixed_artifact_hash_guard_and_missing_counter_unchanged(tmp_path):
    path=tmp_path/'fixed.json'; pin=b.write_once(path, {'a':1})
    assert b.read(path,pin['sha256'])[0]=={'a':1}
    with pytest.raises(ValueError,match='Control hash changed'): b.read(path,'0'*64)
    assert s.read_counter_snapshot(tmp_path/'COUNTERS.json')==(None,None)
    assert s.LIMITS=={'plan':(600,64<<30),'extract':(960,64<<30),'fit':(1260,2<<30)}
    assert (s.LOG_LIMIT,s.ROLE_READ_LIMIT,s.FIT_READ_LIMIT)==(4<<20,256<<30,2<<30)
