"""Tiny owned CPU child and temporary counters only; never production jobs."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

HERE = Path(__file__).resolve().parents[2]/'spatial_probe_actual_preparation_v1'
sys.path.insert(0, str(HERE))
import prepare_bindings as b
import supervise_once as s


def child(tmp_path, wall=3):
    return s.supervise([sys.executable,'-c','import time;time.sleep(.02)'],
        cwd=tmp_path, env=dict(os.environ), cpus=sorted(os.sched_getaffinity(0))[:2],
        descriptors=(), attempt=tmp_path, wall=wall, rss=1<<30, rss_reader=lambda _:0)


def test_final_real_control_lock_wait_is_included_in_wall_admission(tmp_path, monkeypatch):
    b.write_control(tmp_path/'COUNTERS.json', {'limit_exceeded':None}, replace=True)
    finish=s.OwnedSession.finish
    ready=threading.Event(); threads=[]
    def finished_then_hold(self, **kwargs):
        code=finish(self, **kwargs)
        def holder():
            with b.control_lock(tmp_path):
                ready.set(); time.sleep(.65)
        thread=threading.Thread(target=holder);threads.append(thread);thread.start()
        assert ready.wait(1)
        return code
    monkeypatch.setattr(s.OwnedSession, 'finish', finished_then_hold)
    try: result=child(tmp_path, wall=.5)
    finally:
        for thread in threads: thread.join(2)
    assert result['exit_code']==0 and result['process_cleanup']['live_owned_remaining']==0
    assert result['wall_seconds']>.5
    assert result['limit_reason']=='WALL_LIMIT', result


def test_first_freshness_failure_survives_later_final_failure(tmp_path, monkeypatch):
    calls=[]
    def snapshot(path,on_error):
        calls.append(1)
        if len(calls)==1:
            exc=OSError('first freshness observation')
            on_error('counter_freshness',exc);raise exc
        exc=TimeoutError('later final timeout');on_error('counter_lock',exc);raise exc
    monkeypatch.setattr(s,'read_counter_snapshot',snapshot)
    result=child(tmp_path)
    assert len(calls)>=2 and result['limit_reason']=='MONITOR_UNREADABLE'
    assert result['first_monitor_error']['operation']=='counter_freshness'
    assert result['first_monitor_error']['message']=='first freshness observation'


def test_final_valid_limit_is_not_lost(tmp_path, monkeypatch):
    calls=[]
    def snapshot(path,on_error):
        calls.append(1)
        return ({'limit_exceeded':'IO_LIMIT'} if len(calls)>1 else {}, time.time())
    monkeypatch.setattr(s,'read_counter_snapshot',snapshot)
    result=child(tmp_path)
    assert len(calls)>=2 and result['limit_reason']=='IO_LIMIT'
    assert result['first_monitor_error'] is None
