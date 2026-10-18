"""Independent CPU ownership checks. No NVML/GPU/model/trainer invocation."""
import copy
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

SOURCE=Path('/mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v3-20261018/resources/st_rollout_resume_smoke_v1.py')
spec=importlib.util.spec_from_file_location('independent_ddp_ownership_v3',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def wait_until(fn,seconds=6):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        value=fn()
        if value:return value
        time.sleep(.02)
    raise AssertionError('Owned CPU fixture did not reach its bounded state')


def test_real_nested_setsid_orphan_gate_and_foreign_gpu_refusal(tmp_path,monkeypatch):
    libc=ctypes.CDLL(None,use_errno=True);prior=ctypes.c_int()
    assert libc.prctl(37,ctypes.byref(prior),0,0,0)==0
    assert libc.prctl(36,1,0,0,0)==0
    child=None;foreign=None;tracked={}
    rank_file=tmp_path/'rank.json';exit_flag=tmp_path/'exit_parent'
    rank_code="import os,subprocess,sys,time,json;from pathlib import Path;leaf=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);Path(sys.argv[1]).write_text(json.dumps({'rank':os.getpid(),'leaf':leaf.pid}));time.sleep(120)"
    root_code="import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],start_new_session=True);flag=Path(sys.argv[3]);exec('while not flag.exists(): time.sleep(.01)')"
    try:
        child=subprocess.Popen([sys.executable,'-c',root_code,rank_code,str(rank_file),str(exit_flag)],start_new_session=True)
        wait_until(rank_file.exists);ids=json.loads(rank_file.read_text());rank=ids['rank'];leaf=ids['leaf']
        job={'child':child,'start_ticks':None,'owned':{},'unit':'explicit_CPU_ownership_fixture','started_unix':time.time()-1}
        table=wait_until(lambda:(rows if rank in (rows:=m.owned_processes(job)) and leaf in rows else None))
        tracked={pid:row['start_ticks'] for pid,row in table.items()}
        assert os.getsid(rank)==rank and os.getsid(rank)!=os.getsid(child.pid)
        # The real heavy-load gate must accept a proven rank outside root SID.
        run=tmp_path/'synthetic_run';(run/'official_runtime').mkdir(parents=True)
        for index in range(4):(run/'official_runtime'/f'initial_engine_rank{index}.json').write_text('{}')
        (run/'initialization.json').write_text('{}')
        identity={'source_sha':m.SOURCE};identity['identity']=m.digest(identity)
        (run/'identity.json').write_text(json.dumps(identity))
        (run/'status.json').write_text(json.dumps({'identity':identity['identity'],'source_sha':m.SOURCE,'started_unix':time.time(),'pid':rank}))
        boundary=run/'checkpoints/step_000000_cpu';boundary.mkdir(parents=True);(boundary/'COMPLETE.json').write_text('{}')
        monkeypatch.setattr(m,'inspect_native',lambda *a,**k:{'explicit':'tiny fixture; no actual native payload'})
        assert m.group_ready(run,job) is True
        ep=m.helper();route={'gpus':list(range(8)),'gpu_uuids':['explicit-fixture-GPU-'+str(i) for i in range(8)]}
        foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)
        include_foreign=[False]
        def nvml_fixture(args,**kwargs):
            if any('query-compute-apps' in arg for arg in args):
                text=f"{route['gpu_uuids'][0]}, {rank}, 1\n"
                if include_foreign[0]:text+=f"{route['gpu_uuids'][0]}, {foreign.pid}, 1\n"
                return text
            return ''.join(f'{uuid}, 1\n' for uuid in route['gpu_uuids'])
        monkeypatch.setattr(ep.subprocess,'check_output',nvml_fixture)
        assert len(ep.gpu_monitor(route,m.owned_processes(job)))==8
        include_foreign[0]=True
        with pytest.raises(ValueError,match='Unknown/foreign'):ep.gpu_monitor(route,m.owned_processes(job))
        assert foreign.poll() is None
        exit_flag.touch();child.wait(timeout=5)
        orphaned=wait_until(lambda:(row if (row:=m.process_record(rank))['ppid']!=child.pid else None))
        assert orphaned['start_ticks']==tracked[rank]
        assert rank in m.owned_processes(job) and leaf in m.owned_processes(job)
        prior_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        m.stop_smoke_owned([job])
        assert all(signal.getsignal(sig)==handler for sig,handler in prior_handlers.items())
        assert job['cleanup_complete'] is True
        assert all(row['state']=='Z' for row in m.owned_processes(job).values())
        assert foreign.poll() is None,'Strict owned cleanup must leave foreign task alive'
    finally:
        for pid,stamp in tracked.items():
            try:
                if m.process_record(pid)['start_ticks']==stamp:os.kill(pid,signal.SIGKILL)
            except (ProcessLookupError,FileNotFoundError):pass
        if child is not None:
            if child.poll() is None:child.kill()
            child.wait(timeout=5)
        if foreign is not None:
            if foreign.poll() is None:foreign.kill()
            foreign.wait(timeout=5)
        for pid in tracked:
            try:wait_until(lambda:os.waitpid(pid,os.WNOHANG)[0]==pid,seconds=2)
            except (ChildProcessError,AssertionError):pass
        assert libc.prctl(36,prior.value,0,0,0)==0


def test_reused_birth_stamps_and_unproven_session_are_not_owned(monkeypatch):
    # Synthetic proc records test kernel PID reuse semantics without forcing reuse.
    def row(pid,ppid,start,sid):return {'pid':pid,'ppid':ppid,'start_ticks':start,'sid':sid,'state':'S','rss':0,'pgrp':sid}
    child=SimpleNamespace(pid=100,poll=lambda:None)
    job={'child':child,'start_ticks':10,'owned':{100:row(100,1,10,100),200:row(200,100,20,200)},'owned_sessions':{100:10,200:20}}
    table={100:row(100,1,10,100),200:row(200,1,2000,200),201:row(201,200,2001,200),300:row(300,1,30,300)}
    monkeypatch.setattr(m,'process_table',lambda:copy.deepcopy(table))
    assert set(m.owned_processes(job))=={100}


def test_exited_popen_cannot_seed_unknown_recycled_root(monkeypatch):
    child=SimpleNamespace(pid=100,poll=lambda:0)
    job={'child':child,'start_ticks':None,'owned':{}}
    monkeypatch.setattr(m,'process_table',lambda:{100:{'pid':100,'ppid':1,'start_ticks':999,'sid':100,'state':'S','rss':0,'pgrp':100}})
    assert m.owned_processes(job)=={}
    assert job['start_ticks'] is None
