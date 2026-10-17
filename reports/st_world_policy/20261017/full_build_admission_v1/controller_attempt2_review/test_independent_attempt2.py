"""Narrow independent revision tests; no actual builder or ledger is invoked."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest
OWNER=Path('/mnt/project/VLA-Drive-st-full-future-metadata-preparation-20261017/reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py')
assert hashlib.sha256(OWNER.read_bytes()).hexdigest()=='a866ae4261cad36312355028cdfba3b24f52f64b174433635d0473420ddc84bf'
spec=importlib.util.spec_from_file_location('independent_attempt2',OWNER)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
HELPERS=Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1')
for name,sha in c.MONITORS.items():assert hashlib.sha256((HELPERS/name).read_bytes()).hexdigest()==sha
sys.path.insert(0,str(HELPERS))
import supervise_once as s
import observed_cli as obs

def admission():
    return {'schema':'st_full_future_build_resource_admission_v1','status':'PASS_BOUNDED_DEV_BUILD_ATTEMPT2_ONLY',
        'role':'dev','attempt':2,'prior_attempt':dict(c.PRIOR_DEV_RESULT),'limits':c.limits('dev',2),
        'hostname':c.socket.gethostname(),'reviewer':'independent','preparation':{'path':str(c.PREP),'sha256':c.PREP_SHA},
        'issued_at_unix':1,'latest_start_unix':100,'valid_until_unix':20000}

@pytest.mark.parametrize('change',['ok','third','bool','reset_read','wrong_prior','train'])
def test_retry_exact_budget_and_identity(change):
    a=admission()
    if change=='third':a['attempt']=3
    if change=='bool':a['attempt']=True
    if change=='reset_read':a['limits']['read_bytes']=16<<30
    if change=='wrong_prior':a['prior_attempt']['sha256']='0'*64
    if change=='train':a['role']='train'
    if change=='ok':
        c.validate(a,'dev',10)
        assert a['limits']['read_bytes']+c.PRIOR_DEV_CHARGE==16<<30
    else:
        with pytest.raises(ValueError):c.validate(a,a['role'],10)

@pytest.mark.parametrize('failure',[False,True])
def test_child_final_accounting_is_ram_only(tmp_path,monkeypatch,capsys,failure):
    calls=[]
    def run(**_):
        calls.append('build_fixture')
        if failure:raise ValueError('fixture source failure')
        return {'status':'GUARDED_METADATA_RETURN','artifact':{'status':'COMPLETE'}}
    old_write=obs.write_control
    monkeypatch.setattr(c,'controls',lambda *a,**k:(admission(),{}, {},tmp_path,{},None,obs,SimpleNamespace(run_registered_metadata=run),None))
    monkeypatch.setattr(c.os,'getpgrp',lambda:os.getpid())
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(obs.Counters,'monitor',lambda self:pytest.fail('background monitor forbidden'))
    with (tmp_path/'own-fd').open('wb') as fd:
        monkeypatch.setenv('ST_PROBE_SUPERVISOR_LOCK_FD',str(fd.fileno()))
        rc=c.child_run(SimpleNamespace())
    output=capsys.readouterr();j=c.decode_final(output.out)
    assert rc==(1 if failure else 0)
    assert calls==['build_fixture']
    assert obs.write_control is old_write
    assert not list(tmp_path.glob('CONTROL*')) and not (tmp_path/'UNUSED_RAM_ONLY_COUNTER').exists()
    if failure:assert j['error']['type']=='ValueError' and 'fixture source failure' in output.err

@pytest.mark.parametrize('case',['success','failure','bad_accounting','rss','read','wall','pipe'])
def test_real_owned_cpu_children_no_sidecar_and_io_before_reap(tmp_path,monkeypatch,case):
    code="""import json,time
rd=lambda:{k:int(v) for k,v in (r.split(':') for r in open('/proc/self/io'))}
b=rd();time.sleep(.45);f=rd()
print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':{'status':'GUARDED_METADATA_RETURN'},'counters':{'proc_self_io_baseline':b,'proc_self_io_final':f,'proc_self_io_delta':{k:f[k]-b[k] for k in f}}}))
"""
    if case=='failure':code+='raise SystemExit(7)'
    if case=='bad_accounting':code="import time;time.sleep(.45);print('{bad')"
    if case=='pipe':code="import sys,time;sys.stdout.write('x'*((1<<20)+8));sys.stdout.flush();time.sleep(.45)"
    events=[];real_finish=s.OwnedSession.finish;real_io=c.owned_proc_io
    def finish(self,**kw):events.append('finish');return real_finish(self,**kw)
    def io(owner,supervisor):
        assert 'finish' not in events
        events.append('io');return real_io(owner,supervisor)
    monkeypatch.setattr(s.OwnedSession,'finish',finish);monkeypatch.setattr(c,'owned_proc_io',io)
    monkeypatch.setattr(c,'START',time.monotonic())
    budget=c.limits('dev',2);budget.update(wall_seconds=0 if case=='wall' else 10,read_bytes=1 if case=='read' else 128<<20,rss_bytes=1<<30)
    with (tmp_path/'owned-fd').open('wb') as fd:
        r=c.monitor([sys.executable,'-c',code],runtime=str(tmp_path),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),fd=fd.fileno(),root=tmp_path,s=s,m=SimpleNamespace(process_tree_rss=lambda _:2<<30 if case=='rss' else 32<<20),obs=obs,budget=budget)
    assert events[-1]=='finish' and events.count('finish')==1 and 'io' in events
    assert r['cleanup']['live_owned_remaining']==0
    assert r['owned_proc_io_terminal'] is not None
    assert r['charged_owned_child_rchar_bytes']>=r['owned_proc_io_terminal']['rchar']
    assert not list(tmp_path.glob('CONTROL*'))
    expected={'success':None,'failure':None,'bad_accounting':'CHILD_ACCOUNTING_INVALID','rss':'RSS_LIMIT','read':'PARENT_CHILD_READ_LIMIT','wall':'WALL_LIMIT','pipe':'STDOUT_LIMIT'}[case]
    assert r['failure']==expected
    if case=='failure':assert r['returncode']==7
    if case=='success':assert r['returncode']==0


def test_owned_identity_change_refuses_before_proc_io():
    owner=SimpleNamespace(anchor={'pid':os.getpid()},same=lambda a,b:False)
    with pytest.raises(RuntimeError,match='identity'):
        c.owned_proc_io(owner,SimpleNamespace(proc_identity=lambda _:{}))

@pytest.mark.parametrize('bad',[True,-1,1.5,float('nan')])
def test_final_snapshot_no_fabricated_count(bad):
    value={'schema':'st_full_future_child_result_v2','counters':{'proc_self_io_baseline':{'rchar':0},'proc_self_io_final':{'rchar':bad},'proc_self_io_delta':{'rchar':bad}}}
    with pytest.raises(ValueError):c.decode_final(json.dumps(value))
