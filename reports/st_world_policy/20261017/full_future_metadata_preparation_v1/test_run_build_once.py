"""CPU control fixtures only; no real builder, model, ledger or process launch."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('future_build',Path(__file__).with_name('run_build_once.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def admission(monkeypatch,role='dev'):
    monkeypatch.setattr(m.os,'sched_getaffinity',lambda _: {0,1})
    return {'schema':'st_full_future_build_resource_admission_v1','status':'PASS_BOUNDED_'+role.upper()+'_BUILD_ONLY',
        'role':role,'limits':m.limits(role),'hostname':m.socket.gethostname(),'reviewer':'independent_fixture',
        'preparation':{'path':str(m.PREP),'sha256':m.PREP_SHA},
        'issued_at_unix':1,'latest_start_unix':100,'valid_until_unix':20000}


@pytest.mark.parametrize('role',['dev','train'])
def test_fixed_finite_limits(role,monkeypatch):
    a=admission(monkeypatch,role);m.validate(a,role,10)
    assert a['limits']['cpu_affinity']==[0,1] and a['limits']['gpu'] is False
    assert a['limits']['read_bytes']==(16 if role=='dev' else 512)*(1<<30)
    assert a['limits']['wall_seconds']==(1800 if role=='dev' else 14400)


@pytest.mark.parametrize('mutation',['status','role','hostname','own_review','expired','short','nan','prep','gpu','retry','rss','bool_cpu'])
def test_invalid_admission_never_passes(monkeypatch,mutation):
    a=admission(monkeypatch)
    if mutation=='status':a['status']='APPROVED'
    elif mutation=='role':a['role']='index'
    elif mutation=='hostname':a['hostname']='other'
    elif mutation=='own_review':a['reviewer']='/root/st_indexed_composition'
    elif mutation=='expired':a['latest_start_unix']=9
    elif mutation=='short':a['valid_until_unix']=100
    elif mutation=='nan':a['valid_until_unix']=float('nan')
    elif mutation=='prep':a['preparation']['sha256']='0'*64
    elif mutation=='bool_cpu':a['limits']['cpu_affinity']=[False,1]
    elif mutation=='gpu':a['limits']['gpu']=True
    elif mutation=='retry':a['limits']['automatic_retry']=True
    else:a['limits']['rss_bytes']=8<<30
    with pytest.raises(ValueError):m.validate(a,'dev',10)


def test_both_pipe_buffers_are_bounded_and_failure_retains_prefix():
    buffers={'stdout':bytearray(),'stderr':bytearray()}
    for name in buffers:
        assert m.append_pipe(buffers,name,b'a'*m.MiB)
        assert not m.append_pipe(buffers,name,b'excess')
        assert len(buffers[name])==m.MiB


@pytest.mark.parametrize('bad',[True,-1,1.5])
def test_final_accounting_requires_exact_nonnegative_integer_counts(bad):
    value={'schema':'st_full_future_child_result_v2','counters':{'proc_self_io_baseline':{'rchar':0},
        'proc_self_io_final':{'rchar':bad},'proc_self_io_delta':{'rchar':bad}}}
    with pytest.raises(ValueError):m.decode_final(json.dumps(value))


def test_explicit_second_attempt_retains_prior_charge(monkeypatch):
    a=admission(monkeypatch);a.update(attempt=2,prior_attempt=m.PRIOR_DEV_RESULT,
        status='PASS_BOUNDED_DEV_BUILD_ATTEMPT2_ONLY',limits=m.limits('dev',2))
    m.validate(a,'dev',10)
    assert a['limits']['read_bytes']==14977059003
    a['limits']['read_bytes']=16<<30
    with pytest.raises(ValueError):m.validate(a,'dev',10)


def test_second_attempt_cannot_reuse_train_or_change_prior_identity(monkeypatch):
    a=admission(monkeypatch);a.update(attempt=2,prior_attempt={'path':'other','sha256':'0'*64},
        status='PASS_BOUNDED_DEV_BUILD_ATTEMPT2_ONLY',limits=m.limits('dev',2))
    with pytest.raises(ValueError):m.validate(a,'dev',10)
    a['prior_attempt']=m.PRIOR_DEV_RESULT
    with pytest.raises(ValueError):m.validate(a,'train',10)


def mock_storage(tmp_path,*,charged=0,cap=4096):
    state={'entries':0,'writes':0}
    ctx=SimpleNamespace(limits={'max_disk_bytes':cap},check=lambda **_: {'charged_disk_bytes':charged})
    @contextmanager
    def writing(**_):
        state['entries']+=1;yield ctx
    def atomic(path,value):state['writes']+=1;path.write_text(json.dumps(value))
    return SimpleNamespace(metadata_writing=writing),SimpleNamespace(_atomic=atomic,json_write_bound=lambda _:128),state


def test_one_shot_refuses_existing_marker_before_write(tmp_path):
    path=tmp_path/'CONTROL_BUILD_STARTED.json';path.write_text('original')
    metadata,storage,state=mock_storage(tmp_path)
    with pytest.raises(ValueError,match='One-attempt'):m.publish_guarded(metadata,storage,{},path,{})
    assert state['writes']==0 and path.read_text()=='original'


def test_full_root_rejects_result_before_atomic_write(tmp_path):
    path=tmp_path/'CONTROL_BUILD_RESULT.json';metadata,storage,state=mock_storage(tmp_path,charged=4000)
    with pytest.raises(ValueError,match='headroom'):m.publish_guarded(metadata,storage,{},path,{})
    assert state['writes']==0 and not path.exists()


def test_successful_start_cannot_be_reused(tmp_path):
    path=tmp_path/'CONTROL_BUILD_STARTED.json';metadata,storage,state=mock_storage(tmp_path)
    m.publish_guarded(metadata,storage,{},path,{'attempt':1})
    with pytest.raises(ValueError):m.publish_guarded(metadata,storage,{},path,{'attempt':2})
    assert state['writes']==1 and json.loads(path.read_text())=={'attempt':1}


def test_control_hash_remains_mandatory(tmp_path):
    path=tmp_path/'control.json';path.write_text('{}')
    with pytest.raises(ValueError,match='changed'):m.pinned({'path':str(path),'sha256':'0'*64})


@pytest.mark.parametrize('missing_terminal_io',[False,True])
def test_corrupt_final_counter_does_not_cleanup_reaped_leader_twice(tmp_path,monkeypatch,missing_terminal_io):
    # Real local pipes only, no child dispatch/model/cache/ledger access.
    import os
    reads=[]
    for _ in range(2):
        r,w=os.pipe();os.close(w);reads.append(os.fdopen(r,'rb'))
    child=SimpleNamespace(pid=123,stdout=reads[0],stderr=reads[1])
    class Owner:
        verified=1
        def __init__(self):self.finishes=0
        def discover(self):return []
        def leader_done(self):return True
        def send(self,_):pass
        def finish(self,**_):
            self.finishes+=1
            if self.finishes>1:raise FileNotFoundError('already reaped PID')
            return 0
        def close(self):pass
    owner=Owner()
    @contextmanager
    def lifetime():yield {'signal':None}
    supervisor=SimpleNamespace(lifetime_scope=lifetime,pidfd_open=lambda _:os.open('/dev/null',os.O_RDONLY),
        OwnedSession=lambda _:owner,abort_spawn_setup=lambda *_:pytest.fail('already-owned cleanup'))
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:child)
    def io(*_):
        if missing_terminal_io:raise FileNotFoundError('fixture owned IO unavailable')
        return {'rchar':0}
    monkeypatch.setattr(m,'owned_proc_io',io)
    result=m.monitor(['unused'],runtime=str(tmp_path),env={},fd=0,root=tmp_path,s=supervisor,
        m=SimpleNamespace(process_tree_rss=lambda _:0),obs=SimpleNamespace(proc_io=lambda:m.BOOT_IO),budget=m.limits('dev'))
    assert owner.finishes==1
    if missing_terminal_io:
        assert result['charged_owned_child_rchar_bytes']==m.limits('dev')['read_bytes']
        assert result['monitor_error']['type']=='FileNotFoundError'
    else:
        assert result['failure']=='CHILD_ACCOUNTING_INVALID'
        assert result['monitor_error']['type']=='JSONDecodeError'


@pytest.mark.parametrize('case',['success','child_failure','bad_accounting','pipe','read','wall','rss'])
def test_read_only_proc_monitor_real_small_cpu_child(tmp_path,monkeypatch,case):
    import os,sys,time
    helpers=Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1')
    for name,digest in m.MONITORS.items():assert hashlib.sha256((helpers/name).read_bytes()).hexdigest()==digest
    monkeypatch.syspath_prepend(str(helpers))
    import supervise_once as supervisor
    import observed_cli as observer
    code="""import json,sys,time
read=lambda:{k:int(v) for k,v in (x.split(':') for x in open('/proc/self/io'))}
before=read();time.sleep(.35);after=read()
print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':{'status':'GUARDED_METADATA_RETURN'},'counters':{'proc_self_io_baseline':before,'proc_self_io_final':after,'proc_self_io_delta':{k:after[k]-before[k] for k in after}}}))
"""
    if case=='child_failure':code+='raise SystemExit(3)'
    if case=='bad_accounting':code="import time;time.sleep(.35);print('{corrupt')"
    if case=='pipe':code="import sys,time;sys.stdout.write('x'*((1<<20)+2));sys.stdout.flush();time.sleep(.35)"
    budget=m.limits('dev');budget.update(wall_seconds=10,read_bytes=64<<20,rss_bytes=1<<30)
    if case=='read':budget['read_bytes']=1
    if case=='wall':budget['wall_seconds']=0
    monkeypatch.setattr(m,'START',time.monotonic())
    with (tmp_path/'owned-fd').open('wb') as fd:
        r=m.monitor([sys.executable,'-c',code],runtime=str(tmp_path),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
            fd=fd.fileno(),root=tmp_path,s=supervisor,m=SimpleNamespace(process_tree_rss=lambda _:(2<<30) if case=='rss' else 32<<20),
            obs=observer,budget=budget)
    assert r['cleanup']['live_owned_remaining']==0
    assert not list(tmp_path.glob('CONTROL*'))  # observation publishes no sidecars/temp files
    assert r['charged_owned_child_rchar_bytes']>=0
    expected={'success':None,'child_failure':None,'bad_accounting':'CHILD_ACCOUNTING_INVALID',
        'pipe':'STDOUT_LIMIT','read':'PARENT_CHILD_READ_LIMIT','wall':'WALL_LIMIT','rss':'RSS_LIMIT'}[case]
    assert r['failure']==expected
    if case=='child_failure':assert r['returncode']==3
    if case=='success':assert r['returncode']==0 and r['owned_proc_io_terminal'] is not None
