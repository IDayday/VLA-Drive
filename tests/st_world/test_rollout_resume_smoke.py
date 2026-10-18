"""CPU fixtures only; native values here are tiny explicit synthetic tensors."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
import torch

FILE=Path(__file__).resolve().parents[2]/'resources/st_rollout_resume_smoke_v1.py'
SPEC=importlib.util.spec_from_file_location('resume_smoke',FILE)
s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('immutable fixture conflict')
    path.write_text(json.dumps(value,allow_nan=False))


@pytest.fixture
def isolated(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path/'owned_smoke');monkeypatch.setattr(s,'write',put)
    monkeypatch.setattr(s,'PRIOR_ROOT',tmp_path/'synthetic_prior')
    put(s.PRIOR_ROOT/'control/initial_RESULT.json',{'stats':{'sampled_rchar':16348730,'sampled_read_bytes':24576}})
    return tmp_path


def native(run,step,*,identity='fixture'):
    folder=Path(run)/'checkpoints'/f'step_{step:06d}_1';folder.mkdir(parents=True)
    names=[f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt' for rank in range(4)]
    names+=['mp_rank_00_model_states.pt']+[f'rng_rank{rank}.pt' for rank in range(4)]
    for name in names:(folder/name).write_bytes(b'synthetic tiny native marker')
    state={'identity':identity,'completed':step,'tag':folder.name,'pending_feedback':None,'pending_old_policy':None}
    body={'schema':'st_native_checkpoint_v1','world_size':4,'state':state,
        'files':{name:{'bytes':(folder/name).stat().st_size,'sha256':s.sha(folder/name)} for name in names}}
    body['receipt_identity']=s.digest(body);put(folder/'COMPLETE.json',body)
    return folder,body


def test_exact_latest_boundary_never_silently_falls_back(tmp_path):
    run=tmp_path/'run';put(run/'identity.json',{'identity':'fixture'})
    one,_=native(run,2);four,_=native(run,4)
    with pytest.raises(ValueError,match='latest|Latest'):s.selected_native(run,2)
    assert s.selected_native(run,2,require_latest=False)[0]==one
    assert s.selected_native(run,4)[0]==four
    (four/'rng_rank3.pt').unlink()
    assert s.selected_native(run,2)[0]==one
    assert s.selected_native(run,2)[2]


@pytest.mark.parametrize('bad',['identity','partial','extra','pending'])
def test_corrupt_or_foreign_native_refused(tmp_path,bad):
    folder,receipt=native(tmp_path/'run',2)
    if bad=='identity':
        with pytest.raises(ValueError):s.inspect_native(folder,identity='other')
        return
    if bad=='partial':(folder/'mp_rank_00_model_states.pt').write_bytes(b'x')
    elif bad=='extra':(folder/'foreign.pt').write_bytes(b'x')
    else:
        receipt['state']['pending_feedback']='unfinished'
        receipt['receipt_identity']=s.digest({k:v for k,v in receipt.items() if k!='receipt_identity'})
        (folder/'COMPLETE.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):s.inspect_native(folder,completed=2)


def test_full_logical_tensors_preserve_bf16_bytes_stride_and_moments():
    x=torch.tensor([.1015625,.201171875],dtype=torch.bfloat16)
    state={'master':x.float(),'Adam':{'step':torch.tensor(2.),'exp_avg':x.float(),'exp_avg_sq':x.float().square()},'execution':x}
    first=s.logical_state(state);second=s.logical_state(copy.deepcopy(state))
    assert first==second
    state['Adam']['exp_avg'][1]+=1e-6
    assert s.logical_state(state)!=first
    assert s.logical_state(x)['dtype']=='torch.bfloat16'
    assert s.logical_state(torch.arange(6.).view(2,3).t())==s.logical_state(torch.arange(6.).view(2,3).t().contiguous())
    with pytest.raises(ValueError,match='Nonfinite'):s.logical_state(torch.tensor([float('nan')]))
    with pytest.raises(ValueError,match='Unknown'):s.logical_state(object())


class FakeEngine:
    def __init__(self):
        self.module=torch.nn.Linear(2,1);self.master=torch.tensor([.1234567,.89101112])
        self.moment=torch.tensor([.0003,.0004]);self.global_steps=2;self.global_samples=32
        self.skipped_steps=0;self.dp_world_size=4;self.mp_world_size=1;self.lr_scheduler=None
        self.optimizer=SimpleNamespace(state_dict=lambda:{'single_partition_of_fp32_groups':[self.master],
            'base_optimizer_state':{'state':{0:{'step':torch.tensor(2.),'exp_avg':self.moment,'exp_avg_sq':self.moment.square()}},
                'param_groups':[{'params':[0],'lr':2e-7,'betas':(.9,.95)}]},'partition_count':[4],'group_paddings':[0],'zero_stage':2})
        self.saved=None;self.save_count=self.load_count=0
    def save_checkpoint(self,save_dir,tag=None,client_state=None,save_latest=True):
        self.save_count+=1;self.saved=copy.deepcopy(client_state)
        (Path(save_dir)/tag).mkdir(parents=True,exist_ok=True)
        return 'original_save_return'
    def load_checkpoint(self,load_dir,tag=None,load_module_strict=True,load_optimizer_states=True,load_lr_scheduler_states=True,
                        load_module_only=False,custom_load_fn=None):
        self.load_count+=1;return str(Path(load_dir)/tag),copy.deepcopy(self.saved)


@pytest.mark.parametrize('change',['none','moment','rng'])
def test_actual_save_load_rng_witness_and_original_call_counts(isolated,monkeypatch,change):
    engine=FakeEngine();rng={'torch':torch.get_rng_state(),'generators':{'bc_noise':torch.Generator().manual_seed(42).get_state()}}
    calls={'capture':0,'restore':0,'votes':0}
    def capture(g):calls['capture']+=1;return copy.deepcopy(rng)
    def restore(value,g):calls['restore']+=1;rng.clear();rng.update(copy.deepcopy(value));return 'original_restore_return'
    def collective(fn,label):calls['votes']+=1;return fn()
    trainer=SimpleNamespace(capture_rng=capture,restore_rng=restore,collective_call=collective)
    client={'completed':2,'counts':{'samples':64},'tag':'step_000002_1'};root=s.ROOT/'continuous/observer'
    state,uninstall=s.install_observer(trainer,FakeEngine,root,rank=0,resume=True)
    try:
        captured=trainer.capture_rng({})
        assert engine.save_checkpoint(isolated/'native',tag=client['tag'],client_state=client)=='original_save_return'
        client['counts']['samples']=128
        saved=json.loads((root/'SAVE_step_000002_1_rank0.json').read_text())
        assert saved['snapshot']['client_state']['counts']=={'samples':64}
        put(isolated/'native'/client['tag']/'COMPLETE.json',{'state':engine.saved})
        if change=='moment':engine.moment[0]+=1e-7
        if change=='moment':
            with pytest.raises(ValueError,match='Adam'):engine.load_checkpoint(isolated/'native',tag=client['tag'])
            return
        assert engine.load_checkpoint(isolated/'native',tag=client['tag'])[1]['completed']==2
        if change=='rng':captured['torch']=captured['torch'].roll(1)
        if change=='rng':
            with pytest.raises(ValueError,match='Resume RNG'):trainer.restore_rng(captured,{})
        else:
            assert trainer.restore_rng(captured,{})=='original_restore_return'
            assert state['restore_verified'] and (root/'RESTORE_step2_rank0.json').exists()
        assert engine.save_count==engine.load_count==1 and calls['restore']==1
    finally:uninstall()


def test_exact2_pause_occurs_after_original_save_before_complete(isolated,monkeypatch):
    run=s.ROOT/'interrupted/runs/fixture';monkeypatch.setattr(s,'run_path',lambda arm:run)
    put(s.ROOT/'BINDING.json',{'controller_sha256':s.sha(s.__file__)})
    engine=FakeEngine();trainer=SimpleNamespace(capture_rng=lambda g:{'torch':torch.get_rng_state()},restore_rng=lambda *a:None,
        collective_call=lambda fn,label:fn())
    _,uninstall=s.install_observer(trainer,FakeEngine,s.ROOT/'interrupted/observer',rank=0,resume=False)
    try:
        trainer.capture_rng({});engine.save_checkpoint(run/'checkpoints',tag='step_000002_1',client_state={'completed':2})
        assert engine.save_count==1 and (run/'STOP_REQUESTED').exists()
        pause=json.loads((s.ROOT/'PAUSE_REQUEST.json').read_text())
        assert pause['completed']==2 and pause['COMPLETE_seen'] is False
        assert not (run/'checkpoints/step_000002_1/COMPLETE.json').exists()
    finally:uninstall()


def test_observer_local_failure_votes_before_original_save(isolated):
    engine=FakeEngine();votes=[]
    trainer=SimpleNamespace(capture_rng=lambda g:{'torch':torch.get_rng_state()},restore_rng=lambda *a:None,
        collective_call=lambda fn,label:(votes.append(label),fn())[1])
    _,uninstall=s.install_observer(trainer,FakeEngine,s.ROOT/'continuous/observer',rank=0,resume=False)
    try:
        with pytest.raises(ValueError,match='capture'):engine.save_checkpoint(isolated/'native',tag='step_000002_1',client_state={'completed':2})
        assert engine.save_count==0 and votes
    finally:uninstall()


def test_help_status_never_access_real_model_or_dispatch():
    for command in (['--help'],['status']):
        result=subprocess.run(['/root/miniconda3/envs/ddp/bin/python',str(FILE),*command],capture_output=True,text=True,check=True,timeout=10)
        assert 'SMOKE' in result.stdout or 'WAITING' in result.stdout


@pytest.mark.parametrize('fault',['SIGTERM_before_Popen_return','SIGINT_before_Popen_return','LAUNCH_write'])
def test_real_owned_child_fault_has_one_charged_attempt_no_residual(isolated,monkeypatch,fault):
    import os
    import signal
    ep=s.helper();monkeypatch.setattr(ep,'immutable',put);monkeypatch.setattr(s,'helper',lambda:ep)
    binding={'fixture':'explicit synthetic CPU execution'};put(s.ROOT/'BINDING.json',binding)
    route={'gpus':list(range(8)),'gpu_uuids':['fixture'+str(i) for i in range(8)]}
    monkeypatch.setattr(s,'route_checks',lambda *args,**kw:(route,{'fresh_gpu_v3':lambda *a:None}))
    original=subprocess.Popen;spawned=[]
    def popen(*args,**kwargs):
        args=([ '/root/miniconda3/envs/ddp/bin/python','-c','import time;time.sleep(60)'],)
        child=original(*args,**kwargs);spawned.append(child)
        if fault!='LAUNCH_write':os.kill(os.getpid(),signal.SIGTERM if fault.startswith('SIGTERM') else signal.SIGINT)
        return child
    monkeypatch.setattr(s.subprocess,'Popen',popen)
    def write(path,value):
        if fault=='LAUNCH_write' and Path(path).name=='LAUNCH.json':raise OSError('synthetic failed receipt')
        put(path,value)
    monkeypatch.setattr(s,'write',write)
    try:
        result=s.execute('initial',{'fixture':'approved test only'},binding)
        assert result['status']=='PARTIAL_OR_FAILED' and result['invocations_started']==1
        assert len(spawned)==1 and spawned[0].poll() is not None
        paths=list((s.ROOT/'control').glob('invocations/*/*/ADMISSION.json'))
        assert len(paths)==1 and json.loads(paths[0].with_name('EXIT.json').read_text())['owned_cleanup_complete']
    finally:
        for child in spawned:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()


def test_resource_and_scientific_budgets_are_finite_and_no_implicit_retry():
    assert s.LIMITS['total_new_updates']==8 and s.LIMITS['global_calls']==1024
    assert s.LIMITS['retry_calls']==0 and s.LIMITS['max_invocations']==4
    assert s.LIMITS['prior_invocations']==1 and s.LIMITS['new_invocations']==3
    assert s.LIMITS['prior_wall_seconds']+s.LIMITS['wall_seconds']+120+180==3900
    assert s.LIMITS['max_updates']==4096 and s.LIMITS['stop_after']==4
    assert s.LIMITS['output_bytes']==104*2**30 and s.LIMITS['aggregate_RSS_bytes']==512*2**30


def wait_cpu_fixture(predicate):
    import time
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        value=predicate()
        if value:return value
        time.sleep(.02)
    raise AssertionError('Bounded CPU process fixture did not become ready')


def cpu_job(child):
    return {'child':child,'owned':{},'start_ticks':None}


def test_ddp_ownership_real_nested_setsid_and_foreign_gpu_refusal(tmp_path,monkeypatch):
    import os
    import sys
    grand=tmp_path/'grand';child_pid=tmp_path/'rank'
    grand_code=f"import os,time;from pathlib import Path;Path({str(grand)!r}).write_text(str(os.getpid()));time.sleep(60)"
    rank_code=f"import os,subprocess,sys,time;from pathlib import Path;Path({str(child_pid)!r}).write_text(str(os.getpid()));subprocess.Popen([sys.executable,'-c',{grand_code!r}],start_new_session=True);time.sleep(60)"
    root_code=f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{rank_code!r}],start_new_session=True);time.sleep(60)"
    child=subprocess.Popen([sys.executable,'-c',root_code],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    job=cpu_job(child)
    try:
        wait_cpu_fixture(lambda:grand.exists() and child_pid.exists())
        rank=int(child_pid.read_text());nested=int(grand.read_text())
        owned=s.owned_processes(job)
        assert {child.pid,rank,nested}<=owned.keys()
        assert rank not in s.helper().session_members(child.pid,job['start_ticks'])
        assert all(s.process_record(pid)['sid']==pid for pid in (child.pid,rank,nested))
        ep=s.helper();route={'gpu_uuids':['fixture_GPU'],'gpus':[0]}
        def nvml(pid):
            def output(argv,**kw):
                return f'fixture_GPU, {pid}, 1\n' if '--query-compute-apps=gpu_uuid,pid,used_gpu_memory' in argv else 'fixture_GPU, 1\n'
            return output
        monkeypatch.setattr(ep.subprocess,'check_output',nvml(rank))
        ep.gpu_monitor(route,owned) # Actual owned new-session rank is accepted.
        monkeypatch.setattr(ep.subprocess,'check_output',nvml(foreign.pid))
        with pytest.raises(ValueError,match='Unknown/foreign'):ep.gpu_monitor(route,owned)
        s.stop_smoke_owned([job])
        assert job['cleanup_complete'] and child.poll() is not None
        assert not any(row['state']!='Z' for row in s.owned_processes(job).values())
        assert foreign.poll() is None
    finally:
        s.stop_smoke_owned([job])
        if foreign.poll() is None:foreign.terminate()
        foreign.wait(timeout=5)


def test_ddp_ownership_proven_child_session_survives_root_and_leader_exit(tmp_path):
    import os
    import signal
    import sys
    leader_path=tmp_path/'leader';orphan_path=tmp_path/'orphan';go=tmp_path/'go';release=tmp_path/'release'
    # The rank leader is first observed while PPID-proven. It subsequently
    # forks a same-session child and exits; that new orphan is not SID=root.
    leader_code=f"import os,time;from pathlib import Path;Path({str(leader_path)!r}).write_text(str(os.getpid()));\nwhile not Path({str(go)!r}).exists():time.sleep(.01)\npid=os.fork()\nif pid==0:\n Path({str(orphan_path)!r}).write_text(str(os.getpid()));time.sleep(60)\nelse:\n while not Path({str(release)!r}).exists():time.sleep(.01)\n os._exit(0)"
    root_code=f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{leader_code!r}],start_new_session=True);time.sleep(60)"
    child=subprocess.Popen([sys.executable,'-c',root_code],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    job=cpu_job(child)
    try:
        wait_cpu_fixture(leader_path.exists);leader=int(leader_path.read_text())
        assert leader in s.owned_processes(job)
        os.kill(child.pid,signal.SIGTERM);child.wait(timeout=5)
        go.touch();wait_cpu_fixture(orphan_path.exists);orphan=int(orphan_path.read_text())
        owned=wait_cpu_fixture(lambda:(rows if orphan in (rows:=s.owned_processes(job)) else None))
        assert owned[orphan]['sid']==leader and owned[orphan]['ppid']!=child.pid
        release.touch()
        wait_cpu_fixture(lambda:orphan in s.owned_processes(job) and s.process_record(orphan)['ppid']!=leader)
        s.stop_smoke_owned([job]);assert job['cleanup_complete']
        assert not any(row['state']!='Z' for row in s.owned_processes(job).values())
    finally:s.stop_smoke_owned([job])


def test_ddp_ownership_reused_pid_and_exited_uncaptured_root_are_not_seeds(monkeypatch):
    fake=SimpleNamespace(pid=100,poll=lambda:0)
    row={'pid':100,'ppid':1,'start_ticks':200,'sid':100,'state':'S','rss':1}
    monkeypatch.setattr(s,'process_table',lambda:{100:row})
    assert s.owned_processes(cpu_job(fake))=={}
    job={'child':fake,'start_ticks':100,'owned':{100:{**row,'start_ticks':100}},'owned_sessions':{100:100}}
    assert s.owned_processes(job)=={}
    # A historical session integer without any currently verified live member
    # cannot prove a foreign orphan is ours after that SID has been recycled.
    monkeypatch.setattr(s,'process_table',lambda:{200:{**row,'pid':200,'sid':100,'ppid':1,'start_ticks':300}})
    assert s.owned_processes(job)=={}


def test_ddp_ownership_prior_one_plus_partial_intents_consume_total_cap(tmp_path,monkeypatch):
    control=tmp_path/'control'
    for unit in ('initial_continuous','initial_interrupted'):
        (control/'invocations'/unit/'partial').mkdir(parents=True)
    calls=[]
    def claim(root,unit,**kwargs):
        calls.append(unit);path=Path(root)/'invocations'/unit/'partial';path.mkdir(parents=True);return path
    monkeypatch.setattr(s,'helper',lambda:SimpleNamespace(claim_invocation=claim))
    s.claim_smoke_invocation(control,'resume_interrupted',binding_sha='synthetic',argv=['CPU fixture only'])
    assert calls==['resume_interrupted']
    with pytest.raises(ValueError,match='including prior failed launch'):
        s.claim_smoke_invocation(control,'resume_interrupted',binding_sha='synthetic',argv=[])
    assert calls==['resume_interrupted']


def test_ddp_ownership_gpu_snapshot_race_refresh_still_refuses_foreign(monkeypatch):
    calls=[]
    def monitor(route,owned):
        calls.append(set(owned))
        if 123 not in owned:raise ValueError('Unknown/foreign process on endpoint GPU; stopping owned work only')
        return {'fixture_GPU':1}
    ep=SimpleNamespace(gpu_monitor=monitor)
    monkeypatch.setattr(s,'owned_processes',lambda job:{123:{'start_ticks':1}})
    assert s.monitor_smoke_gpu(ep,{},[{'synthetic':'owned lineage'}],{})=={'fixture_GPU':1}
    assert calls==[set(),{123}]
    calls.clear();monkeypatch.setattr(s,'owned_processes',lambda job:{})
    with pytest.raises(ValueError,match='Unknown/foreign'):
        s.monitor_smoke_gpu(ep,{},[{'synthetic':'different lineage'}],{})
    assert calls==[set(),set()]


def test_ddp_ownership_group_ready_uses_cross_session_live_rank(tmp_path,monkeypatch):
    run=tmp_path/'run';identity={'source_sha':s.SOURCE};identity['identity']=s.digest(identity)
    put(run/'identity.json',identity);put(run/'initialization.json',{})
    for rank in range(4):put(run/'official_runtime'/f'initial_engine_rank{rank}.json',{})
    put(run/'status.json',{'identity':identity['identity'],'source_sha':s.SOURCE,'started_unix':2,'pid':123})
    (run/'checkpoints/step_000000_fixture').mkdir(parents=True);put(run/'checkpoints/step_000000_fixture/COMPLETE.json',{})
    monkeypatch.setattr(s,'owned_processes',lambda job:{123:{'synthetic':'new session rank'}})
    monkeypatch.setattr(s,'inspect_native',lambda *a,**kw:{})
    assert s.group_ready(run,{'started_unix':1})
    monkeypatch.setattr(s,'owned_processes',lambda job:{})
    with pytest.raises(ValueError,match='owned invocation'):s.group_ready(run,{'started_unix':1})


def test_ddp_ownership_prior_launch_pins_and_io_are_not_reset(isolated):
    prior=s.PRIOR_ROOT;attempt=prior/'control/invocations/initial_continuous/1'
    result={'stats':{'sampled_rchar':16348730,'sampled_read_bytes':24576},'invocations_started':1,
        'status':'PARTIAL_OR_FAILED','stop_reason':'Unknown/foreign process on endpoint GPU'}
    (prior/'control/initial_RESULT.json').write_text(json.dumps(result))
    for name,value in [('ADMISSION.json',{'fixture':'charged'}),('LAUNCH.json',{'fixture':'actual Popen'}),
                       ('EXIT.json',{'owned_cleanup_complete':True})]:put(attempt/name,value)
    record=s.prior_failed_launch()
    assert record['consumed_trainer_invocations']==1 and record['consumed_optimizer_updates']==0
    assert record['wall_seconds_reserved']==60
    assert s.pinned(record['initial_result'])['stats']==result['stats']
    (prior/'continuous/runs/unknown_actual_state').mkdir(parents=True)
    with pytest.raises(ValueError,match='real run state'):s.prior_failed_launch()


@pytest.mark.parametrize('bad',['valid','observer','runtime','rank','snapshot','native_pin','RNG','foreign_save'])
def test_load_witness_cannot_be_replaced_by_foreign_kind_only(isolated,monkeypatch,bad):
    run=isolated/'run';folder=run/'checkpoints/step_000002_1';put(folder/'COMPLETE.json',{'synthetic':'native fixture'})
    monkeypatch.setattr(s,'selected_native',lambda *a,**k:(folder,{},[]))
    monkeypatch.setattr(s,'consumed_receipt',lambda *a,**k:{'fixture':'no actual metric/cache access'})
    root=s.ROOT/'interrupted/observer';base={'observer_sha256':s.sha(s.__file__),'runtime_sha':s.SOURCE,'rank':0}
    save_path=root/f'SAVE_{folder.name}_rank0.json';save={**base,'snapshot':{'synthetic':'logical byte hash fixture'},'rng':{'synthetic':'saved RNG'}}
    put(save_path,save)
    loaded={**base,'kind':'STRICT_LIVE_NATIVE_MATCH','native_receipt':s.pin(folder/'COMPLETE.json'),
        'save_audit':s.pin(save_path),'snapshot_sha256':s.digest(save['snapshot'])}
    restored={**base,'kind':'EXACT_ACTUAL_RESTORED_RNG','rng':save['rng']}
    if bad=='observer':loaded['observer_sha256']='0'*64
    elif bad=='runtime':loaded['runtime_sha']='0'*40
    elif bad=='rank':restored['rank']=1
    elif bad=='snapshot':loaded['snapshot_sha256']='0'*64
    elif bad=='native_pin':loaded['native_receipt']['sha256']='0'*64
    elif bad=='RNG':restored['rng']={'synthetic':'wrong RNG'}
    elif bad=='foreign_save':
        other=root/'foreign.json';put(other,save);loaded['save_audit']=s.pin(other)
    put(root/'LOAD_step2_rank0.json',loaded);put(root/'RESTORE_step2_rank0.json',restored)
    if bad=='valid':assert s.verify_load_witness(root,run,0)['load']['path'].endswith('LOAD_step2_rank0.json')
    else:
        with pytest.raises(ValueError):s.verify_load_witness(root,run,0)


def test_cleanup_false_cannot_return_execution_complete(isolated,monkeypatch):
    import shutil
    ep=s.helper();monkeypatch.setattr(ep,'immutable',put);monkeypatch.setattr(s,'helper',lambda:ep)
    binding={'fixture':'normal CPU completion with negative cleanup witness'};put(s.ROOT/'BINDING.json',binding)
    route={'gpus':list(range(8)),'gpu_uuids':['fixture'+str(i) for i in range(8)]}
    monkeypatch.setattr(s,'route_checks',lambda *a,**k:(route,{'fresh_gpu_v3':lambda *a:None}))
    monkeypatch.setattr(s,'group_ready',lambda *a:True);monkeypatch.setattr(ep,'gpu_monitor',lambda *a:{})
    monkeypatch.setattr(s,'run_path',lambda arm:s.ROOT/arm/'run')
    put(s.ROOT/'PAUSE_REQUEST.json',{'run':str(s.run_path('interrupted')),'completed':2})
    monkeypatch.setattr(s,'validate_result',lambda *a:{'synthetic':'complete'})
    monkeypatch.setattr(shutil,'disk_usage',lambda *a:SimpleNamespace(free=2**50))
    original=subprocess.Popen;children=[]
    def popen(*a,**kw):
        child=original(['/root/miniconda3/envs/ddp/bin/python','-c','pass'],**kw);children.append(child);return child
    monkeypatch.setattr(s.subprocess,'Popen',popen)
    real_stop=ep.stop_owned
    def incomplete(jobs):
        real_stop(jobs)
        for job in jobs:job['cleanup_complete']=False
    monkeypatch.setattr(ep,'stop_owned',incomplete)
    try:
        result=s.execute('initial',{'fixture':'no actual admission'},binding)
        assert result['status']=='PARTIAL_OR_FAILED' and result['stop_reason']=='OWNED_CLEANUP_UNCONFIRMED'
        assert len(children)==2 and all(child.poll() is not None for child in children)
    finally:real_stop([{'child':child,'start_ticks':None,'owned':{}} for child in children])


@pytest.mark.parametrize('bad',['valid','marker_replaced','source','tag','run','binding'])
def test_only_exact_owned_step2_pause_can_be_acknowledged(isolated,monkeypatch,bad):
    run=s.ROOT/'interrupted/run';run.mkdir(parents=True);folder=run/'checkpoints/step_000002_1'
    monkeypatch.setattr(s,'run_path',lambda arm:run)
    put(s.ROOT/'BINDING.json',{'controller_sha256':s.sha(s.__file__)})
    marker=run/'STOP_REQUESTED';marker.write_text(s.PAUSE_MARKER)
    record={'run':str(run),'completed':2,'source_sha':s.SOURCE,'observer_sha256':s.sha(s.__file__),
        'binding_sha256':s.digest(json.loads((s.ROOT/'BINDING.json').read_text())),
        'native_directory':str(folder),'native_tag':folder.name,'marker':s.pin(marker)}
    if bad=='marker_replaced':marker.write_text('other operator stop; never blindly acknowledge')
    elif bad=='source':record['source_sha']='0'*40
    elif bad=='tag':record['native_tag']='step_000003_1'
    elif bad=='run':record['run']=str(isolated/'other')
    elif bad=='binding':record['binding_sha256']='0'*64
    put(s.ROOT/'PAUSE_REQUEST.json',record)
    if bad=='valid':assert s.validate_pause_marker(run,folder)['path'].endswith('PAUSE_REQUEST.json')
    else:
        with pytest.raises(ValueError):s.validate_pause_marker(run,folder)


def owned_pause_fixture(monkeypatch):
    run=s.ROOT/'interrupted/run';run.mkdir(parents=True);folder=run/'checkpoints/step_000002_1'
    monkeypatch.setattr(s,'run_path',lambda arm:run)
    put(folder/'COMPLETE.json',{'synthetic':'complete native2 receipt'})
    put(s.ROOT/'BINDING.json',{'controller_sha256':s.sha(s.__file__)})
    marker=run/'STOP_REQUESTED';marker.write_text(s.PAUSE_MARKER)
    put(s.ROOT/'PAUSE_REQUEST.json',{'run':str(run),'completed':2,'source_sha':s.SOURCE,'observer_sha256':s.sha(s.__file__),
        'binding_sha256':s.digest(json.loads((s.ROOT/'BINDING.json').read_text())),
        'native_directory':str(folder),'native_tag':folder.name,'marker':s.pin(marker)})
    (s.ROOT/'control').mkdir(exist_ok=True)
    return run,folder,marker


@pytest.mark.parametrize('case',['own_ack','replacement_during_rename','new_stop_after_ack'])
def test_owned_pause_archive_does_not_ack_or_destroy_new_operator_stop(isolated,monkeypatch,case):
    run,folder,marker=owned_pause_fixture(monkeypatch)
    if case=='replacement_during_rename':
        original=Path.rename
        def replace_then_move(path,target):
            if path==marker:path.write_text('new operator stop before rename')
            return original(path,target)
        monkeypatch.setattr(Path,'rename',replace_then_move)
        with pytest.raises(ValueError,match='changed during archive'):s.archive_owned_pause(run,folder)
        assert marker.read_text()=='new operator stop before rename'
        assert (s.ROOT/'control/OWNED_STEP2_STOP_REQUESTED.archived').read_text()==marker.read_text()
        assert not (s.ROOT/'control/OWNED_PAUSE_ACK.json').exists()
        return
    s.archive_owned_pause(run,folder)
    assert not marker.exists()
    if case=='new_stop_after_ack':
        marker.write_text('new operator stop after owned archive')
        with pytest.raises(ValueError,match='New operator STOP'):s.validate_pause_ack(run,folder)
        assert marker.read_text()=='new operator stop after owned archive'
    else:assert s.validate_pause_ack(run,folder)['trainer_acknowledge_stop'] is False


@pytest.mark.parametrize('bad',['valid','same_reviewer','lead_self_review','implementer_self_review'])
def test_two_local_independent_reviews_do_not_require_new_formal_gate(isolated,monkeypatch,bad):
    manifest=isolated/'manifest.json';put(manifest,{'explicit':'synthetic metadata'})
    monkeypatch.setattr(s,'MANIFEST',manifest)
    binding={'source_sha':s.SOURCE,'controller_sha256':s.sha(s.__file__),'limits':s.LIMITS,'root':str(s.ROOT),'manifest':s.pin(manifest)}
    evidence=isolated/'proof.txt';evidence.write_text('synthetic CPU test only')
    reviews={}
    for kind in ('correctness','resources'):
        name='fixture_'+kind
        if bad=='same_reviewer':name='fixture_same'
        elif bad=='lead_self_review' and kind=='resources':name='fixture_lead'
        elif bad=='implementer_self_review' and kind=='resources':name=s.IMPLEMENTER
        path=isolated/(kind+'.json');put(path,{'status':'PASS','review_kind':kind,'reviewer':name,
            'binding_sha256':s.digest(binding),'evidence_artifacts':[s.pin(evidence)]})
        reviews[kind]=s.pin(path)
    path=isolated/'ADMISSION.json';put(path,{'schema':'st_rollout_resume_smoke_admission_v1','status':'APPROVED',
        'issuer':'fixture_lead','binding':binding,'binding_sha256':s.digest(binding),'reviews':reviews})
    if bad=='valid':assert s.admission(path,s.sha(path))[1]==binding
    else:
        with pytest.raises(ValueError):s.admission(path,s.sha(path))


@pytest.mark.parametrize('source',['old','new'])
def test_cold_route_initial_process_imports_real_exact_runtime_metadata(source):
    probe=FILE.parents[1]/'reports/st_world_policy/20261018/rollout_scale_native_resume_smoke_v1/cold_route_v2/cold_route_probe.py'
    result=subprocess.run(['/root/miniconda3/envs/ddp/bin/python','-I',str(probe),'--source',source],
        cwd='/tmp',env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':''},text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    expected='REPRODUCED_ModuleNotFoundError_tools' if source=='old' else 'REAL_FROZEN_ENVIRONMENT_IMPORT_OK'
    assert json.loads(result.stdout)['outcome']==expected
