"""CPU metadata mocks only: never SSH, launch GPU or edit the live campaign."""
import copy,json,time
from pathlib import Path
import pytest,yaml
from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha
from tools.d46_priority_rl import world16_owner as w


def put(path,value):path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

@pytest.fixture
def setup(monkeypatch,tmp_path):
    monkeypatch.setattr(w,'ROOT',tmp_path)
    output=tmp_path/'runs/D46_D0_WORLD16';output.mkdir(parents=True)
    old=tmp_path/'runs/D46_D0';old.mkdir(parents=True)
    cfg=dict(arm='D0',global_scenes=64,group=16,inner_updates=2,target_epochs=6.)
    config=tmp_path/'cfg.yaml';config.write_text(yaml.safe_dump(cfg))
    identity=dict(world=16,qualification=False,parent_model_sha256=w.MODEL_SHA,recipe=cfg,native_world_transition=dict(schema='d46_exact_native_READY_world8_to16_v1',old_owner_may_adopt=False,source_qualification=False,target_qualification=False,source_checkpoint=str(old/'checkpoints/update000020_inner0')))
    identity['sha256']=fingerprint(identity);put(output/'identity.json',identity)
    receipt=dict(target_identity_sha256=identity['sha256'],qualification=False);receipt['sha256']=fingerprint(receipt);put(output/'WORLD_TRANSITION.json',receipt)
    science=tmp_path/'SCI';science.mkdir()
    p=dict(schema=w.PLAN,stage_id='D0_stage16',arm='D0',output=str(output),science_source=str(science),source_sha='abc',config=str(config),config_sha256=file_sha(config),identity_sha256=identity['sha256'],transition_receipt_sha256=file_sha(output/'WORLD_TRANSITION.json'),hosts=['training-mock0','training-mock1'],master_addr='172.16.1.1',master_port=29785,target_epochs=6.,worker_source=str(tmp_path/'OPS'),owner_deadline_unix=None)
    path=tmp_path/'PLAN.json';put(path,p)
    control=dict(execution_holds={'D0':'world16 registered'},world16_stages={'D0':dict(plan=str(path),plan_sha256=file_sha(path))})
    put(tmp_path/'operations/EXECUTION_POLICY.json',control)
    campaign=dict(runs={'D0':dict(arm='D0',output=str(old),config=str(config),config_sha256=file_sha(config),source='oldscience',status='HELD',jobs=[],qualification=str(tmp_path/'oldqual'),qualification_jobs=[])},evaluation_world=32,evaluations={})
    put(tmp_path/'CAMPAIGN.json',campaign)
    monkeypatch.setattr(w.subprocess,'check_output',lambda args,**kw:'abc'if'rev-parse'in args else'')
    plan=w.load_plan(path)
    point=dict(checkpoint=str(output/'checkpoints/update000020_inner0'),files_sha256='sealed_old_allstate',updates=20,next_inner=0,phase='READY',valid_scene_exposures=640,scene_cursor=640)
    monkeypatch.setattr(w,'latest',lambda p,cache=None:copy.deepcopy(point))
    monkeypatch.setattr(w,'allowed_hosts',lambda **kw:plan['hosts'])
    monkeypatch.setattr(w,'capacity',lambda host,minimum:list(range(8)))
    calls=[]
    def launch(host,source,job,cmd,gpus,**kw):
        calls.append((host,str(job),cmd,kw));value=dict(host=host,source=str(source),job=str(job),command=cmd,gpus=gpus,at=time.time())
        put(Path(job)/'LAUNCH.json',value);return value
    monkeypatch.setattr(w,'launch',launch)
    return plan,point,calls,campaign,control


def test_hold_and_old_job_exit_are_mandatory(setup,monkeypatch):
    p,_,_,campaign,control=setup
    policy=w.ROOT/'operations/EXECUTION_POLICY.json';put(policy,dict(control,execution_holds={}))
    with pytest.raises(ValueError,match='hold'):w.load_plan(p['_path'])
    put(policy,control);campaign['runs']['D0']['jobs']=[dict(job=str(w.ROOT/'old_live_job'))];put(w.ROOT/'CAMPAIGN.json',campaign)
    with pytest.raises(ValueError,match='actually exit'):w.load_plan(p['_path'])


def test_true_native_resume_no_qualification_cap_and_two_job_reservations(setup):
    p,point,calls,_,_=setup;state=w.initial(p);w.tick(p,state)
    assert len(calls)==2 and state['status']=='RUNNING_OR_ADMITTING'
    for _,_,cmd,kw in calls:
        assert cmd[cmd.index('--resume')+1]==point['checkpoint']
        assert '--stop-after-update'not in cmd and'--qualification'not in cmd and'--standalone'not in cmd
        assert 'tools.d46_world16.train'in cmd and'--nnodes=2'in cmd
        assert kw['share_min_free_mib']is None
    saved=w.read(w.owner_root(p)/'STATE.json')
    assert len(saved['active_jobs'])==2 and saved['reserved_hosts']==p['hosts']
    assert p.get('owner_deadline_unix')is None


def test_actual_failure_preserved_stops_only_own_peer_never_restarts(setup):
    p,_,calls,_,_=setup;state=w.initial(p);w.tick(p,state)
    nodes=state['attempts'][0]['nodes'];put(Path(nodes[1]['job'])/'RESULT.json',dict(returncode=1,reason='actual native failure'))
    w.tick(p,state)
    assert state['status']=='FAILED_REQUIRES_REVIEW'and len(calls)==2
    assert(Path(nodes[0]['job'])/'STOP_REQUESTED').is_file()
    w.tick(p,state);assert len(calls)==2 and len(state['attempts'])==1


def test_resource75_is_prechild_bounded_and_preserves_resume(setup,monkeypatch):
    p,point,calls,_,_=setup;original=w.launch
    def refusal(*args,**kw):
        value=original(*args,**kw);put(Path(value['job'])/'RESULT.json',dict(returncode=75,child_not_started=True,end_unix=time.time()))
        return value
    monkeypatch.setattr(w,'launch',refusal);state=w.initial(p);w.tick(p,state)
    assert len(calls)==1 and state['attempts'][0]['nodes'][1]['phase']=='NOT_LAUNCHED_PEER_REFUSED'
    w.tick(p,state);assert state['status']=='RESOURCE_BACKOFF'and state['resource75_retries']==1
    state['retry_not_before']=0
    # Fixed clock moves past the actual resource refusal; no new scientific state.
    real=time.time;monkeypatch.setattr(w.time,'time',lambda:real()+61);monkeypatch.setattr(w,'launch',original)
    w.tick(p,state)
    assert len(calls)==3 and state['attempts'][-1]['resume']==point['checkpoint']


def test_crash_adopts_first_node_and_launches_only_missing_peer(setup):
    p,point,calls,_,_=setup;state=w.initial(p)
    a=dict(number=1,created=time.time(),resume=point['checkpoint'],nodes=[])
    for i,host in enumerate(p['hosts']):
        job=w.ROOT/'jobs/WORLD16_FORMAL'/p['stage_id']/'attempt001'/f'node{i}'
        a['nodes'].append(dict(host=host,job=str(job),source=p['science_source'],command=w.command(p,i,point['checkpoint']),phase='LAUNCH_INTENT'if i==0 else'NOT_LAUNCHED'))
    node=a['nodes'][0];put(Path(node['job'])/'RUNNING.json',dict(host=node['host'],command=node['command'],cwd=node['source'],start_unix=time.time()))
    state['attempts']=[a];w.tick(p,state)
    assert len(calls)==1 and calls[0][0]==p['hosts'][1]
    assert a['nodes'][0]['receipt']['adopted_actual_worker']and len(state['active_jobs'])==2


def test_main_ingest_same_logical_arm_and_stage_eval_mailbox(setup,monkeypatch):
    p,_,_,campaign,control=setup;state=w.initial(p);w.persist(p,state)
    w.ingest(campaign,control)
    assert list(campaign['runs'])==['D0']and campaign['runs']['D0']['output'].endswith('/D46_D0')
    assert campaign['world16_reserved_hosts']==p['hosts']and campaign['runs']['D0']['world16_stage']['output']==p['output']
    from tools.d46_priority_rl import owner
    assert owner.startup_hosts(campaign)==set(p['hosts'])
    cp=Path(p['output'])/'checkpoints/update002000_inner0';cp.mkdir(parents=True)
    identity=w.read(Path(p['output'])/'identity.json');put(cp/'identity.json',identity)
    put(cp/'files.json',{'identity.json':file_sha(cp/'identity.json')});put(cp/'COMPLETE',dict(world=16,updates=2000,files_sha256=file_sha(cp/'files.json')))
    req=dict(epoch=.5,checkpoint=str(cp),updates=2000,source_run=p['identity_sha256'],seed=2101,protocol='single_B1_cleanx_ODE_K10_shift5_canonical_v1.1')
    put(Path(p['output'])/'evaluation_requests/epoch0.5.json',req)
    state['evaluation_requests']=w.requests(p);w.persist(p,state);w.ingest(campaign,control)
    assert campaign['evaluations']['D0_epoch0.5']['world']==32 and campaign['evaluations']['D0_epoch0.5']['source']==p['science_source']
    summary=w.ROOT/'SUMMARY.json';put(summary,dict(qualification=False,scenes=12146,invalid_scenes=0,PDMS=90.5,paired={'new_zero':12}))
    campaign['evaluations']['D0_epoch0.5'].update(status='COMPLETE',summary=str(summary))
    from tools.d44_native_rl.campaign import selection_index
    selected=selection_index(campaign)
    assert selected['runs']['D0']['evaluated_checkpoints']==1
    assert selected['runs']['D0']['best_score_checkpoint']['checkpoint']==str(cp)
    # The main loop has a stage branch before any legacy dispatch.
    text=Path(owner.__file__).read_text();assert text.index("if run.get('world16_stage'):")<text.index("if run['status'] in ('FAILED_REQUIRES_REVIEW','COMPLETE')")


def test_finite_completion_uses_original_6E_not_fixed_updates(setup):
    p,point,calls,_,_=setup;state=w.initial(p);w.tick(p,state)
    for node in state['attempts'][0]['nodes']:put(Path(node['job'])/'RESULT.json',{'returncode':0})
    point.update(valid_scene_exposures=6*103288,phase='READY',updates=19368,next_inner=0)
    w.tick(p,state)
    assert state['status']=='COMPLETE_AT_REGISTERED_6E'and len(calls)==2 and not state['active_jobs']and not state['reserved_hosts']


def test_evaluation_pool_RL2_only_017_and_stage_reservations(monkeypatch,tmp_path):
    from tools.d44_native_rl import campaign as c
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'allowed_hosts',lambda **kw:['training-rl-zt2'])
    monkeypatch.setattr(c,'free_evaluation_gpus',lambda host,**kw:list(range(8)))
    calls=[]
    def launch(host,source,job,cmd,gpus,**kw):
        calls.append((host,gpus));return dict(host=host,job=str(job),gpus=gpus)
    monkeypatch.setattr(c,'launch',launch)
    state=dict(runs={},evaluations={'D0_epoch1':dict(status='PENDING',world=4,source='frozen',request={'checkpoint':'sealed'},jobs={},output=str(tmp_path/'evaluation'))},world16_stage_jobs=[],world16_reserved_hosts=[])
    c.evaluation_tick(state)
    assert calls==[('training-rl-zt2',[0]),('training-rl-zt2',[1]),('training-rl-zt2',[7])]
    calls.clear();state['world16_reserved_hosts']=['training-rl-zt2']
    state['evaluations']['another']=dict(status='PENDING',world=4,source='frozen',request={'checkpoint':'sealed'},jobs={},output=str(tmp_path/'another'))
    c.evaluation_tick(state);assert calls==[]
