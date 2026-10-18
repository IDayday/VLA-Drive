"""Synthetic CPU controller faults; no model, image, metric cache or real score."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import pytest

FILE=Path(__file__).resolve().parents[2]/'resources/st_endpoint_supervisor_v1.py'
SPEC=importlib.util.spec_from_file_location('endpoint_supervisor',FILE)
s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)

def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))
    return {'path':str(path),'sha256':s.sha(path)}

def immutable(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Immutable conflict')
        return
    put(path,value)

@pytest.fixture
def isolated(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'OUTPUT',tmp_path/'job');monkeypatch.setattr(s,'ANALYSIS',tmp_path/'analysis')
    monkeypatch.setattr(s,'immutable',immutable)
    (s.OUTPUT/'control').mkdir(parents=True)
    return tmp_path

def admission_fixture(tmp_path,monkeypatch):
    static={'fixture':'frozen'};reference=put(tmp_path/'static.json',static)
    monkeypatch.setattr(s,'STATIC',{'fixture':reference['sha256']})
    binding={'source_sha':s.SOURCE,'run_id':s.RUN_ID,'controller_sha256':s.sha(s.__file__),
             'limits':deepcopy(s.LIMITS),'sealed':{'fixture':reference}}
    report=tmp_path/'proof.txt';report.write_text('Explicitly synthetic review evidence')
    refs={}
    for kind in ('evidence','correctness','resources'):
        refs[kind]=put(tmp_path/(kind+'.json'),{'schema':'st_endpoint_supervisor_review_v1','status':'PASS',
            'review_kind':kind,'binding_sha256':s.digest(binding),'reviewer':'fixture_'+kind,
            'evidence_artifacts':[{'path':str(report),'sha256':s.sha(report)}]})
    value={'schema':'st_rollout_endpoint_admission_v1','status':'APPROVED','issuer':'fixture_lead',
        'binding':binding,'binding_sha256':s.digest(binding),'stable_binding_sha256':s.digest(s.stable_binding(binding)),'reviews':refs}
    ref=put(tmp_path/'ADMISSION.json',value)
    return ref,value

def test_admission_exact_independent_roles_and_nonjson_evidence(tmp_path,monkeypatch):
    reference,receipt=admission_fixture(tmp_path,monkeypatch)
    assert s.validate_admission(reference['path'],reference['sha256'])[1]==receipt['binding']
    Path(receipt['reviews']['resources']['path']).write_text('{}')
    with pytest.raises(ValueError,match='pin'):s.validate_admission(reference['path'],reference['sha256'])

@pytest.mark.parametrize('change',['source','controller','budget','duplicate_reviewer','implementer','placeholder'])
def test_no_stale_or_self_signed_approval(tmp_path,monkeypatch,change):
    reference,receipt=admission_fixture(tmp_path,monkeypatch)
    if change=='source':receipt['binding']['source_sha']='0'*40
    elif change=='controller':receipt['binding']['controller_sha256']='0'*64
    elif change=='budget':receipt['binding']['limits']['score_global_calls']=2000
    elif change=='placeholder':receipt['status']='WAITING_CHECKPOINT'
    else:
        ref=receipt['reviews']['correctness'];review=json.loads(Path(ref['path']).read_text())
        review['reviewer']='fixture_evidence' if change=='duplicate_reviewer' else s.IMPLEMENTER
        receipt['reviews']['correctness']=put(ref['path'],review)
    reference=put(reference['path'],receipt)
    with pytest.raises(ValueError):s.validate_admission(reference['path'],reference['sha256'])

def test_real_small_v5_helpers_reused_without_top_level_dispatch():
    helpers=s.mature_helpers()
    assert set(helpers)=={'sha','check_route_time_v3','fresh_environment_v3','shared_lock_mount_v3','cpu_capacity_v3','fresh_gpu_v3'}
    helpers['check_route_time_v3']({'valid_from_unix':0.,'valid_until_unix':100.},10.,20.)
    with pytest.raises(AssertionError):helpers['check_route_time_v3']({'valid_from_unix':0.,'valid_until_unix':20.},10.,20.)

def test_only_exact_frozen_commands_budget_and_single_model_path():
    root=Path(__file__).resolve().parents[2]
    document=json.loads((root/'reports/st_world_policy/20261018/rollout_scale_endpoint_evaluation_plan_v1/COMMANDS.json').read_text())
    binding={'checkpoint':{'tag':'step_004096_123'},'analysis_plan':{'path':'/fixture/SEALED_PLAN.json'}}
    for rank in range(4):
        argv=s.argv_for('export',{'commands':document},binding,rank)
        assert argv[:7]==['timeout','--signal=TERM','--kill-after=60s','7800s','taskset','-c',f'{rank*2},{rank*2+1}']
        assert argv[argv.index('--rank')+1]==str(rank) and argv[argv.index('--world-size')+1]=='4'
        assert argv[argv.index('--max-scenes')+1]=='1696' and argv[argv.index('--seed')+1]=='42'
        assert argv[argv.index('--checkpoint-tag')+1]=='step_004096_123'
    score=s.argv_for('score',{'commands':document},binding)
    assert score[score.index('--max-calls')+1]=='1728' and score[score.index('--workers')+1]=='4'
    assert score[score.index('--output')+1]==str(s.OUTPUT/'official_score')
    analysis=s.argv_for('analysis',{'commands':document},binding)
    assert analysis[analysis.index('--control-score')+1].endswith('/C1_step004096_dev1696_seed42_v1/official_score')
    assert analysis[analysis.index('--scope')+1]=='FORMAL'

def test_durable_two_invocation_cap_includes_unknown_failed_admissions(isolated):
    root=s.OUTPUT/'control'
    one=s.claim_invocation(root,'score',resume=False,binding_sha='x',argv=['fixture'])
    assert (one/'ADMISSION.json').exists() and not (one/'LAUNCH.json').exists()
    with pytest.raises(ValueError,match='resume'):s.claim_invocation(root,'score',resume=False,binding_sha='x',argv=['fixture'])
    with pytest.raises(ValueError,match='changed'):s.claim_invocation(root,'score',resume=True,binding_sha='other',argv=['fixture'])
    s.claim_invocation(root,'score',resume=True,binding_sha='x',argv=['fixture'])
    with pytest.raises(ValueError,match='exhausted'):s.claim_invocation(root,'score',resume=True,binding_sha='x',argv=['fixture'])
    s.claim_invocation(root,'export_rank3',resume=True,binding_sha='x',argv=['fixture'])

def test_two_loader_slots_release_only_own_load_audit(isolated,monkeypatch):
    ticks=[0];starts=[]
    class Child:
        def __init__(self,rank):self.pid=9000+rank;self.rank=rank;self.returncode=None
        def poll(self):
            if ticks[0]>=5:self.returncode=0
            return self.returncode
        def wait(self):return 0
    def launch(rank):
        starts.append((rank,ticks[0]));attempt=s.OUTPUT/'attempts'/str(rank);attempt.mkdir(parents=True)
        return {'child':Child(rank),'rank':rank,'unit':f'export_rank{rank}','attempt':attempt,'loaded':False,'load_expected':{}}
    monkeypatch.setattr(s,'session_members',lambda pid,*args:{pid:{'rss':1,'start_ticks':pid,'cpu_seconds':0,'state':'R'}})
    monkeypatch.setattr(s,'gpu_monitor',lambda *args:{})
    monkeypatch.setattr(s,'load_finished',lambda rank,owned,**kwargs:ticks[0]>=rank+2)
    monkeypatch.setattr(s.time,'sleep',lambda interval:ticks.__setitem__(0,ticks[0]+1))
    monkeypatch.setattr(s,'stop_owned',lambda jobs:[job.update(cleanup_complete=True) for job in jobs])
    result=s.monitor([],'export',{},pending=[0,1,2,3],launch=launch)
    assert starts==[(0,0),(1,0),(2,2),(3,3)]
    assert result['status']=='PROCESSES_EXITED' and result['invocations_started']==4

def load_fixture(isolated):
    expected={'registration':str(s.TRAIN_REG),'tag':'step_004096_123','run_identity':'a'*64,
        'receipt_identity':'b'*64,'native_receipt_sha256':'c'*64,'native_checkpoint_sha256':'d'*64,
        'selected_scenes_sha256':'f'*64,'manifest_sha256':'1'*64}
    checkpoint={**{key:expected[key] for key in ('tag','run_identity','receipt_identity','native_receipt_sha256','native_checkpoint_sha256')},
        'completed':4096,'parent_checkpoint_sha256':'e'*64,'scope':'FORMAL','weights_usable_for_formal_init':True}
    checkpoint['sha256']=s.digest(checkpoint)
    body={'checkpoint':checkpoint,'source_sha':s.SOURCE,'world_size':4,'seed':42,
        'precision':'FP32_master_eval_TF32_off','solver':'original_Euler_10','candidates_per_scene':1,
        'requested_scenes':1696,'targets_read':False,'teacher_cache_read':False,'scorer':None,
        'selected_scenes_sha256':expected['selected_scenes_sha256'],
        'observations':{'manifest_sha256':expected['manifest_sha256']}}
    signature={**body,'identity':s.digest(body)}
    put(s.OUTPUT/'export/IDENTITY.json',signature)
    invocation={'pid':123,'started_unix':10.,'identity':signature['identity'],'arguments':{'rank':0,'world_size':4,
        'output':str(s.OUTPUT/'export'),'registration':str(s.TRAIN_REG),'checkpoint_tag':expected['tag'],
        'device':'cuda:0','seed':42,'max_scenes':1696}}
    audit={'schema':'st_flow_fp32_restoration_audit_v1','checkpoint_record_sha256':checkpoint['sha256'],
        'parent_checkpoint_sha256':checkpoint['parent_checkpoint_sha256'],'native_checkpoint_sha256':checkpoint['native_checkpoint_sha256'],
        'frozen_encoder_source':'unchanged_original_bound_S0_FP32_masters','training_BF16_encoder_reused':False,
        'original_parent_modified':False,'optimizer_resume_claim':False,'training_feedback_required':False,
        'all_parameters_frozen':True,'strict_action_load':True,'master_dtype':'float32',
        'action_buffers':'unchanged_bound_parent_values_verified_against_native'}
    path=s.OUTPUT/'export/attempts/a'
    put(path/'LOAD_AUDIT.json',audit);put(path/'INVOCATION.json',invocation)
    return expected,path,invocation,audit

@pytest.mark.parametrize('change',['valid','rank','pid','stale','identity','tag','bf16','not_strict','fake_audit','manifest','population'])
def test_load_audit_exact_identity_and_master_provenance(isolated,change):
    expected,path,invocation,audit=load_fixture(isolated)
    if change=='rank':assert not s.load_finished(1,{123:{}},expected=expected);return
    if change=='pid':assert not s.load_finished(0,{456:{}},expected=expected);return
    if change=='stale':assert not s.load_finished(0,{123:{}},expected=expected,after=11.);return
    if change=='identity':invocation['identity']='9'*64
    elif change=='tag':invocation['arguments']['checkpoint_tag']='step_004095_123'
    elif change=='bf16':audit['master_dtype']='bfloat16'
    elif change=='not_strict':audit['strict_action_load']=False
    elif change=='fake_audit':audit={'fixture':'loaded'}
    elif change in ('manifest','population'):
        ref=s.OUTPUT/'export/IDENTITY.json';signature=json.loads(ref.read_text())
        if change=='manifest':signature['observations']['manifest_sha256']='9'*64
        else:signature['selected_scenes_sha256']='9'*64
        signature['identity']=s.digest({k:v for k,v in signature.items() if k!='identity'})
        invocation['identity']=signature['identity'];put(ref,signature)
    put(path/'INVOCATION.json',invocation);put(path/'LOAD_AUDIT.json',audit)
    if change=='valid':assert s.load_finished(0,{123:{}},expected=expected)
    else:
        with pytest.raises(ValueError):s.load_finished(0,{123:{}},expected=expected)

def test_unknown_or_reused_gpu_pid_refuses_without_signaling(monkeypatch):
    monkeypatch.setattr(s.subprocess,'check_output',lambda argv,**k:'GPU-a, 123, 20\n' if '--query-compute-apps=gpu_uuid,pid,used_gpu_memory' in argv else 'GPU-a, 20\n')
    monkeypatch.setattr(s,'process_tree',lambda pid:{pid:{'start_ticks':99}})
    route={'gpu_uuids':['GPU-a'],'gpus':[0]}
    with pytest.raises(ValueError,match='Unknown/foreign'):s.gpu_monitor(route,{})
    with pytest.raises(ValueError,match='Unknown/foreign'):s.gpu_monitor(route,{123:{'start_ticks':1}})
    assert s.gpu_monitor(route,{123:{'start_ticks':99}})=={'GPU-a':20*2**20}

def test_cpu_phase_checks_only_owned_exporters_leaves_new_gpu_users_alone(isolated,monkeypatch):
    folder=s.OUTPUT/'control/invocations/export_rank0/a'
    put(folder/'ADMISSION.json',{'fixture':'charged'})
    with pytest.raises(ValueError,match='Unresolved'):s.ensure_exports_exited(s.OUTPUT/'control')
    put(folder/'EXIT.json',{'owned_cleanup_complete':True,'owned_IDs':{'123':{'start_ticks':7}}})
    monkeypatch.setattr(s,'process_tree',lambda pid:{})
    s.ensure_exports_exited(s.OUTPUT/'control')
    monkeypatch.setattr(s,'process_tree',lambda pid:{pid:{'start_ticks':7,'state':'R'}})
    with pytest.raises(ValueError,match='still alive'):s.ensure_exports_exited(s.OUTPUT/'control')
    monkeypatch.setattr(s,'process_tree',lambda pid:{pid:{'start_ticks':9,'state':'R'}})
    s.ensure_exports_exited(s.OUTPUT/'control')

def test_real_owned_cpu_process_cleanup_on_monitor_fault(isolated,monkeypatch):
    child=subprocess.Popen([sys_executable(),'-c','import time;time.sleep(60)'],start_new_session=True)
    tree=s.process_tree(child.pid);attempt=s.OUTPUT/'attempt';attempt.mkdir()
    job={'child':child,'unit':'score','rank':None,'attempt':attempt,'loaded':False,
         'start_ticks':tree[child.pid]['start_ticks'],'owned':tree}
    monkeypatch.setattr(s,'owned_output_bytes',lambda:2**30+1)
    try:
        result=s.monitor([job],'score',{},pending=[],launch=lambda _:pytest.fail('No additional launch'))
        assert result['status']=='PARTIAL_OR_FAILED' and 'output cap' in result['stop_reason']
        assert child.poll() is not None
        assert json.loads((attempt/'EXIT.json').read_text())['owned_cleanup_complete']
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()

def sys_executable():return '/root/miniconda3/envs/ddp/bin/python'

def test_status_and_help_are_cpu_only_no_runtime_asset_reads():
    for arguments in (['--help'],['--status']):
        completed=subprocess.run([sys_executable(),str(FILE),*arguments],capture_output=True,text=True,timeout=10)
        assert completed.returncode==0,completed.stderr
        if arguments==['--status']:
            value=json.loads(completed.stdout);assert value['status'].startswith('WAITING_CHECKPOINT') and not value['dispatch_started']

def endpoint_fixture(tmp_path,monkeypatch):
    from tools.st_world_policy import check_formal_approval
    monkeypatch.setattr(check_formal_approval,'check_formal_approval',lambda *args,**kwargs:{'formal_launch_allowed':True})
    config={'runtime':{'output_root':str(tmp_path/'runs')},'execution':{'scope':'FORMAL','max_updates':4096,'sample_limit':0}}
    cfg=put(tmp_path/'config.json',config)
    environment=put(tmp_path/'environment.json',{'fixture':'environment'})
    reg=put(tmp_path/'REGISTRATION.json',{'source_sha':s.SOURCE,'run_id':s.RUN_ID,'execution':config['execution'],'artifacts':{'config':cfg,'environment':environment}})
    monkeypatch.setattr(s,'TRAIN_REG',Path(reg['path']));monkeypatch.setattr(s,'TRAIN_REG_SHA',reg['sha256'])
    approval=put(tmp_path/'APPROVAL.json',{'fixture':'approved'})
    body={'source_sha':s.SOURCE,'registration_sha256':reg['sha256'],'config':config,'weights_usable_for_formal_init':True}
    identity=put(tmp_path/'identity.json',{**body,'identity':s.digest(body)})
    launch=put(tmp_path/'LAUNCH.json',{'source_sha':s.SOURCE,'diagnostic_weights_used':False,
        'approval_sha256':approval['sha256'],'registration_sha256':reg['sha256']})
    tag='step_004096_123';folder=Path(config['runtime']['output_root'])/s.RUN_ID/'checkpoints'/tag
    folder.mkdir(parents=True)
    files={}
    names=['mp_rank_00_model_states.pt']+[f'bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(4)]+[f'rng_rank{r}.pt' for r in range(4)]
    for name in names:
        (folder/name).write_bytes(b'explicit fixture, not model state');files[name]={'bytes':(folder/name).stat().st_size,'sha256':s.sha(folder/name)}
    checkpoint={'schema':'st_native_checkpoint_v1','world_size':4,'files':files,'state':{
        'completed':4096,'policy_version':4096,'tag':tag,'identity':s.digest(body),
        'pending_old_policy':None,'pending_feedback':None,'termination_reason':'MAX_STEPS'}}
    checkpoint['receipt_identity']=s.digest(checkpoint)
    native=put(folder/'COMPLETE.json',checkpoint)
    rows=[{'token':f'fixture{i}','log':f'log{i%16}'} for i in range(1696)]
    manifest=put(tmp_path/'MANIFEST.json',{'split':'dev','scenes':rows})
    index=put(tmp_path/'INDEX.json',rows)
    training={'registration':reg,'approval':approval,'identity':identity,'launch':launch}
    draft={'status':'WAITING','formal_launch_allowed':False,'formal_runs':{'C1':{'fixture':'fixed_old'}},
        'execution_amendment':{'status':'PROPOSED_NOT_APPROVED','world':4},
        'score_protocol_sha256':'db383e17b887e08beb358b282fbeab223c80c558164bead81d52a45b1b798143','statistics':{'fixed':True}}
    plan={**draft,'status':'APPROVED','formal_launch_allowed':True,'execution_amendment':{**draft['execution_amendment'],'status':'APPROVED_WORLD4_EXECUTION'},
        'formal_runs':{**draft['formal_runs'],'C1_ROLLOUT_STD':{'registration':reg,'approval':approval,'run_identity_file':identity,'launch':launch}}}
    plan_ref=put(tmp_path/'PLAN.json',plan)
    old=put(tmp_path/'OLD_COMPLETE.json',{'status':'COMPLETE','requested_scenes':1696,'valid_scored':1696,'invalid':0,'unresolved':0,'scenes_csv_sha256':'f'*64})
    binding={'training':training,'environment':environment,'checkpoint':{'tag':tag,'complete':native,'receipt_identity':checkpoint['receipt_identity']},'analysis_plan':plan_ref}
    documents={'fixed_inputs':{'pins':{'DEV_manifest':manifest,'DEV_metric_index':index,'old_score_COMPLETE':old},'legacy_csv_sha256':'f'*64},'analysis_template':draft}
    return binding,documents,folder

def test_exact_endpoint_complete_population_and_no_early_fallback(tmp_path,monkeypatch):
    binding,documents,folder=endpoint_fixture(tmp_path,monkeypatch)
    actual=s.validate_endpoint(binding,documents)
    assert len(actual['scenes'])==1696
    binding['checkpoint']['tag']='step_004095_123'
    with pytest.raises(ValueError,match='exact4096'):s.validate_endpoint(binding,documents)

def test_waiting_endpoint_receipt_and_stale_inventory(tmp_path,monkeypatch):
    binding,documents,folder=endpoint_fixture(tmp_path,monkeypatch)
    complete=(folder/'COMPLETE.json').read_bytes();(folder/'COMPLETE.json').unlink()
    with pytest.raises(s.WaitingCheckpoint):s.validate_endpoint(binding,documents)
    (folder/'COMPLETE.json').write_bytes(complete)
    (folder/'stale_model_states.pt').write_bytes(b'fixture stale')
    with pytest.raises(ValueError,match='inventory'):s.validate_endpoint(binding,documents)

def test_final_scientific_plan_cannot_change_thresholds_or_control(tmp_path,monkeypatch):
    binding,documents,_=endpoint_fixture(tmp_path,monkeypatch)
    plan=json.loads(Path(binding['analysis_plan']['path']).read_text());plan['statistics']['fixed']=False
    binding['analysis_plan']=put(binding['analysis_plan']['path'],plan)
    with pytest.raises(ValueError,match='scientific'):s.validate_endpoint(binding,documents)

def route_fixture(tmp_path,monkeypatch):
    from types import SimpleNamespace
    now=time.time()
    route={'hostname':'fixture-host','host':'training-vlawm-tsinghua-zt6','gpus':[0,1,2,3],
        'gpu_uuids':[f'GPU-{rank:08x}-0000-0000-0000-000000000000' for rank in range(4)],
        'valid_from_unix':now-1,'valid_until_unix':now+9000}
    policy=put(tmp_path/'POLICY.json',{'allowed_hosts':{'training-vlawm-tsinghua-zt6':{'aliases':['fixture-host']}},
        'allowed_aliases':['fixture-host'],'global_dispatch_prohibitions':{}})
    auth=put(tmp_path/'AUTH.json',{'allowed_hosts':['training-vlawm-tsinghua-zt6']})
    monkeypatch.setattr(s,'DISPATCH_POLICY',Path(policy['path']));monkeypatch.setattr(s,'AUTHORIZATION',Path(auth['path']))
    storage=put(tmp_path/'STORAGE.json',{'observed_unix':now,'future_reserved_bytes':2**30})
    binding={'route':route,'dispatch_policy':policy,'authorization':auth,'environment':{},'storage_reservations':storage}
    timing=s.mature_helpers()['check_route_time_v3']
    helpers={'check_route_time_v3':timing,'fresh_environment_v3':lambda ref:None,'shared_lock_mount_v3':lambda:None,
             'cpu_capacity_v3':lambda:{'quota_us':-1,'period_us':1}}
    monkeypatch.setattr(s.socket,'gethostname',lambda:'fixture-host')
    monkeypatch.setattr(s.os,'sched_getaffinity',lambda pid:set(range(12)))
    monkeypatch.setattr(s.subprocess,'check_output',lambda args,**kwargs:s.SOURCE+'\n' if 'rev-parse' in args else '')
    monkeypatch.setattr(s.shutil,'disk_usage',lambda root:SimpleNamespace(free=s.LIMITS['shared_free_floor']+3*2**30))
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda path,*a,**k:'MemAvailable: 1073741824 kB\n' if str(path)=='/proc/meminfo' else original(path,*a,**k))
    return binding,helpers

@pytest.mark.parametrize('change',['valid','placeholder_uuid','expired','stale_storage','quota','foreign_host'])
def test_fresh_route_resources_no_placeholders_or_unconfirmed_capacity(tmp_path,monkeypatch,change):
    binding,helpers=route_fixture(tmp_path,monkeypatch)
    if change=='placeholder_uuid':binding['route']['gpu_uuids'][0]='__LATER__'
    elif change=='expired':binding['route']['valid_until_unix']=time.time()+1
    elif change=='stale_storage':
        binding['storage_reservations']=put(binding['storage_reservations']['path'],{'observed_unix':time.time()-301,'future_reserved_bytes':0})
    elif change=='quota':helpers['cpu_capacity_v3']=lambda:{'quota_us':7000,'period_us':1000}
    elif change=='foreign_host':binding['route']['host']='forbidden'
    if change=='valid':assert s.route_checks(binding,'export',helpers)==binding['route']
    else:
        with pytest.raises((ValueError,AssertionError)):s.route_checks(binding,'export',helpers)

def test_new_32GiB_driver_ceiling_includes_context_overhead(monkeypatch):
    def query(args,**kwargs):
        return 'GPU-a, 123, 1\n' if '--query-compute-apps=gpu_uuid,pid,used_gpu_memory' in args else 'GPU-a, 32769\n'
    monkeypatch.setattr(s.subprocess,'check_output',query)
    monkeypatch.setattr(s,'process_tree',lambda pid:{pid:{'start_ticks':99}})
    with pytest.raises(ValueError,match='driver usage'):
        s.gpu_monitor({'gpu_uuids':['GPU-a'],'gpus':[0]},{123:{'start_ticks':99}})

def test_partial_empty_attempt_directory_still_consumes_budget(isolated):
    root=s.OUTPUT/'control';binding={'fixture':'stable'}
    put(root/'BINDING.json',binding)
    empty=root/'invocations/score/0001';empty.mkdir(parents=True)
    assert not (empty/'ADMISSION.json').exists()
    s.claim_invocation(root,'score',resume=True,binding_sha=s.digest(binding),argv=['fixture'])
    with pytest.raises(ValueError,match='exhausted'):
        s.claim_invocation(root,'score',resume=True,binding_sha=s.digest(binding),argv=['fixture'])
    assert empty.exists()

def test_fresh_storage_after_five_minutes_does_not_change_output_identity(isolated):
    root=s.OUTPUT/'control'
    old={'source_sha':s.SOURCE,'checkpoint':{'tag':'fixed4096'},'route':{'uuid':'fixed','lease':'fixed'},
         'storage_reservations':{'path':'/fixture/old','sha256':'a'*64}}
    fresh={**old,'storage_reservations':{'path':'/fixture/new','sha256':'b'*64}}
    stable=s.stable_binding(old)
    assert stable==s.stable_binding(fresh) and s.digest(old)!=s.digest(fresh)
    immutable(root/'BINDING.json',stable)
    s.claim_invocation(root,'score',resume=False,binding_sha=s.digest(stable),argv=['fixture'])
    immutable(root/'BINDING.json',s.stable_binding(fresh))
    s.claim_invocation(root,'score',resume=True,binding_sha=s.digest(s.stable_binding(fresh)),argv=['fixture'])
    changed={**fresh,'checkpoint':{'tag':'another'}}
    with pytest.raises(ValueError,match='Immutable'):immutable(root/'BINDING.json',s.stable_binding(changed))

@pytest.mark.parametrize('reason',['MAX_STEPS','RUNNING_CHECKPOINT','BUDGET_PAUSED'])
def test_exact_complete4096_reuses_mature_terminal_contract(tmp_path,monkeypatch,reason):
    binding,documents,folder=endpoint_fixture(tmp_path,monkeypatch)
    complete=json.loads((folder/'COMPLETE.json').read_text());complete['state']['termination_reason']=reason
    complete.pop('receipt_identity');complete['receipt_identity']=s.digest(complete)
    binding['checkpoint']['complete']=put(folder/'COMPLETE.json',complete)
    binding['checkpoint']['receipt_identity']=complete['receipt_identity']
    assert s.validate_endpoint(binding,documents)['receipt_identity']==complete['receipt_identity']

def cpu_session(tmp_path,*,threaded=False,fast_exit=False):
    pid_path=tmp_path/'child.pid'
    spawn=("import subprocess,time,pathlib; child=subprocess.Popen(["+repr(sys_executable())+",'-c','import time;time.sleep(60)']);"
           "pathlib.Path("+repr(str(pid_path))+").write_text(str(child.pid)); time.sleep("+("0" if fast_exit else "60")+")")
    code=("import threading,time; t=threading.Thread(target=lambda:exec("+repr(spawn)+"));t.start();time.sleep(60)") if threaded else spawn
    child=subprocess.Popen([sys_executable(),'-c',code],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    initial=s.process_tree(child.pid);start=initial.get(child.pid,{}).get('start_ticks')
    deadline=time.monotonic()+5
    while not pid_path.exists() and time.monotonic()<deadline:time.sleep(.01)
    assert pid_path.exists()
    return child,int(pid_path.read_text()),start,initial

@pytest.mark.parametrize('case',['thread_child','reparented_child'])
def test_complete_owned_session_cleanup_handles_thread_and_root_exit(isolated,case):
    child,descendant,start,initial=cpu_session(isolated,threaded=case=='thread_child',fast_exit=case=='reparented_child')
    job={'child':child,'start_ticks':start,'owned':initial}
    try:
        if case=='thread_child':assert descendant in s.process_tree(child.pid)
        else:child.wait(timeout=5);assert descendant in s.session_members(child.pid,start)
        s.stop_owned([job])
        assert job['cleanup_complete'] and child.poll() is not None
        row=s.process_tree(descendant).get(descendant)
        assert not row or row['state']=='Z'
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
        try:
            row=s.process_tree(descendant).get(descendant)
            if row and row['state']!='Z':os.kill(descendant,signal.SIGKILL)
        except ProcessLookupError:pass

@pytest.mark.parametrize('fault',['launch_receipt','log_open','log_close','Popen_SIGTERM','Popen_SIGINT'])
def test_log_and_post_Popen_receipt_failure_preserves_admission_and_cleanup(isolated,monkeypatch,fault):
    binding,documents,_=endpoint_fixture(isolated/'source',monkeypatch)
    route={'gpus':[0,1,2,3],'gpu_uuids':['fixture0','fixture1','fixture2','fixture3']}
    helpers={'check_route_time_v3':lambda *args:None,'fresh_gpu_v3':lambda *args:None}
    monkeypatch.setattr(s,'mature_helpers',lambda:helpers)
    monkeypatch.setattr(s,'route_checks',lambda *args:route)
    monkeypatch.setattr(s,'gpu_monitor',lambda *args:{})
    monkeypatch.setattr(s,'argv_for',lambda *args:[sys_executable(),'-c','import time;time.sleep(60)'])
    original=s.subprocess.Popen;spawned=[]
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def popen(*args,**kwargs):
        child=original(*args,**kwargs);spawned.append(child)
        if fault.startswith('Popen_SIG'):
            os.kill(os.getpid(),signal.SIGTERM if fault=='Popen_SIGTERM' else signal.SIGINT)
        return child
    monkeypatch.setattr(s.subprocess,'Popen',popen)
    def fail_launch(path,value):
        if fault=='launch_receipt' and Path(path).name=='LAUNCH.json':raise OSError('fixture publication failure after Popen')
        immutable(path,value)
    monkeypatch.setattr(s,'immutable',fail_launch)
    original_open=Path.open
    class CloseFault:
        def __init__(self,stream):self.stream=stream
        def fileno(self):return self.stream.fileno()
        def close(self):self.stream.close();raise OSError('fixture close failure after Popen')
    def log_fault(path,*args,**kwargs):
        if path.name=='OUTPUT.log':
            if fault=='log_open':raise OSError('fixture log open failure before Popen')
            stream=original_open(path,*args,**kwargs)
            return CloseFault(stream) if fault=='log_close' else stream
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',log_fault)
    try:
        result=s.execute('export',{'fixture':'lead'},binding,documents,resume=False)
        count=0 if fault=='log_open' else 1
        assert result['status']=='PARTIAL_OR_FAILED' and result['invocations_started']==count
        assert len(spawned)==count and all(child.poll() is not None for child in spawned)
        if fault.startswith('Popen_SIG'):
            signum=signal.SIGTERM if fault=='Popen_SIGTERM' else signal.SIGINT
            assert result['stop_reason']=='RuntimeError: OPERATOR_SIGNAL_'+str(int(signum))
            assert all(signal.getsignal(sig)==handler for sig,handler in previous_handlers.items())
        assert len(list((s.OUTPUT/'control/invocations/export_rank0').glob('*/ADMISSION.json')))==1
        exited=list((s.OUTPUT/'control/invocations/export_rank0').glob('*/EXIT.json'))
        assert len(exited)==1 and json.loads(exited[0].read_text())['owned_cleanup_complete']
    finally:
        for child in spawned:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()

def test_nvidia_queries_have_bounded_timeout(monkeypatch):
    calls=[]
    def query(args,**kwargs):
        calls.append(kwargs)
        return '' if '--query-compute-apps=gpu_uuid,pid,used_gpu_memory' in args else 'GPU-a, 0\n'
    monkeypatch.setattr(s.subprocess,'check_output',query)
    assert s.gpu_monitor({'gpus':[0],'gpu_uuids':['GPU-a']},{})=={'GPU-a':0}
    assert calls and all(row['timeout']==15 for row in calls)
