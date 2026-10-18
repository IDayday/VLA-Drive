"""Synthetic rejection tests. No real process, GPU or run is dispatched."""
import ast
import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.st_world_policy import check_formal_approval as gate


LAUNCHER = Path('/mnt/project/st-world-policy-20261016/resources/launch_reviewed_rollout_scale_formal_v1.py')
MONITOR = Path('/mnt/project/st-world-policy-20261016/resources/monitor_rollout_scale_formal_v1.py')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launch():
    return load(LAUNCHER, 'independent_formal_launcher')


def reserve(m):
    return {'run_id':m.RUN_ID, 'registration_sha256':m.REG_SHA,
            'own_run_reserve_bytes':209*m.GIB,'shared_free_floor_bytes':1024*m.GIB,
            'unresolved_old_remaining_bytes':4*m.GIB,
            'other_active_remaining_writes':[{'bytes':9*m.GIB}]}


def test_native_directory_ceiling_is_remaining_writes_not_step_reset(launch):
    assert launch.native_admission([],0,False)['maximum_directories'] == 11
    r=launch.native_admission(['zero','512'],512,True)
    assert r['initial_writes']==0 and r['remaining_periodic_steps']==list(range(1024,4097,512))
    assert launch.native_admission([str(i) for i in range(12)],4000,True)['maximum_directories']==15
    with pytest.raises(ValueError): launch.native_admission([str(i) for i in range(13)],4000,True)
    with pytest.raises(ValueError): launch.native_admission([],-1,True)
    with pytest.raises(ValueError): launch.native_admission([],4097,True)


@pytest.mark.parametrize('bad', ['free','identity','bool','negative'])
def test_storage_refuses_invalid_reservations(launch,bad):
    r=reserve(launch); need=(209+1024+4+9)*launch.GIB
    assert launch.validate_storage(r,need)==need
    if bad=='free': free=need-1
    else:
        free=need
        if bad=='identity': r['registration_sha256']='wrong'
        if bad=='bool': r['other_active_remaining_writes'][0]['bytes']=True
        if bad=='negative': r['other_active_remaining_writes'][0]['bytes']=-1
    with pytest.raises(ValueError): launch.validate_storage(r,free)


@pytest.mark.parametrize('kind', ['startup_zero','busy_memory','wrong_uuid','bad_query'])
def test_gpu_admission_refuses_occupied_or_ambiguous_devices(launch,monkeypatch,kind):
    monkeypatch.setattr(launch.os,'sched_getaffinity',lambda _:set(range(16)))
    uuids=['GPU-'+str(i) for i in range(4)]
    def query(cmd,**kwargs):
        if '--query-compute-apps=gpu_uuid,pid' in cmd:
            return uuids[0]+', 12345\n'  # A real startup PID despite 0 MiB and 0%.
        if kind=='bad_query': return 'malformed\n'
        return '\n'.join(f'{i}, {"WRONG" if kind=="wrong_uuid" and i==0 else u}, {128 if kind=="busy_memory" else 0}, 0' for i,u in enumerate(uuids))
    monkeypatch.setattr(launch.subprocess,'check_output',query)
    with pytest.raises((RuntimeError,ValueError,IndexError)):
        launch.fresh_resources('0,1,2,3',uuids,reserve(launch))


def setup_main(m,monkeypatch,tmp_path,*,resume=False,attempts=0,completed=0,control_bad=None):
    root=tmp_path/'root';m.ROOT=root
    folder=root/'formal/consequence/registrations'/m.RUN_ID;folder.mkdir(parents=True)
    run=root/'formal/consequence/runs'/m.RUN_ID
    cfg={'execution':{'scope':'FORMAL','max_updates':4096,'stop_after':4096,'world_size':4,
                     'global_batch':32,'micro_batch':4,'sample_limit':0},
         'runtime':{'max_seconds':28800,'save_every':512,'profile_mode':'none',
                    'output_root':str(run.parent)},'fpo':{'advantage':'scene_center_rollout_std_v1'}}
    lease={'hostnames':[m.HOST],'valid_from_unix':0,'valid_until_unix':1e20,
           'cuda_visible_devices_by_host':{m.HOST:'0,1,2,3'},
           'gpu_uuids_by_host':{m.HOST:['GPU-'+str(i) for i in range(4)]}}
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    cfgpath=folder/'CONFIG.json';leasepath=folder/'LEASE.json';regpath=folder/'REGISTRATION.json'
    write(cfgpath,cfg);write(leasepath,lease)
    write(regpath,{'run_id':m.RUN_ID,'source_sha':m.SOURCE,'artifacts':{'config':{'path':str(cfgpath)},'resource_lease':{'path':str(leasepath)}}})
    reservespath=folder/'RESERVES.json';write(reservespath,reserve(m))
    def art(p):return {'path':str(Path(p).resolve()),'sha256':'a'*64}
    controls=[art(p) for p in [m.__file__,root/'resources/monitor_rollout_scale_formal_v1.py',root/'resources/observe_formal_flow_cpu_v1.py',reservespath]]
    if control_bad is not None:controls[control_bad]['sha256']='b'*64
    for role in ['evidence','correctness','efficiency']:
        write(folder/'reviews'/f'{role.upper()}.json',{'status':'PASS','fingerprint':{'fixture':1},'evidence_artifacts':controls})
    for i in range(attempts): (folder/'invocations'/str(i)).mkdir(parents=True)
    if resume:
        run.mkdir(parents=True);(run/'checkpoints').mkdir()
        write(run/'identity.json',{'registration_sha256':m.REG_SHA,'identity':'id'})
        write(folder/'LAUNCH.json',{'existing':True})
    real_read=m.read
    def read(p):
        if str(p)=='/mnt/project/server_dispatch_policy.json':return {'allowed_aliases':[m.HOST]}
        if Path(p)==root/'resources/AUTHORIZATION_20261016T203442.json':return {'allowed_hosts':[m.HOST.removesuffix('-worker-0')]}
        return real_read(p)
    monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m,'artifact',art)
    monkeypatch.setattr(m,'file_hash',lambda p:m.REG_SHA)
    monkeypatch.setattr(m,'validate_config',lambda cfg:None)
    monkeypatch.setattr(m,'verify_registration',lambda *a,**k:{'fixture':1})
    monkeypatch.setattr(m,'check_formal_approval',lambda *a,**k:{'fixture':'passed'})
    monkeypatch.setattr(m.socket,'gethostname',lambda:m.HOST)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:m.SOURCE+'\n')
    def forbid(*a,**k):raise AssertionError('TEST_FORBIDS_REAL_DISPATCH')
    monkeypatch.setattr(m.subprocess,'Popen',forbid)
    monkeypatch.setattr(m,'resolve_latest',lambda *a,**k:{'selected':{'receipt':{'state':{'completed':completed,'termination_reason':'MAX_STEPS' if completed==4096 else 'RUNNING_CHECKPOINT'}}}})
    def stop_before_resources(*a,**k):raise RuntimeError('TEST_RESOURCE_REJECTION')
    monkeypatch.setattr(m,'fresh_resources',stop_before_resources)
    def safe_open(p,*a,**k):
        if str(p).startswith('/tmp/foresight_gpu_'):p=tmp_path/Path(p).name
        return builtins.open(p,*a,**k)
    monkeypatch.setattr(m,'open',safe_open,raising=False)
    argv=['launcher','--registration',str(regpath),'--remaining-write-reserves',str(reservespath)]
    if resume:argv.append('--resume')
    monkeypatch.setattr(m.sys,'argv',argv)
    return folder,run


@pytest.mark.parametrize('index',range(4))
def test_each_external_control_hash_must_match_review(launch,monkeypatch,tmp_path,index):
    setup_main(launch,monkeypatch,tmp_path,control_bad=index)
    with pytest.raises(AssertionError,match='Final controls'):launch.main()


def test_six_attempt_limit_counts_partial_directories(launch,monkeypatch,tmp_path):
    folder,_=setup_main(launch,monkeypatch,tmp_path,resume=True,attempts=6,completed=2048)
    assert not list(folder.glob('invocations/*/LAUNCH.json'))
    with pytest.raises(AssertionError,match='Six-invocation'):launch.main()


def test_terminal_resume_never_reaches_resources_or_dispatch(launch,monkeypatch,tmp_path):
    setup_main(launch,monkeypatch,tmp_path,resume=True,attempts=1,completed=4096)
    with pytest.raises(RuntimeError,match='Run is terminal'):launch.main()


def test_resource_mismatch_fails_without_creating_an_invocation(launch,monkeypatch,tmp_path):
    folder,_=setup_main(launch,monkeypatch,tmp_path)
    with pytest.raises(RuntimeError,match='TEST_RESOURCE_REJECTION'):launch.main()
    assert not (folder/'invocations').exists()


@pytest.mark.parametrize('bad',['wrong_role','same_reviewer','bad_status','stale_review','bad_evidence_hash'])
def test_mature_three_review_gate_rejects_invalid_signature(monkeypatch,tmp_path,bad):
    fingerprint={'execution':{'scope':'FORMAL'},'fixture':True}
    reg=tmp_path/'registration.json';reg.write_text(json.dumps({'implementer_agents':['/root']}))
    evidence=tmp_path/'evidence.txt';evidence.write_text('bounded synthetic evidence')
    records={}
    for i,role in enumerate(gate.REVIEW_ROLES):
        review={'status':'PASS','role':role,'fingerprint':fingerprint,'reviewer_agent':'/reviewer/'+role,
                'evidence_artifacts':[{'path':str(evidence),'sha256':gate.file_hash(evidence)}]}
        if role=='correctness':
            if bad=='wrong_role':review['role']='efficiency'
            if bad=='same_reviewer':review['reviewer_agent']='/reviewer/evidence'
            if bad=='bad_status':review['status']='FAIL'
            if bad=='stale_review':review['fingerprint']={'other':True}
            if bad=='bad_evidence_hash':review['evidence_artifacts'][0]['sha256']='0'*64
        p=tmp_path/(role+'.json');p.write_text(json.dumps(review));records[role]={'path':str(p),'sha256':gate.file_hash(p)}
    approval=tmp_path/'approval.json';approval.write_text(json.dumps({'schema':'st_world_approval_v1','fingerprint':fingerprint,'reviews':records,'formal_launch_allowed':True}))
    monkeypatch.setattr(gate,'verify_registration',lambda *a,**k:fingerprint)
    with pytest.raises(gate.ApprovalError):gate.check_formal_approval(reg,approval,repo=tmp_path)


@pytest.mark.parametrize('reason',['gpu_other','memory','native','monitor_error','shared_free','reuse_storage_sample'])
def test_monitor_only_requests_owned_stop_and_never_signals(monkeypatch,tmp_path,reason):
    m=load(MONITOR,'independent_formal_monitor')
    expected=tmp_path/'formal';run=expected/'runs/C1_rollout_std_fulltrain_seed42_official_4096_v1'
    invocation=expected/'registrations/C1_rollout_std_fulltrain_seed42_official_4096_v1/invocations/1'
    run.mkdir(parents=True);invocation.mkdir(parents=True)
    reservations={'shared_free_floor_bytes':1024*m.GIB,'unresolved_old_remaining_bytes':4*m.GIB,
                  'other_active_remaining_writes':[{'bytes':292*m.GIB}]}
    (invocation/'LAUNCH.json').write_text(json.dumps({'host':'fixture','pid':123,'output':str(run),'scope':'FORMAL','source_sha':'fixed',
                                                   'storage_reservations':reservations}))
    realpath=Path
    monkeypatch.setattr(m,'Path',lambda p:expected if str(p)=='/mnt/project/st-world-policy-20261016/formal/consequence' else realpath(p))
    monkeypatch.setattr(m.socket,'gethostname',lambda:'fixture')
    monkeypatch.setattr(m,'stat',lambda pid:{'start_ticks':456})
    monkeypatch.setattr(m,'subtree',lambda *a:({123,124},257*m.GIB if reason=='memory' else m.GIB))
    def query(*a,**k):
        if reason=='monitor_error':raise RuntimeError('fixture read error')
        return 'GPU-0, '+('999' if reason=='gpu_other' else '124')+', 0\n'
    monkeypatch.setattr(m.subprocess,'check_output',query)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=(1319 if reason=='shared_free' else 2048)*m.GIB))
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    scans=[]
    def bytes_fixture(p):
        scans.append(str(p))
        return 1024
    monkeypatch.setattr(m,'directory_bytes',bytes_fixture)
    monkeypatch.setattr(m.os,'kill',lambda *a:pytest.fail('Monitor must not signal'))
    monkeypatch.setattr(m.os,'killpg',lambda *a:pytest.fail('Monitor must not signal'))
    if reason=='native':
        for i in range(16):(run/'checkpoints'/str(i)).mkdir(parents=True)
    class Child:
        pid=123;polls=0
        def poll(self):self.polls+=1;return None if self.polls<=2 else 0
        def wait(self):return 0
    result=m.wait_owned(Child(),invocation,run,['GPU-0'])
    if reason=='reuse_storage_sample':
        assert result['stop_request'] is None
        assert not (run/'STOP_REQUESTED').exists()
        assert len(scans)==5  # Initial run+registration+logs, no repeat before 300s, final run+registration.
        assert result['storage_sample_period_seconds']==300
        return
    assert result['stop_request'] is not None
    assert (run/'STOP_REQUESTED').is_file()
    assert len(list(tmp_path.rglob('STOP_REQUESTED')))==1
    if reason=='shared_free':
        assert result['stop_request']['observed']['protected_free_floor_bytes']==1320*m.GIB
        assert 'SHARED_FREE_BELOW_FLOOR_AND_OTHER_REMAINING_WRITES' in result['stop_request']['reasons']


def test_monitor_preserves_operator_reason_and_does_not_create_missing_run(tmp_path):
    m=load(MONITOR,'independent_formal_stop')
    absent=tmp_path/'absent';assert not m.request_stop(absent,{'new':1});assert not absent.exists()
    run=tmp_path/'run';run.mkdir();p=run/'STOP_REQUESTED';p.write_text('operator reason\n')
    assert m.request_stop(run,{'new':1});assert p.read_text()=='operator reason\n'


def test_control_has_no_auto_restart_or_optimizer_clock_override():
    source=LAUNCHER.read_text();tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    # Both Popen calls are outside any retry/while loop: training once, observer once.
    assert not any(isinstance(n,ast.While) for n in ast.walk(main))
    assert source.count('subprocess.Popen(')==2
    assert "'automatic_followup': False" in source
    assert "'--max-updates'" not in source and "'--stop-after'" not in source and "'--warmup'" not in source
    assert "if a.resume:\n        cmd.append('--resume')" in source
