"""Campaign CPU mocks verify budgets, gates and nonduplicating job lifecycle."""
import copy
import json
from pathlib import Path

import pytest

from tools.qwen_world_compare import campaign as c
from starVLA.rl.qwen_world_compare.contracts import fingerprint


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def plan(tmp_path):
    source=Path(c.__file__).resolve().parents[2]
    arms={}
    for arch in c.ARCHITECTURES:
        arms[arch]=dict(config=str(source/f'configs/qwen_world_compare/arch_{arch.lower()}.yaml'),
            config_sha256='cfg'+arch,IL=str(tmp_path/'runs'/arch/'IL'),RL=str(tmp_path/'runs'/arch/'RL'),
            selection_IL=str(tmp_path/'selections'/f'{arch}_IL.json'),selection_RL=str(tmp_path/'selections'/f'{arch}_RL.json'),
            qualification=str(tmp_path/'qualification'/arch/'ENGINEERING_ACCEPTANCE.json'))
    value=dict(root=str(tmp_path),source=str(source),architectures=arms,common=dict(data_root='/sealed/current_data'),
        data_paths={role:'/sealed/current_data/'+role for role in('TRAIN','DEV','NAVTEST')},
        rl_config=str(source/'configs/qwen_world_compare/rl.yaml'),minimum_disk_free_GiB=0,maximum_evaluation_workers=8)
    value['sha256']=fingerprint(value);return value


class Driver:
    def __init__(self,root):
        self.root=Path(root);self.calls=[];self.before_launch=None
        self.hosts=[('host1',dict(gpus=[dict(uuid=f'GPU-{i}',index=i)for i in range(8)])),
                    ('host2',dict(gpus=[dict(uuid=f'GPU-{8+i}',index=i)for i in range(8)]))]

    def choose_host(self,*,excluded_uuids,minimum):
        return next(((h,p)for h,p in self.hosts if not set(g['uuid']for g in p['gpus'])&set(excluded_uuids)),None)

    def launch(self,host,source,command,name,**kwargs):
        if self.before_launch:self.before_launch()
        path=self.root/'jobs'/name;path.mkdir(parents=True,exist_ok=True)
        self.calls.append(dict(host=host,command=command,name=name,**kwargs))
        save(path/'RUNNING.json',dict(status='RUNNING'));return path

    def job_state(self,path):
        path=Path(path)
        for name in('RESULT.json','RUNNING.json'):
            if(path/name).exists():return json.loads((path/name).read_text())
        raise ValueError('Missing receipt')


@pytest.fixture
def instance(plan,monkeypatch):
    monkeypatch.setattr(c,'_checkpoint',lambda *a,**k:None)
    driver=Driver(plan['root']);return c.Campaign(plan,{'core.py':'locked'},driver=driver)


def gate(plan,arch,*,sources=None,**changes):
    value=dict(status='PASS',architecture_id=arch,formal_initialized_from_qualification=False,
        source_files=sources or {'core.py':'locked'},config_sha256=plan['architectures'][arch]['config_sha256'])
    value.update(changes);save(plan['architectures'][arch]['qualification'],value)


def test_exact_qualification_is_a_hard_gate_not_an_automatic_pass(instance,plan):
    state=instance.tick(stages=('il',))
    assert not instance.driver.calls
    assert all(v['status']=='WAITING_QUALIFICATION'for v in state['architectures'].values())
    gate(plan,'A',sources={'core.py':'different'})
    instance.tick(stages=('il',));assert instance.state['architectures']['A']['status']=='BLOCKED_SOURCE_MISMATCH'
    gate(plan,'A',formal_initialized_from_qualification=True)
    instance.tick(stages=('il',));assert not instance.driver.calls


def test_training_dispatch_has_durable_intent_and_disjoint_eight_gpu_hosts(instance):
    def assert_intent():
        state=json.loads(instance.path.read_text());assert list(state['jobs'].values())[-1]['status']=='LAUNCH_INTENT'
    instance.driver.before_launch=assert_intent
    instance._training('A','IL');instance._training('B','IL')
    calls=instance.driver.calls;assert len(calls)==2 and calls[0]['host']!=calls[1]['host']
    assert not set(calls[0]['expected_uuids'])&set(calls[1]['expected_uuids'])
    assert all(len(call['expected_uuids'])==8 for call in calls)
    assert '--qualification'not in calls[0]['command']and'--stop-after'not in calls[0]['command']
    assert '--nproc-per-node=8'in calls[0]['command']


def test_running_resume_family_does_not_start_another_learner_when_checkpoint_advances(instance,monkeypatch):
    instance._training('A','IL');old=list(instance.state['jobs'].values())[-1]
    old['status']='COMPLETE'
    monkeypatch.setattr(c,'_checkpoint',lambda run,stage,updates=None:None if updates else dict(path='/native/update001000',seal=dict(updates=1000)))
    instance._training('A','IL');assert len(instance.driver.calls)==2
    monkeypatch.setattr(c,'_checkpoint',lambda run,stage,updates=None:None if updates else dict(path='/native/update002000',seal=dict(updates=2000)))
    instance._training('A','IL');assert len(instance.driver.calls)==2
    assert instance.driver.calls[-1]['command'][-2:]==['--resume','/native/update001000']


def test_prechild_resource_retry_is_bounded_and_preserves_command(instance):
    for _ in range(3):
        instance._training('A','IL');job=list(instance.state['jobs'].values())[-1]
        save(Path(job['receipt'])/'RESULT.json',dict(status='REFUSED_RESOURCE',child_not_started=True,returncode=75))
        instance._poll()
    instance._training('A','IL')
    assert len(instance.driver.calls)==3
    assert instance.state['architectures']['A']['status']=='BLOCKED_RESOURCE_RETRY_LIMIT'
    assert all(call['command']==instance.driver.calls[0]['command']for call in instance.driver.calls)


@pytest.mark.parametrize('status,extra',[('FAILED',dict(returncode=1)),('REFUSED_RESOURCE',dict(child_not_started=False)),('FORCED_EXIT_REVIEW_REQUIRED',dict(returncode=-9))])
def test_scientific_or_unproven_failure_never_automatically_retries(instance,status,extra):
    instance._training('A','IL');job=list(instance.state['jobs'].values())[-1]
    save(Path(job['receipt'])/'RESULT.json',dict(status=status,**extra));instance._poll()
    instance.tick(stages=('il',));assert len(instance.driver.calls)==1
    assert instance.state['architectures']['A']['status']in('FAILED','BLOCKED_REVIEW_REQUIRED')


def test_dispatch_gap_is_blocked_not_duplicated_on_owner_resume(instance,plan):
    def fail(*a,**k):raise TimeoutError('lost SSH reply before admission receipt')
    instance.driver.launch=fail
    with pytest.raises(TimeoutError):instance._training('A','IL')
    assert list(instance.state['jobs'].values())[-1]['status']=='LAUNCH_INTENT'
    restarted=c.Campaign(plan,{'core.py':'locked'},driver=instance.driver)
    restarted._poll();restarted._training('A','IL')
    assert list(restarted.state['jobs'].values())[-1]['status']=='BLOCKED_DISPATCH_RECEIPT'


def test_campaign_resume_refuses_source_or_plan_changes(instance,plan):
    with pytest.raises(ValueError,match='source'):c.Campaign(plan,{'core.py':'other'},driver=instance.driver)
    altered=copy.deepcopy(plan);altered['sha256']='changed_budget'
    with pytest.raises(ValueError):c.Campaign(altered,{'core.py':'locked'},driver=instance.driver)


def selections(*,same_il=False,same_rl=False,rl_zero=False):
    il_rows=[dict(updates=5000,checkpoint='/il5000',evaluation='/dev_il5000'),dict(updates=100000,checkpoint='/il100000',evaluation='/dev_il100000')]
    il=dict(evaluated_candidates=il_rows,selected=il_rows[-1]if same_il else il_rows[0])
    rl_rows=[dict(updates=0,checkpoint=None,evaluation=il['selected']['evaluation']),dict(updates=20000,checkpoint='/rl20000',evaluation='/dev_rl20000'),dict(updates=50000,checkpoint='/rl50000',evaluation='/dev_rl50000')]
    rl=dict(evaluated_candidates=rl_rows,selected=rl_rows[0]if rl_zero else rl_rows[-1]if same_rl else rl_rows[1])
    return il,rl


@pytest.mark.parametrize('same_il,same_rl,rl_zero,count',[(False,False,False,4),(True,True,False,2),(True,False,True,2)])
def test_identical_model_roles_share_inference_but_keep_each_selection_binding(same_il,same_rl,rl_zero,count):
    il,rl=selections(same_il=same_il,same_rl=same_rl,rl_zero=rl_zero)
    requests=c.final_policy_requests(il,rl,'/IL_SELECTION.json','/RL_SELECTION.json')
    assert len(requests)==count
    assert {role for r in requests for role in r['bindings']}=={'il_endpoint','il_selected','rl_endpoint','rl_selected'}
    assert all(r['selection']in('/IL_SELECTION.json','/RL_SELECTION.json')for r in requests)
    if rl_zero:
        q=next(r for r in requests if'rl_selected'in r['bindings'])
        assert q['stage']=='IL'and q['checkpoint']==il['selected']['checkpoint']


def test_partial_requested_stage_completion_is_not_full_experiment_completion(instance):
    for arm in instance.state['architectures'].values():arm['status']='IL_COMPLETE'
    assert c.selected_stages_complete(instance.state,c.ARCHITECTURES,['il'])
    assert not c.selected_stages_complete(instance.state,c.ARCHITECTURES,['il','rl','evaluate'])
    result=c.report(instance.plan);assert result['complete']is False
    assert all(row['RL50k_Navtest']is None for row in result['architectures'].values())


def test_actual_registered_plan_retains_all_full_budgets_and_milestones():
    source=Path(c.__file__).resolve().parents[2]
    p=c.make_plan(source/'configs/qwen_world_compare/campaign.yaml',source)
    assert p['IL_updates']==100000 and p['RL_updates']==50000
    assert p['IL_DEV_updates']==list(range(5000,100001,5000))
    assert p['RL_DEV_updates']==[0,5000,10000,20000,35000,50000]
    assert p['logical_RL_candidates_per_arm']==800000 and p['logical_IL_scene_exposures_per_arm']==3200000
    assert set(p['architectures'])==set('ABCD')


def test_IL_endpoint_without_all_twenty_DEV_points_cannot_start_RL(instance,plan,monkeypatch):
    gate(plan,'A');calls=[]
    def training(arch,stage):
        calls.append(stage);return dict(path='/native100k')
    monkeypatch.setattr(instance,'_training',training)
    monkeypatch.setattr(instance,'_dev_window',lambda *a,**k:[dict(updates=n)for n in c.IL_MILESTONES[:-1]])
    instance._advance_architecture('A',('il','rl','evaluate'))
    assert calls==['IL']and instance.state['architectures']['A']['status']=='WAITING_IL_DEV'


def test_RL_endpoint_is_required_before_any_Navtest(instance,plan,monkeypatch):
    gate(plan,'A');il,_=selections();il['selected'].update(checkpoint='/own_A_IL_parent',updates=5000)
    monkeypatch.setattr(instance,'_training',lambda arch,stage:dict(path='/native100k')if stage=='IL'else None)
    monkeypatch.setattr(instance,'_dev_window',lambda arch,stage,*a,**k:[{}]*(20 if stage=='IL'else 6))
    import starVLA.rl.qwen_world_compare.parent_selection as selection
    monkeypatch.setattr(selection,'select_il_parent',lambda *a,**k:il)
    monkeypatch.setattr(instance,'_evaluation',lambda *a,**k:pytest.fail('No Navtest before full50k'))
    instance._advance_architecture('A',('il','rl','evaluate'))
    assert instance.state['architectures']['A']['IL_parent_update']==5000
    assert'RL_selected_update'not in instance.state['architectures']['A']


def test_one_architecture_artifact_failure_does_not_stop_other_architectures(instance,monkeypatch):
    visited=[]
    def advance(arch,stages):
        visited.append(arch)
        if arch=='A':raise ValueError('invalid A checkpoint seal')
        instance.state['architectures'][arch]['status']='WAITING_QUALIFICATION'
    monkeypatch.setattr(instance,'_advance_architecture',advance)
    instance.tick()
    assert visited==list('ABCD')and instance.state['architectures']['A']['status']=='FAILED'
    assert instance.state['architectures']['B']['status']=='WAITING_QUALIFICATION'


def test_completed_scene_files_are_not_summarized_before_resumed_worker_exits(instance,monkeypatch):
    arch='A';stage='IL';step=5000;output=instance.root/'evaluations/ARCH_A/DEV_IL_update005000'
    prefix='A_DEV_IL_update005000'
    for rank in range(8):
        logical=f'{prefix}_rank{rank:02d}';save(output/f'rank{rank}.json',dict(status='COMPLETE'))
        instance.state['jobs'][logical]=dict(logical=logical,family=logical,status='COMPLETE',architecture='A')
    base=prefix+'_rank00';resume=base+'_scene_resume001'
    instance.state['jobs'][resume]=dict(logical=resume,family=base,status='RUNNING',architecture='A')
    monkeypatch.setattr(instance,'_export',lambda *a,**k:(Path('/sealed_export'),{}))
    monkeypatch.setattr(instance,'_gpu_routes',lambda:[])
    import tools.qwen_world_compare.evaluate as evaluation
    calls=[]
    def summary(*a):
        calls.append(a);return dict(sha256='sealed',scenes=1696,model_sha256='model',PDMS=1.)
    monkeypatch.setattr(evaluation,'summarize',summary)
    assert instance._evaluation(arch,stage,step,'/native')is None and not calls
    instance.state['jobs'][resume]['status']='COMPLETE'
    assert instance._evaluation(arch,stage,step,'/native')['summary']['scenes']==1696
    assert len(calls)==1


def test_own_IL_selection_cannot_start_formal_RL_before_real_RL_acceptance(instance,plan,monkeypatch):
    gate(plan,'A');il,_=selections();calls=[]
    monkeypatch.setattr(instance,'_training',lambda arch,stage:calls.append(stage)or dict(path='/native100k'))
    monkeypatch.setattr(instance,'_dev_window',lambda *a,**k:[{}]*20)
    import starVLA.rl.qwen_world_compare.parent_selection as selection
    monkeypatch.setattr(selection,'select_il_parent',lambda *a,**k:il)
    def qualification(arch,parent):
        assert parent is il
        instance.state['architectures'][arch]['status']='WAITING_RL_QUALIFICATION'
        return False
    monkeypatch.setattr(instance,'_qualify_rl',qualification)
    instance._advance_architecture('A',('il','rl','evaluate'))
    assert calls==['IL']and instance.state['architectures']['A']['status']=='WAITING_RL_QUALIFICATION'


def test_real_RL_qualification_is_finite_and_separate_from_formal_output(instance,plan,monkeypatch):
    il,_=selections();monkeypatch.setattr(instance,'_cpu_host',lambda:'host1')
    instance._qualify_rl('A',il)
    first=instance.driver.calls[-1]['command']
    assert '--qualification'in first and first[first.index('--stop-after-update')+1]=='4'
    assert first[first.index('--save-every')+1]=='1'
    assert first[first.index('--output')+1]!=plan['architectures']['A']['RL']
    assert first[first.index('--il-checkpoint')+1]==il['selected']['checkpoint']
    assert not any('resume'==p for p in first)
    stages=['baseline','prepare_resume','resume','verify']
    for previous,expected in zip(stages,stages[1:]):
        instance._last('A_RL_qualification_'+previous)['status']='COMPLETE'
        instance._qualify_rl('A',il)
        assert instance._last('A_RL_qualification_'+expected)['status']=='RUNNING'
    assert len(instance.driver.calls)==4
    resumed=instance.driver.calls[2]['command']
    assert resumed[resumed.index('--resume')+1].endswith('update000002_inner2')
    assert instance.state['architectures']['A']['RL_qualification']['formal_initialized_from_qualification']is False


def test_scientific_jobs_keep_old_snapshot_and_only_CPU_qualifier_uses_ops_snapshot(instance,monkeypatch):
    instance.controller_source=Path('/frozen/controller')
    captures=[]
    old=instance.driver.launch
    def launch(host,source,command,name,**kwargs):
        captures.append(str(source));return old(host,source,command,name,**kwargs)
    instance.driver.launch=launch
    instance._launch('A','A_train',['old train'],'host1',kind='training')
    instance._launch('A','A_RL_qualhelper',['new helper'],'host1',kind='rl_qualification_helper')
    assert captures==[instance.plan['source'],'/frozen/controller']
    entries=list(instance.state['jobs'].values())
    assert entries[0]['scientific_source']==entries[1]['scientific_source']==instance.plan['source']
