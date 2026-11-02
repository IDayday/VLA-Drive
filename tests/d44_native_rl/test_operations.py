from argparse import Namespace
import fcntl
import json
from pathlib import Path
import pytest
from tools.d44_native_rl.campaign import epoch_update
from tools.qwen_world_compare import job_worker as worker
from starVLA.rl.qwen_world_compare.contracts import fingerprint


def test_scene_clock_operational_stops_include_every_epochs_padding():
    assert epoch_update('D0',.25,64)==808
    assert epoch_update('D0',.5,64)==1614
    assert epoch_update('D0',8,64)==25824
    assert epoch_update('D4',.25,32)==1614
    assert epoch_update('D6',1,64)==6456
    assert epoch_update('D6',4,64)==16140
    assert epoch_update('D6',12,64)==41964


def test_owner_has_real_output_module_and_sealed_native_parent(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'D44_ROOT',tmp_path)
    output=tmp_path/'runs/D0';output.mkdir(parents=True)
    value=dict(schema='d44_native_rl_v1',architecture_id='D',recipe=dict(parent_checkpoint=str(tmp_path/'IL44_PARENT'),training_uses_navtest=False),
        data=dict(NAVTEST_overlap=0),parent_model_sha256=worker.D44_PARENT_SHA)
    value['sha256']=fingerprint(value);(output/'identity.json').write_text(json.dumps(value))
    argv=['python','-m','tools.d44_native_rl.train','--output',str(output)]
    assert worker.known_owner_args(argv)
    value['parent_model_sha256']='other';body=dict(value);body.pop('sha256');value['sha256']=fingerprint(body)
    (output/'identity.json').write_text(json.dumps(value));assert not worker.known_owner_args(argv)


def test_shared_slot_fallback_does_not_steal_or_leak_partial_locks(tmp_path,monkeypatch):
    primary=tmp_path/'primary';shared=tmp_path/'shared';primary.mkdir();shared.mkdir()
    monkeypatch.setattr(worker,'PRIMARY_LEASE_ROOT',primary);monkeypatch.setattr(worker,'SHARED_LEASE_ROOT',shared)
    devices={i:dict(index=i,uuid=f'GPU-test{i}',total_MiB=81920,free_MiB=81920,utilization=0)for i in range(2)}
    monkeypatch.setattr(worker,'_admission_sample',lambda:(devices,[]))
    existing=(shared/'GPU-test1.lock').open('a');fcntl.flock(existing,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args=Namespace(share_min_free_mib=41000,minimum_free_mib=None,lease_slot='shared');locks=[]
    try:
        worker._gpu_admission(args,'testhost',dict(permanent_hosts=['testhost']),[0,1],['GPU-test0','GPU-test1'],{},locks)
        assert args.lease_slot=='primary'and len(locks)==2
        # shared slot0 was temporarily acquired, then correctly released.
        probe=(shared/'GPU-test0.lock').open('a');fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB);probe.close()
        third=Namespace(share_min_free_mib=41000,minimum_free_mib=None,lease_slot='shared')
        with pytest.raises(worker.ResourceRefused,match='Both qualified'):worker._gpu_admission(third,'testhost',dict(permanent_hosts=['testhost']),[0,1],['GPU-test0','GPU-test1'],{},[])
    finally:
        for stream in locks:stream.close()
        existing.close()


def test_existing_world_multiscene_owner_requires_exact_il_parent_and_output():
    for module in ('tools.qwen_world_multiscene.train','tools.qwen_world_verified_quality.train'):
        args=['python','-m',module,'--output','/mnt/project/qwen-world-performance-rl-20261103/qualification/known',
              '--il-checkpoint','/mnt/project/qwen-world-rl-optimization-20261030/IL_PARENT']
        assert worker.known_owner_args(args)
        assert not worker.known_owner_args(args[:-1]+['/other/IL_PARENT'])
        assert not worker.known_owner_args(args[:args.index('--output')]+['--output','/other/run']+args[args.index('--il-checkpoint'):])


def test_profiled_third_slot_is_opt_in_locked_and_cannot_admit_fourth(tmp_path,monkeypatch):
    roots={key:tmp_path/key for key in ('PRIMARY_LEASE_ROOT','SHARED_LEASE_ROOT','PROFILED_THIRD_LEASE_ROOT')}
    for key,path in roots.items():path.mkdir();monkeypatch.setattr(worker,key,path)
    devices={0:dict(index=0,uuid='GPU-test',total_MiB=81920,free_MiB=47000,utilization=70)}
    owners=[dict(uuid='GPU-test',pid=i,used_MiB=17000)for i in (1,2)]
    monkeypatch.setattr(worker,'_admission_sample',lambda:(devices,owners));monkeypatch.setattr(worker,'_known_owners',lambda *args:None)
    held=[];locks=[]
    for name in ('PRIMARY_LEASE_ROOT','SHARED_LEASE_ROOT'):
        f=(roots[name]/'GPU-test.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(f)
    try:
        args=Namespace(share_min_free_mib=40960,minimum_free_mib=None,lease_slot='shared',profiled_d0_third_slot=False)
        with pytest.raises(worker.ResourceRefused):worker._gpu_admission(args,'test',dict(permanent_hosts=['test']),[0],['GPU-test'],{},[])
        args.profiled_d0_third_slot=True
        worker._gpu_admission(args,'test',dict(permanent_hosts=['test']),[0],['GPU-test'],{},locks)
        assert args.lease_slot=='profiled_third'
        owners.append(dict(uuid='GPU-test',pid=3,used_MiB=27000))
        with pytest.raises(worker.ResourceRefused):worker._gpu_admission(args,'test',dict(permanent_hosts=['test']),[0],['GPU-test'],{},[])
    finally:
        for f in held+locks:f.close()


def test_profiled_resume_rejects_other_scopes_and_low_headroom(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'D44_ROOT',tmp_path);output=tmp_path/'runs/D0';output.mkdir(parents=True)
    recipe=dict(parent_checkpoint=str(tmp_path/'IL44_PARENT'),training_uses_navtest=False,arm='D0',global_scenes=64,group=16,text_rank=16,film_last_layers=0,train_action_io=False,post_feedback=False,checkpoint_policy='all_off')
    value=dict(schema='d44_native_rl_v1',architecture_id='D',recipe=recipe,data=dict(NAVTEST_overlap=0),parent_model_sha256=worker.D44_PARENT_SHA)
    def save():
        body={k:v for k,v in value.items()if k!='sha256'};value['sha256']=fingerprint(body);(output/'identity.json').write_text(json.dumps(value))
    save();cmd=['python','-m','torch.distributed.run','-m','tools.d44_native_rl.train','--output',str(output),'--resume','CP']
    worker.validate_profiled_d0_resume(cmd,40960)
    with pytest.raises(ValueError):worker.validate_profiled_d0_resume(cmd,32768)
    recipe['arm']='D6';save()
    with pytest.raises(ValueError):worker.validate_profiled_d0_resume(cmd,40960)


def test_deployment_selector_never_uses_partial_results(tmp_path):
    from tools.d44_native_rl.campaign import selection_index
    run=tmp_path/'run';run.mkdir()
    cases={}
    for i,(pdms,newzero,status)in enumerate([(91.0,20,'COMPLETE'),(90.9,2,'COMPLETE'),(94.,0,'RUNNING')]):
        summary=tmp_path/f's{i}.json';summary.write_text(json.dumps(dict(qualification=False,scenes=12146,invalid_scenes=0,PDMS=pdms,paired=dict(new_zero=newzero))))
        cases[str(i)]=dict(status=status,request=dict(checkpoint=str(run/str(i)),epoch=i+1),summary=str(summary))
    r=selection_index(dict(runs=dict(D0=dict(output=str(run))),evaluations=cases))['runs']['D0']
    assert r['evaluated_checkpoints']==2
    assert r['best_score_checkpoint']['PDMS']==91.0
    assert r['best_stable_proxy_checkpoint']['PDMS']==90.9


def test_profiled_qualification_requires_measured_scope_and_bounded_updates(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'D44_ROOT',tmp_path)
    c=dict(global_scenes=32,group=16,text_rank=16,film_last_layers=0,train_action_io=False,checkpoint_policy='all_off',inner_updates=2,arm='D3_FPO',parent_checkpoint=str(tmp_path/'IL44_PARENT'))
    monkeypatch.setattr(worker.subprocess,'check_output',lambda *a,**k:json.dumps(c))
    cmd=['python','-m','torch.distributed.run','--nproc_per_node=2','-m','tools.d44_native_rl.train','--qualification','--config','C','--output',str(tmp_path/'qualification/FPO'),'--stop-after-update','10']
    worker.validate_profiled_native_qualification(cmd,49152)
    with pytest.raises(ValueError):worker.validate_profiled_native_qualification(cmd,41000)
    with pytest.raises(ValueError):worker.validate_profiled_native_qualification(cmd[:-1]+['100000'],49152)
    c['group']=32
    with pytest.raises(ValueError):worker.validate_profiled_native_qualification(cmd,49152)
    assert not worker.frozen_evaluation_owner(['python','-m','tools.qwen_world_multiscene.train'])
    assert not worker.frozen_evaluation_owner(['python','-m','tools.qwen_world_compare.evaluate'])


def test_prechild_qualification_validation_writes_real_failure_not_hanging(tmp_path,monkeypatch):
    import sys
    monkeypatch.setattr(sys,'argv',['job_worker','--receipt-root',str(tmp_path),'--gpus','','--cwd',str(tmp_path),
        '--command-json',json.dumps(['python','-m','tools.d44_native_rl.train']),
        '--share-min-free-mib','41000','--profiled-native-qualification-third-slot'])
    assert worker.main()==2
    value=json.loads((tmp_path/'RESULT.json').read_text())
    assert value['status']=='FAILED_BEFORE_CHILD'and value['child_not_started']is True
    assert 'child_pid'not in value and not(tmp_path/'RUNNING.json').exists()


def test_an_input_directory_does_not_turn_an_unknown_process_into_an_owner():
    assert not worker.known_owner_args(['python','-m','unrelated.project','--data','/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST'])


def test_waiting_p0_soft_yields_only_owned_lower_priority_stable_job(tmp_path,monkeypatch):
    from tools.d44_native_rl import campaign
    monkeypatch.setattr(campaign,'ROOT',tmp_path);(tmp_path/'jobs').mkdir()
    monkeypatch.setattr(campaign,'authorization',lambda:(dict(permanent_hosts=['fixed']),None))
    monkeypatch.setattr(campaign,'allowed_hosts',lambda **kwargs:['fixed'])
    done=tmp_path/'jobs/done';done.mkdir();(done/'RESULT.json').write_text(json.dumps(dict(returncode=0)))
    lower=tmp_path/'jobs/lower';lower.mkdir();cmd=['python','-m','tools.d44_native_rl.train','--output','OWN']
    (lower/'RUNNING.json').write_text(json.dumps(dict(command=cmd,cwd='SCI')))
    value=dict(runs=dict(D0=dict(status='WAITING_FOR_AUTHORIZED_CAPACITY',jobs=[dict(job=str(done))]),
        D1=dict(priority=1,status='RUNNING_OR_ADMITTING',source='SCI',source_sha='HASH',jobs=[dict(job=str(lower),host='fixed',command=cmd)])))
    campaign.yield_owned_lower_priority(value)
    assert(lower/'STOP_REQUESTED').exists()and len(value['priority_yields'])==1
    campaign.yield_owned_lower_priority(value)
    assert len(value['priority_yields'])==1


def test_evaluation_shares_known_capacity_but_protects_whole_il_node(monkeypatch):
    from types import SimpleNamespace
    from tools.d44_native_rl import campaign
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps('0, 81000, 0\n1, 50000, 100\n')))
    assert campaign.free_evaluation_gpus('training-vla-zt',shared=False)==[]
    assert campaign.free_evaluation_gpus('other_authorized',shared=True)==[0,1]
