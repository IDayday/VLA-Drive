"""Two finite CPU checks; no process probe, dispatch, locks or running-owner edit."""
import copy,json
from types import SimpleNamespace
from pathlib import Path
import pytest
from tools.qwen_world_performance import launch as L,continuation as C,evaluate as E


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))


def test_explicit_v2_admits_only_second_known_native_and_preserves_v1_cap(tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir();alias=tmp_path/'transparent';alias.symlink_to(root,target_is_directory=True)
    monkeypatch.setattr(C,'ROOT',root);monkeypatch.setattr(C,'ALIAS',alias)
    p1=dict(C.PROTOCOL,output_alias=str(alias));p2=dict(C.PROTOCOL_V2,output_alias=str(alias))
    monkeypatch.setattr(C,'PROTOCOL',p1);monkeypatch.setattr(C,'PROTOCOL_V2',p2)
    run=dict(id='S4',root=str(root/'runs/S4'),execution_output_root=str(alias/'runs/S4'),resource_protocol=p2,
             deployment_candidates=[dict(host='h',lease_slot=s,minimum_free_MiB=32768)for s in ('primary','shared','research')])
    C.validate_resource_fields(run)
    owner=lambda out:dict(entry='tools.qwen_world_performance.train',visible=True,output=str(out))
    sample=dict(owners=[owner(root/'runs/S0')for _ in range(8)])
    assert C.shared_sample_ready(sample,run)[0]
    assert not C.shared_sample_ready(sample,dict(run,resource_protocol=p1))[0]
    sample['owners']+=[owner(root/'runs/S1')for _ in range(8)]
    assert not C.shared_sample_ready(sample,run)[0]
    assert not C.shared_sample_ready(dict(owners=[owner(run['root'])]),run)[0]
    hidden=dict(owners=[dict(owner(root/'runs/S0'),visible=False)])
    assert not C.shared_sample_ready(hidden,run)[0]
    too_weak=copy.deepcopy(run);too_weak['deployment_candidates'][1]['minimum_free_MiB']=32767
    with pytest.raises(ValueError):C.validate_resource_fields(too_weak)


def test_actual_candidate_holds_loading_then_admits_v2_shared_lane_without_dispatch(tmp_path,monkeypatch):
    from starVLA.rl.qwen_world_safety_exploration import resources,contracts
    from tools.qwen_world_safety_exploration import eval_owner
    monkeypatch.setattr(L,'ROOT',tmp_path);monkeypatch.setattr(E,'ROOT',tmp_path);monkeypatch.setattr(C,'ROOT',tmp_path)
    monkeypatch.setattr(L,'bootstrap',lambda:None)
    monkeypatch.setattr(contracts,'allowed_hosts',lambda **kw:['h'])
    monkeypatch.setattr(resources,'gpu_activity_issue',lambda *a:None)
    monkeypatch.setattr(eval_owner,'lease_free',lambda p:True)
    old=dict(id='S0',root=str(tmp_path/'runs/S0'),source='frozen5e',config='cfg0',trainer_module='tools.qwen_world_performance.train')
    new=dict(id='S4',priority=0,root=str(tmp_path/'runs/S4'),resource_protocol=C.PROTOCOL_V2,
             deployment_candidates=[dict(host='h',lease_slot='shared',minimum_free_MiB=32768)])
    uuids=[str(i)for i in range(8)]
    sample=dict(gpus=[dict(uuid=u,free_MiB=60000)for u in uuids],worker_python_exists=True,MemAvailable_bytes=200*2**30,CPU_affinity=64,load1=5,
                owners=[dict(uuid=u,visible=True,output=old['root'],entry=old['trainer_module'],pid=10+i,start_ticks=99+i,used_MiB=17000)for i,u in enumerate(uuids)])
    monkeypatch.setattr(resources,'probe',lambda h:sample)
    job=tmp_path/'jobs/S0/a1';command=['model','-m',old['trainer_module'],'--config','cfg0','--output',old['root']]
    dispatch=dict(host='h',worker_pid=1,worker_start_ticks=2,command=command)
    write(job/'DISPATCH.json',dispatch);write(tmp_path/'resources/NATIVE_HOST_CLAIMS.json',dict(h=dict(job=str(job),arm='S0')))
    cls=L.owner_class();obj=SimpleNamespace(runs=[new],state=dict(runs={}),registry=dict(runs=[old,new]))
    assert cls.candidate(obj,new) is None
    write(job/'RUNNING.json',dict(dispatch,status='RUNNING',child_pid=3,child_start_ticks=4,cwd='frozen5e',gpu_uuids=uuids))
    assert cls.candidate(obj,new)[0]['lease_slot']=='shared'
    assert cls.candidate(obj,dict(new,resource_protocol=C.PROTOCOL)) is None
    sample['owners'][0]['visible']=False
    assert cls.candidate(obj,new) is None


def test_evaluation_fallback_preserves_admitted_gpu_mask_only(monkeypatch):
    import os,sys,types
    observed=[]
    fake=types.ModuleType('tools.qwen_world_safety_exploration.evaluate')
    fake.main=lambda:observed.append(os.environ.get('CUDA_VISIBLE_DEVICES'))
    monkeypatch.setitem(sys.modules,fake.__name__,fake)
    monkeypatch.setattr(sys,'argv',['evaluate','--rank','0'])
    monkeypatch.setattr(E,'bootstrap',lambda:os.environ.__setitem__('CUDA_VISIBLE_DEVICES',''))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    E.main();assert observed[-1]=='3'and os.environ['CUDA_VISIBLE_DEVICES']=='3'
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    E.main();assert observed[-1] is None and 'CUDA_VISIBLE_DEVICES'not in os.environ
