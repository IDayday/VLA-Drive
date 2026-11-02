import copy,json,time
from pathlib import Path
import numpy as np
import pytest
from tools.qwen_world_performance import evaluate as E,continuation as C,select_checkpoints as S
from tools.qwen_world_performance import launch as L
from test_world_performance_operations import dump
from starVLA.rl.qwen_world_research_data.manifest import sealed


def test_running_native_releases_only_startup_hold_and_stable_selector_is_bounded(tmp_path,monkeypatch):
    monkeypatch.setattr(E,'ROOT',tmp_path);job=tmp_path/'jobs/own/attempt';job.mkdir(parents=True)
    run=dict(id='own',source='fixed-science',root=str(tmp_path/'runs/own'),config='fixed-config',trainer_module='tools.qwen_world_performance.train')
    command=['model','-m',run['trainer_module'],'--config','fixed-config','--output',run['root']]
    dispatch=dict(host='h',worker_pid=1,worker_start_ticks=2,command=command)
    dump(job/'DISPATCH.json',dispatch);claim=dict(job=str(job),arm='own')
    sample=dict(gpus=[dict(uuid=str(i))for i in range(8)],owners=[dict(uuid=str(i),visible=True,output=run['root'],entry=run['trainer_module'],pid=10+i,start_ticks=20+i,used_MiB=17000)for i in range(8)])
    assert E.native_claim_startup_hold('h',claim,sample,run)[0]
    dump(job/'RUNNING.json',dict(dispatch,status='RUNNING',child_pid=3,child_start_ticks=4,cwd='fixed-science',gpu_uuids=[str(i)for i in range(8)]))
    assert not E.native_claim_startup_hold('h',claim,sample,run)[0]
    unknown=copy.deepcopy(sample);unknown['owners'][0]['visible']=False
    assert E.native_claim_startup_hold('h',claim,unknown,run)[0]
    ref=np.array([1.,.95,.2,0.]);points=[dict(updates=1,PDMS=90.,scores=ref),dict(updates=2,PDMS=90.08,scores=np.array([.99,.95,.21,0.])),dict(updates=3,PDMS=89.8,scores=np.array([1.,.95,.2,0.]))]
    selected=S.choose_points(points,ref)
    assert selected['best_score_checkpoint']['updates']==2 and selected['best_stable']['updates']==1 and selected['last']['updates']==3
    summary=sealed(dict(schema='world_rl_research_evaluation_summary_v1',status='COMPLETE',cohort='NAVTEST',scenes=12145,expected_scenes=12146,physical_logs=44,invalid_scenes=0,seed=2101,train_feedback_allowed=False,PDMS=90.))
    with pytest.raises(ValueError,match='COMPLETE12146'):S.validate_summary(summary)


def test_real_alias_same_run_migration_requires_exact_intent_and_no_scientific_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(C,'ROOT',tmp_path/'new');monkeypatch.setattr(L,'ROOT',tmp_path/'new')
    alias=tmp_path/'old-view';alias.symlink_to(tmp_path/'new',target_is_directory=True);monkeypatch.setattr(C,'ALIAS',alias)
    monkeypatch.setattr(C,'PROTOCOL',dict(C.PROTOCOL,output_alias=str(alias)))
    root=tmp_path/'new/runs/own';root.mkdir(parents=True);(alias/'runs/own/proof').write_text('same real output')
    assert(root/'proof').read_text()=='same real output'and(root/'proof').stat().st_ino==(alias/'runs/own/proof').stat().st_ino
    run=dict(id='own',root=str(root),source='fixed-source',source_commit='5e522',config_sha256='fixed-cfg-sha',
        execution_output_root=str(alias/'runs/own'),resource_protocol=C.PROTOCOL,
        deployment_candidates=[dict(host='training-vlawm-tsinghua-zt5',lease_slot='shared',minimum_free_MiB=32768)])
    C.validate_resource_fields(run)
    assert C.shared_sample_ready(dict(owners=[dict(entry='tools.old.train',visible=True,output=str(tmp_path/'old-I'))]))[0]
    assert not C.shared_sample_ready(dict(owners=[dict(entry='tools.new.train',visible=True,output=str(alias/'runs/own'))]))[0]
    cp=dict(path=str(root/'checkpoints/cp'),updates=5,world=8,next_inner=1)
    monkeypatch.setattr(C,'inspect_native',lambda r,p:cp)
    dump(root/'latest.json',dict(checkpoint=cp['path']));dump(root/'identity.json',dict(sha256='same-run'))
    job=tmp_path/'new/jobs/own/attempt001';command=['model','--output',run['root']]
    running=dict(status='RUNNING',host='old-host',worker_pid=1,worker_start_ticks=2,command=command,cwd='fixed-source')
    dump(job/'RUNNING.json',running);dump(job/'DISPATCH.json',running);node=dict(job=str(job),host='old-host')
    intent=C.make_migration_intent(run,node,'training-vlawm-tsinghua-zt5');dump(job/'OWN_NATIVE_CONTINUATION.json',intent);(job/'STOP_REQUESTED').touch()
    result=dict(running,returncode=0,end_unix=time.time()+1)
    assert C.verified_migration(run,node,result,cp)['run_identity']=='same-run'
    assert C.verified_migration(run,node,dict(result,returncode=1),cp)is None
    assert C.same_path(alias/'runs/own',run['root'])
