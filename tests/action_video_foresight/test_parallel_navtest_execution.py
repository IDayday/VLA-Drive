from pathlib import Path
import json
import pytest
from tools.planning_interface_transfer import watch_navtest as queue
from tools.planning_interface_transfer import navtest_execution as execution
from tests.action_video_foresight.test_planning_interface_navtest_queue import registered, complete_checkpoint


def test_parallel_future_jobs_do_not_wait_for_failed_75k(registered, monkeypatch):
    r,path = registered
    for m in r['models']: complete_checkpoint(m,80000)
    queue.atomic(Path(r['protocol']['root'])/'status.json',dict(status='FAILED'))
    overlay=dict(queue_registration_identity=r['identity'],max_parallel_future_tasks=3,
                 source_worktree=r['source_worktree'],identity='allocation')
    monkeypatch.setattr(execution,'load',lambda _:overlay)
    calls=[]
    class Child:
        pid=12345
        def poll(self):return None
    def spawn(command,**kw): calls.append(command);return Child()
    monkeypatch.setattr(queue.subprocess,'Popen',spawn)
    queue.watch(path,once=True,execution='execution.json')
    assert len(calls)==3
    assert {x[x.index('--task')+1] for x in calls}=={f'{m["arm"]}_080000' for m in r['models']}
    status=queue.read(Path(r['root'])/'status.json')
    assert status['awaiting75k'] is False
    assert len(status['active_tasks'])==3


def test_foreign_overlay_is_rejected_before_dispatch(registered,monkeypatch):
    r,path=registered
    monkeypatch.setattr(execution,'load',lambda _:dict(queue_registration_identity='foreign'))
    with pytest.raises(ValueError,match='different queue'):queue.watch(path,once=True,execution='foreign')


def test_overlay_rejects_changed_scene_partition_count(tmp_path,monkeypatch):
    r=dict(schema='planning_interface_navtest_execution_v1',source_worktree=str(tmp_path),
           source_sha='locked',world_size=8,max_parallel_future_tasks=3,asset_hashes={},
           allocations={},authorized_hosts={})
    r['identity']=execution.identity_hash(r)
    p=tmp_path/'allocation.json';p.write_text(json.dumps(r))
    monkeypatch.setattr(execution,'source_identity',lambda _: 'locked')
    with pytest.raises(ValueError,match='allocation/source'):execution.load(p)


def test_overlay_rejects_unauthorized_device(tmp_path,monkeypatch):
    r=dict(schema='planning_interface_navtest_execution_v1',source_worktree=str(tmp_path),
           source_sha='locked',world_size=4,max_parallel_future_tasks=3,asset_hashes={},
           allocations={'75000':{'A_ACTION':{'host':'host','gpus':[0,1,2,3]}}},
           authorized_hosts={'host':{'allowed_gpus':[4,5,6,7]}})
    r['identity']=execution.identity_hash(r)
    p=tmp_path/'allocation.json';p.write_text(json.dumps(r))
    monkeypatch.setattr(execution,'source_identity',lambda _: 'locked')
    with pytest.raises(ValueError,match='Unauthorized GPU'):execution.load(p)


def test_killed_export_meter_preserves_failed_cost_evidence(tmp_path):
    p=tmp_path/'run/status.json'
    original=dict(status='RUNNING',start_unix=10,gpu_count=1,real_optimizer_updates=0,
                  kind='foresight_current_camera_export',inference_scenes=19)
    queue.atomic(p,original)
    execution.close_dead_export_meter(p,3610,'verified child SIGKILL')
    result=queue.read(p)
    assert result['status']=='FAILED' and result['gpu_hours']==1
    assert result['inference_scenes']==19
    assert queue.read(result['termination_receipt'])['original']==original
    execution.close_dead_export_meter(p,7210,'later observation')
    assert queue.read(p)['gpu_hours']==1
