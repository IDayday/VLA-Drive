from types import SimpleNamespace
import fcntl
from pathlib import Path
import pytest
from tools.qwen_world_postcot import job_worker as worker
from tools.qwen_world_postcot import fresh_loss_owner as owner


@pytest.fixture
def admission(monkeypatch,tmp_path):
    primary=tmp_path/'primary';shared=tmp_path/'shared';primary.mkdir();shared.mkdir()
    monkeypatch.setattr(worker,'PRIMARY_LEASE_ROOT',primary)
    monkeypatch.setattr(worker,'SHARED_LEASE_ROOT',shared)
    devices={0:dict(index=0,uuid='GPU-test',total_MiB=81000,free_MiB=52000,utilization=50)}
    occupants=[dict(pid=123,uuid='GPU-test',used_MiB=28000)]
    monkeypatch.setattr(worker,'_admission_sample',lambda:(devices,occupants))
    monkeypatch.setattr(worker,'_known_owners',lambda *args:None)
    args=SimpleNamespace(share_min_free_mib=49152,minimum_free_mib=49152,lease_slot='primary')
    return args,primary,shared,devices,occupants


def release(locks):
    for lock in locks:lock.close()


def test_primary_slot_can_share_without_releasing_current_shared_owner(admission):
    args,primary,shared,_,_=admission;locks=[]
    with(shared/'GPU-test.lock').open('a')as existing:
        fcntl.flock(existing,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            worker._gpu_admission(args,'host',dict(permanent_hosts=['host']),[0],['GPU-test'],{},locks)
            assert len(locks)==1 and Path(locks[0].name).parent==primary
            with(shared/'GPU-test.lock').open('a')as another:
                with pytest.raises(BlockingIOError):fcntl.flock(another,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:release(locks)


def test_busy_primary_is_refused_not_overwritten(admission):
    args,primary,_,_,_=admission;locks=[]
    with(primary/'GPU-test.lock').open('a')as existing:
        fcntl.flock(existing,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(worker.ResourceRefused,match='lease became busy'):
            worker._gpu_admission(args,'host',dict(permanent_hosts=['host']),[0],['GPU-test'],{},locks)
    assert not locks


def test_new_slot_keeps_unknown_owner_and_headroom_checks(admission,monkeypatch):
    args,_,_,_,_=admission;locks=[]
    def unknown(*args):raise worker.ResourceRefused('Unknown GPU owner')
    monkeypatch.setattr(worker,'_known_owners',unknown)
    with pytest.raises(worker.ResourceRefused,match='Unknown GPU owner'):
        worker._gpu_admission(args,'host',dict(permanent_hosts=['host']),[0],['GPU-test'],{},locks)
    assert not locks


def test_headroom_rechecked_under_primary_slot(admission,monkeypatch):
    args,_,_,devices,occupants=admission;locks=[];calls=[]
    def sample():
        calls.append(True)
        if len(calls)>1:
            devices[0]['free_MiB']=47000;occupants[0]['used_MiB']=33000
        return devices,occupants
    monkeypatch.setattr(worker,'_admission_sample',sample)
    try:
        with pytest.raises(worker.ResourceRefused,match='headroom changed'):
            worker._gpu_admission(args,'host',dict(permanent_hosts=['host']),[0],['GPU-test'],{},locks)
    finally:release(locks)
    assert len(calls)==2


def test_default_sharing_still_uses_existing_shared_slot(admission):
    args,_,shared,_,_=admission;args.lease_slot=None;locks=[]
    try:
        worker._gpu_admission(args,'host',dict(permanent_hosts=['host']),[0],['GPU-test'],{},locks)
        assert Path(locks[0].name).parent==shared
    finally:release(locks)


def test_scheduler_uses_free_primary_with_shared_occupied(monkeypatch):
    monkeypatch.setattr(owner,'held_shared_uuids',lambda root=None:set()if root else{'GPU-test'})
    monkeypatch.setattr(owner,'allowed_hosts',lambda:['host'])
    monkeypatch.setattr(owner.c,'choose_training_host',lambda *args,**kwargs:dict(host='host',uuids=['GPU-test'],share=49152))
    assert owner.training_route(dict(jobs={}))['lease_slot']=='primary'
