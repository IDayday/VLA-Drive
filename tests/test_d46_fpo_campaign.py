"""Preemption must fence predecessors and respect live shared rank locks."""
import fcntl
import json
import time
from pathlib import Path
from tools.d46_fpo_campaign import owner


def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))


def test_preempted_worker_cannot_be_requeued_while_own_rank_lock_live(tmp_path,monkeypatch):
    monkeypatch.setattr(owner,'ROOT',tmp_path)
    host='training-vlawm-zt4';put(tmp_path/'operations/PREEMPTED_HOSTS.json',{host:dict(at=time.time())})
    job=tmp_path/'old_job';job.mkdir();output=tmp_path/'evaluation';output.mkdir()
    scene=output/'scene_proof.json';scene.write_text('retained exact scene')
    last=dict(host=host,job=str(job),at=time.time()-1000)
    case=dict(status='RUNNING',output=str(output),jobs={'0':[last]})
    state=dict(cases={'case':case})
    with(output/'rank0.lock').open('a')as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);owner.recover_preempted(state)
        assert case['jobs']['0']==[last]and not(job/'PREEMPTIBLE_FENCED_REQUEUE.json').exists()
    owner.recover_preempted(state)
    assert case['jobs']['0']==[]and case['status']=='PENDING'
    assert(job/'STOP_REQUESTED').exists()and(job/'PREEMPTIBLE_FENCED_REQUEUE.json').exists()
    assert not(job/'RESULT.json').exists()and scene.read_text()=='retained exact scene'
    assert case['preempted_attempts'][0]['previous']==last
    owner.recover_preempted(state);assert len(case['preempted_attempts'])==1


def test_dispatch_receipt_alone_is_initialization_not_actual_training(tmp_path,monkeypatch):
    monkeypatch.setattr(owner,'ROOT',tmp_path);out=tmp_path/'run';job=tmp_path/'job';job.mkdir()
    put(out/'identity.json',dict(sha256='run'));launches=tmp_path/'launches.json'
    put(launches,dict(receipts=[dict(job=str(job),host='training-vlawm-zt',at=time.time())]))
    r=dict(id='FPO',output=str(out),world=2,hosts=['training-vlawm-zt'],launch_receipts=str(launches),start_update=10)
    plan=dict(routes={'FPO':r});s=owner.route_status(plan)
    assert s['actual_training']==0 and s['routes']['FPO']['status']=='INITIALIZING'
    for rank in range(2):
        p=out/f'metrics_rank{rank}.jsonl';p.write_text(json.dumps(dict(run='run',updates=11,at=time.time()))+'\n')
    s=owner.route_status(plan);assert s['actual_training']==1 and (tmp_path/'operations/ACTUAL_START_FPO.json').is_file()
