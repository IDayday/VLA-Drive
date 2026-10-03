"""Queue/identity tests only; no synthetic result is presented as model PDMS."""
from pathlib import Path

import pytest

from tools.action_video_foresight import navtest_milestones as nm
from tools.foresight.lock_navtest import validate_lock
from starVLA.model.modules.vehicle_joint.initialization import identity_hash


@pytest.fixture
def queue(tmp_path, monkeypatch):
    root=tmp_path/'observer';models={}
    for arm in nm.ARMS:
        run=tmp_path/'students'/arm
        nm.atomic(run/'identity.json',{'identity':arm})
        nm.atomic(run/'status.json',{'identity':arm,'completed':49000,'status':'RUNNING'})
        models[arm]=dict(arm=arm,training_run=str(run),run_identity=arm,
                         hostname=arm,host=arm,gpus=list(range(8)))
    config=dict(models=models,artifact_root=str(root),canonical_hostname=nm.socket.gethostname(),
                worker_python='/usr/bin/python3',source_worktree=str(tmp_path),poll_seconds=.01)
    reg=dict(config=config,identity='registered')
    monkeypatch.setattr(nm,'load_registration',lambda path:reg)
    return config,root,tmp_path/'registration.json'


def test_all_thirty_wait_until_exact_complete_checkpoint(queue,monkeypatch):
    config,root,path=queue
    def no_launch(*args,**kwargs):
        raise AssertionError('No ready checkpoint')
    monkeypatch.setattr(nm,'launch',no_launch)
    nm.watch(path,once=True)
    states=nm.read(root/'TASKS.json')
    assert len(states)==30 and set(states.values())=={'WAITING'}
    assert nm.read(root/'status.json')['optimizer_updates']==0


def test_no_need_to_wait_for_training_end(queue,monkeypatch):
    config,root,path=queue;job=root/'jobs/S4_050000'
    nm.atomic(job/'snapshot.json',{'test':True});nm.atomic(job/'status.json',{'status':'QUEUED'})
    launched=[]
    class Child:
        pid=999999
        def poll(self):return None
    def launch(command,*args,**kwargs):
        launched.append(command);return Child()
    monkeypatch.setattr(nm,'launch',launch)
    nm.watch(path,once=True)
    assert len(launched)==1 and 'S4_050000' in launched[0]
    assert nm.read(Path(config['models']['S4']['training_run'])/'status.json')['status']=='RUNNING'
    monkeypatch.setattr(nm,'alive',lambda *args:True)
    nm.watch(path,once=True)
    assert len(launched)==1


def test_missed_checkpoint_is_not_replaced(queue):
    config,root,path=queue
    nm.atomic(Path(config['models']['S0']['training_run'])/'status.json',
              {'identity':'S0','status':'RUNNING','completed':50200})
    nm.watch(path,once=True)
    states=nm.read(root/'TASKS.json')
    assert states['S0_050000']=='MISSED_CHECKPOINT'
    assert states['S0_060000']=='WAITING'


def test_stop_flag_stops_launch_not_checkpoint_preservation(queue,monkeypatch):
    config,root,path=queue;root.mkdir();(root/'STOP_SCHEDULING').touch()
    saved=[]
    def preserve(model,update,job):saved.append((model['arm'],update));return None
    monkeypatch.setattr(nm,'preserve',preserve)
    nm.watch(path,once=True)
    assert len(saved)==30 and nm.read(root/'status.json')['status']=='PAUSED'


def test_duplicate_observer_rejected(queue):
    _,root,path=queue
    with nm.lease(root/'observer.lock'):
        with pytest.raises(BlockingIOError):nm.watch(path,once=True)


def test_foreign_run_status_rejected(queue):
    config,_,path=queue
    nm.atomic(Path(config['models']['S0']['training_run'])/'status.json',
              {'identity':'foreign','completed':49999})
    with pytest.raises(ValueError,match='status identity'):nm.watch(path,once=True)


def test_other_live_gpu_group_blocks_next_milestone(queue,monkeypatch):
    _,root,path=queue
    job=root/'jobs/S0_060000';nm.atomic(job/'snapshot.json',{'test':True})
    nm.atomic(job/'status.json',{'status':'QUEUED'})
    def no_launch(*args,**kwargs):raise AssertionError('GPU slot already active')
    monkeypatch.setattr(nm,'launch',no_launch)
    with nm.lease(root/'jobs/S0_050000/gpu.lock'):
        nm.watch(path,once=True)


def lock_fixture(tmp_path):
    current={'identity':'current-id','index_sha256':'index-id','split':'navtest'}
    nm.atomic(tmp_path/'current/identity.json',current);nm.atomic(tmp_path/'metric.json',[])
    config=dict(current_root=str(tmp_path/'current'),metric_index=str(tmp_path/'metric.json'),
                evaluation_source='eval-source',observer_source='observer-source',authorization='explicit user request')
    model=dict(run_identity='run-S4',training_source_sha='train-source',arm='S4')
    cp=dict(run_identity='run-S4',training_source_sha='train-source',scope='formal',
            completed=50000,arm='C1',sha256='checkpoint-sha',
            model_class='starVLA.model.framework.ddp_action_video_foresight.DDPActionVideoForesight')
    return config,model,cp,current


def test_new_s_arm_lock_works_with_unchanged_exporter(tmp_path):
    config,model,cp,current=lock_fixture(tmp_path)
    lock=nm.build_lock(config,model,50000,cp)
    validate_lock(lock,cp,'eval-source',current,12146,42,10)
    assert lock['experimental_arm']=='S4' and cp['arm']=='C1'
    for changed in (dict(cp,completed=50200),dict(cp,run_identity='other')):
        with pytest.raises(ValueError):nm.build_lock(config,model,50000,changed)
    with pytest.raises(ValueError):nm.build_lock(config,model,40000,dict(cp,completed=40000))


def test_strict_export_coverage_and_failure_preservation(tmp_path):
    config,model,cp,current=lock_fixture(tmp_path);lock=nm.build_lock(config,model,50000,cp)
    bank=tmp_path/'bank'
    manifest=dict(checkpoint=cp,current_identity=current,evaluation_source='eval-source',limit=0,world_size=8,
        protocol=dict(precision='FP32',tf32=False,steps=10,sampling_seed=42,future_conditioning=False,
                      scorer=None,candidates_per_scene=1,auxiliary_heads_removed=True))
    nm.atomic(bank/'identity.json',manifest)
    assert not nm.export_complete(bank,lock)
    for rank in range(8):
        nm.atomic(bank/f'shard_{rank}.json',dict(identity_sha256=identity_hash(manifest),failed=0,
                  status='complete',completed=len(range(rank,12146,8))))
    assert nm.export_complete(bank,lock)
    failed=nm.read(bank/'shard_0.json');failed['failed']=1;nm.atomic(bank/'shard_0.json',failed)
    with pytest.raises(ValueError,match='failed scenes retained'):nm.export_complete(bank,lock)


def test_registration_tamper_rejected(tmp_path):
    nm.atomic(tmp_path/'registration.json',dict(identity='wrong',config={}))
    with pytest.raises(ValueError,match='registration'):nm.load_registration(tmp_path/'registration.json')


def test_only_exact_requested_tasks():
    config={'models':{'S0':{'arm':'S0'}}}
    assert nm.model_for(config,'S0_050000')[1]==50000
    for key in ('S0_050200','C0_050000','S0_50000','S0_025000'):
        with pytest.raises(ValueError):nm.model_for(config,key)


def test_gpu_headroom_never_kills_a_process(monkeypatch):
    calls=[]
    def query(command,**kwargs):calls.append(command);return '0, 19999\n1, 60000\n'
    monkeypatch.setattr(nm.subprocess,'check_output',query)
    assert not nm.gpu_headroom({'minimum_free_mib':20000},{'gpus':[0,1]})
    assert len(calls)==1 and calls[0][0]=='nvidia-smi'
