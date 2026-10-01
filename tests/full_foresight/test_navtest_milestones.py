"""CPU orchestration contracts; no mock result is a driving benchmark."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
from tools.full_foresight import navtest_schedule as ns
from tools.full_foresight import navtest_milestones as nm


@pytest.fixture
def checkpoint(tmp_path):
    run=tmp_path/'run';folder=run/'checkpoints/periodic_070000';folder.mkdir(parents=True)
    ns.atomic(run/'identity.json',{'identity':'training-id'})
    ns.atomic(folder/'COMPLETE.json',{'identity':'training-id','tag':folder.name,
                                     'completed':70000,'exposure':2239840})
    for rank in range(8):
        (folder/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').write_bytes(b'optimizer')
        (folder/f'rng_rank{rank}.pt').write_bytes(b'rng')
    (folder/'mp_rank_00_model_states.pt').write_bytes(b'model')
    model={'arm':'C0','run_identity':'training-id','training_run':str(run)}
    return model,folder,tmp_path/'job'


def test_exact_snapshot_survives_rolling_gc(checkpoint):
    model,source,job=checkpoint;s=ns.preserve(model,70000,job)
    target=job/'frozen_student/checkpoints'/s['tag']
    assert (source/'mp_rank_00_model_states.pt').stat().st_ino==(target/'mp_rank_00_model_states.pt').stat().st_ino
    shutil.rmtree(source)
    assert (target/'mp_rank_00_model_states.pt').read_bytes()==b'model'
    assert ns.preserve(model,70000,job)==s


def test_checkpoint_metadata_publication_resume(checkpoint):
    model,source,job=checkpoint;ns.preserve(model,70000,job)
    (job/'snapshot.json').unlink();(job/'models.json').unlink();shutil.rmtree(source)
    s=ns.preserve(model,70000,job)
    assert s['metadata_recovered'] and s['completed']==70000
    assert ns.read(job/'models.json')[0]['checkpoint_tag']=='periodic_070000'


def test_never_substitute_newer_checkpoint(checkpoint):
    model,source,job=checkpoint;shutil.rmtree(source)
    other=source.with_name('periodic_070200');other.mkdir()
    ns.atomic(other/'COMPLETE.json',{'tag':other.name,'completed':70200})
    assert ns.find_exact(model['training_run'],70000) is None
    assert ns.preserve(model,70000,job) is None


@pytest.mark.parametrize('key,value',[('completed',70001),('tag','periodic_070200'),('identity','foreign')])
def test_reject_wrong_checkpoint_identity(checkpoint,key,value):
    model,source,job=checkpoint;r=ns.read(source/'COMPLETE.json');r[key]=value;ns.atomic(source/'COMPLETE.json',r)
    with pytest.raises(ValueError):ns.preserve(model,70000,job)


def test_unfinished_save_is_not_captured(checkpoint):
    model,source,job=checkpoint;(source/'COMPLETE.json').unlink()
    assert ns.preserve(model,70000,job) is None


@pytest.mark.parametrize('name',['rng_rank7.pt','bf16_zero_pp_rank_7_mp_rank_00_optim_states.pt','mp_rank_00_model_states.pt'])
def test_missing_shard_is_rejected(checkpoint,name):
    model,source,job=checkpoint;(source/name).unlink()
    with pytest.raises(ValueError):ns.preserve(model,70000,job)


def test_link_failure_never_mutates_training(checkpoint,monkeypatch):
    model,source,job=checkpoint;original={p.name:p.read_bytes() for p in source.iterdir()}
    def fail(*args):raise FileNotFoundError('synthetic rolling race')
    monkeypatch.setattr(ns.os,'link',fail)
    with pytest.raises(FileNotFoundError):ns.preserve(model,70000,job)
    assert {p.name:p.read_bytes() for p in source.iterdir()}==original
    assert not (job/'frozen_student').exists() and not list(job.glob('.snapshot_*'))


def test_single_observer_lease(tmp_path):
    path=tmp_path/'observer.lock'
    with ns.lease(path):
        assert ns.busy(path)
        with pytest.raises(BlockingIOError):
            with ns.lease(path):pass
    assert not ns.busy(path)


def test_host_development_handoff_does_not_launch(tmp_path):
    run=tmp_path/'students/run';run.mkdir(parents=True);model={'training_run':str(run)};c={'campaign_root':str(tmp_path)}
    ns.atomic(run/'status.json',{'status':'PAUSED'})
    ns.atomic(tmp_path/'priority_queues/run.json',{'status':'RUNNING'})
    assert not ns.host_ready(c,model)
    ns.atomic(run/'status.json',{'status':'RUNNING'});assert ns.host_ready(c,model)
    ns.atomic(run/'status.json',{'status':'COMPLETE'});assert not ns.host_ready(c,model)
    ns.atomic(tmp_path/'priority_queues/run.json',{'status':'MAIN_TRAINING_COMPLETE_EVALUATION_TRACKED_SEPARATELY'})
    assert ns.host_ready(c,model)


def test_remote_argv_is_quoted_not_shell_interpolated():
    cfg={'worker_python':'/usr/bin/python3','source_worktree':'/some/source with space'}
    cmd=nm.remote_command(cfg,{'host':'approved-host'},['task','--registration','/some/path with space','--task','C0_070000'])
    assert cmd[-1]=="cd '/some/source with space' && exec /usr/bin/python3 -u -m tools.full_foresight.navtest_milestones task --registration '/some/path with space' --task C0_070000"


def test_run_ids_and_noise_protocol_independent_of_queue():
    reg={'identity':'1234567890abrest'}
    assert nm.task_id(reg,'C0_070000',1)!=nm.task_id(reg,'C0_070000',2)
    assert nm.environment('3')['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert nm.environment('3')['CUDA_VISIBLE_DEVICES']=='3'


def test_cpu_resume_uses_new_meter_id_and_same_inputs(tmp_path):
    job=tmp_path/'job';cfg={'scoring_python':'python','devkit':'devkit','metric_index':'index','current_root':'current',
       'campaign_root':'campaign','cpu_workers':16,'task_timeout_seconds':21600};reg={'identity':'abcdef123456'}
    first=nm.score_command(cfg,reg,'C0_070000',job,1)
    ns.atomic(job/'scores/identity.json',{'unit_test':True})
    resumed=nm.score_command(cfg,reg,'C0_070000',job,1,1)
    assert '--resume' not in first and '--resume' in resumed
    assert first[first.index('--run-id')+1]!=resumed[resumed.index('--run-id')+1]
    for key in ['--predictions','--current-index','--metric-index','--devkit']:
        assert first[first.index(key)+1]==resumed[resumed.index(key)+1]


def test_registration_tampering_is_rejected(tmp_path):
    p=tmp_path/'registration.json';ns.atomic(p,{'identity':'bad','config':{},'created_unix':0,'asset_hashes':{}})
    with pytest.raises(ValueError,match='registration'):ns.load_registration(p)


def test_unknown_config_does_not_silently_pass():
    with pytest.raises(ValueError,match='Unknown/missing'):ns.validate({'schema':ns.SCHEMA,'ignored_option':True})


def test_strict_full_population_score_validation_real_assets():
    # Read-only replay of already completed real results, not a new model score.
    art=Path('/mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_latest_three_20260930_v2')
    if not (art/'completion.json').exists():pytest.skip('Existing complete real audit unavailable')
    s,rows=ns.validate_scores(art/'C0_scores_v1',art/'cache_snapshot.csv',66200)
    assert len(rows)==12146 and s['failed']==0
    with pytest.raises(ValueError):ns.validate_scores(art/'C0_scores_v1',art/'cache_snapshot.csv',70000)


def test_pressure_release_only_exact_owned_process(tmp_path):
    # Real CPU dummy processes, no CUDA/pressure workload.
    script=tmp_path/'dummy_pressure.py';script.write_text('import time\ntime.sleep(60)\n')
    root=tmp_path/'campaign';alloc=root/'allocations';alloc.mkdir(parents=True)
    ledger=alloc/'test_run_1.json';log=alloc/'test_run_1_pressure_gpu0.log'
    cfg={'campaign_root':str(root),'gpus':[0],'pressure_python':sys.executable,'pressure_script':str(script)}
    model={'training_run':'/unused/students/test_run'};job=tmp_path/'job'
    children=[]
    try:
        with log.open('w') as stream:
            owned=subprocess.Popen([sys.executable,'-u',str(script),'dummy'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(owned)
        other=subprocess.Popen([sys.executable,'-u',str(script),'dummy'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        children.append(other);time.sleep(.1)
        ns.atomic(ledger,{'host':nm.socket.gethostname(),'restored_pressure':[{'parent_pid':other.pid,'gpu':0}]})
        assert nm.release_owned_pressure(cfg,model,job)==[] and other.poll() is None
        ns.atomic(ledger,{'host':nm.socket.gethostname(),'restored_pressure':[{'parent_pid':owned.pid,'gpu':0},{'parent_pid':other.pid,'gpu':0}]})
        records=nm.release_owned_pressure(cfg,model,job)
        owned.wait(timeout=5)
        assert [r['pid'] for r in records]==[owned.pid] and other.poll() is None
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
            child.wait(timeout=5)


def queue_fixture(tmp_path,monkeypatch):
    models=[]
    root=tmp_path/'observer'
    for arm in ['C0','C1','C4']:
        run=tmp_path/'students'/arm;ns.atomic(run/'identity.json',{'identity':arm})
        ns.atomic(run/'status.json',{'status':'RUNNING','completed':69000})
        model={'arm':arm,'training_run':str(run),'run_identity':arm,'host':arm,'hostname':arm}
        models.append(model)
    config={'artifact_root':str(root),'models':models,'updates':[70000,80000,90000,100000],
            'evaluation_source_sha':'test-source',
            'source_worktree':str(tmp_path),'campaign_root':str(tmp_path),'worker_python':sys.executable,
            'poll_seconds':.01,'watch_timeout_seconds':600,'evaluation_gpu_hours':150}
    reg={'config':config,'identity':'test-schedule','created_unix':time.time()}
    monkeypatch.setattr(nm,'load_registration',lambda _:reg)
    monkeypatch.setattr(nm,'host_ready',lambda c,m:True)
    monkeypatch.setattr(nm,'usage',lambda c,i:0)
    return config,reg


def test_watch_waits_for_future_milestones(tmp_path,monkeypatch):
    cfg,reg=queue_fixture(tmp_path,monkeypatch)
    def no_launch(*a,**k):raise AssertionError('No completed milestone can launch')
    monkeypatch.setattr(nm.subprocess,'Popen',no_launch)
    nm.watch(tmp_path/'registration.json',once=True)
    state=ns.read(Path(cfg['artifact_root'])/'status.json')
    assert state['status']=='RUNNING' and state['completed_tasks']==0 and len(state['tasks'])==12
    assert set(state['tasks'].values())=={'WAITING'}


def test_watch_missed_checkpoint_is_explicit(tmp_path,monkeypatch):
    cfg,reg=queue_fixture(tmp_path,monkeypatch)
    ns.atomic(Path(cfg['models'][0]['training_run'])/'status.json',{'status':'RUNNING','completed':70200})
    nm.watch(tmp_path/'registration.json',once=True)
    states=ns.read(Path(cfg['artifact_root'])/'TASKS.json')
    assert states['C0_070000']=='MISSED_CHECKPOINT' and states['C0_080000']=='WAITING'


def test_active_group_blocks_next_model_task(tmp_path,monkeypatch):
    cfg,reg=queue_fixture(tmp_path,monkeypatch);root=Path(cfg['artifact_root'])
    job=root/'jobs/C0_080000';ns.atomic(job/'snapshot.json',{'unit_test':True});ns.atomic(job/'status.json',{'status':'QUEUED'})
    monkeypatch.setattr(nm,'preserve',lambda *a:None)
    def no_launch(*a,**k):raise AssertionError('An active GPU group owns the host slot')
    monkeypatch.setattr(nm.subprocess,'Popen',no_launch)
    with ns.lease(root/'jobs/C0_070000/gpu_group.lock'):
        nm.watch(tmp_path/'registration.json',once=True)
    assert ns.read(root/'status.json')['active_models']==['C0']


def test_stop_scheduling_preserves_but_never_launches(tmp_path,monkeypatch):
    cfg,reg=queue_fixture(tmp_path,monkeypatch);root=Path(cfg['artifact_root']);root.mkdir()
    (root/'STOP_SCHEDULING').touch()
    def no_launch(*a,**k):raise AssertionError('Stopped observer cannot launch')
    monkeypatch.setattr(nm.subprocess,'Popen',no_launch)
    nm.watch(tmp_path/'registration.json',once=True)
    assert ns.read(root/'status.json')['status']=='PAUSED'


def test_changed_schedule_cannot_reuse_observer_root(tmp_path,monkeypatch):
    cfg,reg=queue_fixture(tmp_path,monkeypatch);root=Path(cfg['artifact_root'])
    ns.atomic(root/'status.json',{'registration':'different'})
    with pytest.raises(ValueError,match='different schedule'):nm.watch(tmp_path/'registration.json',once=True)


def test_pressure_never_restored_during_training(tmp_path,monkeypatch):
    run=tmp_path/'students/run';job=tmp_path/'job';ns.atomic(run/'status.json',{'status':'RUNNING'})
    ns.atomic(job/'pressure_releases.json',[{'gpu':0,'verified_cmdline':['never']}])
    def no_query(*a,**k):raise AssertionError('No pressure starts on training GPUs')
    monkeypatch.setattr(nm.subprocess,'check_output',no_query)
    nm.restore_owned_pressure({}, {'training_run':str(run)},job)


def test_queued_task_launch_record_is_per_job_and_restart_is_idempotent(tmp_path,monkeypatch):
    cfg,reg=queue_fixture(tmp_path,monkeypatch);root=Path(cfg['artifact_root'])
    job=root/'jobs/C0_070000'
    ns.atomic(job/'snapshot.json',{'unit_test':True});ns.atomic(job/'status.json',{'status':'QUEUED'})
    monkeypatch.setattr(nm,'preserve',lambda *a:None)
    calls=[]
    class Child:
        pid=12345
        def poll(self):return None
    def launch(cmd,**kwargs):calls.append(cmd);return Child()
    monkeypatch.setattr(nm.subprocess,'Popen',launch)
    monkeypatch.setattr(nm,'live_owner',lambda record,*args:record['pid']==12345)
    nm.watch(tmp_path/'registration.json',once=True)
    assert len(calls)==1 and ns.read(job/'launch.json')['pid']==12345
    assert not (root/'jobs/C4_100000/launch.json').exists()
    nm.watch(tmp_path/'registration.json',once=True)
    assert len(calls)==1


def test_recover_missing_model_manifest_with_existing_snapshot(checkpoint):
    model,source,job=checkpoint;ns.preserve(model,70000,job)
    (job/'models.json').unlink();shutil.rmtree(source)
    ns.preserve(model,70000,job)
    assert ns.read(job/'models.json')[0]['checkpoint_tag']=='periodic_070000'
