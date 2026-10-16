import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.analysis.fm_transfer import dispatch_final_evaluation as dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint


def fixture_lock(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    monkeypatch.setattr(dispatch,'EVALUATION_SOURCE',source)
    atomic_json(source/'SOURCE_COMPLETE.json',dict(git_archive_extraction_complete=True))
    for name in ('simwam_recipe','fm_transfer'):
        p=source/'scripts/analysis'/name/'evaluate_policy.py';p.parent.mkdir(parents=True);p.write_text('qualified worker')
    pop=tmp_path/'population.json';atomic_json(pop,dict(identity_sha256='fixed-pop'))
    monkeypatch.setattr(dispatch,'POPULATION',pop)
    models={}
    for name in dispatch.MODELS:
        export=tmp_path/name;export.mkdir()
        (export/'pytorch_model.pt').write_bytes(name.encode());(export/'config.yaml').write_text('configuration')
        info=dict(qualification=False,run_identity=name,new_SFT=dict(own=name.rsplit('_',1)[0]),
            stage=name.rsplit('_',1)[1],updates=20000,weight_sha256=file_sha(export/'pytorch_model.pt'))
        info=dispatch.publish(export/'export.json',info);atomic_json(export/'COMPLETE',info)
        models[name]=dict(export=str(export),export_receipt_sha256=file_sha(export/'export.json'),
            config_sha256=file_sha(export/'config.yaml'),**{k:info[k] for k in
            ['run_identity','new_SFT','stage','updates','weight_sha256']})
    value=dict(schema='fm_transfer_final_four_models_v1',status='LOCKED_BEFORE_FINAL_NAVTEST',
        formal_budget_complete=True,models=models,inference_seeds=[2101],training_seeds=[42],
        modes=['E_NATIVE','E_SHIFT5'],Navtest_scores_read_for_selection=False,
        evaluation_source=str(source),population_identity='fixed-pop',
        sampler_sha256=file_sha(source/'scripts/analysis/simwam_recipe/evaluate_policy.py'),
        transfer_evaluation_sha256=file_sha(source/'scripts/analysis/fm_transfer/evaluate_policy.py'))
    lock=tmp_path/'LOCK.json';dispatch.publish(lock,value)
    return lock,source,value


def test_lock_selects_only_registered_own_endpoint_and_unchanged_worker(tmp_path,monkeypatch):
    lock,source,value=fixture_lock(tmp_path,monkeypatch)
    _,model=dispatch.validate_selection(lock,'A_DDP_TIME_RL',source)
    cmd=dispatch.worker_command(source,model,tmp_path/'output',lock,17,24)
    assert cmd[2]=='scripts.analysis.fm_transfer.evaluate_policy'
    assert cmd[cmd.index('--shard')+1]=='17' and cmd[cmd.index('--shards')+1]=='24'
    assert cmd[cmd.index('--cohort')+1]=='NAVTEST' and '--max-scenes' not in cmd
    assert cmd[cmd.index('--final-lock')+1]==str(lock)
    (Path(model['export'])/'pytorch_model.pt').write_bytes(b'changed-model')
    with pytest.raises(ValueError,match='changed after lock'):
        dispatch.validate_selection(lock,'A_DDP_TIME_RL',source)


@pytest.mark.parametrize('field,value',[
    ('formal_budget_complete',False),('inference_seeds',[2101,2102]),
    ('modes',['E_SHIFT5']),('Navtest_scores_read_for_selection',True),
    ('population_identity','different-population')])
def test_partial_or_changed_scientific_contract_refused(tmp_path,monkeypatch,field,value):
    lock,source,body=fixture_lock(tmp_path,monkeypatch)
    body[field]=value;dispatch.publish(lock,body)
    with pytest.raises(ValueError):dispatch.validate_selection(lock,'A_DDP_TIME_RL',source)


def test_changed_worker_and_export_receipt_refused(tmp_path,monkeypatch):
    lock,source,body=fixture_lock(tmp_path,monkeypatch)
    worker=source/'scripts/analysis/fm_transfer/evaluate_policy.py';worker.write_text('changed')
    with pytest.raises(ValueError,match='source changed'):dispatch.validate_selection(lock,'A_DDP_TIME_RL',source)


def test_global_shards_keep_exact_membership_without_duplicate_gpu():
    mapping=dispatch.partition(24,list(range(8,16)),list(range(8)))
    assert mapping==list(zip(range(8,16),range(8)))
    assigned=[list(range(12146))[rank::24] for rank in range(24)]
    assert sorted(x for shard in assigned for x in shard)==list(range(12146))
    for ranks,gpus in [([0,0],[0,1]),([0,1],[0,0]),([24],[0]),([0],[8]),([0,1],[0])]:
        with pytest.raises(ValueError):dispatch.partition(24,ranks,gpus)


def inventory(*,free=70000,used=0,owners=None):
    return dict(gpus={0:dict(uuid='physical0',free_MiB=free,used_MiB=used)},owners=owners or [])


def test_admission_requires_two_samples_visible_owners_and_unique_uuid():
    empty=inventory();assert dispatch.admit_devices(empty,empty,[(0,0)],set())=={'physical0'}
    known=inventory(used=16000,owners=[dict(GPU_uuid='physical0',visible=True)])
    assert dispatch.admit_devices(known,known,[(0,0)],set())=={'physical0'}
    for first,second,reserved in [(empty,empty,{'physical0'}),(inventory(free=20000),empty,set()),
            (empty,inventory(used=70000),set()),
            (known,inventory(used=16000,owners=[dict(GPU_uuid='physical0',visible=False)]),set())]:
        with pytest.raises(RuntimeError):dispatch.admit_devices(first,second,[(0,0)],reserved)


def attempt(tmp_path):
    queue=tmp_path/'queue';queue.mkdir();output=tmp_path/'output';output.mkdir()
    contract=dict(shards=24,model='A_DDP_TIME_RL',output=str(output))
    atomic_json(output/'dispatch_contract.json',contract)
    path=output/'dispatches/old_rank0.json'
    path.parent.mkdir()
    record=dispatch.publish(path,dict(state='STARTED',rank=0,output=str(output),
        contract_sha256=fingerprint(contract),GPU_uuid='physical0',pid=123,start_ticks=456,host='host'))
    global_path=queue/'workers/A_DDP_TIME_RL/old_rank0.json';global_path.parent.mkdir(parents=True)
    dispatch.publish(global_path,{k:v for k,v in record.items() if k!='sha256'})
    return queue,output,contract,record


def test_retry_preserves_scene_and_attempt_evidence_and_requires_proven_exit(tmp_path):
    queue,output,contract,record=attempt(tmp_path)
    evidence=output/'saved_scene.pt';evidence.write_bytes(b'valid scene payload')
    with pytest.raises(ValueError,match='explicit'):dispatch.previous_attempts(queue,output,contract,[0],False,lambda r:False)
    with pytest.raises(RuntimeError,match='still alive'):dispatch.previous_attempts(queue,output,contract,[0],True,lambda r:True)
    def unknown(record):raise TimeoutError('SSH observation timed out')
    with pytest.raises(TimeoutError):dispatch.previous_attempts(queue,output,contract,[0],True,unknown)
    parents,reserved=dispatch.previous_attempts(queue,output,contract,[0],True,lambda r:False)
    assert parents=={0:[record['sha256']]} and not reserved
    assert evidence.read_bytes()==b'valid scene payload' and (output/'dispatches/old_rank0.json').exists()
    atomic_json(output/'rank0.json',dict(status='ASSIGNED_COMPLETE'))
    with pytest.raises(ValueError,match='Completed'):dispatch.previous_attempts(queue,output,contract,[0],True,lambda r:False)


def test_unresolved_launch_is_not_automatically_retriable(tmp_path):
    with pytest.raises(RuntimeError,match='Unresolved launch'):
        dispatch.live(dict(state='LAUNCHING'))


def test_cross_model_loading_reservation_and_changed_shard_plan(tmp_path):
    queue,output,contract,_=attempt(tmp_path)
    parents,reserved=dispatch.previous_attempts(queue,output,contract,[1],False,lambda r:True)
    assert parents=={} and reserved=={'physical0'}
    with pytest.raises(ValueError,match='identity'):
        dispatch.previous_attempts(queue,output,dict(contract,shards=8),[1],False,lambda r:False)


def test_missing_lock_stops_before_gpu_probe_or_model_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(dispatch,'require_dispatch',lambda:'allowed-host')
    def forbidden(*a,**kw):pytest.fail('No GPU activity before final lock')
    monkeypatch.setattr(dispatch,'gpu_snapshot',forbidden);monkeypatch.setattr(dispatch.subprocess,'Popen',forbidden)
    monkeypatch.setattr('sys.argv',['dispatch','--lock',str(tmp_path/'LOCK.json'),'--model','A_DDP_TIME_RL',
        '--shards','8','--ranks','0','--gpus','0'])
    with pytest.raises(FileNotFoundError):dispatch.main()


def test_launch_receipt_precedes_child_and_preserves_exact_worker_flags(tmp_path,monkeypatch):
    queue=tmp_path/'final_navtest';queue.mkdir();source=tmp_path/'source';source.mkdir()
    lock=queue/'LOCK.json';lock.write_text('{}')
    chosen=dict(export='own-SFT-export',weight_sha256='weight',export_receipt_sha256='receipt')
    monkeypatch.setattr(dispatch,'require_dispatch',lambda:'allowed-host')
    monkeypatch.setattr(dispatch,'validate_selection',lambda *a:(dict(sha256='selection'),chosen))
    monkeypatch.setattr(dispatch,'gpu_snapshot',inventory);monkeypatch.setattr(dispatch.time,'sleep',lambda t:None)
    captures=[]
    def launch(cmd,**kw):
        path=next((tmp_path/'evaluations/A_DDP_TIME_RL_final_NAVTEST/dispatches').glob('*rank0.json'))
        assert dispatch.authenticated(path)['state']=='LAUNCHING'
        captures.append((cmd,kw));return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(dispatch.subprocess,'Popen',launch)
    monkeypatch.setattr('sys.argv',['dispatch','--lock',str(lock),'--source',str(source),
        '--model','A_DDP_TIME_RL','--shards','24','--ranks','0','--gpus','0'])
    dispatch.main()
    assert len(captures)==1
    cmd,kw=captures[0]
    assert kw['env']['PYTHONPATH']==str(source) and kw['env']['CUDA_VISIBLE_DEVICES']=='0'
    assert kw['cwd']==str(source) and cmd[cmd.index('--shards')+1]=='24'
    records=list((queue/'workers/A_DDP_TIME_RL').glob('*.json'))
    value=dispatch.authenticated(records[0]);assert value['state']=='STARTED' and value['pid']==os.getpid()
    assert value['parent_attempts']==[] and value['GPU_uuid']=='physical0'
