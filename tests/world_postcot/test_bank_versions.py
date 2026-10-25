import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from starVLA.model.modules.qwen_world_postcot.candidate_bank import CandidateBank,CandidateBankWriter,CandidateRecord,trajectory_hash
from starVLA.model.modules.qwen_world_postcot.post_action import OFFICIAL_METRIC_NAMES
from starVLA.rl.qwen_world_postcot.batching import BatchBuilder
from starVLA.rl.qwen_world_postcot.bank_versions import BankVersions
from starVLA.rl.qwen_world_postcot.contracts import file_sha,fingerprint
from tools.qwen_world_postcot.train_il import consistent_cpu_data_error,completed_update_guard


def make_bank(path,offset=0):
    identity=dict(split='TRAIN',source_data_manifest_sha256='data',source_records_sha256='records',score_protocol_sha256='protocol')
    with CandidateBankWriter(path,identity,['s0','s1'],minimum_candidates=2) as writer:
        for token in ['s0','s1']:
            for index in range(2):
                poses=torch.zeros(8,3);poses[:,0]=torch.arange(1,9)*2+offset+index*.1
                labels=dict(zip(OFFICIAL_METRIC_NAMES,[1.-index,1.,1.,.75,1.,1.]))
                writer.append_many([CandidateRecord(scene_id=token,source_log='unit_log',split='TRAIN',timestamp_us=1,
                    observation_hash='observation_'+token,trajectory_fp32=poses.tolist(),trajectory_sha256=trajectory_hash(poses.numpy()),
                    raw_factor_labels=labels,valid_masks={key:True for key in labels},raw_total_score=.9*(1-index),
                    score_protocol_sha256='protocol',evaluator_revision={'source':'unit-protocol'},metric_version='canonical_v1.1',
                    context_sha256='unit-context',candidate_source='coherent_variant',actor_checkpoint_sha256=None,sampling={})])
        writer.finalize()
    return CandidateBank(path)


def builder(bank):
    data=SimpleNamespace(sha='data',manifest={'records_sha256':'records'},tokens=['s0','s1'])
    return BatchBuilder(data,None,None,bank,42)


def refresh_config(path):
    return dict(enabled=True,produce_every_updates=10000,adopt_every_updates=5000,new_candidates_per_scene=2,max_candidates=16,registry=str(path))


def source_entry(path,step,bank,parent,identity):
    source=path/f'source{step}';(source/'native').mkdir(parents=True)
    model=source/'native/mp_rank_00_model_states.pt';model.write_bytes(b'unit-test-checkpoint-not-a-real-policy')
    (source/'identity.json').write_text(json.dumps(identity));files={'native/mp_rank_00_model_states.pt':file_sha(model)}
    (source/'files.json').write_text(json.dumps(files));(source/'COMPLETE').write_text(json.dumps(dict(status='COMPLETE',updates=step,
        identity_sha256=fingerprint(identity),files_sha256=file_sha(source/'files.json'),model_file='native/mp_rank_00_model_states.pt')))
    return dict(source_updates=step,source_checkpoint=str(source),source_model_sha256=file_sha(model),parent_bank_hash=parent,
        bank_root=str(Path(bank.manifest_path).parent),bank_hash=bank.bank_hash,data_manifest_sha256='data',score_protocol_sha256='protocol')


def test_boundary_lock_adoption_rng_and_checkpoint_restore_ignore_later_registry(tmp_path):
    initial=make_bank(tmp_path/'initial');new=make_bank(tmp_path/'new',1);later=make_bank(tmp_path/'later',2)
    current=builder(initial);registry=tmp_path/'REGISTRY.json';cfg=refresh_config(registry);identity=dict(qualification=False,world=1)
    versions=BankVersions(tmp_path/'run',cfg,current,identity)
    versions.adopt(current,0)
    assert current.bank.bank_hash==initial.bank_hash
    assert versions.adopt(current,4999) is None
    versions.adopt(current,10000) # Missing registry locks the original version.
    entry=source_entry(tmp_path,10000,new,initial.bank_hash,identity)
    registry.write_text(json.dumps(dict(schema='world_postcot_bank_registry_v1',entries=[entry])))
    versions.adopt(current,10000)
    assert current.bank.bank_hash==initial.bank_hash
    rng={name:getattr(current,name).get_state().clone() for name in ('aux_rng','future_rng','candidate_rng')}
    global_rng=torch.get_rng_state().clone()
    versions.adopt(current,15000)
    assert current.bank.bank_hash==new.bank_hash
    assert all(torch.equal(value,getattr(current,name).get_state()) for name,value in rng.items())
    assert torch.equal(global_rng,torch.get_rng_state())
    state=current.state_dict();stream=io.BytesIO();torch.save(state,stream);stream.seek(0)
    expected_pair=current.bank.sample_pairs(['s0'],generator=current.candidate_rng).record_ids
    expected_aux=torch.randperm(10,generator=current.aux_rng);expected_future=torch.rand(10,generator=current.future_rng)
    next_entry=source_entry(tmp_path,20000,later,new.bank_hash,identity)
    registry.write_text(json.dumps(dict(schema='world_postcot_bank_registry_v1',entries=[entry,next_entry])))
    restored=builder(CandidateBank(tmp_path/'initial'));restored.load_state_dict(torch.load(stream,weights_only=False))
    migrated=BankVersions(tmp_path/'new_output',cfg,builder(CandidateBank(tmp_path/'initial')),identity)
    migrated.adopt(restored,15000) # Recreates the saved locked boundary in a new output directory.
    assert restored.bank.bank_hash==new.bank_hash
    assert restored.bank.sample_pairs(['s0'],generator=restored.candidate_rng).record_ids==expected_pair
    assert torch.equal(torch.randperm(10,generator=restored.aux_rng),expected_aux)
    assert torch.equal(torch.rand(10,generator=restored.future_rng),expected_future)
    migrated.adopt(restored,20000)
    assert restored.bank.bank_hash==later.bank_hash
    current.bank.close();restored.bank.close();new.close();later.close()


def test_registry_wrong_data_and_corrupt_saved_binding_fail(tmp_path):
    initial=make_bank(tmp_path/'initial');new=make_bank(tmp_path/'new',1);current=builder(initial)
    registry=tmp_path/'REGISTRY.json';identity=dict(qualification=False);versions=BankVersions(tmp_path/'run',refresh_config(registry),current,identity)
    versions.adopt(current,0);entry=source_entry(tmp_path,10000,new,initial.bank_hash,identity)
    entry['data_manifest_sha256']='different'
    registry.write_text(json.dumps(dict(schema='world_postcot_bank_registry_v1',entries=[entry])))
    with pytest.raises(RuntimeError,match='data/score'):versions.adopt(current,10000)
    state=current.state_dict();state['bank_binding']['selected']['bank_hash']='changed'
    with pytest.raises(ValueError,match='boundary binding'):current.load_state_dict(state)
    current.bank.close();new.close()


def test_cpu_failure_and_optimizer_skip_nonfinite_guards(monkeypatch):
    consistent_cpu_data_error(None,torch.device('cpu'))
    with pytest.raises(RuntimeError,match='no targets/cursor substituted'):consistent_cpu_data_error('broken target',torch.device('cpu'))
    engine=SimpleNamespace(global_steps=1,skipped_steps=0,optimizer=SimpleNamespace(overflow=False),get_global_grad_norm=lambda:2.5)
    assert completed_update_guard(engine,0,0,torch.device('cpu'))['grad_norm']==2.5
    for field,value in [('global_steps',0),('skipped_steps',1)]:
        before=getattr(engine,field);setattr(engine,field,value)
        with pytest.raises(FloatingPointError):completed_update_guard(engine,0,0,torch.device('cpu'))
        setattr(engine,field,before)
    engine.optimizer.overflow=True
    with pytest.raises(FloatingPointError):completed_update_guard(engine,0,0,torch.device('cpu'))
    engine.optimizer.overflow=False;engine.get_global_grad_norm=lambda:float('nan')
    with pytest.raises(FloatingPointError):completed_update_guard(engine,0,0,torch.device('cpu'))
    # Simulate a CPU read failure confined to the other rank. Every healthy
    # peer sees the same fixed error collective and fails before denominators.
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:2)
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda value,**kwargs:value.fill_(1))
    def gather(output,value):output[:]=[None,'rank1 target failed']
    monkeypatch.setattr(torch.distributed,'all_gather_object',gather)
    with pytest.raises(RuntimeError,match='rank1 target failed'):consistent_cpu_data_error(None,torch.device('cpu'))
