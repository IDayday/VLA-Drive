"""Synthetic/config-only review of the proposed two-invocation T1 check."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.st_world_policy.training_config import validate_config, official_runtime_policy
from tools.st_world_policy.budget_extension import _recipe, resolve_extension
from tools.foresight.student_state import learning_rate, validate_rank_batches
from tools.ddpolicy_vehicle.training_state import epoch_batches
from test_budget_extension import setup, resolve

PREP=Path(__file__).resolve().parents[1]/'official_T1_native4_preparation_v1'


def proposed():return json.loads((PREP/'PROPOSED_CONFIG_STEP2.json').read_text())


def test_actual_proposed_config_uses_history_current_only_original_schedule_and_official_backend():
    cfg=proposed();before=deepcopy(cfg)
    st=validate_config(cfg);policy=official_runtime_policy(cfg)
    assert cfg==before
    assert st.cell=='T1' and st.history and not st.future and not st.delta
    assert cfg['data'].get('teacher_provider') is None
    assert cfg['execution']==dict(family='temporal',global_batch=4,max_updates=20000,micro_batch=1,
        precision_policy='S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off',sample_limit=64,
        scope='SMOKE',seed=42,stop_after=2,world_size=2)
    assert cfg['optimizer']['warmup']==1000 and cfg['runtime']['save_every']==2
    assert not cfg['runtime']['gradient_checkpointing']
    assert policy['backend']['optimizer']=='torch.optim.AdamW'
    assert policy['backend']['master_dtype']=='float32'
    assert policy['backend']['attention']=='flash_attention_2'
    validate_rank_batches(64,4,2,1)


def test_phase2_then4_original_sampler_and_schedule_have_sixteen_exposures():
    cfg=proposed();ex=cfg['execution'];opt=cfg['optimizer']
    batches=epoch_batches(64,4,42,0)
    phase2=batches[:2];phase4=batches[2:4]
    assert sum(map(len,phase2))==8 and sum(map(len,phase4))==8
    assert len(set(i for batch in phase2+phase4 for i in batch))==16
    assert all(len(batch[rank::2])==2 for batch in phase2+phase4 for rank in (0,1))
    lr=lambda step:learning_rate(step,opt['lr'],opt['minimum_lr'],1000,20000)
    assert [lr(step) for step in (2,3)]==pytest.approx([3e-8,4e-8])
    assert [lr(step) for step in (2,3)]!=[lr(step) for step in (0,1)]
    extended=deepcopy(cfg);extended['execution']['stop_after']=4
    assert _recipe(cfg)==_recipe(extended)


@pytest.mark.parametrize('field',['history','future','teacher_root','pca_path','batch','world','schedule','warmup','backend'])
def test_phase_extension_cannot_change_any_proposed_learning_dependency(field):
    cfg=proposed();child=deepcopy(cfg);child['execution']['stop_after']=4
    if field in ('history','future'):child['space_time'][field]=not cfg['space_time'][field]
    elif field in ('teacher_root','pca_path'):child['data'][field]+='/other'
    elif field=='batch':child['execution']['global_batch']=8
    elif field=='world':child['execution']['world_size']=1
    elif field=='schedule':child['execution']['max_updates']=4
    elif field=='warmup':child['optimizer']['warmup']=1
    else:child['runtime']['official_execution']['preset']='other'
    assert _recipe(cfg)!=_recipe(child)


@pytest.mark.parametrize('mutation',['source','formal_weights','operator_paused','max_steps','rng_corrupt','base','data'])
def test_native_continuation_rejects_changed_boundary_or_provenance(tmp_path,monkeypatch,mutation):
    cfg,reg,folder=setup(tmp_path,monkeypatch)
    if mutation=='source':reg['source_fingerprint']='foreign'
    elif mutation=='formal_weights':reg['weights_usable_for_formal_init']=True
    elif mutation=='rng_corrupt':(folder/'rng_rank0.pt').write_bytes(b'changed')
    elif mutation in ('operator_paused','max_steps'):
        from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256
        path=folder/'COMPLETE.json';receipt=json.loads(path.read_text())
        receipt['state']['termination_reason']='OPERATOR_PAUSED' if mutation=='operator_paused' else 'MAX_STEPS'
        receipt['receipt_identity']=identity_hash({k:v for k,v in receipt.items() if k!='receipt_identity'})
        path.write_text(json.dumps(receipt));cfg['continuation']['parent_receipt_sha256']=file_sha256(path)
    if mutation in ('base','data'):
        with pytest.raises(ValueError):resolve_extension(cfg,reg,repo=tmp_path,
            data_identity={'observations':'other' if mutation=='data' else 'same'},
            base_binding='other' if mutation=='base' else 'base')
    else:
        with pytest.raises(ValueError):resolve(cfg,reg,tmp_path)
