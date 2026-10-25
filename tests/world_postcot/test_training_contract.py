"""CPU contracts; native multi-GPU DeepSpeed acceptance is a separate audit."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from tools.qwen_world_postcot.train_il import groups, learning_scale, learning_scales, initialize_joint_optimizer, verify_generic_qwen_identity, validate, optimizer_group_contract
from starVLA.rl.qwen_world_postcot import checkpoint as cp
from starVLA.rl.qwen_world_postcot.batching import update_denominators
from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint
from starVLA.rl.qwen_world_postcot.loading import inspect_checkpoint, validate_rl_parent_selection
from starVLA.model.modules.qwen_world_postcot.readouts import valid_scene_mse
from starVLA.model.modules.qwen_world_postcot.post_action import post_action_loss


def config():
    return yaml.safe_load(Path('configs/qwen_world_postcot/il.yaml').read_text())


def test_recipe_rejects_ignored_and_unknown_configuration_fields():
    assert validate(config())['FM_repeat']==1
    for path,value in [('scheduler.type','constant'),('optimizer.name','SGD'),('flow.training_loss','velocity_MSE'),
                       ('flow.sigma_formula','uniform'),('dev_every',250),('not_wired',True),('optimizer.not_wired',True),
                       ('model.action.num_layers',2),('model.memory.global_tokens',4)]:
        cfg=config();parts=path.split('.');node=cfg
        for part in parts[:-1]:node=node[part]
        node[parts[-1]]=value
        with pytest.raises(ValueError):validate(cfg)


def test_identity_normalizes_actual_qwen_integer_keys_without_discarding_contracts():
    live=dict(config={'spatial_grid':(6,8)},qwen_config={'id2label':{0:'LABEL_0',1:'LABEL_1'}})
    on_disk=json.loads(json.dumps(live))
    assert live!=on_disk
    assert cp.canonical_identity(live)==on_disk
    changed=copy.deepcopy(on_disk);changed['config']['spatial_grid'][0]=5
    assert cp.identity_diff(on_disk,changed)==[dict(field='config.spatial_grid',expected=[6,8],actual=[5,8])]


def test_deployment_construction_does_not_mutate_sealed_qwen_identity(monkeypatch):
    from transformers import Qwen3VLConfig
    from starVLA.model.modules.qwen_world_postcot.qwen_encoder import SelectedQwenEncoder
    from starVLA.rl.qwen_world_postcot import loading

    # Exercise the real Transformers config/parser and native encoder on meta:
    # no GPU and no allocation of the actual 2B backbone's stored weights.
    qwen=Qwen3VLConfig(text_config=dict(hidden_size=2048,num_hidden_layers=28,
        num_attention_heads=16,num_key_value_heads=8,intermediate_size=6144,head_dim=128,
        rope_scaling=dict(rope_type='default',mrope_section=[24,20,20],mrope_interleaved=True)),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
                           out_hidden_size=2048)).to_dict()
    qwen['attn_implementation']='sdpa'
    original=copy.deepcopy(qwen)
    with torch.device('meta'):
        encoder=SelectedQwenEncoder(None,checkpoint_config=qwen)
    assert encoder.num_layers==28 and qwen==original

    info=dict(identity=dict(qwen_config=qwen,model_config=dict(unit_only=True),train_heads=False),
              seal=dict(model_file='native/model.pt'))
    receipt_before=copy.deepcopy(info)
    monkeypatch.setattr(loading,'inspect_checkpoint',lambda *args,**kwargs:info)
    monkeypatch.setattr(loading.torch,'load',lambda *args,**kwargs:dict(module={}))

    def mutating_constructor(path,*,config,checkpoint_config,train_heads):
        # Simulate a downstream config consumer that owns and changes its args.
        Qwen3VLConfig.from_dict(checkpoint_config)
        checkpoint_config.pop('attn_implementation',None)
        config.pop('unit_only')
        return nn.Module()

    monkeypatch.setattr(loading,'QwenWorldPostCoT',mutating_constructor)
    _,returned=loading.load_deployment_checkpoint('/unit-checkpoint',device='meta')
    assert returned is info and returned==receipt_before


class TinyGroups(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen=nn.Module();self.qwen.model=nn.Module();self.qwen.model.visual=nn.Module()
        self.qwen.model.visual.patch=nn.Linear(1,1)
        self.qwen.model.visual.merger=nn.Linear(1,1)
        self.qwen.model.language_model=nn.Linear(1,1)
        self.actor=nn.Linear(1,1)


def test_optimizer_group_contract_is_ordered_and_yaml_key_order_invariant():
    cfg=config();model=TinyGroups();a=optimizer_group_contract(model,groups(model,cfg))
    cfg['learning_rates']=dict(reversed(list(cfg['learning_rates'].items())))
    b=optimizer_group_contract(model,groups(model,cfg))
    assert a==b
    assert [row['name'] for row in a]==['qwen_language_merger','qwen_vision','action_memory_aux_heads']
    assert all(row['parameters'] and all(p['dtype']=='torch.float32' and p['name'] for p in row['parameters']) for row in a)


def test_interface_zeros_lr_not_just_gradient_and_keeps_original_timeline():
    cfg=config();model=TinyGroups();pgs=groups(model,cfg)
    assert {p['name'] for p in pgs}==set(cfg['learning_rates'])
    assert sum(len(p['params']) for p in pgs)==len(list(model.parameters()))
    optimizer=torch.optim.AdamW(pgs,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,learning_scales(pgs,cfg))
    before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('qwen.')}
    for step in range(1000):
        for name,p in model.named_parameters():p.grad=torch.zeros_like(p) if name.startswith('qwen.') else torch.ones_like(p)
        assert all(g['lr']==0 for g in optimizer.param_groups if g['name'].startswith('qwen_'))
        optimizer.step();schedule.step()
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters() if n in before)
    assert schedule.last_epoch==1000
    for g in optimizer.param_groups:
        assert g['lr']==pytest.approx(cfg['learning_rates'][g['name']]*learning_scale(1000,cfg))
    for p in model.parameters():p.grad=torch.ones_like(p)
    optimizer.step();schedule.step()
    assert all(not torch.equal(before[n],p) for n,p in model.named_parameters() if n in before)


def test_qualification_joint_diagnostic_does_not_freeze_qwen_lr():
    cfg=config();pgs=groups(TinyGroups(),cfg)
    schedule=learning_scales(pgs,cfg,qualification_joint=True)
    assert all(fn(0)==learning_scale(0,cfg) for fn in schedule)
    assert all(fn(70000)==learning_scale(70000,cfg) for fn in schedule)


def test_joint_boundary_initializes_only_qwen_moments_once_without_scheduler_reset():
    cfg=config();model=TinyGroups();pgs=groups(model,cfg)
    optimizer=torch.optim.AdamW(pgs);schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,learning_scales(pgs,cfg))
    for _ in range(3):
        for p in model.parameters():p.grad=torch.ones_like(p)
        optimizer.step();schedule.step()
    actor_before=[{k:v.clone() for k,v in optimizer.state[p].items()} for p in model.actor.parameters()]
    scheduler_before=copy.deepcopy(schedule.state_dict())
    state=dict(updates=1000,qwen_optimizer_initialized=False)
    engine=type('Engine',(),{'optimizer':type('ZeRO',(),{'optimizer':optimizer})()})()
    event=initialize_joint_optimizer(engine,state)
    assert event['after_updates']==1000 and state['qwen_optimizer_initialized']
    assert {row['name'] for row in event['groups']}=={'qwen_vision','qwen_language_merger'}
    for name,p in model.named_parameters():
        if name.startswith('qwen.'):
            assert all(torch.count_nonzero(v)==0 for v in optimizer.state[p].values())
    for before,p in zip(actor_before,model.actor.parameters()):
        assert all(torch.equal(before[k],v) for k,v in optimizer.state[p].items())
    assert schedule.state_dict()==scheduler_before
    assert initialize_joint_optimizer(engine,state) is None


def test_initial_generic_weights_are_rehashed_not_accepted_by_path_config(tmp_path):
    weights=tmp_path/'model.safetensors';weights.write_bytes(b'unit-test-weights-not-a-real-Qwen')
    receipt=dict(repository='Qwen/Qwen3-VL-2B-Instruct',revision='fixed-source-revision',path=str(tmp_path),
                 files={weights.name:file_sha(weights)},official_weight_sha_verified=True)
    path=tmp_path/'receipt.json';path.write_text(json.dumps(receipt))
    assert verify_generic_qwen_identity(path,tmp_path)['all_files_reverified']
    weights.write_bytes(b'changed-under-the-same-path')
    with pytest.raises(ValueError,match='file changed'):verify_generic_qwen_identity(path,tmp_path)


def denominator_batch(n=4):
    return dict(action_valid=torch.tensor([[1]*8,[1]*7+[0],[0]*8,[1]*8],dtype=torch.bool)[:n],
        dino_valid=torch.tensor([[1,0],[0,0],[1,1],[1,1]],dtype=torch.bool)[:n],
        future_valid=torch.tensor([[0,0],[1,0],[0,0],[1,1]],dtype=torch.bool)[:n],
        objects=dict(valid=torch.tensor([[1,0],[0,0],[1,1],[0,0]],dtype=torch.bool)[:n],
                     coverage_valid=torch.tensor([0,1,0,0],dtype=torch.bool)[:n]),
        candidate_labels=torch.tensor([[[1.,1.,1.,.8,1.,1.],[0.,1.,1.,.2,1.,1.]],
            [[.5,1.,1.,.5,1.,1.],[0.,1.,1.,.2,1.,1.]],
            [[1.,0.,1.,.5,1.,1.],[1.,1.,1.,.5,1.,1.]],
            [[1.,1.,1.,.5,1.,1.],[0.,0.,1.,.5,1.,1.]]])[:n],
        candidate_valid=torch.tensor([[[1]*6,[1]*6],[[1]*6,[1]*6],[[1]*6,[1]*6],[[0]*6,[0]*6]],dtype=torch.bool)[:n],
        candidate_pair_mask=torch.tensor([[1,1],[1,1],[1,1],[1,1]],dtype=torch.bool)[:n])


def sliced(batch,indices):
    return {key:({k:v[indices] for k,v in value.items()} if isinstance(value,dict) else value[indices])
            for key,value in batch.items()}


def test_full_update_denominators_masks_soft_pairs_and_microbatch_equivalence(monkeypatch):
    batch=denominator_batch();chunks=[sliced(batch,slice(i,i+1)) for i in range(4)]
    den=update_denominators(chunks,torch.device('cpu'))
    assert den['action_elements']==69
    assert den['dino']==3 and den['future']==2
    assert den['object']==dict(existence=3,classification=2,box=2)
    assert den['post']==dict(factor=3,pair=2)
    theta=torch.tensor(.37,requires_grad=True)
    pred=theta*torch.arange(1.,9.).reshape(4,2)
    logits=theta*torch.arange(1.,49.).reshape(4,2,6)/20
    target=torch.ones(4,2)
    full=valid_scene_mse(pred,target,batch['dino_valid'])+post_action_loss(
        logits,batch['candidate_labels'],batch['candidate_valid'],batch['candidate_pair_mask'])['loss']
    expected=torch.autograd.grad(full,theta,retain_graph=True)[0]
    # Two ranks, two accumulation microbatches each, unequal valid counts. Each
    # local loss is world-scaled; the distributed optimizer averages ranks.
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:2)
    summed=0
    for i,chunk in enumerate(chunks):
        summed=summed+valid_scene_mse(pred[i:i+1],target[i:i+1],chunk['dino_valid'],scene_denominator=den['dino'])
        summed=summed+post_action_loss(logits[i:i+1],chunk['candidate_labels'],chunk['candidate_valid'],
            chunk['candidate_pair_mask'],scene_denominators=den['post'])['loss']
    actual=torch.autograd.grad(summed/2,theta)[0]
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(summed/2,full)


class Builder:
    def __init__(self):self.generator=torch.Generator().manual_seed(501)
    def state_dict(self):return {'generator':self.generator.get_state()}
    def load_state_dict(self,state):self.generator.set_state(state['generator'])


class NativeSaveDouble:
    """File/protocol double only; does not claim actual DeepSpeed resume."""
    global_steps=2
    def save_checkpoint(self,path,**kwargs):
        assert kwargs['exclude_frozen_parameters'] is False
        folder=Path(path)/kwargs['tag'];folder.mkdir()
        torch.save({'module':{'weight':torch.tensor([2.])},'global_steps':self.global_steps},folder/'mp_rank_00_model_states.pt')
        torch.save({'optimizer':'fp32-test'},folder/'zero_pp_rank_0_mp_rank_00_optim_states.pt')
        return True
    def load_checkpoint(self,path,**kwargs):
        assert kwargs['load_optimizer_states'] and kwargs['load_lr_scheduler_states'] and kwargs['load_module_strict']
        self.global_steps=torch.load(Path(path)/kwargs['tag']/'mp_rank_00_model_states.pt',weights_only=False)['global_steps']
        return str(Path(path)/kwargs['tag']),{}


def test_boundary_receipt_full_inventory_rng_cursor_roundtrip_and_partial_retry(tmp_path):
    identity=dict(world=1,qualification=False);state=dict(updates=2,cursor=64,phase='interface')
    stale=tmp_path/'.saving/update000002.interrupted';stale.mkdir(parents=True)
    random.seed(13);np.random.seed(13);torch.manual_seed(13)
    builder=Builder();engine=NativeSaveDouble()
    saved=cp.save(tmp_path,engine,state,identity,builder)
    expected=(random.random(),np.random.rand(),torch.rand(2),torch.rand(2,generator=builder.generator))
    info=inspect_checkpoint(saved)
    assert info['seal']['updates']==2 and stale.exists()
    random.seed(99);np.random.seed(99);torch.manual_seed(99);builder.generator.manual_seed(99)
    engine.global_steps=0
    restored=cp.restore(saved,engine,identity,builder)
    actual=(random.random(),np.random.rand(),torch.rand(2),torch.rand(2,generator=builder.generator))
    assert restored==state and engine.global_steps==2 and actual[:2]==expected[:2]
    assert all(torch.equal(a,b) for a,b in zip(actual[2:],expected[2:]))
    with pytest.raises(RuntimeError,match='overwrite'):cp.save(tmp_path,engine,state,identity,builder)
    opt=Path(saved)/'native/zero_pp_rank_0_mp_rank_00_optim_states.pt';opt.write_bytes(b'corrupt')
    # Model-only deployment validates its used bytes plus sealed full inventory;
    # full training resume also rehashes every optimizer shard and rejects this.
    inspect_checkpoint(saved,verify_files='model')
    with pytest.raises(ValueError,match='changed'):inspect_checkpoint(saved)
    opt.unlink()
    with pytest.raises(ValueError,match='missing'):inspect_checkpoint(saved,verify_files='model')


def selection_fixture(tmp_path,*,qualification=False):
    rows=[dict(checkpoint=f'/experiment/update{i:06d}',updates=i,model_sha256=f'model{i}',PDMS=i/100000)
          for i in range(5000,100001,5000)]
    # Exact tie takes the later endpoint, using the fixed registered rule.
    rows[-2]['PDMS']=1.
    selected={k:rows[-1][k] for k in ('checkpoint','updates','model_sha256')}
    value=dict(schema='world_postcot_DEV_selection_v1',completed_IL_updates=100000,Navtest_used=False,
               evaluated_candidates=rows,selected=selected)
    path=tmp_path/'selection.json'
    def write(v):path.write_text(json.dumps(dict(v,sha256=fingerprint(v))))
    write(value)
    info=dict(identity=dict(qualification=qualification),path=selected['checkpoint'],
              seal=dict(updates=selected['updates']),model_sha256=selected['model_sha256'])
    return path,value,info,write


def test_rl_requires_complete_formal_il_and_correct_dev_checkpoint_hash(tmp_path):
    path,value,info,write=selection_fixture(tmp_path)
    assert validate_rl_parent_selection(path,info)['selected']['updates']==100000
    bad=copy.deepcopy(value);bad['selected']['model_sha256']='other';write(bad)
    with pytest.raises(ValueError,match='identity'):validate_rl_parent_selection(path,info)
    write(value);info['identity']['qualification']=True
    with pytest.raises(ValueError,match='Qualification'):validate_rl_parent_selection(path,info)
    info['identity']['qualification']=False;bad=copy.deepcopy(value);bad['evaluated_candidates'].pop();write(bad)
    with pytest.raises(ValueError,match='incomplete'):validate_rl_parent_selection(path,info)
