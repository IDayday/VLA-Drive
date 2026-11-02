"""Genuine two-scene features/ego; source per-scene reductions remain separate."""
import copy
import torch
import pytest
from world_postcot.test_scope_rl import TinyModel
from test_world_performance_nominal import Service
from test_world_safety_exploration_groups import cfg as group_cfg
from starVLA.rl.qwen_world_safety_exploration.conditions import GroupMemoryConditions
from starVLA.rl.qwen_world_safety_exploration.group_runtime import collect_group
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit,canonical_components
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.qwen_world_multiscene.actor import MultiSceneScopeActor
from starVLA.rl.qwen_world_multiscene.conditions import PackedConditions
from starVLA.rl.qwen_world_multiscene.objective import source_probabilities
from starVLA.rl.qwen_world_multiscene.diagnostics import compare_execution


class Data:
    tokens=['t0','t1'];sha='different_real_observation_fixture'
    def observation(self,token):
        return dict(token=token,images={},navigation='forward',proprio=[1.+self.tokens.index(token),.1,0.,0.,1.,0.,0.,0.])


def make():
    torch.manual_seed(42);c=group_cfg(16);m=TinyModel();features=[m.qwen.features]
    #Different feature values AND token counts. This is not a repeated scene.
    second=dict(selected_hidden_states={i:torch.randn(1,11,64)for i in (1,2,3)},
        final_hidden=torch.randn(1,11,64),visual_indices=torch.arange(9)[None],
        visual_mask=torch.ones(1,9,dtype=torch.bool),camera_ids=torch.tensor([[0,0,0,1,1,1,2,2,2]]),
        visual_positions=torch.tensor([[[0,0,j%3]for j in range(9)]]),
        text_indices=torch.tensor([[9,10]]),text_mask=torch.ones(1,2,dtype=torch.bool),merged_visual_features=torch.randn(1,9,64))
    features.append(second)
    m.qwen.forward=lambda batch:features[Data.tokens.index(batch['token'])]
    a=MultiSceneScopeActor(m.actor,c,scene_provider=m.scene_provider,predictive_provider=m.predictive_provider)
    a._loss=ResearchLoss(c)
    conditions=GroupMemoryConditions(m,Data(),'same_IL_fixture',a,group_size=16,processor=lambda obs:dict(token=obs[0]['token']))
    object.__setattr__(a,'conditions',conditions)
    with torch.no_grad():
        next(a.scene_provider.parameters()).add_(.01)
        for name,p in a.named_parameters():
            if name.endswith('lora_B'):p.normal_(std=.002)
    samples=[]
    for i,token in enumerate(Data.tokens):
        service=Service(primary_ep=(.8,1.)if i==0 else(1.,1.))
        r=collect_group(a,conditions.get(token,a.device),token,'TRAIN_log',service,torch.Generator().manual_seed(42+i),
            group_size=16,variant='R_CANONICAL',version=dict(run='CPU_two_scene',update=0,adapter_sha256='same'))
        samples.append(r)
    materials=[torch.cat((r['rewards'].double()[...,None],canonical_components(r['records'],16)['components']),-1)for r in samples]
    all_material=torch.cat(materials*4,0)
    for r in samples:
        credit=GroupCredit(16).advantages(r['rewards'],canonical_components(r['records'],16)['components'],all_material=all_material)
        r.update(advantages=credit['advantages'],adv_mask=credit['adv_mask'])
    packed=PackedConditions(conditions,selected_layers=a.scene_provider.selected_layers).pack([r['cond_bg']for r in samples],a.device)
    return a,conditions,samples,packed


def test_two_scene_context_and_source_reductions(capsys):
    torch.set_num_threads(1);a,conditions,samples,packed=make()
    assert samples[0]['adv_mask'].all()and not samples[1]['adv_mask'].any()
    assert packed['base_ego'].shape==(2,8)and not torch.equal(packed['base_ego'][0],packed['base_ego'][1])
    assert packed['frozen_features']['visual_mask'].sum(1).tolist()==[6,9]
    #Original one-scene method remains exact on repeated recomputation.
    for r in samples:
        with torch.no_grad():p=a.action_chain_logprobs(r['cond_bg'],r['chain'],r['timesteps'],r['deltas'],stoch_idx=r['stoch_idx'])
        assert torch.equal(p,r['old_logp'])
    result=compare_execution(a,samples,packed,loss=a._loss);print(result)
    assert result['per_scene_policy_losses'][1]==0 #PG0 does not change scene0 denominator; anchor still exists.
    assert result['current_logprob_max_abs']==result['reference_logprob_max_abs']==0
    assert result['loss_absolute_error']<1e-8 and result['fresh_ratio_within_1e_4']
    assert all(x['cosine']>.99 for x in result['gradient_scopes'].values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in a.parameters()if p.requires_grad)
    #No numeric acceptance threshold is silently relaxed. Report explicitly
    #whether BF16 packed fresh ratios pass the actual trainer guard.
    assert isinstance(result['fresh_ratio_within_1e_4'],bool)


def test_pack_order_and_repeated_scene_rejection():
    torch.set_num_threads(1);a,conditions,samples,packed=make()
    with pytest.raises(ValueError,match='Repeated one-scene'):
        PackedConditions(conditions,selected_layers=a.scene_provider.selected_layers).pack([samples[0]['cond_bg']]*2,a.device)
    bad=copy.copy(packed);bad['scene_index']=packed['scene_index'].roll(1)
    with pytest.raises(ValueError,match='carrier/index'):a.materialize(bad)
    bad=list(reversed(samples))
    with pytest.raises(ValueError,match='order'):source_probabilities(a,bad,packed)
