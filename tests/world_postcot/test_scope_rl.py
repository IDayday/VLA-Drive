import copy
from pathlib import Path
import pytest
import torch
from torch import nn

from starVLA.model.modules.qwen_world_postcot.actor import CleanXActor
from starVLA.model.modules.qwen_world_postcot.memory import SceneMemoryProvider,PredictiveMemory
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY,TIME,SCOPE_SCHEMA,expected_scope_recipe,time_paths
from starVLA.rl.qwen_world_postcot.scope_conditions import MemoryLearningConditions
from starVLA.rl.qwen_world_postcot.rl_conditions import FrozenWorldConditions
from starVLA.rl.qwen_world_postcot.deployment import load_scope_trainable_state
from starVLA.rl.qwen_world_postcot.rl_actor import install_deployment_lora
from starVLA.rl.qwen_world_postcot.scope_runtime import ScopeRuntime
from starVLA.rl.simwam_recipe.vendor.lora import LoRALinear
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.recovery_state import validate_boundary,rollout_digest
from starVLA.rl.qwen_world_postcot.contracts import fingerprint
from scripts.analysis.simwam_recipe.train import parameter_hash


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(1.));self.calls=0
        self.features=dict(selected_hidden_states={i:torch.randn(1,8,64) for i in (1,2,3)},
            final_hidden=torch.randn(1,8,64),visual_indices=torch.arange(6)[None],
            visual_mask=torch.ones(1,6,dtype=torch.bool),camera_ids=torch.tensor([[0,0,1,1,2,2]]),
            visual_positions=torch.tensor([[[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4],[0,0,5]]]),
            text_indices=torch.tensor([[6,7]]),text_mask=torch.ones(1,2,dtype=torch.bool),
            merged_visual_features=torch.randn(1,6,64))
    def forward(self,batch):self.calls+=1;return self.features


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__();self.qwen=FakeQwen()
        self.scene_provider=SceneMemoryProvider(input_dim=64,hidden_dim=32,num_layers=1,num_heads=4,
            selected_layers=(1,2,3))
        self.predictive_provider=PredictiveMemory(input_dim=32,hidden_dim=16,num_layers=1,num_heads=4)
        self.actor=CleanXActor(hidden_dim=32,num_layers=2,num_heads=4,ffn_dim=128)
        with torch.no_grad():
            for block in self.actor.blocks:
                block.modulation[-1].weight.normal_(std=.02)
                block.modulation[-1].bias.fill_(.05)
            self.actor.output_modulation[-1].weight.normal_(std=.02)
    def encode_condition(self,batch,ego):return self.predictive_provider(self.scene_provider(self.qwen(batch),ego))


class Data:
    tokens=['train'];sha='unit_train_only'
    def observation(self,token):
        assert token=='train'
        return dict(images={},navigation='forward',proprio=[1.,0.,0.,0.,1.,0.,0.,0.])


def make(scope):
    torch.manual_seed(42);model=TinyModel();cfg=expected_scope_recipe(scope)
    actor=ScopeRecipeActor(model.actor,cfg,scene_provider=model.scene_provider,predictive_provider=model.predictive_provider)
    processor=lambda observations:dict(current=True)
    cond=(MemoryLearningConditions(model,Data(),'IL40k',actor,processor=processor) if scope==MEMORY
          else FrozenWorldConditions(model,Data(),'IL40k',processor=processor))
    object.__setattr__(actor,'conditions',cond)
    return actor,cond,cfg,model


def batch(actor,cond,update=0):
    descriptor=cond.get('train',torch.device('cpu'))
    with torch.no_grad():
        chain,times,deltas,indices=actor.sample_action_chain(descriptor,8,generator=torch.Generator().manual_seed(91))
        old=actor.action_chain_logprobs(descriptor,chain,times,deltas,stoch_idx=indices).detach()
        audit=copy.deepcopy(actor._probability_audit['current'])
    rewards=torch.arange(8,dtype=torch.float32)[None]/10
    std=rewards.double().std(unbiased=False)
    offset=rewards.double()-rewards.double()[:,:1]
    advantages=((offset-offset.mean(1,keepdim=True))/(std+1e-8)).float()[0]
    return dict(cond_bg=descriptor,chain=chain,timesteps=times,deltas=deltas,stoch_idx=indices,
        old_logp=old,old_probability=audit,advantages=advantages,adv_mask=torch.ones(8),
        rewards=rewards,global_std=float(std),raw_group_std=float(std),advantage_statistics={},
        records=[],token='train',log='unit_train',behavior_version=dict(run='unit',update=update))


def step(actor,optimizer,rollout,scheduler=None):
    optimizer.zero_grad(set_to_none=True);loss,_=actor(rollout);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters() if p.requires_grad)
    optimizer.step()
    if scheduler:scheduler.step()


@pytest.mark.parametrize('scope',[MEMORY,TIME])
def test_scope_initial_ode_logprob_and_frozen_closure(scope):
    actor,cond,cfg,_=make(scope);actor.record_sampling_terms=True
    b=batch(actor,cond)
    assert len(actor.sampling_terms)==3
    audited=actor._probability_audit['current']
    for i,(mean,std) in enumerate(actor.sampling_terms):assert torch.equal(mean,audited['mean'][:,i])
    current=actor.action_chain_logprobs(b['cond_bg'],b['chain'],b['timesteps'],b['deltas'],stoch_idx=b['stoch_idx'])
    assert torch.equal((current-b['old_logp']).exp(),torch.ones_like(current))
    with torch.no_grad():
        x,*_=actor.sample_action_chain(b['cond_bg'],8,deterministic=True,init_actions=b['chain'][:,0])
        ref,*_=actor.sample_action_chain(b['cond_bg'],8,deterministic=True,init_actions=b['chain'][:,0],velocity_use_ref=True)
    assert torch.equal(x,ref)
    assert all(not p.requires_grad for p in cond.qwen.parameters())
    if scope==TIME:
        assert len(actor.closure['adapters'])==5
        assert all('.self_attn.' not in n and '.cross_attn.' not in n for n in actor.closure['trainable_names'])
    else:assert all(not p.requires_grad for p in actor.ref_scene_provider.parameters())


def test_memory_gradient_cache_and_permanent_reference_are_separate():
    actor,cond,cfg,_=make(MEMORY);r=batch(actor,cond);optimizer=ScopeRuntime().make_optimizer(actor,cfg)
    frozen=parameter_hash(actor,trainable=False)
    ref=actor.action_chain_logprobs(r['cond_bg'],r['chain'],r['timesteps'],r['deltas'],stoch_idx=r['stoch_idx'],use_ref=True).detach()
    before=actor.current_memory_builds
    for _ in range(3):step(actor,optimizer,r)
    assert actor.current_memory_builds-before==3
    assert cond.qwen.calls==1
    for prefix in ['scene_provider.','predictive_provider.']:
        assert sum(float(p.grad.float().norm()) for n,p in actor.named_parameters() if n.startswith(prefix))>0
    assert parameter_hash(actor,trainable=False)==frozen
    ref_after=actor.action_chain_logprobs(r['cond_bg'],r['chain'],r['timesteps'],r['deltas'],stoch_idx=r['stoch_idx'],use_ref=True).detach()
    assert torch.equal(ref,ref_after)
    assert actor._prepared_scope is None
    assert [g['lr'] for g in optimizer.param_groups]==[5e-5,5e-6]


def test_time_projection_lora_matches_explicit_formula_and_gradients():
    torch.manual_seed(4);base=nn.Linear(6,9).double();module=LoRALinear(base,3,6).double()
    with torch.no_grad():module.lora_B.normal_(std=.03)
    x=torch.randn(2,5,6,dtype=torch.float64,requires_grad=True)
    y=module(x);g=torch.autograd.grad(y.square().sum(),(x,module.lora_A,module.lora_B))
    xx=x.detach().clone().requires_grad_();a=module.lora_A.detach().clone().requires_grad_();b=module.lora_B.detach().clone().requires_grad_()
    expected=nn.functional.linear(xx,base.weight,base.bias)+2*nn.functional.linear(nn.functional.linear(xx,a),b)
    gg=torch.autograd.grad(expected.square().sum(),(xx,a,b))
    torch.testing.assert_close(y,expected,rtol=0,atol=1e-14)
    for left,right in zip(g,gg):torch.testing.assert_close(left,right,rtol=0,atol=1e-12)


def test_full_time_scope_counts_and_every_projection_is_actually_called():
    with torch.device('meta'):
        head=CleanXActor();paths=time_paths(head)
    assert len(paths)==23
    actor,cond,cfg,_=make(TIME);called=[];handles=[]
    for row in actor.closure['adapters']:
        path=row['path'];handles.append(actor.action_expert.get_submodule(path).register_forward_hook(lambda m,i,o,p=path:called.append(p)))
    r=batch(actor,cond);optimizer=ScopeRuntime().make_optimizer(actor,cfg)
    for _ in range(3):step(actor,optimizer,r)
    for h in handles:h.remove()
    assert set(called)=={r['path'] for r in actor.closure['adapters']}
    assert all(float(p.grad.norm())>0 for p in actor.parameters() if p.requires_grad)
    assert cfg['grpo']['lora']['r']==32


@pytest.mark.parametrize('scope',[MEMORY,TIME])
def test_checkpoint_inner_resume_preserves_trainable_parameters_optimizer_and_chain(tmp_path,scope):
    a,conditions,cfg,_=make(scope);r=batch(a,conditions)
    optimizer=ScopeRuntime().make_optimizer(a,cfg);scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.,total_iters=50000)
    for _ in range(2):step(a,optimizer,r,scheduler)
    identity=dict(schema=SCOPE_SCHEMA,recipe=cfg,world=1,sha256='unit',projected_condition_identity=conditions.sha)
    state=dict(updates=2,collections=1,next_inner=2,phase='INNER_COMMITTED',scene_cursor=1,raw_scene_exposures=1,new_candidates=8)
    checkpoint.EXECUTION_IDENTITY=None;checkpoint.MIGRATION_RECEIPT=None
    path=checkpoint.save(tmp_path,a,optimizer,scheduler,torch.Generator().manual_seed(5),state,r,identity)
    old_digest=rollout_digest(r)
    for _ in range(2):step(a,optimizer,r,scheduler)
    b,bc,_,_=make(scope);bo=ScopeRuntime().make_optimizer(b,cfg);bs=torch.optim.lr_scheduler.ConstantLR(bo,factor=1.,total_iters=50000)
    restored,rr=checkpoint.restore(path,b,bo,bs,torch.Generator(),identity)
    assert restored==state and rollout_digest(rr)==old_digest
    for _ in range(2):step(b,bo,rr,bs)
    for name,value in a.state_dict().items():assert torch.equal(value,b.state_dict()[name]),name
    for key,left in optimizer.state_dict()['state'].items():
        for field,v in left.items():assert torch.equal(v,bo.state_dict()['state'][key][field])


@pytest.mark.parametrize('scope',[MEMORY,TIME])
def test_scope_export_load_keeps_nonzero_memory_and_lora(scope):
    actor,cond,cfg,model=make(scope);r=batch(actor,cond);optimizer=ScopeRuntime().make_optimizer(actor,cfg)
    for _ in range(3):step(actor,optimizer,r)
    saved={n:p.detach().clone() for n,p in actor.named_parameters() if p.requires_grad}
    _,_,_,deployment=make(scope)
    # make() installs adapters into deployment.actor already, as a real export loader does.
    load_scope_trainable_state(deployment,saved,scope)
    for n,p in actor.action_expert.state_dict().items():assert torch.equal(p,deployment.actor.state_dict()[n])
    if scope==MEMORY:
        for module in ['scene_provider','predictive_provider']:
            for n,p in getattr(actor,module).state_dict().items():assert torch.equal(p,getattr(deployment,module).state_dict()[n])
    bad=dict(saved);bad['qwen.fake']=torch.tensor(0.)
    with pytest.raises(ValueError):load_scope_trainable_state(deployment,bad,scope)
