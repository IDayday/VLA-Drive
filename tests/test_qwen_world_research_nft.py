"""Forward-process NFT math, EMA separation, and exact native recovery."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch

from starVLA.rl.qwen_world_research_algorithms.nft import (
    BehaviorEMA,NFTLoss,make_forward_batch,nft_objective,optimality_probabilities,
)
from starVLA.rl.qwen_world_research_algorithms import nft_checkpoint
from starVLA.rl.qwen_world_postcot.contracts import fingerprint


def tensor(value,requires_grad=False):
    return torch.full((2,8,3),value,dtype=torch.float64,requires_grad=requires_grad)


def test_nft_cleanx_velocity_equivalence_has_sigma_squared_weight():
    clean=tensor(.2);noise=tensor(-.7);sigma=torch.tensor([.1,.9],dtype=torch.float64)
    state=(1-sigma[:,None,None])*clean+sigma[:,None,None]*noise
    prediction=tensor(.3,True)
    velocity=(state-prediction)/sigma[:,None,None]
    target=noise-clean
    loss,_=nft_objective(velocity,velocity.detach(),target,torch.ones(2,dtype=torch.float64))
    expected=((prediction-clean).square()/sigma[:,None,None].square()).mean()
    torch.testing.assert_close(loss,expected,rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(torch.autograd.grad(loss,prediction,retain_graph=True)[0],
                               torch.autograd.grad(expected,prediction)[0],rtol=1e-14,atol=1e-14)


def test_negative_branch_is_nonnegative_and_reverses_initial_learning_direction():
    positive=tensor(.3,True);negative=tensor(.3,True);old=tensor(.3);target=tensor(-.1)
    lp,_=nft_objective(positive,old,target,torch.ones(2,dtype=torch.float64))
    ln,_=nft_objective(negative,old,target,torch.zeros(2,dtype=torch.float64))
    gp=torch.autograd.grad(lp,positive)[0];gn=torch.autograd.grad(ln,negative)[0]
    assert lp>=0 and ln>=0
    torch.testing.assert_close(gn,-gp,rtol=0,atol=0)


def test_tied_probability_half_has_zero_gradient_at_behavior_initialization():
    current=tensor(.3,True)
    loss,_=nft_objective(current,tensor(.3),tensor(-.4),torch.full((2,),.5,dtype=torch.float64))
    gradient=torch.autograd.grad(loss,current)[0]
    assert torch.equal(gradient,torch.zeros_like(gradient))


def test_old_velocity_and_optimality_cannot_receive_training_gradient():
    current=tensor(.3,True);old=tensor(.2,True);prob=torch.tensor([.1,.9],dtype=torch.float64,requires_grad=True)
    loss,_=nft_objective(current,old,tensor(-.4),prob)
    loss.backward()
    assert current.grad is not None and old.grad is None and prob.grad is None


def test_probability_transform_centers_within_group_and_uses_global_scale():
    r=torch.tensor([[.1,.3,.5],[.8,.8,.8]],dtype=torch.float64)
    p=optimality_probabilities(r,.2)
    torch.testing.assert_close(p[0],torch.tensor([0.,.5,1.]),rtol=0,atol=1e-6)
    assert torch.equal(p[1],torch.full((3,),.5))
    assert bool(((p>=0)&(p<=1)).all())


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.tensor([.25,-.5],dtype=torch.float64))
        self.frozen=torch.nn.Parameter(torch.tensor([3.],dtype=torch.float64),requires_grad=False)
        self.device=torch.device('cpu')


def test_ema_inference_exchange_restores_current_weights_even_on_exception():
    model=Tiny();ema=BehaviorEMA(model,.9)
    with torch.no_grad():model.p.add_(1)
    before=model.p.detach().clone();frozen=model.frozen.detach().clone()
    with pytest.raises(RuntimeError,match='injected'):
        with ema.use(model):
            assert torch.equal(model.p,ema.values['p'])
            raise RuntimeError('injected')
    assert torch.equal(model.p,before) and torch.equal(model.frozen,frozen)
    ema.update(model)
    torch.testing.assert_close(ema.values['p'],.9*(before-1)+.1*before,rtol=1e-14,atol=1e-14)
    assert ema.updates==1


def identity():
    body=dict(schema='world_postcot_cleanx_scope_RL_v1',training_protocol='world_nft_forward_v1',
              world=1,recipe=dict(max_steps=4))
    return dict(body,sha256=fingerprint(body))


def step(model,optimizer,scheduler,generator):
    optimizer.zero_grad()
    random.random();np.random.rand();torch.rand(1);target=torch.randn((2,),generator=generator,dtype=torch.float64)
    loss=(model.p-target).square().sum();loss.backward();optimizer.step();scheduler.step()


def objects():
    model=Tiny()
    # Frozen parameters are excluded in the real runtime optimizer.
    optimizer=torch.optim.AdamW([model.p],lr=.01)
    scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.,total_iters=4)
    generator=torch.Generator().manual_seed(122)
    return model,optimizer,scheduler,generator,BehaviorEMA(model,.9)


def test_mid_inner_native_checkpoint_restores_ema_pending_rng_and_next_update(tmp_path):
    model,optimizer,scheduler,generator,ema=objects()
    step(model,optimizer,scheduler,generator)
    state=dict(updates=1,collections=1,next_inner=1,ema_updates=0,phase='INNER_COMMITTED',
               scene_cursor=1,raw_scene_exposures=1,new_candidates=8)
    pending=dict(schema='world_nft_ODE_collection_v1',ema_version=0,
        clean_actions=torch.arange(192,dtype=torch.float32).reshape(8,8,3),
        rewards=torch.linspace(0,1,8).reshape(1,8),optimality=torch.linspace(0,1,8),
        cond_bg=dict(token='scene',schema='test'))
    path=nft_checkpoint.save(tmp_path,model,optimizer,scheduler,generator,state,pending,ema,identity())
    saved_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    step(model,optimizer,scheduler,generator)
    expected=model.p.detach().clone();expected_opt=copy.deepcopy(optimizer.state_dict())
    other,opt,sched,gen,other_ema=objects()
    resumed,resumed_pending=nft_checkpoint.restore(path,other,opt,sched,gen,other_ema,identity())
    assert random.getstate()==saved_rng[0]
    assert np.array_equal(np.random.get_state()[1],saved_rng[1][1])
    assert torch.equal(torch.get_rng_state(),saved_rng[2])
    assert resumed==state and torch.equal(resumed_pending['clean_actions'],pending['clean_actions'])
    assert torch.equal(other_ema.values['p'],ema.values['p'])
    step(other,opt,sched,gen)
    assert torch.equal(other.p,expected)
    for key,value in opt.state_dict()['state'][0].items():assert torch.equal(value,expected_opt['state'][0][key])
    complete=json.loads((Path(path)/'COMPLETE').read_text())
    transaction=json.loads((Path(path)/'transaction.json').read_text())
    assert complete['schema']==2 and complete['training_protocol']=='world_nft_forward_v1'
    assert transaction['schema']=='world_nft_checkpoint_transaction_v1'


def test_nft_validator_rejects_wrong_ema_or_old_sde_payload():
    model,opt,sched,gen,ema=objects();state=dict(updates=1,collections=1,next_inner=1,ema_updates=0,
        phase='INNER_COMMITTED',scene_cursor=1,raw_scene_exposures=1,new_candidates=8)
    with pytest.raises(ValueError,match='pending samples'):
        nft_checkpoint.validate(state,dict(schema='old_SDE_chain',ema_version=0),ema,identity())
    ema.updates=1
    with pytest.raises(ValueError,match='EMA'):
        nft_checkpoint.validate(state,dict(schema='old_SDE_chain',ema_version=0),ema,identity())


@pytest.mark.parametrize('minimum',[0.,-.1])
def test_cleanx_weighting_rejects_zero_sigma(minimum):
    with pytest.raises(ValueError,match='sigma'):
        nft_objective(tensor(.1,True),tensor(.1),tensor(0.),torch.ones(2),
                      sigma=torch.full((2,),minimum),loss_type='x0_self_normalized')


class TinyFlow(Tiny):
    @contextmanager
    def _context_call(self,condition):
        yield condition

    def action_velocity(self,state,timestep,condition):
        return self.p[0].expand_as(state)


def test_forward_batch_uses_ema_policy_then_restores_current_graph():
    from test_qwen_world_research_policy import cfg
    model=TinyFlow();ema=BehaviorEMA(model,.9)
    with torch.no_grad():model.p.add_(1)
    c=cfg();c['grpo']['train']['kl_beta']=0
    c['research'].update(policy_objective='nft',nft_sigma_min=.05,nft_beta=1.,nft_loss='velocity_mse')
    r=dict(clean_actions=torch.zeros(8,8,3),cond_bg={},optimality=torch.linspace(0,1,8))
    batch=make_forward_batch(model,r,ema,torch.Generator().manual_seed(32),c)
    assert torch.equal(batch['old_velocity'],torch.full((8,8,3),.25))
    assert model.p[0]==1.25 and not batch['old_velocity'].requires_grad
    assert bool(((batch['sigma']>=.05)&(batch['sigma']<1)).all())
    loss,metrics=NFTLoss(c)._policy_loss(model,batch)
    loss.backward()
    assert bool(torch.isfinite(model.p.grad).all())
    assert metrics['nft_positive_loss']>0 and metrics['nft_negative_loss']>0
