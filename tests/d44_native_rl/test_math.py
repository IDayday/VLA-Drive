import torch
import pytest
from torch.distributions import Normal,Independent,kl_divergence
from starVLA.rl.d44_native_rl.gaussian_objective import gaussian_terms,clipped_loss
from starVLA.rl.d44_native_rl.safety_credit import raw_credit,normalize
from starVLA.rl.d44_native_rl.adaptive_kl import AdaptiveKL
from starVLA.rl.d44_native_rl.scheduler import scene_lr
from starVLA.rl.d44_native_rl.expanded_scope import install
from starVLA.rl.d44_native_rl.native_cleanx_aux import verified_index


@pytest.mark.parametrize('batch,group',[(1,8),(4,16),(8,32)])
def test_gaussian_joint_density_and_gradient(batch,group):
    torch.manual_seed(42);old=torch.randn(batch,group,3,8,3);new=(old+.01).requires_grad_();ref=old-.02
    std=torch.tensor([.1,.2,.3]);nxt=old+torch.randn_like(old)*std[:,None,None]
    ratio,kl=gaussian_terms(new,old,ref,std,nxt)
    current=Independent(Normal(new,std[:,None,None]),2);behavior=Independent(Normal(old,std[:,None,None]),2)
    reference=Independent(Normal(ref,std[:,None,None]),2)
    torch.testing.assert_close(ratio,current.log_prob(nxt)-behavior.log_prob(nxt),atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(kl,kl_divergence(current,reference),atol=2e-5,rtol=2e-5)
    advantage=torch.randn(batch,group)
    loss,_=clipped_loss(ratio,advantage);loss=loss+.001*kl.mean()
    gradient=torch.autograd.grad(loss,new,retain_graph=True)[0]
    oracle_ratio=(current.log_prob(nxt)-behavior.log_prob(nxt)).exp();a=advantage[...,None]
    oracle=torch.maximum(-a*oracle_ratio,-a*oracle_ratio.clamp(.9,1.1)).mean()+.001*kl_divergence(current,reference).mean()
    oracle_gradient=torch.autograd.grad(oracle,new)[0]
    torch.testing.assert_close(gradient,oracle_gradient,atol=1e-5,rtol=1e-5);assert gradient.abs().sum()>0


def test_fpo_energy_velocity_sign_and_awm_positive_weight():
    from starVLA.rl.d44_native_rl.fpo import velocity_energy,awm_loss
    torch.manual_seed(42);clean=torch.randn(16,8,3);eps=torch.randn_like(clean);sigma=torch.linspace(.2,.8,16)
    state=(1-sigma[:,None,None])*clean+sigma[:,None,None]*eps
    pred=(clean+.02*eps).requires_grad_()
    energy=velocity_energy(state,pred,eps-clean,sigma)
    oracle=((pred-clean)/sigma[:,None,None]).square().mean((-1,-2))
    torch.testing.assert_close(energy,oracle,atol=1e-6,rtol=1e-5)
    grad=torch.autograd.grad(energy.mean(),pred,retain_graph=True)[0]
    reference=torch.autograd.grad(oracle.mean(),pred,retain_graph=True)[0];torch.testing.assert_close(grad,reference)
    loss=awm_loss(energy,-torch.ones(16),pred.detach(),pred)
    assert loss==0 and torch.equal(torch.autograd.grad(loss,pred)[0],torch.zeros_like(pred))


def test_invalid_std_rejected():
    x=torch.zeros(8,3,8,3)
    with pytest.raises(ValueError):gaussian_terms(x,x,x,torch.tensor([.1,0,.1]),x)


@pytest.mark.parametrize('group',[8,16,32])
def test_safety_priority_and_ties(group):
    f=torch.ones(8,group,6);raw,feasible=raw_credit(f);a,std=normalize(raw);assert std==0 and torch.equal(a,torch.zeros_like(a))
    f[:,0,0]=0.;f[:,0,3]=1.;f[:,1:,3]=0.;raw,_=raw_credit(f)
    assert (raw[:,1:]>=raw[:,:1]).all()
    f[...,:3]=0;f[:,0,0]=.5;raw,_=raw_credit(f);assert(raw[:,0]>raw[:,1]).all()


def test_stage2_feasible_priority():
    f=torch.ones(1,16,6);f[0,0,0]=0;f[0,1,3:5]=0
    raw,_=raw_credit(f,endpoint=torch.zeros(1,16),stage2=True)
    assert raw[0,1]>=raw[0,0]


def test_dual_once_and_scheduler():
    d=AdaptiveKL();old=d.beta;d.update(24*.03);assert old<d.beta<=old*1.05
    old=d.beta;d.ema=0;d.update(0);assert d.beta<old and d.collections==2
    assert scene_lr(.60)==1 and scene_lr(4)==.05 and scene_lr(12)==.05


def test_new_lora_zero_delta_and_nonzero_second_gradient():
    m=torch.nn.Sequential(torch.nn.Linear(3,10));before=m[0](torch.ones(2,3)).detach()
    install(m,['0'],8,16);torch.testing.assert_close(m(torch.ones(2,3)),before,rtol=0,atol=0)
    opt=torch.optim.AdamW([p for p in m.parameters()if p.requires_grad],lr=.01)
    m(torch.ones(2,3)).sum().backward();assert m[0].lora_B.grad.abs().sum()>0
    opt.step();opt.zero_grad();m(torch.ones(2,3)).sum().backward();assert m[0].lora_A.grad.abs().sum()>0


def test_teacher_strict_high_includes_comfort_and_real_nominal_improvement():
    good=dict(no_at_fault_collisions=1.,drivable_area_compliance=1.,time_to_collision_within_bound=1.,comfort=1.,score=.95)
    records=[dict(metrics=dict(good,comfort=.5)),dict(metrics=good)]
    assert verified_index(records,0.,improvement=0.)==1
    assert verified_index(records[:1],0.,improvement=0.)is None
    assert verified_index(records,.94) is None
    assert verified_index(records,.90)==1


def test_endpoint_invalid_or_stationary_does_not_override_safety():
    from starVLA.rl.d44_native_rl.two_stage_reward import endpoint_quality
    req=torch.zeros(16,8,3);gt=torch.zeros(8,3)
    q,w=endpoint_quality(gt,req);assert w==0
    qbad,wbad=endpoint_quality(gt*float('nan'),req);assert wbad==0 and torch.isfinite(qbad).all()
    factors=torch.ones(1,16,6);factors[:,0,0]=0
    a,_=raw_credit(factors,endpoint=q[None],endpoint_confidence=w.reshape(1,1),stage2=True)
    assert torch.isfinite(a).all()and(a[:,1:]>=a[:,:1]).all()
