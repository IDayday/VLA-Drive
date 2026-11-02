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
    loss,_=clipped_loss(ratio,torch.randn(batch,group));loss.backward();assert new.grad.abs().sum()>0


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
