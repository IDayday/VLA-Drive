"""Pinned official AUGUstus math checks. CPU only, no model/scorer/weights."""
import hashlib
from pathlib import Path
import pytest,torch
from starVLA.rl.d46_augustus.kernel import step,stats,policy_loss
from starVLA.rl.d46_augustus.vendor.fm_sde_logprob import fm_sde_step_with_logprob_x0,fm_sde_logprob_from_mean_std
from starVLA.rl.d46_augustus.vendor.jit_grpo import jit_grpo_step_logratio_kl,jit_grpo_policy_loss


def test_pinned_anisotropic_kernel_three_steps_and_coordinate_mean_identity():
    directory=Path(__file__).resolve().parents[2]/'starVLA/rl/d46_augustus/vendor'
    for name,digest in [('jit_grpo.py','e951c54ad1bfa13cff12f0676eaefa6e89f2ca029b9ee3169306dfdc0120db47'),('fm_sde_logprob.py','6f20c279660a89f0545634fdd6c4ee174f298ed106b20045b219c62f82ce2585')]:
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
    gen=torch.Generator().manual_seed(42);state=torch.randn(8,8,3,generator=gen);scale=torch.tensor([.05,.05,.02])
    for t,nxt in zip([1.,2/3,1/3],[2/3,1/3,0.]):
        old=torch.randn(8,8,3,generator=gen);z=torch.randn(8,8,3,generator=gen)
        actual,mean,std=step(old,state,t,nxt,noise=z)
        tt=torch.full((8,),t);nt=torch.full((8,),nxt)
        _,_,mu,rawstd=fm_sde_step_with_logprob_x0(old,state,tt,nt,noise_level=scale,schedule='lambda_t',prev_sample=state)
        assert torch.equal(mean,mu)and torch.equal(std,rawstd.expand_as(state))
        assert torch.equal(actual,mu+rawstd*z)
        torch.testing.assert_close(std[0,0],scale*(torch.tensor(t)*torch.tensor(t-nxt)).sqrt(),rtol=1e-6,atol=1e-8)
        new=(old+.0001*torch.randn(8,8,3,generator=gen)).requires_grad_();ref=old-.0001
        ratio,kl=stats(new,old,ref,state,actual,t,nxt)
        _,_,mu_new,std_new=fm_sde_step_with_logprob_x0(new,state,tt,nt,noise_level=scale,schedule='lambda_t',prev_sample=actual)
        _,_,mu_ref,_=fm_sde_step_with_logprob_x0(ref,state,tt,nt,noise_level=scale,schedule='lambda_t',prev_sample=actual)
        logp_new=fm_sde_logprob_from_mean_std(actual,mu_new,std_new);logp_old=fm_sde_logprob_from_mean_std(actual,mu,rawstd)
        torch.testing.assert_close(ratio,logp_new-logp_old,rtol=1e-3,atol=2e-6)
        torch.testing.assert_close(kl,(.5*((mu_new-mu_ref)/std_new).square()).mean((1,2)),rtol=1e-6,atol=1e-8)
        assert ratio.shape==kl.shape==(8,)and ratio.dtype==torch.float32
        state=actual.detach()


def test_exact_official_policy_loss_gradient_and_reference_isolation():
    torch.manual_seed(42)
    lr=(.4*torch.randn(8,3)).requires_grad_();kl=torch.rand(8,3,requires_grad=True);adv=torch.tensor([10.,-10.,1.,-1.,.2,-.2,0.,0.])
    a,m=policy_loss(lr,adv,kl);b,_=jit_grpo_policy_loss(lr,adv,kl=kl,clip_range=.2,kl_coef=.001)
    assert torch.equal(a,b)
    ga=torch.autograd.grad(a,[lr,kl],retain_graph=True);gb=torch.autograd.grad(b,[lr,kl])
    assert all(torch.equal(x,y)for x,y in zip(ga,gb))
    assert torch.equal(m['reference_KL_joint'],24*m['reference_KL_coordinate_mean'])
    state=torch.randn(8,8,3);old=torch.randn_like(state,requires_grad=True);ref=old.detach().clone().requires_grad_();new=old.detach().clone().requires_grad_();noise=torch.ones_like(state)
    nxt,_,_=step(old.detach(),state,1/3,0.,noise=noise)
    r,k=stats(new,old,ref,state,nxt,1/3,0.);loss,_=policy_loss(r[:,None],torch.ones(8),k[:,None]);loss.backward()
    assert old.grad is None and ref.grad is None and new.grad is not None and torch.isfinite(new.grad).all()
    assert new.grad.sum()<0  # Gradient descent reinforces positive standard innovation.


def test_explicit_private_noise_no_RNG_consumption_and_validation():
    torch.manual_seed(7);x=torch.randn(8,8,3);z=torch.randn_like(x);before=torch.get_rng_state().clone()
    out,_,_=step(x,x,1.,2/3,noise=z)
    assert torch.equal(before,torch.get_rng_state())and out.dtype==torch.float32
    torch.manual_seed(42);a,_,_=step(x,x,1.,2/3)
    torch.manual_seed(42);b,_,_,_=fm_sde_step_with_logprob_x0(x,x,torch.ones(8),torch.full((8,),2/3),noise_level=torch.tensor([.05,.05,.02]),schedule='lambda_t')
    assert torch.equal(a,b)
    for t,nxt in[(0.,0.),(.5,.5),(.2,.3),(1.01,0.),(.3,-.1)]:
        with pytest.raises(ValueError,match='denoising'):step(x,x,t,nxt,noise=z)
    with pytest.raises(ValueError,match='Noise must'):step(x,x,1.,0.,noise=z[:1])
