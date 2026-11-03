"""Independent Gaussian re-scoring for the optional per-pose covariance."""
import pytest,torch
from starVLA.rl.d46_augustus.physical_noise import calibrated_scale,codec_jacobian
from starVLA.rl.d46_augustus.kernel import step,stats


def test_unchanged_codec_maps_calibrated_noise_into_physical_units():
    low=-torch.arange(1,9,dtype=torch.float64)[:,None]*torch.tensor([[.5,.1,.01]])
    high=-low
    scale,physical=calibrated_scale(low,high)
    torch.testing.assert_close(scale.double()*codec_jacobian(),physical,rtol=1e-7,atol=1e-10)
    assert scale.shape==(8,3)and(scale>0).all()and scale[0,0]<scale[-1,0]
    with pytest.raises(ValueError):calibrated_scale(low,low)


def test_pose_covariance_is_identical_in_sampler_and_independent_normal_ratio():
    gen=torch.Generator().manual_seed(42);old=torch.randn(3,8,3,generator=gen);state=torch.randn(3,8,3,generator=gen)
    scale=torch.linspace(.003,.05,8)[:,None]*torch.tensor([[1.,.8,.3]])
    noise=torch.randn(3,8,3,generator=gen);t,nxt=.6818181872367859,.5555555820465088
    sampled,oldmean,std=step(old,state,t,nxt,noise_scale=scale,noise=noise)
    torch.testing.assert_close(std[0],scale*(t*(t-nxt))**.5,rtol=1e-6,atol=1e-8)
    new=(old+.0001).requires_grad_();_,mean2,std2=step(new,state,t,nxt,noise_scale=scale,noise=torch.zeros_like(noise))
    logratio,kl=stats(new,old,old,state,sampled,t,nxt,noise_scale=scale)
    independent=(torch.distributions.Normal(mean2,std2).log_prob(sampled)-torch.distributions.Normal(oldmean,std).log_prob(sampled)).mean((1,2))
    torch.testing.assert_close(logratio,independent,rtol=1e-3,atol=5e-5)
    torch.testing.assert_close(kl,(.5*((mean2-oldmean)/std).square()).mean((1,2)),rtol=1e-6,atol=1e-8)
    assert torch.isfinite(torch.autograd.grad(logratio.sum(),new)[0]).all()
