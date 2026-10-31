"""Research-only pure loss: native-proxy surrogate with conditional Gaussian gate.

No runtime entry/registration. This is NOT exact full-policy importance sampling
or a full Flow-DPPO reproduction: native mean-clamped log-density stays a proxy.
"""
import json,math
from pathlib import Path
import torch


def gaussian_kl(old_mean,new_mean,old_std,new_std):
    """[candidate,step,H,A],scalar-perstep std;exact nominal equal-Gaussian KL."""
    if old_mean.shape!=new_mean.shape or old_mean.ndim!=4:raise ValueError('Expected matched[B,S,H,A] means')
    if old_std.shape!=(old_mean.shape[1],) or not torch.equal(old_std,new_std) or not bool((old_std>0).all()):raise ValueError('Same strictly positive perstep Gaussian variance required')
    if not torch.isfinite(old_mean).all() or not torch.isfinite(new_mean).all() or not torch.isfinite(old_std).all():raise ValueError('Finite Gaussian kernel parameters required')
    elem=(new_mean.double()-old_mean.detach().double()).square()/(2*old_std.detach().double()[None,:,None,None].square())
    return elem.sum((-1,-2)),elem.mean((-1,-2)),old_mean.shape[-2]*old_mean.shape[-1]


def proxy_kl_mask_loss(new_proxy_logp,old_proxy_logp,advantages,mask,old_mean,new_mean,old_std,new_std,*,delta,units):
    if not math.isfinite(delta) or delta<=0 or units not in ('sum_nats','mean_nats_per_coordinate'):raise ValueError('Explicit positive delta with named units required')
    if new_proxy_logp.shape!=old_proxy_logp.shape or new_proxy_logp.shape!=advantages.shape or mask.shape!=advantages.shape[:1]:raise ValueError('Matched[B,S] proxy/A and[B] mask required')
    sums,means,dimension=gaussian_kl(old_mean,new_mean,old_std,new_std)
    if sums.shape!=new_proxy_logp.shape:raise ValueError('Transition means/proxy step layout differ')
    logdiff=new_proxy_logp-old_proxy_logp.detach();ratio=logdiff.exp()
    if not torch.isfinite(ratio).all():raise FloatingPointError('Prototype refuses nonfinite proxy ratio')
    divergence=sums if units=='sum_nats' else means
    # Eq18/code-style asymmetry;ratio direction is still sampled/native-proxy.
    # Official code excludes equality;this prototype follows that exact boundary.
    blocked=((advantages.detach()*logdiff.detach())>0)&(divergence.detach()>=delta)
    per=torch.where(blocked,torch.zeros_like(ratio),-advantages.detach()*ratio)
    loss=(per*mask.detach()[:,None]).sum()/(mask.detach().sum().clamp_min(1)*ratio.shape[1])
    return loss,dict(blocked=blocked,KL_sum_nats=sums.detach(),KL_mean_nats_per_coordinate=means.detach(),dimensions=dimension,
        source_ratio_is_exact_IS=False,mask_guarantees_global_postupdate_KL_bound=False)


def self_check():
    dtype=torch.float64;old=torch.zeros((2,1,8,3),dtype=dtype);new=old+.01;std=torch.ones(1,dtype=dtype)
    ksum,kmean,d=gaussian_kl(old,new,std,std)
    reference=torch.distributions.kl_divergence(torch.distributions.Normal(old,std.reshape(1,1,1,1)),torch.distributions.Normal(new,std.reshape(1,1,1,1))).sum((-1,-2))
    assert torch.allclose(ksum,reference) and torch.allclose(ksum,d*kmean)
    # One fixed Gaussian shift/two old actions:ratio clipping decisions differ,
    # yet conditional KL is identical and below the chosen direct threshold.
    ratio=torch.tensor([[1.02],[1/1.02]],dtype=dtype);A=torch.tensor([[1.],[-1.]],dtype=dtype);lp=ratio.log().requires_grad_();mask=torch.ones(2,dtype=dtype)
    loss,meta=proxy_kl_mask_loss(lp,torch.zeros_like(lp),A,mask,old,new,std,std,delta=1e-4,units='mean_nats_per_coordinate')
    assert not meta['blocked'].any();loss.backward();assert (lp.grad!=0).all()
    # A high conditional shift can be missed by eps .005 when the sampled
    # ratio is close to1;direct KL+outward-sign mask blocks that gradient.
    lp=torch.tensor([[math.log(1.001)],[math.log(.999)]],dtype=dtype,requires_grad=True)
    loss,meta=proxy_kl_mask_loss(lp,torch.zeros_like(lp),A,mask,old,old+.1,std,std,delta=1e-4,units='mean_nats_per_coordinate')
    assert meta['blocked'].all();loss.backward();assert (lp.grad==0).all()
    # Opposite-sign corrective surrogate updates remain allowed at high KL.
    lp=torch.tensor([[math.log(.999)],[math.log(1.001)]],dtype=dtype,requires_grad=True)
    loss,meta=proxy_kl_mask_loss(lp,torch.zeros_like(lp),A,mask,old,old+.1,std,std,delta=1e-4,units='mean_nats_per_coordinate')
    assert not meta['blocked'].any();loss.backward();assert (lp.grad!=0).all()
    # Unit conversion must preserve every decision,not silently change gate.
    a,ma=proxy_kl_mask_loss(lp,torch.zeros_like(lp),A,mask,old,old+.1,std,std,delta=1e-4,units='mean_nats_per_coordinate')
    b,mb=proxy_kl_mask_loss(lp,torch.zeros_like(lp),A,mask,old,old+.1,std,std,delta=24e-4,units='sum_nats')
    assert torch.equal(ma['blocked'],mb['blocked']) and torch.equal(a,b)
    return dict(status='PASS',CPU_only=True,dimensions=24,checks=['analytic Gaussian KL matches torch distribution KL','sum/mean unit consistency','lowKL preserves gradients that ratio can clip','highKL blocks near1 outward ratio','highKL corrective gradient remains','maskdetached+ratio gradient behavior'],runtime_or_GPU_entry_created=False)


if __name__=='__main__':
    value=self_check();p=Path(__file__).with_name('FLOW_DIRECT_PROXIMAL_CPU_CHECK_20261031.json');p.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
