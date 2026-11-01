"""Two CPU checks: dense covariance reference and actual Memory actor replay."""
import copy

import pytest
import torch
from torch.distributions import MultivariateNormal, kl_divergence

from starVLA.rl.qwen_world_safety_exploration.endpoint_policy import (
    endpoint_log_prob,same_covariance_mean_kl,replay_endpoint_mean,
)
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec


def test_joint_logp_KL_full_Kronecker_reference_and_gradients():
    rng=torch.Generator().manual_seed(42)
    for rho in (0.,.8):
        for ramp in (False,True):
            sigma=torch.tensor([1.486417944,.935414349,.078619349],dtype=torch.float64)
            if ramp:sigma=sigma[None,:]*(torch.arange(1,9,dtype=torch.float64)/8)[:,None]
            scale=sigma.expand(8,3).reshape(-1)
            lag=(torch.arange(8)[:,None]-torch.arange(8)[None,:]).abs()
            temporal=torch.tensor(rho,dtype=torch.float64).pow(lag)
            covariance=scale[:,None]*torch.kron(temporal,torch.eye(3,dtype=torch.float64))*scale[None,:]
            action=torch.randn(2,8,3,generator=rng,dtype=torch.float64,requires_grad=True)
            mean=torch.randn(2,8,3,generator=rng,dtype=torch.float64,requires_grad=True)
            ref=torch.randn(2,8,3,generator=rng,dtype=torch.float64,requires_grad=True)
            mu=mean.detach().clone().requires_grad_()
            current=MultivariateNormal(mu.reshape(2,24),covariance_matrix=covariance)
            reference=MultivariateNormal(ref.detach().reshape(2,24),covariance_matrix=covariance)
            actual,diag=endpoint_log_prob(action,mean,sigma,rho,return_diagnostics=True)
            expected=current.log_prob(action.detach().reshape(2,24))
            torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-12)
            torch.testing.assert_close(diag['covariance_logdet'],torch.linalg.slogdet(covariance).logabsdet,atol=1e-12,rtol=1e-12)
            assert torch.equal(diag['mean24_diagnostic'],actual/24)
            g1=torch.autograd.grad(actual.sum(),mean,retain_graph=True)[0]
            g2=torch.autograd.grad(expected.sum(),mu,retain_graph=True)[0]
            torch.testing.assert_close(g1,g2,atol=1e-9,rtol=1e-11)
            k=same_covariance_mean_kl(mean,ref,sigma,rho)
            expected_kl=kl_divergence(current,reference)
            torch.testing.assert_close(k,expected_kl,atol=1e-9,rtol=1e-11)
            kg1=torch.autograd.grad(k.sum(),mean,retain_graph=True)[0]
            kg2=torch.autograd.grad(expected_kl.sum(),mu)[0]
            torch.testing.assert_close(kg1,kg2,atol=1e-9,rtol=1e-11)
            assert torch.autograd.grad(actual.sum(),action,allow_unused=True,retain_graph=True)[0] is None
            assert torch.autograd.grad(k.sum(),ref,allow_unused=True)[0] is None
    for sigma,rho in [(torch.tensor([1.,0.,1.]),.8),(torch.ones(8,3).requires_grad_(),.8),(torch.ones(3),1.),(torch.ones(3),-.1)]:
        with pytest.raises(ValueError):endpoint_log_prob(action,mean,sigma,rho)


def test_actual_scope_terminal_replay_stops_prefix_and_collection_grad_rebuilds_memory():
    from world_postcot.test_scope_rl import make
    from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY
    torch.set_num_threads(2)
    actor,conditions,_,_=make(MEMORY)
    descriptor=conditions.get('train',torch.device('cpu'))
    prefix=torch.nn.Parameter(torch.randn(8,8,3));stored=1.3*prefix;stored.retain_grad()
    target=torch.nn.Parameter(torch.zeros(8,8,3))
    time=torch.full((8,),357.14285,requires_grad=True)
    delta=torch.tensor(-.35714285,requires_grad=True)
    before=actor.current_memory_builds
    frozen={n:p.detach().clone() for n,p in actor.named_parameters() if not p.requires_grad}
    sigma=torch.tensor([1.486417944,.935414349,.078619349])
    for inner in range(2):
        mean,normalized=replay_endpoint_mean(actor,descriptor,stored,time,delta,return_normalized=True)
        assert torch.equal(mean,ActionCodec.denorm_odo(normalized))
        assert actor._prepared_scope is None and actor.current_memory_builds==before+inner+1
        objective=-endpoint_log_prob(target,mean,sigma,.8).mean();objective.backward()
        assert prefix.grad is stored.grad is target.grad is time.grad is delta.grad is None
        for scope in ('action_expert.','scene_provider.','predictive_provider.'):
            grads=[p.grad for n,p in actor.named_parameters() if n.startswith(scope) and p.requires_grad and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
        assert conditions.qwen.calls==1
        for n,p in actor.named_parameters():
            if n in frozen:assert torch.equal(p,frozen[n]) and p.grad is None
        actor.zero_grad(set_to_none=True)
    with pytest.raises(ValueError):replay_endpoint_mean(actor,descriptor,stored,time,torch.tensor(-.1))
