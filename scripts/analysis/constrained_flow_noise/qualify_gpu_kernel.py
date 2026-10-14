"""Real full DDP chain, flat layout, hidden gradient and graph acceptance."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy
from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.constrained_flow_noise.noise_policy import legacy_std_range
from starVLA.rl.constrained_flow_noise.rollout_graph import configure
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.rollout import SamplingSpec
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.policy_adapter import sample_legacy_sde


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--condition-cache',required=True)
    p.add_argument('--registration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();host=require_dispatch();configure_numerics();torch.cuda.set_device(0)
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    tokens=json.loads(Path(a.registration).read_text())['calibration_tokens'][:2]
    identity=json.loads((Path(a.condition_cache)/'identity.json').read_text())
    store=ConditionStore(a.condition_cache,identity,readonly=False)
    raw=torch.cat([store.get(t) for t in tokens]).cuda()
    evidence=[]
    for variant in ('C_SDE','N_FLOW_NOISE'):
        actor,_,_=make_noise_policy(variant,legacy_std_range()[:2])
        for batch in (1,2):
            configure(actor,False)
            t0=time.monotonic();eager=actor.sample_chain(raw[:batch],tokens[:batch],list(range(batch)),version={'acceptance':1})
            torch.cuda.synchronize();seconds=time.monotonic()-t0
            with torch.no_grad():
                serial=actor.saved_statistics(raw[:batch],eager['chain'],chunk=1)
                flat=actor.saved_statistics(raw[:batch],eager['chain'])
            gap=float((flat['elementwise_logp']-serial['elementwise_logp']).abs().max())
            ratio=(flat['mean_transition_logp']-eager['old_elementwise_logp'].mean((-2,-1))).exp()
            ratio_error=float((ratio-1).abs().max())
            if gap>1e-3 or ratio_error>1e-4:raise ValueError(f'Unqualified flat chain layout: {gap}, {ratio_error}')
            if batch==1 and variant=='C_SDE':
                with torch.no_grad():
                    old=sample_legacy_sde(actor.policy,actor.condition(raw[:1]),eager['chain'][:,:,0],
                         innovation_generator=torch.Generator(device='cuda').manual_seed(eager['innovation_keys'][0]),
                         spec=SamplingSpec(group_size=16,temporal_noise_correlation=.8))
                if not torch.equal(old['chain'],eager['chain']):raise ValueError('C_SDE changed the old sampler')
            configure(actor,True)
            graphed=actor.sample_chain(raw[:batch],tokens[:batch],list(range(batch)),version={'acceptance':1})
            graph_error=float((graphed['chain']-eager['chain']).abs().max())
            if graph_error!=0:raise ValueError('Same-layout graph must exactly preserve eager chain')
            t0=time.monotonic();actor.sample_chain(raw[:batch],tokens[:batch],list(range(batch)),version={'acceptance':1})
            torch.cuda.synchronize();graph_seconds=time.monotonic()-t0
            evidence.append(dict(variant=variant,B=batch,G=16,K=10,flat_elementwise_logp_max_abs=gap,
                  initial_mean_ratio_max_abs_from_one=ratio_error,graph_chain_max_abs=graph_error,
                  eager_seconds=seconds,warm_graph_seconds=graph_seconds))
        configure(actor,False)
        if variant=='N_FLOW_NOISE':
            x=eager['chain'][:1,:,0][:,0];condition=actor.condition(raw[:1])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float32):
                original=actor.policy.action_model._euler_sample(x.clone(),condition)
                means=x.clone()
                for step in range(10):
                    means=actor.distribution(means,torch.tensor([step*100],device='cuda'),condition,
                                             torch.tensor([step],device='cuda')).mean
            if not torch.equal(original,means):raise ValueError('Noise-off Euler means changed original ODE')
            sigma=actor.distribution(x,torch.zeros(1,device='cuda',dtype=torch.long),condition,
                                     torch.zeros(1,device='cuda',dtype=torch.long)).std
            sigma.mean().backward()
            noise_grad={n:float(p.grad.float().norm()) if p.grad is not None else None for n,p in actor.noise_head.named_parameters()}
            dit_grad=sum(float(p.grad.float().norm()) for p in actor.policy.action_model.model.parameters() if p.grad is not None)
            if not all(v is not None and v>0 for v in noise_grad.values()) or dit_grad<=0:
                raise ValueError('Variance branch did not reach every noise parameter and shared DiT')
            evidence.append(dict(variance_only_noise_gradients=noise_grad,shared_DiT_gradient_norm_sum=dit_grad,noise_off_ODE_exact=True))
            actor.zero_grad(set_to_none=True)
            with torch.no_grad():
                initial=actor.saved_statistics(raw[:1],eager['chain'][:1])['mean_transition_logp']
                last=list(actor.noise_head.parameters())[-1];saved=last.clone();last.add_(.01)
                sigma_changed=actor.saved_statistics(raw[:1],eager['chain'][:1])['mean_transition_logp'];last.copy_(saved)
                bias=actor.policy.action_model.action_decoder.layer2.bias;saved=bias.clone();bias.add_(.01)
                mean_changed=actor.saved_statistics(raw[:1],eager['chain'][:1])['mean_transition_logp'];bias.copy_(saved)
            if torch.equal(initial,sigma_changed) or torch.equal(initial,mean_changed):raise ValueError('Ratio ignored mean or sigma changes')
            evidence.append(dict(sigma_only_ratio_changed=True,mean_only_ratio_changed=True))
        del actor,eager,flat,serial;torch.cuda.empty_cache()
    atomic_json(out,dict(status='PASS',host=host,checks=evidence,peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
                         scope='full model arithmetic/graph acceptance, no driving improvement claim'))


if __name__=='__main__':main()
