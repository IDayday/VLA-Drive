"""Finite native-3D RL qualification; synthetic arithmetic is not driving evidence."""
import argparse
import copy
import json
from pathlib import Path
import torch
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from starVLA.model.modules.qwen_sourceflow.cached_action_runner import CachedActionRunner
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.qwen_sourceflow.recipe import expected_recipe
from starVLA.rl.simwam_recipe.source_advantages import advantages
from starVLA.rl.simwam_recipe.vendor.lora import merged_state_dict


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    args=dict(action_dim=3,hidden_dim=32,ffn_dim=64,text_dim=40,freq_dim=16,
        num_heads=2,attn_head_dim=12,num_layers=3,eps=1e-6,use_gradient_checkpointing=False)
    head=ActionDiT(**args).eval();reference=copy.deepcopy(head)
    condition=dict(context=torch.randn(8,5,40),context_mask=torch.ones(8,5,dtype=torch.bool),
        visual_mask=torch.ones(8,7,dtype=torch.bool),visual_kv=[dict(k=torch.randn(8,7,24),v=torch.randn(8,7,24)) for _ in range(3)])
    actor=NativeRecipeActor(head,expected_recipe('R_CANONICAL'))
    x=torch.randn(8,8,3);t=torch.full((8,),333.3)
    torch.testing.assert_close(actor.action_velocity(x,t,condition),CachedActionRunner(reference)(x,t,condition),atol=0,rtol=0)
    optimizer=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=5e-5,betas=(.9,.95),weight_decay=.01)
    checks=[]
    for label in ['zero_LoRA','nonzero_LoRA']:
        sampled=[];original=actor._sde_mean_std
        def record(*args,**kwargs):
            mean,std=original(*args,**kwargs);sampled.append((mean.detach().clone(),std));return mean,std
        actor._sde_mean_std=record
        chain,times,deltas,indices=actor.sample_action_chain(condition,8,generator=torch.Generator().manual_seed(345))
        del actor._sde_mean_std
        old=actor.action_chain_logprobs(condition,chain,times,deltas,stoch_idx=indices).detach()
        audit=actor._probability_audit['current']
        if len(sampled)!=3 or indices.tolist()!=[7,8,9]:raise AssertionError('Source stochastic window')
        for i,(mean,std) in enumerate(sampled):
            torch.testing.assert_close(mean,audit['mean'][:,i],atol=0,rtol=0)
            torch.testing.assert_close(torch.tensor(std,dtype=torch.float32),audit['std'][i],atol=0,rtol=0)
        stats=advantages(torch.tensor([[.2,.4,.7,.9,1.,0.,.3,.8]]))
        rollout=dict(cond_bg=condition,chain=chain,timesteps=times,deltas=deltas,stoch_idx=indices,
            old_logp=old,advantages=stats['advantages'],adv_mask=stats['adv_mask'])
        old_copy=old.clone();chain_copy=chain.clone();adapters={n:p.clone() for n,p in actor.named_parameters() if p.requires_grad}
        frozen={n:p.clone() for n,p in actor.named_parameters() if not p.requires_grad}
        ratios=[];gradient_counts=[]
        for inner in range(4):
            optimizer.zero_grad(set_to_none=True);loss,metrics=actor(rollout)
            ratios.append(float((actor._probability_audit['current']['source_clamped_logprob'].mean((-1,-2))-old).exp().sub(1).abs().max()))
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in actor.parameters() if p.requires_grad):
                raise AssertionError('Missing native action LoRA gradient')
            counts={suffix:sum(p.grad is not None and bool(torch.count_nonzero(p.grad))
                for n,p in actor.named_parameters() if n.endswith(suffix)) for suffix in ['lora_A','lora_B']}
            if inner>0 and any(v!=len(actor.closure['adapters']) for v in counts.values()):
                raise AssertionError('A/B task gradients remain zero after the first inner update')
            gradient_counts.append(counts)
            optimizer.step()
        if ratios[0]!=0 or max(ratios[1:])<=0:raise AssertionError('Fresh/updated policy ratios')
        for n,p in actor.named_parameters():
            if n in frozen:torch.testing.assert_close(p,frozen[n],atol=0,rtol=0)
        changed={n:not torch.equal(p,adapters[n]) for n,p in actor.named_parameters() if p.requires_grad}
        if not all(changed.values()):raise AssertionError('Native A/B did not update')
        torch.testing.assert_close(old,old_copy,atol=0,rtol=0);torch.testing.assert_close(chain,chain_copy,atol=0,rtol=0)
        checks.append(dict(scope=label,sampling_recompute_mean_std='exact',four_inner_ratio_errors=ratios,
            changed_parameters=len(changed),nonzero_task_gradients=gradient_counts,frozen_reference_and_base=True))
    merged=ActionDiT(**args).eval();merged.load_state_dict(merged_state_dict(actor.action_expert),strict=True)
    torch.testing.assert_close(actor.action_velocity(x,t,condition),CachedActionRunner(merged)(x,t,condition),atol=2e-6,rtol=2e-5)
    s=expected_recipe('R_SOURCE');c=expected_recipe('R_CANONICAL')
    s['variant']='R_CANONICAL';s['grpo']['reward']['progress_weight']=5.
    assert s==c
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(status='PASS',qualification_only=True,checks=checks,
        merged_velocity='PASS',config_diff='reward_and_name_only',anchor=[.2,1.,1.]),indent=2)+'\n')
    print(out.read_text())


if __name__=='__main__':main()
