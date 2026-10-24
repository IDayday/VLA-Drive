"""Finite CPU safety arithmetic/LoRA acceptance; not driving efficacy evidence."""
import argparse
import copy
import json
from pathlib import Path
import torch
from starVLA.rl.qwen_sourceflow.safe_advantages import component_advantages, event_costs, EventMultipliers, factorized_advantages
from starVLA.rl.qwen_sourceflow.safe_recipe import expected_recipe, COSTS
from starVLA.rl.qwen_sourceflow.recipe import expected_recipe as baseline_recipe
from starVLA.rl.constrained_flow_noise.component_advantages import component_advantages as old_components
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.simwam_recipe.source_advantages import advantages
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);checks=[]
    reward=torch.tensor([[.9,.9,.9,.9,.9,.9,.9,.9],[.5,.6,.7,.8,.9,1.,.4,.3]])
    costs=torch.zeros(2,8,3,dtype=torch.float64);costs[0,0,0]=1;costs[0,1,1]=1;costs[1,2,2]=1
    weights=torch.tensor([1.,1.,1.]);new,stats=component_advantages(reward,costs,weights)
    old_costs=torch.cat((costs.repeat_interleave(2,1),torch.zeros(2,16,2)),dim=-1)
    old,_=old_components(reward.repeat_interleave(2,1),old_costs,torch.tensor([1.,1.,1.,.1,.1]),[.01,1e-8,1e-8,1e-8,1e-8,1e-8])
    torch.testing.assert_close(new,old[:,::2],atol=0,rtol=0);checks.append('G8/C3 matches prior G16/C5 population-moment kernel under duplicated candidates and zero added costs')
    stronger,_=component_advantages(reward,costs,torch.tensor([2.,1.,1.]))
    assert stronger[0,0]<new[0,0] and stronger[0,2]>new[0,2]
    delta=(stronger-new).double();torch.testing.assert_close(delta,-stats['component_z'][...,1],atol=3e-7,rtol=1e-6)
    checks.append('Increasing safety lambda decreases violating relative advantage; no post whitening')
    zero,_=component_advantages(torch.full((2,8),.9),torch.zeros(2,8,3),weights)
    assert torch.equal(zero,torch.zeros_like(zero));checks.append('Exact tied components remain zero')
    small=torch.tensor([[.8,.8,.8,.8,.8,.8,.8,.800001]])
    tiny,_=component_advantages(small,torch.zeros(1,8,3),weights)
    assert float(tiny.abs().max())<1e-3;checks.append('Absolute 0.01 reward floor does not inflate micro-progress')
    rec=[dict(status='OK',metrics=dict(zip(COSTS,[1.,1.,1.]))) for _ in range(8)]
    rec[0]['metrics'][COSTS[0]]=.5; assert event_costs(rec)[0,0,0]==1
    checks.append('NC=0.5 counts as a violation; no invalid feedback fallback')
    d=EventMultipliers([.25,.25,.25]);before=d.snapshot();d.update(torch.tensor([.5,0.,.25]),0)
    assert d.snapshot()[0]>before[0] and d.snapshot()[1]<before[1] and d.snapshot()[2]==before[2]
    clone=EventMultipliers([.25,.25,.25]);clone.load_state_dict(d.state_dict())
    d.update(torch.tensor([.4,.2,.9]),1);clone.update(torch.tensor([.4,.2,.9]),1)
    assert tensor_digest(d.state_dict())==tensor_digest(clone.state_dict());checks.append('Independent dual signs, nonnegative projection, exact Adam state continuation')
    try:d.update(torch.ones(3),1)
    except ValueError:pass
    else:raise AssertionError('Repeated dual collection accepted')
    checks.append('Four inner reuse cannot update dual four times')
    baseline=baseline_recipe('R_CANONICAL');x1=expected_recipe('R_CANONICAL_X1')
    normalized=copy.deepcopy(x1);normalized['variant']='R_CANONICAL';normalized['grpo']['train']['kl_dim_weights']=[.2,1.,1.]
    assert normalized==baseline;checks.append('X1 entire resolved diff is name and longitudinal anchor only')
    # Actual shared source action sampler/loss, with native3D and nonzero adapters.
    architecture=dict(action_dim=3,hidden_dim=32,ffn_dim=64,text_dim=40,freq_dim=16,num_heads=2,attn_head_dim=12,num_layers=3,eps=1e-6,use_gradient_checkpointing=False)
    head=ActionDiT(**architecture).eval()
    condition=dict(context=torch.randn(8,5,40),context_mask=torch.ones(8,5,dtype=torch.bool),visual_mask=torch.ones(8,7,dtype=torch.bool),
        visual_kv=[dict(k=torch.randn(8,7,24),v=torch.randn(8,7,24)) for _ in range(3)])
    paper_records=[]
    for ep in [.2,.3,.4,.5,.6,.7,.8,.9]:
        paper_records.append(dict(status='OK',metrics=dict(no_at_fault_collisions=1.,drivable_area_compliance=1.,
            ego_progress=ep,time_to_collision_within_bound=1.,comfort=1.,score=(5*ep+7)/12)))
    paper_rewards=torch.tensor([[r['metrics']['score'] for r in paper_records]])
    paper_a,paper_stats=factorized_advantages(paper_rewards,paper_records)
    vector=paper_stats['metric_vector'];std=vector.std(1,unbiased=False,keepdim=True)
    z=(vector-vector.mean(1,keepdim=True))/(std+1e-8)
    combined=(z*vector.new_tensor([2.,2.,5.,5.,2.])).sum(-1)
    variance=paper_rewards.double().var(1,unbiased=False,keepdim=True)
    oracle=combined/torch.sqrt(combined.var(1,unbiased=False,keepdim=True)+1e-8)*variance/(variance+.01)
    torch.testing.assert_close(paper_a,oracle.float().reshape(-1),atol=2e-7,rtol=1e-6)
    failed=copy.deepcopy(paper_records)
    for r in failed:r['metrics']['no_at_fault_collisions']=0.;r['metrics']['score']=0.
    no_signal,_=factorized_advantages(torch.zeros(1,8),failed)
    assert not no_signal.any()
    checks.append('DriveWLA Eq8/9 NumPy-style independent tensor oracle; all-zero PDMS shrinkage=0 despite varying EP')
    details=[]
    for variant in ('R_CANONICAL_X1','R_CANONICAL_SCADV3','R_CANONICAL_WLA_ADV'):
        actor=NativeRecipeActor(copy.deepcopy(head),expected_recipe(variant));actor.record_sampling_terms=True
        chain,times,deltas,indices=actor.sample_action_chain(condition,8,generator=torch.Generator().manual_seed(345))
        old=actor.action_chain_logprobs(condition,chain,times,deltas,stoch_idx=indices).detach();sampled=actor.sampling_terms
        for i,(mean,std) in enumerate(sampled):
            torch.testing.assert_close(mean,actor._probability_audit['current']['mean'][:,i],atol=0,rtol=0)
            torch.testing.assert_close(torch.tensor(std,dtype=torch.float32),actor._probability_audit['current']['std'][i],atol=0,rtol=0)
        adv=(new[1] if variant.endswith('SCADV3') else paper_a if variant.endswith('WLA_ADV') else advantages(reward[1:])['advantages'])
        batch=dict(cond_bg=condition,chain=chain,timesteps=times,deltas=deltas,stoch_idx=indices,old_logp=old,advantages=adv,adv_mask=torch.ones(8))
        fixed=tensor_digest(batch);opt=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=5e-5,betas=(.9,.95),weight_decay=.01)
        frozen={n:p.clone() for n,p in actor.named_parameters() if not p.requires_grad};ratios=[]
        for inner in range(4):
            opt.zero_grad();loss,_=actor(batch);ratios.append(float((actor._probability_audit['current']['source_clamped_logprob'].mean((-1,-2))-old).exp().sub(1).abs().max()))
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters() if p.requires_grad);opt.step()
        assert ratios[0]==0 and ratios[-1]>0 and tensor_digest(batch)==fixed
        assert all(torch.equal(p,frozen[n]) for n,p in actor.named_parameters() if n in frozen)
        assert all(float(s['step'])==4 for s in opt.state.values())
        details.append(dict(variant=variant,ratios=ratios,all_LoRA_gradients_finite=True,optimizer_steps=4,frozen_base_reference=True))
    checks.append('All admitted adapters execute actual source K10/G8 sampler and four native3D LoRA optimizer updates; old chain fixed')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='PASS',qualification_only=True,checks=checks,native_action_tests=details,no_driving_efficacy_claim=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())


if __name__=='__main__':main()
