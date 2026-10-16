"""Real raw-observation/cache/deployment and source-chain LoRA plumbing gates.

The synthetic reward vector below is exclusively a probability/gradient test;
it is not driving feedback, cannot train a formal model and is never scored as
an RL result. All qualification weights remain isolated.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
import torch
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.fm_transfer.deployment import SourceActionHead
from starVLA.rl.fm_transfer.rl_adapter import load_supervised_policy,make_transfer_actor,TransferConditions
from starVLA.rl.simwam_recipe.contracts import ROOT as OLD_ROOT,SFT_EXPORT,POPULATION,load_recipe,fingerprint,seed_for
from starVLA.rl.simwam_recipe.source_advantages import advantages
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state
from starVLA.rl.simwam_recipe.vendor.lora import merged_state_dict
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();host=require_dispatch();configure_numerics();started=time.monotonic();torch.cuda.set_device(0)
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    os.environ['BASE_VLM']='/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-WorldAction'
    from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
    from starVLA.rl.simwam_recipe.contracts import RAW_CONDITION
    cache=ConditionStore(RAW_CONDITION)
    # VLAAgent has an environment-selected backend. Pin the already validated
    # original execution, not its unrelated fallback default (SDPA).
    os.environ['VLM_ATTN_IMPLEMENTATION']=cache.identity['execution']['attention']
    from infer import VLAAgent
    from starVLA.model.framework.baseline_qwen import extract_baseline_action_conditions
    agent=VLAAgent(SFT_EXPORT,device='cuda',qwen_forward_mode='optimized')
    model=agent.model.eval().requires_grad_(False)
    policy,sft_id=load_supervised_policy(a.checkpoint,device='cuda',qualification=True)
    policy.eval().requires_grad_(False)
    cfg=copy.deepcopy(agent.model_config)
    cfg.datasets.vla_data.data_root='/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/dataset'
    cfg.datasets.vla_data.datalist_path='/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/navtrain_tokens.json'
    cfg.enable_image_aug=0;cfg.w_depth=0
    dataset=KeyedDataset(cfg);pop=json.loads(Path(POPULATION).read_text());token=pop['train_tokens'][0]
    obs=prepare_policy_observation([dataset[(token,seed_for('fm_transfer_deployment',2101,token))]])
    if policy.architecture=='ddp':model.action_model=policy.head
    else:
        deployment=SourceActionHead(cfg).to('cuda').eval()
        deployment.policy.load_state_dict(policy.state_dict(),strict=True)
        model.action_model=deployment.requires_grad_(False)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        hidden,pos=model.encode_policy_features(obs.examples(),feature_output='normalized')
        raw=extract_baseline_action_conditions(hidden,pos['action'])
    cached=cache.get(token).to('cuda')
    torch.testing.assert_close(raw,cached,rtol=0,atol=0)
    root=torch.randn((1,8,4),device='cuda',dtype=raw.dtype,generator=torch.Generator(device='cuda').manual_seed(2101))
    with torch.no_grad(),torch.autocast('cuda',enabled=False):
        expected=policy.sample_ode(cached.float(),root,grid='native')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        direct=model.predict_action_infer_1d(obs.examples(),initial_noise=root)['normalized_actions']
    torch.testing.assert_close(torch.from_numpy(direct),expected.float().cpu(),rtol=0,atol=0)
    architecture=policy.architecture
    del agent,model,policy,dataset,obs,hidden,pos
    if architecture!='ddp':del deployment
    gc.collect();torch.cuda.empty_cache()
    recipe=load_recipe('configs/flow_rl/simwam_source_ddp.yaml')
    actor,sft_id=make_transfer_actor(a.checkpoint,recipe,device='cuda',qualification=True)
    conditions=TransferConditions(actor,sft_id);context=conditions.get(token,'cuda')
    cond={'context':context.expand(8,-1,-1).contiguous()};gen=torch.Generator().manual_seed(42)
    with torch.no_grad():
        chain,times,deltas,indices=actor.sample_action_chain(cond,8,generator=gen)
        old=actor.action_chain_logprobs(cond,chain,times,deltas,stoch_idx=indices).detach()
        old_mean=actor._probability_audit['current']['mean'].clone()
    if chain.shape!=(8,11,8,4) or indices.tolist()!=[7,8,9]:raise ValueError('Changed source chain')
    rew=torch.tensor([[0.,.1,.2,.3,.4,.5,.6,.7]],device='cuda')
    stat=advantages(rew)
    rollout=dict(cond_bg=cond,chain=chain.detach(),timesteps=times,deltas=deltas,stoch_idx=indices,
        old_logp=old,rewards=rew,**{k:stat[k] for k in ['advantages','adv_mask','global_std','raw_group_std']})
    parameters=[v for v in actor.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    before=adapter_state(actor);steps=[]
    for k in range(4):
        opt.zero_grad(set_to_none=True);loss,metrics=actor(rollout)
        current=actor._probability_audit['current'];ratio=(current['source_clamped_logprob'].mean((-1,-2))-old).exp()
        error=float((ratio-1).abs().max())
        if k==0:
            if error>1e-4:raise ValueError('Fresh source ratio mismatch')
            torch.testing.assert_close(current['mean'],old_mean,rtol=0,atol=0)
        loss.backward()
        norms={n:float(p.grad.norm()) for n,p in actor.named_parameters() if p.requires_grad and p.grad is not None}
        if len(norms)!=len(parameters) or not all(torch.isfinite(torch.tensor(v)) for v in norms.values()):raise ValueError('Missing/nonfinite LoRA gradient')
        if k>=1 and not all(v>0 for v in norms.values()):raise ValueError('LoRA A/B did not connect after B initialization')
        opt.step();steps.append(dict(inner=k+1,loss=float(loss.detach()),ratio_max_error=error,
            gradient_tensors=len(norms),positive_gradients=sum(v>0 for v in norms.values())))
    after=adapter_state(actor)
    if not all(torch.count_nonzero(after[n]-before[n]) for n in before):raise ValueError('Some LoRA tensors never updated')
    reference_frozen=all(not p.requires_grad and p.grad is None for p in actor.ref_action_expert.parameters())
    merged=merged_state_dict(actor.action_expert)
    from starVLA.rl.fm_transfer.model import make_policy
    plain=make_policy(architecture,device='cuda').eval().requires_grad_(False)
    plain.head.load_state_dict(merged,strict=True)
    with torch.no_grad():
        # Merge changes the FP32 matmul order; measure rather than assert bits.
        x=chain[:,7];step_time=times[7].reshape(1).expand(8)
        actual=actor.action_velocity(x,step_time,cond)
        expected=plain.velocity_sigma(x,step_time/1000,plain.project(cached.float()).expand(8,-1,-1).contiguous())
        merge_error=float((actual-expected).abs().max())
        torch.testing.assert_close(actual,expected,rtol=5e-4,atol=5e-4)
    result=dict(status='PASS',host=host,architecture=architecture,SFT=sft_id,
        original_observation_cache_raw_exact=True,original_VLAAgent_native_ODE_exact=True,
        source_chain_shape=list(chain.shape),stochastic_steps=indices.tolist(),source_recipe='R_SOURCE',
        LoRA_tensors=len(parameters),all_adapters_updated=True,permanent_reference_frozen=reference_frozen,
        four_inner_test=steps,merge_velocity_max_error=merge_error,wall_seconds=time.monotonic()-started,
        qualification_only=True,driving_scores_computed=0,
        reward_scope='synthetic arithmetic test on real model generated chains; not a driving learning result')
    if not reference_frozen:raise ValueError('Reference trained')
    atomic_json(out,result);print(json.dumps(result))


if __name__=='__main__':main()
