"""Full native D: independent vs packed G8/G16/G32, FP32/BF16 evidence.

The FP32 oracle and cross-candidate perturbation test check semantics. BF16
execution differences and their joint-density amplification remain explicit;
no cross-layout old probability is ever used for training.
"""
import argparse
import os
from pathlib import Path
import torch
import copy
from starVLA.rl.d44_native_rl.parent import load
from starVLA.rl.d44_native_rl.actor import D44Actor
from starVLA.rl.d44_native_rl.conditions import DConditions
from starVLA.rl.d44_native_rl.config import source_config
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.deployment import inspect
from starVLA.rl.d44_native_rl.gaussian_objective import gaussian_terms
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
from starVLA.rl.qwen_world_compare.contracts import require_dispatch
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');configure_numerics();torch.set_num_threads(2);require_dispatch()
    identity,_,_=inspect(a.checkpoint);cfg=identity['recipe'];data=DataStream(cfg,64);model,info=load(cfg['parent_checkpoint'],'cuda',readouts=cfg['post_feedback'])
    actor=D44Actor(model,source_config(cfg),cfg);restore_adapters(actor,torch.load(Path(a.checkpoint)/'optimizer.pt',map_location='cpu',weights_only=False)['adapters'])
    conditions=DConditions(model,data.data,info);object.__setattr__(actor,'conditions',conditions)
    material=conditions.materialize(conditions.get(data.tokens[0]),actor.device);params=[p for p in actor.parameters()if p.requires_grad]
    model.decision.activation_checkpointing=True;model.decision.checkpoint_short_suffix=True
    times,deltas=actor.infer_action_scheduler.build_inference_schedule(num_inference_steps=10,device=actor.device,dtype=torch.float32,shift_override=5.)
    grid=times/1000
    result=[]
    for group in (8,16,32):
        model.decision.pack_shared_candidates=True
        with torch.no_grad():
            chain,_,_,_=actor.sample_action_chain(dict(material,context=torch.zeros(group,1,1,device='cuda')),8,generator=torch.Generator().manual_seed(42))
        actual_sampling=copy.deepcopy(actor.sampling_terms)
        state=chain[:,[7,8,9]].contiguous()
        modes={}
        for precision in ('float32','bfloat16'):
            for packed in (False,True):
                actor.zero_grad(set_to_none=True);model.decision.pack_shared_candidates=packed
                ctx=torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bfloat16')
                with ctx:
                    prefix=model.prepare_prefix(material['current']);first_key=prefix.layer_kv[0][0];first_key.retain_grad()
                    selected=prefix.index_select(torch.zeros(group,device='cuda',dtype=torch.long))
                    pred=torch.stack([model.denoise_x0(state[:,j].contiguous(),grid[k].expand(group),selected)for j,k in enumerate((7,8,9))],1)
                weight=torch.linspace(-1,1,group*3*24,device='cuda').reshape(group,3,8,3)
                (pred.float()*weight).mean().backward()
                grads=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten().cpu()for p in params])
                modes[(precision,packed)]=dict(pred=pred.detach().cpu(),gradient=grads,key_gradient=first_key.grad.detach().float().cpu())
                # Other candidates cannot change candidate0. Same layout/shape,
                # so this test does not conflate cross-candidate leakage with
                # changed BF16 GEMM batch layouts.
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bfloat16'):
                    changed=state.clone();changed[1:]+=2
                    prefix2=model.prepare_prefix(material['current']).index_select(torch.zeros(group,device='cuda',dtype=torch.long))
                    alternative=torch.stack([model.denoise_x0(changed[:,j].contiguous(),grid[k].expand(group),prefix2)for j,k in enumerate((7,8,9))],1)
                    actor.ref_action_expert.decision.pack_shared_candidates=packed
                    reference_prefix=actor.ref_action_expert.prepare_prefix(material['ref_current']).index_select(torch.zeros(group,device='cuda',dtype=torch.long))
                    reference=torch.stack([actor.ref_action_expert.denoise_x0(state[:,j].contiguous(),grid[k].expand(group),reference_prefix)for j,k in enumerate((7,8,9))],1)
                means=[];references=[];std=[]
                for j,k in enumerate((7,8,9)):
                    mu,s=actor._sde_mean_std(state[:,j],(state[:,j]-pred[:,j].detach())/grid[k],float(times[k])/1000,float(deltas[k]));means.append(mu.detach().cpu());std.append(s)
                    rm,_=actor._sde_mean_std(state[:,j],(state[:,j]-reference[:,j])/grid[k],float(times[k])/1000,float(deltas[k]));references.append(rm.cpu())
                modes[(precision,packed)].update(mean=torch.stack(means,1),reference=torch.stack(references,1),std=torch.tensor(std))
                cross=float((alternative[0].float().cpu()-pred[0].detach().cpu()).abs().max())
                modes[(precision,packed)]['cross_candidate_error']=cross
                del prefix,prefix2,reference_prefix,pred,alternative,reference,selected;actor.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        row=dict(group=group,checkpoint=a.checkpoint,sigma=[float(grid[k])for k in(7,8,9)],source='full D44 model, nonzero D5 adapters, actual TRAIN image condition',
            actual_sampled_candidates=group,scorer_calls=0,optimizer_updates=0,precision={})
        for precision in ('float32','bfloat16'):
            x=modes[(precision,False)];y=modes[(precision,True)]
            relative=lambda left,right:float((left-right).norm()/left.norm().clamp_min(1e-30))
            row['precision'][precision]=dict(output_max_abs=float((x['pred']-y['pred']).abs().max()),gradient_relative=relative(x['gradient'],y['gradient']),
                prefix_key_gradient_relative=relative(x['key_gradient'],y['key_gradient']),
                cross_candidate_error=max(x['cross_candidate_error'],y['cross_candidate_error']))
        old=torch.stack([m.cpu()for m,s in actual_sampling],1);std=modes[('bfloat16',True)]['std'];nxt=chain[:,[8,9,10]].cpu()
        row['native_transition_density']={}
        for (precision,packed),value in modes.items():
            logr,kl=gaussian_terms(value['mean'],old,value['reference'],std,nxt)
            row['native_transition_density'][precision+'_'+str(packed)]=dict(mean_std=std.tolist(),max_abs_joint_logratio=float(logr.abs().max()),reference_KL_joint=float(kl.mean()))
        row['BF16_vs_FP32_output_error']={str(pack):float((modes[('float32',pack)]['pred']-modes[('bfloat16',pack)]['pred']).abs().max())for pack in (False,True)}
        f=row['precision']['float32'];row['semantic_status']='PASS'if f['output_max_abs']<=3e-5 and f['gradient_relative']<=1e-3 and f['prefix_key_gradient_relative']<=1e-3 and all(v['cross_candidate_error']==0 for v in row['precision'].values())else'FAIL'
        row['matched_packed_sampler_max_mean_error']=float((old-modes[('bfloat16',True)]['mean']).abs().max())
        result.append(row);atomic_json(a.output,dict(status='RUNNING',records=result))
    atomic_json(a.output,dict(status='PASS'if all(r['semantic_status']=='PASS'for r in result)else'FAIL',records=result,
        interpretation='FP32 semantics and fixed-layout independence; BF16 differences retained. Only matching packed sampler/current paths are used in formal Gaussian PPO.'))


if __name__=='__main__':main()
