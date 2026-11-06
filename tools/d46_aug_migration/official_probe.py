"""Bounded official AUG model / JaT diagnostic on real TRAIN video inputs.

Uses pinned upstream model, adapter and JaT kernel without its Ray launcher.
This is not a reproduction of full CoT-GRPO, Stage II, or its NAVTEST score.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import require_dispatch

SOURCE='edf3422304e4b2987e839a9212cf1176782e7380'
WEIGHTS={'SFT':'c9215f97e61b93a4f995e7e3041ab702e66298b7d3adae1a3d5c1c038584d136',
         'stage1_94.0':'c25e2c80170f0eea04f7602c2583d422c8c7484d24510aba4c7ffab63a0ec0f3'}


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('upstream','checkpoint','data','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--adam-updates',type=int,choices=(0,1),default=0)
    a=p.parse_args();host=require_dispatch();torch.set_num_threads(2)
    root=Path(a.upstream);cp=Path(a.checkpoint);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if(out/'SUMMARY.json').exists():raise FileExistsError('Completed probe already exists')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==SOURCE
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).strip()
    h=hashlib.sha256()
    with(cp/'model.safetensors').open('rb')as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==WEIGHTS[cp.name]
    manifest=json.loads((Path(a.data)/'MANIFEST.json').read_text())
    assert manifest['role']=='NAVTRAIN'and manifest['NAVTEST_overlap']==0
    sft=root/'AUGUstus_SFT';grpo=root/'AUGUstus_GRPO'
    sys.path[:0]=[str(sft),str(grpo)];os.environ['TRANSFUSION_ROOT']=str(sft)
    os.environ['TRANSFUSION_QWEN_MODEL_NAME']=str(cp)
    os.environ['TRANSFUSION_STATS_FILE']=str(cp/'action_norm_stats.json')
    os.environ['TRANSFUSION_FREEZE_VIT']='1'
    from recipe.transfusion import transfusion_policy as upstream_policy
    from recipe.transfusion.algorithms.diffusion_adapter import TransfusionDiffusionAdapter
    from recipe.transfusion.algorithms.jit_grpo import jit_grpo_step_logratio_kl,jit_grpo_policy_loss
    from model.dataset import TransfusionDataset
    # The upstream loader warns about uncovered config-only weights; this
    # diagnostic rejects them, so random/zero backbone cannot look like SFT.
    load_original=upstream_policy._load_state_dict_into
    def strict_load(*args,**kwargs):
        result=load_original(*args,**kwargs)
        if result!=(0,0):raise ValueError('Official model/state mismatch '+repr(result))
        return result
    upstream_policy._load_state_dict_into=strict_load
    cfg=yaml.safe_load((cp/'config.yaml').read_text())
    started=time.monotonic();device=torch.device('cuda',0);torch.cuda.set_device(device)
    model=upstream_policy.build_transfusion_model(cfg,ckpt_path=str(cp),transfusion_root=str(sft))
    if a.adam_updates:
        assert cp.name=='SFT','Only official SFT can initialize diagnostic Adam'
        model.float()  # Explicit FP32 optimizer masters, BF16 autocast below.
        model.qwen_model.gradient_checkpointing_enable()
        torch.cuda.set_per_process_memory_fraction(.8,device)
    model.to(device).eval()
    policy=upstream_policy.TransfusionPolicy(model,model.processor,traj_token=8,action_dim=3)
    dataset=TransfusionDataset(str(Path(a.data)/'samples.json'),model.processor,traj_token=8,action_dim=3,is_training=False)
    samples=json.loads((Path(a.data)/'samples.json').read_text())
    from starVLA.rl.simwam_recipe.reward_readouts import RewardService
    service=RewardService(Path(__file__).resolve().parents[2],workers=1,cache_roots=manifest['canonical_metric_cache_roots'])
    rows=[];update=None
    def infer(i):
        model.eval()
        torch.manual_seed(2101)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            value=policy.rollout([dataset[i]],generate_cot=False,num_steps=1,device=device)
        trajectory=np.asarray(value['trajectories'])
        records=service.score((samples[i]['token'],samples[i]['log'],trajectory))
        return dict(token=samples[i]['token'],metrics=records[0]['metrics'],trajectory=trajectory.tolist())
    try:
        for i in range(len(samples)):
            require_dispatch();row=infer(i);rows.append(row)
            atomic_json(out/f'scene{i:03}.json',row);print(json.dumps(dict(index=i,metrics=row['metrics'])),flush=True)
        if a.adam_updates:
            adapter=TransfusionDiffusionAdapter(model);batch=policy._to_device(policy.collate([dataset[0]]),device)
            records=[];trajectories=[]
            for i in range(8):
                torch.manual_seed(42+i)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    context=adapter.prepare_context(batch)
                    record=adapter.sample_sde_with_logprob(context,num_steps=3,noise_level=[.05,.05,.02],noise_schedule='lambda_t')
                    trajectory=model.apply_action_offset(model.denormalize_actions(record['x0_normed']))
                records.append(record);trajectories.append(trajectory.cpu().numpy()[0]);del context
            scored=service.score((samples[0]['token'],samples[0]['log'],np.stack(trajectories)))
            from .audit_split_credit import released_functions
            group_zscore,reverse_travel,_=released_functions(root)
            reverse_metres=[reverse_travel(t)for t in trajectories]
            reward_values=np.array([5*r['metrics']['score']*(backward==0)
                for r,backward in zip(scored,reverse_metres)],dtype=np.float64)
            reward=torch.from_numpy(reward_values).to(device)
            advantage=torch.from_numpy(group_zscore(reward_values).astype(np.float32)).to(device)
            # Upstream checkpointing is conditional on module.training; eval()
            # above is correct for rollout, but silently disables it in updates.
            model.train();model._visual.eval()
            if not(model._text_model.training and model._text_model.gradient_checkpointing):
                raise RuntimeError('Official update requires active layer checkpointing')
            if any(isinstance(m,torch.nn.Dropout)and m.training and m.p for m in model.modules()):
                raise ValueError('Stochastic dropout would invalidate cached behavior comparison')
            optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
            optimizer.zero_grad(set_to_none=True);fresh=[];kls=[];initial_clipped=[]
            for i,record in enumerate(records):
                require_dispatch()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    context=adapter.prepare_context(batch,grad=True)
                    prediction=adapter.recompute_x0_preds(context,record['x_in'],record['sigma'])
                ratios=[];reference=[]
                for s in range(3):
                    ratio,kl=jit_grpo_step_logratio_kl(prediction[s],record['x0_old'][s],record['x_in'][s],record['next_sample'][s],record['sigma'][s],record['sigma_prev'][s],noise_level=record['noise_level'],sigma_max=record['sigma_max'],schedule=record['noise_schedule'])
                    ratios.append(ratio);reference.append(kl)
                ratio=torch.stack(ratios,1);kl=torch.stack(reference,1)
                # Before the only update, behavior and reference are both SFT.
                loss,_=jit_grpo_policy_loss(ratio,advantage[i:i+1],kl=kl,clip_range=.2,kl_coef=.001)
                fresh.append(float((ratio.exp()-1).abs().max()));kls.append(float(kl.detach().mean()))
                initial_clipped.append(float(((ratio.exp()<.8)|(ratio.exp()>1.2)).float().mean()))
                (loss/8).backward();del prediction,context,loss,ratio,kl,ratios,reference
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters()if p.requires_grad],1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite official gradient')
            readiness=dict(actual_Adam_updates=0,gradient_norm=float(norm),fresh_ratio_max_error=max(fresh),
                reference_KL_before_update=kls,initial_ratio_clip_fraction=initial_clipped,
                peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                note='BF16 cached rollout and full-sequence gradient paths are not bitwise equivalent. '
                     'The previous local 1e-3 diagnostic threshold is not an upstream requirement. '
                     'Keep the original loss and record this difference; refuse any already-clipped fresh sample.')
            atomic_json(out/'UPDATE_READINESS.json',readiness)
            if max(initial_clipped)>0:raise FloatingPointError('Fresh official sample already outside the registered 0.2 trust region')
            optimizer.step();update=dict(actual_Adam_updates=1,scenes=1,group=8,inner=1,gradient_norm=float(norm),
                fresh_ratio_max_error=max(fresh),reference_KL_before_update=kls,rewards=reward.tolist(),
                initial_ratio_clip_fraction=initial_clipped,layer_checkpointing_active=True,optimizer_foreach=False,after=infer(0))
            update.update(advantage_rule='released_split_group_population_std_no_epsilon',
                reverse_travel_m=reverse_metres,advantages=advantage.tolist(),
                reward_rule='released_stage1_5PDMS_reverse_gate_1cm')
        atomic_json(out/'SUMMARY.json',dict(status='COMPLETE_OFFICIAL_TRAIN_DIAGNOSTIC',host=host,upstream_sha=SOURCE,
            checkpoint_sha=h.hexdigest(),rows=rows,update=update,seconds=time.monotonic()-started,
            peak_GPU_bytes=torch.cuda.max_memory_allocated(),NAVTEST_used=False,formal_training_started=False,
            limits='8 TRAIN scenes; native official one-step action inference, no CoT. Optional single-GPU action-only JaT update uses FP32 master/BF16 autocast, no FSDP. Not the paper full run or a NAVTEST result.'))
    finally:service.close()


if __name__=='__main__':main()
