"""Full action-parameter gradient directions on sixteen sealed behavior groups.

No policy update is performed here. Native BF16 leaf gradients are accumulated
in FP32 observer buffers; this is distinguished from the actual ZeRO optimizer
gradients measured in the learning qualification.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from starVLA.rl.flow_grpo.action_head_policy import install_frozen_features
from starVLA.rl.flow_grpo.advantages import behavior_advantages
from starVLA.rl.flow_grpo.loading import weight_path
from starVLA.rl.flow_grpo.math import clipped_surrogate, conditional_kl, reduce_dimensions
from starVLA.rl.flow_grpo.model import make_reference
from starVLA.rl.flow_grpo.observation import PolicyObservation
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.rollout import SamplingSpec, RolloutBatch, evaluate_transitions
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, file_sha, fingerprint, seed_for
from starVLA.rl.safe_flow_improvement.reference_retention import original_replay_loss
from starVLA.rl.safe_flow_improvement.sampling import load_cached_policy


def norm_sq(values):
    return sum(float(v.double().square().sum()) for v in values)


def product(left, right):
    return sum(float((l.float()*r.float()).sum(dtype=torch.float64)) for l,r in zip(left,right))


def grouping(name):
    if 'qwen_proj' in name:
        return 'qwen_proj'
    match = re.search(r'transformer_blocks\.(\d+)', name)
    return 'DiT_block_'+match.group(1) if match else name.split('.')[1]


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('population','cohorts','diagnostic','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--behavior')
    p.add_argument('--behavior-sha256')
    p.add_argument('--config',default='configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml')
    p.add_argument('--source-config',default='/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml')
    a=p.parse_args()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)));configure_numerics();torch.set_num_threads(4)
    pop,cohorts=[json.loads(Path(x).read_text()) for x in (a.population,a.cohorts)]
    root=Path(a.diagnostic);identity=json.loads((root/'manifest.json').read_text())
    tokens=pop['diagnostic_random_tokens'][:8]+[t for t in cohorts['diagnostic_tokens'] if t not in set(pop['diagnostic_random_tokens'])][:8]
    if not set(tokens)<=set(pop['train_tokens']) or identity['behavior_sha256']!=(a.behavior_sha256 or SFT_SHA256):
        raise ValueError('Use genuine matching TRAIN behavior chains')
    rows=[];seals={}
    for token in tokens:
        path=root/'scenes'/token[:2]/(token+'.pt')
        seal=json.loads(path.with_suffix('.json').read_text())
        if seal['identity_sha256']!=identity['sha256'] or seal['payload_sha256']!=file_sha(path):
            raise ValueError('Missing or changed behavior chain')
        rows.append(torch.load(path,weights_only=True,map_location='cpu'));seals[token]=seal['payload_sha256']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    registration=dict(tokens=tokens,behavior_identity=identity['sha256'],payload_sha256=fingerprint(seals),
                      fixed_reward_scale=.005,advantage_clip=5.,clip_range=.02,
                      source_sha256=file_sha(__file__),no_optimizer_updates=True,
                      gradient_scope='complete_359_action_tensors_native_BF16_leaf_FP32_observer_sum_FP64_statistics',
                      deployment_ODE_KL=False,joint_sum_is_offline_only=True)
    atomic_json(out/'registration.json',registration)
    policy,store,cfg,sft,behavior=load_cached_policy(a.config,a.source_config,behavior=a.behavior,behavior_sha256=a.behavior_sha256)
    reference=make_reference(policy)
    if behavior!=SFT_SHA256:
        with torch.serialization.safe_globals([set]):
            weights=torch.load(weight_path(cfg['sft_checkpoint']),weights_only=True,mmap=True,map_location='cpu')
        weights=weights.get('module',weights)
        reference.action_model.load_state_dict({k.removeprefix('action_model.'):v for k,v in weights.items() if k.startswith('action_model.')},strict=True)
        del weights
    install_frozen_features(policy,store);install_frozen_features(reference,store)
    names,parameters=zip(*[(n,p) for n,p in policy.named_parameters() if p.requires_grad])
    assert len(parameters)==359
    reward=torch.tensor([[f['metrics']['score'] for f in r['samples']['legacy_sde']['feedback']] for r in rows],device='cuda',dtype=torch.float32)
    grouped,_=behavior_advantages(reward,normalization='group')
    global_adv,global_stats=behavior_advantages(reward,normalization='global_batch')
    fixed=((reward.double()-reward.double().mean(1,keepdim=True))/.005).clamp(-5,5).float()
    advantages={'RL_group':grouped,'RL_global_batch':global_adv,'RL_fixed_scale':fixed}
    sums={}; details=[];started=time.monotonic()
    for scene,(token,row) in enumerate(zip(tokens,rows)):
        value=store.get(token);data=row['samples']['legacy_sde']['sample'];spec=SamplingSpec(**data['sampling_spec'])
        obs=PolicyObservation(((),),(value['lang'],),(value['state'].numpy().copy(),),(token,))
        rollout=RolloutBatch(obs,(token,),torch.arange(16),0,0,torch.arange(11,device='cuda').float()/10,
                             data['chain'][None].cuda(),data['elementwise_logprob'][None].cuda(),
                             torch.ones(1,16,10,1,1,device='cuda',dtype=torch.bool),torch.ones(1,16,10,device='cuda',dtype=torch.bool),spec,identity)
        with torch.no_grad():
            ref=evaluate_transitions(reference,obs,rollout,layout='flat_saved_chain')
        stats=evaluate_transitions(policy,obs,rollout,layout='flat_saved_chain')
        current=reduce_dimensions(stats['elementwise_logprob'],rollout.dimension_mask)
        losses={};pg_terms={}
        for kind,adv in advantages.items():
            term,ratio=clipped_surrogate(current,rollout.old_logprob,adv[scene:scene+1,:,None],.02)
            losses[kind]=term.mean();pg_terms[kind]=term
        kl=reduce_dimensions(conditional_kl(stats['mean'],stats['std'],ref['mean'],ref['std'],temporal_correlation=.8),rollout.dimension_mask).mean()
        losses['reference_conditional_KL']=kl
        gradients={}
        for kind,loss in losses.items():
            grads=torch.autograd.grad(loss,parameters,retain_graph=True)
            if any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Nonfinite native parameter gradient')
            gradients[kind]=[g.detach().cpu() for g in grads]
            del grads
        step_records=[]
        for step in range(10):
            grads=torch.autograd.grad(pg_terms['RL_group'][:,:,step].mean()/10,parameters,retain_graph=step<9)
            gn=math.sqrt(norm_sq(grads));total_norm=math.sqrt(norm_sq(gradients['RL_group']))
            step_dot=sum(float((g.detach().float()*whole.to(g.device).float()).sum(dtype=torch.float64)) for g,whole in zip(grads,gradients['RL_group']))
            step_records.append(dict(generation_step=step,norm=gn,cosine_with_total=step_dot/(gn*total_norm) if gn*total_norm>0 else None))
            del grads
        replay=original_replay_loss(policy,value['raw'].cuda(),value['action'][None].cuda().float(),seed_for('D2_replay',42,token))
        gradients['SFT_replay']=[g.detach().cpu() for g in torch.autograd.grad(replay,parameters)]
        norms={k:math.sqrt(norm_sq(g)) for k,g in gradients.items()}
        groups={}
        for kind,grads in gradients.items():
            buckets={}
            for name,g in zip(names,grads):
                group=grouping(name);buckets[group]=buckets.get(group,0.)+float(g.double().square().sum())
            groups[kind]={k:math.sqrt(v) for k,v in buckets.items()}
            if kind not in sums:
                sums[kind]=[g.float()/len(tokens) for g in grads]
            else:
                for total,g in zip(sums[kind],grads):total.add_(g.float(),alpha=1/len(tokens))
        cosines={left+'/'+right:product(gradients[left],gradients[right])/(norms[left]*norms[right]) if norms[left]*norms[right]>0 else None
                 for i,left in enumerate(gradients) for right in list(gradients)[i+1:]}
        logratio=(current-rollout.old_logprob).detach();joint_logratio=logratio*32
        details.append(dict(token=token,subset=row['diagnostic_subpopulation'],norms=norms,groups=groups,cosines=cosines,per_generation_step=step_records,
                            reward_std=float(reward[scene].double().std(unbiased=False)),
                            recomputed_behavior_ratio_mean=float(ratio.detach().mean()),
                            recomputed_behavior_clip_fraction=float((ratio.detach().sub(1).abs()>.02).float().mean()),
                            dimension_mean_logratio_max_abs=float(logratio.abs().max()),joint_sum_logratio_max_abs=float(joint_logratio.abs().max()),
                            first_ratio_near_one_is_normal=True))
        atomic_json(out/'per_scene.json',details)
        print(json.dumps(dict(scene=scene+1,total=16,token=token,norms=norms)),flush=True)
        del gradients,stats,ref,losses,pg_terms,replay,rollout,current,kl
    norms={k:math.sqrt(norm_sq(g)) for k,g in sums.items()}
    cosines={l+'/'+r:product(sums[l],sums[r])/(norms[l]*norms[r]) if norms[l]*norms[r]>0 else None
             for i,l in enumerate(sums) for r in list(sums)[i+1:]}
    report=dict(registration,status='COMPLETE_NATIVE_GRADIENT_DIAGNOSTIC',scenes=16,global_advantage_statistics=global_stats,
                aggregate_norms=norms,aggregate_cosines=cosines,wall_seconds=time.monotonic()-started,
                clipping_note='No optimizer step here; actual pre-clip/scale observations are in the eight-GPU qualification and historical inner-epoch logs',
                native_observer_is_not_ZeRO_FP32_accumulation=True)
    atomic_json(out/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
