"""Per-step PG/anchor gradients on a saved real global batch, without new samples."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from starVLA.rl.simwam_recipe.contracts import require_dispatch,file_sha,registered_stochastic_steps
from starVLA.rl.simwam_recipe.trainer_adapter import make_actor
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
from starVLA.rl.simwam_recipe.checkpoint import device_tree
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json


def anchor_channel_contributions(squared_mean_difference, weights):
    """Diagnostic contributions with the actor's actual horizon/channel weights."""
    if squared_mean_difference.ndim != 4 or tuple(weights.shape) != tuple(squared_mean_difference.shape[-2:]):
        raise ValueError('Expected candidate, step, horizon, channel and horizon/channel weights')
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Invalid anchor diagnostic weights')
    return (squared_mean_difference.detach()*weights).sum(2).mean(0)/weights.sum()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--collection',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--gradient-strata-registration');a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve the previous diagnostic: '+str(output))
    started=time.monotonic()
    require_dispatch();configure_numerics();run=Path(a.run);checkpoint=Path(a.checkpoint)
    identity=json.loads((run/'identity.json').read_text());cfg=identity['recipe'];world=identity['world']
    if json.loads((checkpoint/'identity.json').read_text())['sha256']!=identity['sha256']:
        raise ValueError('Checkpoint and run identities differ')
    if file_sha(checkpoint/'files.json')!=json.loads((checkpoint/'COMPLETE').read_text())['files_sha256']:
        raise ValueError('Checkpoint seal changed')
    seal=json.loads((checkpoint/'files.json').read_text())
    if file_sha(checkpoint/'optimizer.pt')!=seal['optimizer.pt']:raise ValueError('Checkpoint changed')
    actor,_,_=make_actor(cfg);saved=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
    optimizer_steps={int(v['step']) for v in saved['optimizer']['state'].values()}
    if optimizer_steps!={a.collection*cfg['grpo']['train']['num_inner_epochs']}:
        raise ValueError('Diagnostic requires the completed four-inner checkpoint for the requested collection')
    restore_adapters(actor,saved['adapters']);parameters=[p for p in actor.parameters() if p.requires_grad]
    count=sum(p.numel() for p in parameters)
    strata=None
    if a.gradient_strata_registration:
        from scripts.analysis.simwam_recipe.saved_gradient_strata import SavedGradientStrata
        strata=SavedGradientStrata(a.gradient_strata_registration,run,checkpoint,a.collection,identity,count)
    pg_gradient=torch.zeros(3,count);anchor_gradient=torch.zeros_like(pg_gradient)
    per_step=[];batches=[];categories=dict(safe_recovery_vs_single_SFT=0,safe_other_positive=0,
        positive_still_unsafe=0,positive_unsafe_despite_safe_alternative=0,positive_micro_progress_only=0)
    clip=[];raw_clip=[];tail=[];innovations=[];density_ratio=[];movement=[]
    for rank in range(world):
        path=run/'behavior'/f'collection{a.collection:06}'/f'rank{rank}.pt'
        cpu_row=torch.load(path,map_location='cpu',weights_only=False)
        if strata is not None:strata.validate(rank,cpu_row)
        row=device_tree(cpu_row,actor.device)
        if row['behavior_version']['run']!=identity['sha256']:raise ValueError('Unmatched behavior identity')
        if row['stoch_idx'].tolist()!=registered_stochastic_steps(cfg['variant']):
            raise ValueError('Saved batch window differs from registered recipe')
        logp,terms=actor.action_chain_logprobs(row['cond_bg'],row['chain'],row['timesteps'],row['deltas'],stoch_idx=row['stoch_idx'],return_terms=True)
        raw=actor._probability_audit['current']['raw_gaussian_logprob'].mean((-1,-2))
        tail.append(actor._probability_audit['current']['source_clamp_hit_rate'])
        with torch.no_grad():
            _,reference=actor.action_chain_logprobs(row['cond_bg'],row['chain'],row['timesteps'],row['deltas'],stoch_idx=row['stoch_idx'],return_terms=True,use_ref=True)
        ratio=(logp-row['old_logp']).exp();unclamped_ratio=(raw-row['old_probability']['raw_gaussian_logprob'].mean((-1,-2))).exp()
        advantage=row['advantages'].clamp(-5,5)[:,None];mask=row['adv_mask'][:,None]
        loss=torch.maximum(-ratio*advantage,-ratio.clamp(.995,1.005)*advantage)
        pg=(loss*mask).sum(0)/mask.sum().clamp_min(1)
        weights=actor.kl_weight_matrix(8,4,actor.device)
        d2=(terms['mean']-reference['mean']).square()
        anchor=(d2*weights).sum((-1,-2)).mean(0)/weights.sum()
        rank_basis=torch.zeros(2,3,count) if strata is not None else None
        for step in range(3):
            for kind,(target,scalar) in enumerate([(pg_gradient,pg[step]),(anchor_gradient,50*anchor[step])]):
                g=torch.autograd.grad(scalar/world,parameters,retain_graph=True)
                flat=torch.cat([v.detach().float().flatten().cpu() for v in g])
                target[step]+=flat
                if strata is not None:
                    strata.add(rank,kind,step,flat);rank_basis[kind,step]=flat
        if strata is not None:
            actual,metrics=actor(row)
            torch.testing.assert_close(actual,pg.mean()+50*anchor.mean(),atol=1e-7,rtol=2e-5)
            actual_g=torch.autograd.grad(actual/world,parameters)
            flat=torch.cat([v.detach().float().flatten().cpu() for v in actual_g])
            reconstructed=rank_basis.sum(0).mean(0)
            torch.testing.assert_close(flat,reconstructed,atol=1e-7,rtol=3e-4)
            strata.oracle+=flat
            strata.oracle_max_abs.append(float((flat-reconstructed).abs().max()))
        d=d2.detach().mean((0,2));v=(terms['vel']-reference['vel']).detach().square().mean((0,2))
        channel_anchor=anchor_channel_contributions(d2,weights)
        torch.testing.assert_close(channel_anchor.sum(-1),anchor.detach(),rtol=1e-5,atol=1e-9)
        movement.append(dict(mean_squared_by_step_channel=d.cpu().tolist(),velocity_squared_by_step_channel=v.cpu().tolist(),
                             weighted_anchor_by_step_channel=channel_anchor.cpu().tolist(),
                             anchor_by_step=anchor.detach().cpu().tolist(),std=terms['std'].detach().cpu().tolist()))
        clip.append(((ratio-1).abs()>.005).float().mean(0).tolist())
        raw_clip.append(((unclamped_ratio-1).abs()>.005).float().mean(0).tolist());density_ratio.append(ratio.detach().cpu().tolist())
        audits=row['noise_audit'][1:] # Exclude the un-clipped root Gaussian.
        innovations.append(dict(clipped=sum(x['outside_source_clip'] for x in audits),total=sum(x['count'] for x in audits)))
        metrics=[r['metrics'] for r in row['records']];positive=row['advantages'].cpu().numpy()>0
        safe=np.array([all(m[k]==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')) for m in metrics])
        sft=Path('/mnt/project/simwam-recipe-20261015/SFT_NATIVE_TRAIN_seed2101/scenes')/row['token'][:2]/(row['token']+'.json')
        base=json.loads(sft.read_text())['metrics'];base_safe=all(base[k]==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'))
        categories['safe_recovery_vs_single_SFT']+=int((positive&safe).sum()) if not base_safe else 0
        categories['safe_other_positive']+=int((positive&safe).sum()) if base_safe else 0
        categories['positive_still_unsafe']+=int((positive&~safe).sum())
        categories['positive_unsafe_despite_safe_alternative']+=int((positive&~safe).sum()) if safe.any() else 0
        constant_safety=all(len({m[k] for m in metrics})==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort','driving_direction_compliance'))
        span=np.ptp([m['score'] for m in metrics])
        if constant_safety and 0<span<.005:categories['positive_micro_progress_only']+=int(positive.sum())
        batches.append(dict(rank=rank,sha256=file_sha(path),behavior=row['behavior_version'],advantage_mask_fraction=float(row['adv_mask'].mean())))
    d=np.mean([m['mean_squared_by_step_channel'] for m in movement],axis=0)
    v=np.mean([m['velocity_squared_by_step_channel'] for m in movement],axis=0)
    channel_anchor=np.mean([m['weighted_anchor_by_step_channel'] for m in movement],axis=0)
    for j,index in enumerate(registered_stochastic_steps(cfg['variant'])):
        pn=float(pg_gradient[j].norm());an=float(anchor_gradient[j].norm());both=pn>0 and an>0
        per_step.append(dict(step=index,PG_gradient_norm=pn,beta50_anchor_gradient_norm=an,
            cosine=float(torch.nn.functional.cosine_similarity(pg_gradient[j],anchor_gradient[j],dim=0)) if both else None,
            x_mean_anchor=float(np.mean([m['anchor_by_step'][j] for m in movement])),
            x_kl_normalized_diagnostic=float(np.mean([m['anchor_by_step'][j]/(2*m['std'][j]**2) for m in movement])),
            logical_anchor_contribution=dict(x=channel_anchor[j,0],y=channel_anchor[j,1],heading_pair=channel_anchor[j,2]+channel_anchor[j,3]),
            mean_shift_RMS_by_native_channel=np.sqrt(d[j]).tolist(),velocity_shift_RMS_by_native_channel=np.sqrt(v[j]).tolist(),
            source_PPO_clip_fraction=float(np.mean(clip,axis=0)[j]),raw_density_PPO_clip_fraction_diagnostic=float(np.mean(raw_clip,axis=0)[j])))
    output.parent.mkdir(parents=True,exist_ok=True)
    effective=[]
    # ||B A||_F^2 = tr((B^T B)(A A^T)); exact low-rank identity avoids
    # materializing 96 full projection matrices merely for a norm diagnostic.
    for name,A in saved['adapters'].items():
        if not name.endswith('.lora_A'):continue
        B=saved['adapters'][name[:-len('lora_A')]+'lora_B']
        aa=A.double()@A.double().T;bb=B.double().T@B.double()
        norm=float((torch.trace(bb@aa).clamp_min(0).sqrt())*(cfg['grpo']['lora']['alpha']/cfg['grpo']['lora']['r']))
        effective.append(dict(projection=name[:-len('.lora_A')],effective_weight_update_Frobenius=norm))
    torch.cuda.synchronize(actor.device)
    result=dict(status='COMPLETE_SAVED_BATCH_DIAGNOSTIC',run_identity=identity['sha256'],checkpoint=str(checkpoint),collection=a.collection,
        analysis_source_sha256=file_sha(__file__),variant=cfg['variant'],optimizer_updates=next(iter(optimizer_steps)),
        anchor_weights_actual=weights.detach().cpu().tolist(),
        probability_checkpoint_relation='Checkpoint after four inner updates; saved old behavior precedes them, so clipping can be a real update effect.',
        cumulative_worker_wall_seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(actor.device),
        new_candidates=0,new_scoring_calls=0,world=world,steps=per_step,batches=batches,positive_candidate_categories=categories,
        positive_categories_note='Recovery relative to the fixed single SFT_NATIVE seed2101 result, not all-noise safety; micro-progress is an overlapping tag',
        probability_tail_clamp_fraction=float(np.mean(tail)),innovation_clip_fraction=sum(x['clipped'] for x in innovations)/sum(x['total'] for x in innovations),
        source_current_velocity_forwards=actor._velocity_calls,reference_velocity_forwards=actor._reference_calls,
        effective_LoRA_weight_updates=effective,
        theory='Gradients are source per-random-step proxies; x_kl_normalized is diagnostic only and is not the legacy correlated C_SDE KL')
    if strata is not None:
        result['gradient_strata']=strata.finish(pg_gradient,anchor_gradient)
        for name,param in actor.named_parameters():
            if param.grad is not None:raise ValueError('Diagnostic unexpectedly retained parameter gradients')
            if param.requires_grad and not torch.equal(param.detach().cpu(),saved['adapters'][name]):
                raise ValueError('Read-only diagnostic changed an adapter')
        result['adapters_unchanged']=True
        result['cumulative_worker_wall_seconds']=time.monotonic()-started
    atomic_json(output,result);print(json.dumps(dict(status=result['status'],output=str(output),new_candidates=0)))


if __name__=='__main__':main()
