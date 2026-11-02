"""Finite real-model coherent collection/scoring/four-inner probe, not training.

The campaign owner allocates the GPU and leases. This entry never dispatches.
It uses an independent IL70 actor/Adam0 and cannot initialize a formal run.
"""
import argparse,copy,time
from pathlib import Path
import torch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_performance.data_protocol import load_manifest
from starVLA.rl.qwen_world_safety_exploration.credit import canonical_components,GroupCredit
from starVLA.rl.qwen_world_multiscene.runtime import FastPackedRuntime
from starVLA.rl.qwen_world_multiscene.collection import collect_packed
from starVLA.rl.qwen_world_multiscene.conditions import PackedConditions
from starVLA.rl.qwen_world_multiscene.recovery import validate_packs


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('config','manifest','output','il-checkpoint','parent-selection','data'):p.add_argument('--'+key,required=True)
    p.add_argument('--tokens',nargs='+');a=p.parse_args();a.qualification=True;a.qualification_force_auxiliary=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.cuda.set_device(0)
    rt=FastPackedRuntime();cfg=rt.load_recipe(a);actor,_,_=rt.make_actor(cfg,a,torch.device('cuda',0))
    manifest,logs=load_manifest(a.manifest);n=rt.spec['microbatch_scenes'];g=rt.group_size
    tokens=a.tokens if a.tokens is not None else manifest['tokens'][:n]
    if len(tokens)!=n or len(set(tokens))!=n or any(t not in logs for t in tokens):raise ValueError('Distinct registered TRAIN scenes required')
    service=rt.make_reward_service();gen=torch.Generator().manual_seed(42);start=time.monotonic()
    try:
        leaves,pack=collect_packed(actor,rt.conditions,[rt.conditions.get(t,actor.device)for t in tokens],tokens,[logs[t]for t in tokens],service,gen,
            version=dict(run='COHERENT_PACK_QUALIFICATION_NOT_FORMAL',update=0,adapter_sha256=rt.parent['model_sha256']),pack_index=0)
        rewards=torch.cat([s['rewards']for s in leaves]);components=torch.cat([canonical_components(s['records'],g,device=actor.device)['components']for s in leaves])
        stats=GroupCredit(g).advantages(rewards,components,all_material=torch.cat((rewards.double().unsqueeze(-1),components),-1))
        for i,s in enumerate(leaves):s.update(advantages=stats['advantages'][i*g:(i+1)*g],adv_mask=stats['adv_mask'][i*g:(i+1)*g],valid_scene=True)
        execution=copy.deepcopy(cfg['research']['packed_execution'])
        validate_packs(dict(samples=leaves,packed_groups=[pack],packed_execution=execution),execution)
        opt=rt.make_optimizer(actor,cfg);records=[]
        for inner in range(4):
            opt.zero_grad(set_to_none=True)
            raw=PackedConditions(rt.conditions,selected_layers=actor.scene_provider.selected_layers).pack([s['cond_bg']for s in leaves],actor.device)
            torch.cuda.synchronize();tick=time.monotonic()
            loss,metrics=actor(dict(packed_loss=True,samples=leaves,packed_condition=raw,weights=[1/n]*n))
            audit=actor._probability_audit['current'];ratio=(audit['source_clamped_logprob'].mean((-1,-2))-torch.cat([s['old_logp']for s in leaves])).exp()
            error=float((ratio-1).abs().max())
            if inner==0 and error>1e-4:raise ValueError('Real coherent packed fresh ratio failed: '+str(error))
            loss.backward();torch.cuda.synchronize();elapsed=time.monotonic()-tick
            named={k:v for k,v in actor.named_parameters()if v.requires_grad}
            if any(v.grad is None or not torch.isfinite(v.grad).all()for v in named.values()):raise ValueError('Missing/nonfinite real packed gradient')
            norms={prefix:sum(float(v.grad.double().square().sum())for k,v in named.items()if k.startswith(prefix))**.5
                for prefix in ('action_expert.','scene_provider.','predictive_provider.')}
            norm=float(torch.nn.utils.clip_grad_norm_(list(named.values()),cfg['gradient_clipping_effective']));opt.step()
            records.append(dict(inner=inner,fresh_ratio_max_abs=error,loss=float(loss),preclip_gradient_norm=norm,
                gradient_scopes=norms,used_lrs={x['name']:x['lr']for x in opt.param_groups},
                forward_backward_seconds=elapsed,metrics={k:float(v)for k,v in metrics.items()}))
        receipt=dict(schema='world_coherent_packed_real_GPU_qualification_v1',status='PASS',formal_training=False,
            real_full_IL70=True,parent_model_sha256=rt.parent['model_sha256'],trainable_numel=actor.closure['trainable_numel'],
            packed_execution=execution,scenes=tokens,canonical_queries=n*g,Adam_steps=4,
            normalization='single-device qualification scene population only; formal trainer uses all valid B*G/8ranks',
            iterations=records,cache=rt.conditions.statistics(),scoring_batch_sizes=[g]*n,
            qualification_only='Independent actor/Adam0; no new weights exported or admitted as formal start',
            qualification_schedule='Four-step finite stress probe at registered peak LR, no warmup; full native training retains original exposure-clock schedule',
            peak_GPU_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.monotonic()-start)
        atomic_json(out/'COMPLETE.json',receipt);print('PASS '+str(out))
    finally:service.close()


if __name__=='__main__':main()
