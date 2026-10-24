"""Actual common independent ODE pool, separate reference seeds and cost ledger."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from starVLA.rl.qwen_flow_exploration.protocol import ROOT,read_protocol,require_budget_lock
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def diagnostics(reference,records):
    rewards=np.array([r['raw_reward'] for r in records],dtype=np.float64)
    ref=np.array([r['raw_reward'] for r in reference],dtype=np.float64)
    keys=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')
    safe=np.array([all(r['native_subscores'][k]==1 for k in keys) for r in records])
    stable_safe=all(all(r['native_subscores'][k]==1 for k in keys) for r in reference)
    stratum=('stable_high' if stable_safe and (ref>=.9).all() else 'stable_zero' if (ref<=1e-6).all()
             else 'low_nonzero' if ref.mean()<.6 else 'other_or_unstable')
    other=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort','driving_direction_compliance')
    progress_only=(np.ptp(rewards)>0 and all(np.ptp([r['native_subscores'][k] for r in records])==0 for k in other))
    opportunities={}
    for K in (8,16,24,32):
        available=safe[:K];best=float(rewards[:K][available].max()) if available.any() else None
        opportunities[K]=dict(feasible=bool(available.any()),oracle_PDMS=float(rewards[:K].max()),
            feasible_oracle_PDMS=best,safe_improvement_ge005=bool(best is not None and best-ref.mean()>=.005))
    return dict(stratum=stratum,finite_reference_joint_safe=stable_safe,reference_mean=float(ref.mean()),
                all_zero=bool((rewards==0).all()),all_one=bool((rewards==1).all()),
                tied=bool(np.ptp(rewards)==0),other_tied=bool(np.ptp(rewards)==0 and not((rewards==0).all() or(rewards==1).all())),
                reward_std=float(rewards.std()),small_variance_nontied=bool(0<rewards.std()<.005),
                progress_only=bool(progress_only),opportunities=opportunities)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,required=True)
    a=p.parse_args();protocol=read_protocol(a.protocol);require_budget_lock(protocol,phase='DIAGNOSTIC_LOCKED')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if not 0<=a.rank<a.world:raise ValueError('Bad common bank shard')
    ledger_path=root/f'ledger_rank{a.rank}.json';env=None;start=time.monotonic();complete=[]
    try:
        env=NativeEnvironment(protocol,experiment_id='common_base_ODE_K32',ledger_path=ledger_path)
        work=[(split,token) for split in ('TRAIN','DEV_SELECT') for token in protocol['diagnostic'][split]]
        for split,token in work[a.rank::a.world]:
            dest=root/split/f'{token}.pt';seal=dest.with_suffix('.json')
            if seal.exists():
                old=json.loads(seal.read_text())
                if old['protocol_sha256']!=protocol['sha256'] or old['payload_sha256']!=file_sha(dest):
                    raise ValueError('Changed common bank candidate cache')
                env.ledger.add(cache_reused_candidates=37,logical_scoring_cost=37)
                complete.append(str(dest));continue
            condition=env.condition(token,split)
            root_list=[];seeds=[]
            for seed in protocol['diagnostic']['reference_seeds']:
                actual=seed_for('three_route_independent_reference',seed,token)
                root_list.append(torch.randn((8,3),generator=torch.Generator().manual_seed(actual)));seeds.append(seed)
            for i in range(32):
                actual=seed_for('three_route_initial_candidate_bank',42,token+':'+str(i))
                root_list.append(torch.randn((8,3),generator=torch.Generator().manual_seed(actual)));seeds.append(actual)
            roots=torch.stack(root_list)
            generated=env.sample_ode(condition,roots)
            records=env.score_actions(token,generated,split,proposal_source='common_base_ODE',seed=seeds,
                latent_or_proposal_parameters=[dict(root=z.tolist(),noise_seed=seed) for z,seed in zip(roots,seeds)])
            reference,candidates=records[:5],records[5:]
            obs=env.observation(token,split);summary=diagnostics(reference,candidates)
            payload=dict(schema='common_native_candidate_pool_v1',protocol_sha256=protocol['sha256'],
                scene_token=token,log_id=records[0]['log_id'],split=split,checkpoint_id=protocol['base_model_sha256'],
                condition_id=fingerprint(dict(base=protocol['base_model_sha256'],data=protocol['data_manifest_sha256'],token=token)),
                reference_records=reference,candidate_records=candidates,initial_roots=roots[5:],proprio=obs['proprio'],
                diagnostics=summary,cost_ledger=dict(reference_candidates=5,pool_candidates=32,
                   actual_scored_candidates=37,actual_scorer_batch_calls=1,model_velocity_forwards=100,actual_Qwen_prefills=1))
            dest.parent.mkdir(parents=True,exist_ok=True);atomic_torch(dest,payload)
            atomic_json(seal,dict(status='COMPLETE',protocol_sha256=protocol['sha256'],payload_sha256=file_sha(dest),
                        diagnostics=summary,split=split,checkpoint_id=protocol['base_model_sha256']))
            complete.append(str(dest));env.ledger.save()
            atomic_json(root/f'progress_rank{a.rank}.json',dict(rank=a.rank,completed=len(complete),assigned=len(work[a.rank::a.world]),at=time.time()))
        atomic_json(root/f'COMPLETE_rank{a.rank}.json',dict(status='COMPLETE',rank=a.rank,world=a.world,
                     protocol_sha256=protocol['sha256'],paths=complete,ledger=env.ledger.snapshot(),wall_seconds=time.monotonic()-start))
    except BaseException:
        import traceback
        atomic_json(root/f'failure_rank{a.rank}.json',dict(error=traceback.format_exc(),wall_seconds=time.monotonic()-start))
        raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
