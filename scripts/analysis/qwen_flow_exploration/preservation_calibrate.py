"""TRAIN-only frozen-SDE budgets, one FM-scale and bounded P1 drift calibration."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_flow_exploration.preservation.archive import FrozenArchive
from starVLA.rl.qwen_flow_exploration.preservation.objective import official_costs, calibrate_replay_coefficient
from starVLA.rl.qwen_flow_exploration.preservation.update import PreservationUpdate
from starVLA.rl.qwen_flow_exploration.preservation.diagnostics import physical_drift, select_conservative_trial
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state, restore_adapters
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def gradients(actor):
    return [p.grad.detach().cpu().double().clone() if p.grad is not None else torch.zeros_like(p,dtype=torch.float64,device='cpu')
            for p in actor.parameters() if p.requires_grad]


def norm(values):
    return sum(float(v.square().sum()) for v in values)**.5


def make_optimizer(env, multiplier, total_updates):
    optimizer=torch.optim.AdamW([p for p in env.actor.parameters() if p.requires_grad],lr=5e-5*multiplier,
                               betas=(.9,.95),eps=1e-8,weight_decay=.01)
    env.actor._loss.optimizer=optimizer;env.actor._loss.learning_rate=5e-5*multiplier
    scheduler=env.actor._loss._build_scheduler('constant',total_updates,max(1,round(total_updates*.03)))
    return optimizer,scheduler


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--protocol',required=True);p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    p.add_argument('--reference-pool',help='Reuse only a sealed complete initial-base same-kernel TRAIN pool')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--reference-scenes',type=int,default=128)
    p.add_argument('--G',type=int,default=16);p.add_argument('--probe-B',type=int,default=4)
    p.add_argument('--planned-rounds',type=int,default=64,help='Locked post-profile finite training window used for the 3% scheduler calibration')
    args=p.parse_args();protocol=read_protocol(args.protocol);out=Path(args.output)
    if out.exists():raise FileExistsError('Preserve prior calibration; use a new experiment identity')
    if args.reference_scenes!=128 or args.G!=16 or not 1<=args.probe_B<=8 or not 1<=args.planned_rounds<=128:raise ValueError('Locked finite TRAIN reference128/G16 calibration')
    out.mkdir(parents=True)
    archive_path=Path(args.archive).resolve();archive=FrozenArchive.from_state_dict(json.loads(archive_path.read_text()))
    if not archive.scenes or not set(archive.scenes)<=set(protocol['splits']['TRAIN']):raise ValueError('Shared fixed ODE TRAIN bank required')
    env=NativeEnvironment(protocol,trainable='lora',ledger_path=out/'physical_ledger.json',experiment_id='preservation_calibration_only')
    started=time.monotonic();initial=adapter_state(env.actor);initial_hash=tensor_digest(initial)
    if any(v.count_nonzero() for k,v in initial.items() if k.endswith('lora_B')):raise ValueError('Calibration starts from initial common base, not qualification weights')
    env.set_behavior_identity(dict(experiment='preservation_reference_SDE',version=0,policy_sha256=initial_hash))
    tokens=protocol['splits']['TRAIN'][:128]
    generator=torch.Generator(device='cpu').manual_seed(seed_for('preservation_same_kernel_reference_SDE',args.seed,'unfiltered128'))
    try:
        rollouts,groups=[],[]
        if args.reference_pool:
            pool=torch.load(args.reference_pool,map_location='cpu',weights_only=False)
            if pool['protocol_sha256']!=protocol['sha256'] or pool['base_model_sha256']!=protocol['base_model_sha256'] or pool['initial_adapter_hash']!=initial_hash or pool['tokens']!=tokens:
                raise ValueError('Reference pool is not the same initial base/kernel/population')
            rollouts,groups=pool['rollouts'],pool['record_groups']
            if len(groups)!=128 or any(len(x)!=16 for x in groups):raise ValueError('Incomplete frozen reference pool')
            env.ledger.add(cache_reused_candidates=2048,logical_scoring_cost=2048)
        else:
            for i,token in enumerate(tokens):
                condition=env.condition(token)
                rollout,records=env.sample_source_sde(condition,16,generator,token=token,proposal_source='initial_base_reference_source_SDE')
                rollouts.append(rollout);groups.append(records);del condition
                atomic_json(out/'reference_progress.json',dict(completed_scenes=i+1,total_scenes=128,actual_candidates=(i+1)*16,
                            reference_actor_unmodified=True,not_a_completed_budget=True))
        if tensor_digest(adapter_state(env.actor))!=initial_hash:raise ValueError('Reference calibration updated the actor')
        pool=dict(schema='initial_base_source_SDE_reference_pool_v1',status='COMPLETE',protocol_sha256=protocol['sha256'],
                  base_model_sha256=protocol['base_model_sha256'],initial_adapter_hash=initial_hash,tokens=tokens,
                  sampler='same initial source rigorous SDE K10 shift5 stochastic[7,8,9] G16 chunk4',rollouts=rollouts,record_groups=groups)
        atomic_torch(out/'reference_pool.pt',pool)
        costs=torch.stack([official_costs([r['native_subscores'] for r in rows],protocol['safety_metrics']) for rows in groups])
        budgets=costs.mean((0,1)).tolist()
        B=args.probe_B;probe_rollouts=rollouts[:B];probe_groups=groups[:B]
        updater=PreservationUpdate('P5',seed=args.seed,archive=archive,replay_coefficient=1.,budgets=budgets)
        prepared=updater.prepare(probe_rollouts,probe_groups,collection=0)
        env.actor.zero_grad(set_to_none=True)
        for i,rollout in enumerate(probe_rollouts):
            condition=env.condition(rollout['token'])
            for loss,_,weight in env.policy_chunks(rollout,prepared['advantages'][i],prepared['mask'][i],condition):
                (loss*weight/B).backward();env.ledger.add(backward_calls=1)
            del condition
        policy_grad=gradients(env.actor);policy_norm=norm(policy_grad)
        env.actor.zero_grad(set_to_none=True)
        scene_rng=torch.Generator().manual_seed(seed_for('preservation_scale_probe_scenes',args.seed,'TRAIN'))
        target_rng=torch.Generator().manual_seed(seed_for('preservation_scale_probe_targets',args.seed,'TRAIN'))
        fm_rng=torch.Generator(device='cpu').manual_seed(seed_for('preservation_scale_probe_FM',args.seed,'TRAIN'))
        replay_tokens=archive.sample_scenes(B,generator=scene_rng)
        for token in replay_tokens:
            target,_=archive.sample([token],generator=target_rng,device=env.device)
            condition=env.condition(token);loss,_=env.source_fm_loss(condition,target,generator=fm_rng)
            (loss/B).backward();env.ledger.add(backward_calls=1);del condition
        replay_grad=gradients(env.actor);replay_norm=norm(replay_grad)
        scale=calibrate_replay_coefficient(policy_norm,replay_norm)
        if scale['coefficient'] is None:raise ValueError('No finite PG/FM calibration signal in fixed TRAIN probe; no fabricated coefficient')
        cosine=sum(float((a*b).sum()) for a,b in zip(policy_grad,replay_grad))/(policy_norm*replay_norm)
        env.actor.zero_grad(set_to_none=True);del policy_grad,replay_grad
        def deployment_probe():
            values=[]
            for token in tokens[:B]:
                condition=env.condition(token)
                noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for('preservation_P1_drift_probe',14001,token)))
                raw=env.sample_ode(condition,noise);values.append(env.codec.denorm_odo(raw)[0].detach().cpu().numpy());del condition
            return values
        baseline=deployment_probe();total_updates=args.planned_rounds*4
        def shadow(arm,multiplier):
            restore_adapters(env.actor,initial)
            op,sc=make_optimizer(env,multiplier,total_updates)
            update=PreservationUpdate(arm,seed=args.seed,archive=archive if arm=='P5' else None,
                replay_coefficient=scale['coefficient'] if arm=='P5' else 0.,budgets=budgets if arm=='P5' else None)
            fixed=update.prepare(probe_rollouts,probe_groups,collection=0)
            logs=update.update(env,probe_rollouts,fixed,op,scheduler=sc,
                    after_inner=lambda inner,row,custom:env.ledger.add(optimizer_updates=1,update_seconds=row['seconds']))
            return physical_drift(baseline,deployment_probe()),logs
        p5_drift,p5_logs=shadow('P5',1.)
        trials=[]
        for ratio in [.2,.5]:
            drift,logs=shadow('P1',ratio)
            trials.append(dict(split='TRAIN',learning_rate=5e-5*ratio,LR_multiplier=ratio,
                experiment_id=f'P1_shadow_LR{ratio}',drift=drift,actual_optimizer_updates=len(logs)))
        selected=select_conservative_trial(trials,p5_drift)
        selected_ratio=next(t['LR_multiplier'] for t in trials if t['experiment_id']==selected['selected_experiment_id'])
        restore_adapters(env.actor,initial)
        if tensor_digest(adapter_state(env.actor))!=initial_hash:raise AssertionError('Failed to discard shadow-learning weights')
        record=dict(schema='preservation_TRAIN_calibration_v1',status='COMPLETE',qualification=False,
            protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],
            TRAIN_token_hash=fingerprint(protocol['splits']['TRAIN']),shared_ODE_K32_pool_hash=archive.provenance['input_pool_hash'],
            archive_path=str(archive_path),archive_file_sha256=file_sha(archive_path),archive_state_hash=archive.digest(),
            replay_coefficient=scale['coefficient'],replay_gradient_calibration=dict(**scale,policy_norm=policy_norm,replay_norm=replay_norm,cosine=cosine,TRAIN_scenes=B),
            cost_metrics=protocol['safety_metrics'],cost_formula='1-official_metric',cost_budgets=budgets,epsilon_budget=0.,
            reference_SDE_scenes=128,reference_SDE_candidates=2048,reference_SDE_pool_sha256=file_sha(out/'reference_pool.pt'),
            costs_reference_identity='unmodified common-base source SDE, same kernel as P0-P5; not ODE candidate budgets',
            P1_learning_rate_multiplier=selected_ratio,P1_matching=selected,P1_trials=trials,P5_shadow_drift=p5_drift,
            shadow_optimizer_updates=12,shadow_weights_used_for_formal=False,
            planned_optimizer_updates=total_updates,warmup_updates=max(1,round(total_updates*.03)),
            formal_initialization_must_be_common_base_zero_B=True,wall_seconds=time.monotonic()-started,
            reference_sampling_and_archive_construction_costs_separate=True,physical_ledger=env.ledger.snapshot())
        record['sha256']=fingerprint(record);atomic_json(out/'calibration.json',record)
        print(json.dumps({k:record[k] for k in ['status','cost_budgets','replay_coefficient','P1_learning_rate_multiplier','wall_seconds']}))
    finally:
        env.close()


if __name__=='__main__':main()
