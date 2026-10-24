"""Finite real native P5 rehearsal/dual/inner-resume qualification, not efficacy."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import torch
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_flow_exploration.checkpoint import save_state, load_state
from starVLA.rl.qwen_flow_exploration.preservation.archive import FrozenArchive
from starVLA.rl.qwen_flow_exploration.preservation.objective import official_costs
from starVLA.rl.qwen_flow_exploration.preservation.update import PreservationUpdate
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state, restore_adapters
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.flow_grpo.transactions import atomic_json


def frozen_hash(environment):
    result={}
    for name,module in [('model',environment.model),('actor',environment.actor)]:
        digest=hashlib.sha256()
        for key,value in list(module.named_parameters())+list(module.named_buffers()):
            if not value.requires_grad:
                digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        result[name]=digest.hexdigest()
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--B',type=int,default=2);parser.add_argument('--G',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();start=time.monotonic();out=Path(args.output)
    if out.exists():raise FileExistsError('Qualification attempts must have independent output identity')
    out.mkdir(parents=True);protocol=read_protocol(args.protocol)
    if not 1<=args.B<=4 or args.G not in (4,8):raise ValueError('Finite profile only; not a formal screen')
    torch.cuda.reset_peak_memory_stats()
    environment=NativeEnvironment(protocol,trainable='lora',ledger_path=out/'physical_ledger.json',experiment_id='P5_qualification_only')
    environment.set_behavior_identity(dict(experiment='P5_qualification',version=0,policy='fixed_common_base'))
    generator=torch.Generator(device='cpu').manual_seed(seed_for('preservation_profile_SDE',args.seed,'CPU_root'))
    try:
        before=frozen_hash(environment)
        rollouts,groups,pool=[],[],[]
        tokens=protocol['smoke']['TRAIN']
        for token in tokens[:args.B]:
            condition=environment.condition(token)
            rollout,records=environment.sample_source_sde(condition,args.G,generator,token=token,proposal_source='common_base_SDE_qualification')
            rollouts.append(rollout);groups.append(records);pool.extend(records);del condition
        def make_archive():
            logs={row['scene_token']:row['log_id'] for row in pool}
            return FrozenArchive.build(pool,train_tokens=set(protocol['splits']['TRAIN']),token_logs=logs)
        archive=make_archive()
        # Qualification may find a replay-supporting TRAIN probe, but never
        # removes a failed/tied group from the actual B-scene policy batch.
        extra=0
        while not archive.scenes and extra<min(8,len(tokens)-args.B):
            token=tokens[args.B+extra];condition=environment.condition(token)
            _,records=environment.sample_source_sde(condition,args.G,generator,token=token,proposal_source='common_base_SDE_qualification')
            pool.extend(records);extra+=1;archive=make_archive();del condition
        if not archive.scenes:raise ValueError('No qualified fixed replay target found within finite TRAIN profile budget')
        atomic_json(out/'initial_archive.json',archive.state_dict())
        metrics=protocol['safety_metrics']
        budgets=torch.stack([official_costs([r['native_subscores'] for r in group],metrics) for group in groups]).mean((0,1)).tolist()
        update=PreservationUpdate('P5',seed=args.seed,archive=archive,replay_coefficient=.001,budgets=budgets,metrics=metrics)
        prepared=update.prepare(rollouts,groups,collection=0)
        optimizer=torch.optim.AdamW([p for p in environment.actor.parameters() if p.requires_grad],lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
        environment.actor._loss.optimizer=optimizer;environment.actor._loss.learning_rate=5e-5
        scheduler=environment.actor._loss._build_scheduler('constant',4,1)
        saved={}
        def checkpoint(inner,row,custom):
            environment.ledger.add(optimizer_updates=1,update_seconds=row['seconds'])
            if inner==1:
                state=dict(phase='INNER_COMMITTED',round=0,updates=1,next_inner=1,arm='P5_qualification_only',seed=args.seed)
                payload=dict(adapters=adapter_state(environment.actor),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    updater=custom,rollout_rng=generator.get_state(),batch=dict(rollouts=rollouts,records=groups,prepared=prepared))
                saved['path']=save_state(out,state,payload,protocol['sha256'])
        continuous_logs=update.update(environment,rollouts,prepared,optimizer,scheduler=scheduler,after_inner=checkpoint)
        continuous=dict(adapters=adapter_state(environment.actor),optimizer=copy.deepcopy(optimizer.state_dict()),scheduler=copy.deepcopy(scheduler.state_dict()),updater=copy.deepcopy(update.state_dict()))
        condition=environment.condition(tokens[0])
        future_a,_=environment.sample_source_sde(condition,args.G,generator,token=tokens[0],proposal_source='qualification_after_update')
        state,payload=load_state(saved['path'],protocol['sha256'])
        restore_adapters(environment.actor,payload['adapters']);optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
        update.load_state_dict(payload['updater']);generator.set_state(payload['rollout_rng'])
        resumed_logs=update.update(environment,payload['batch']['rollouts'],payload['batch']['prepared'],optimizer,
            scheduler=scheduler,start_inner=state['next_inner'],after_inner=lambda inner,row,custom:environment.ledger.add(optimizer_updates=1,update_seconds=row['seconds']))
        resumed=dict(adapters=adapter_state(environment.actor),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),updater=update.state_dict())
        if tensor_digest(continuous)!=tensor_digest(resumed):raise AssertionError('Native inner resume changed adapters/Adam/scheduler/dual/custom RNG')
        future_b,_=environment.sample_source_sde(condition,args.G,generator,token=tokens[0],proposal_source='qualification_after_update')
        for key in ('chain','old_logp','rewards'):
            if not torch.equal(future_a[key],future_b[key]):raise AssertionError('Native resume changed subsequent '+key)
        after=frozen_hash(environment)
        if before!=after:raise AssertionError('Frozen base/reference/Qwen/provider parameters or buffers changed')
        names={name:param for name,param in environment.actor.named_parameters() if param.requires_grad}
        moments={name:bool(optimizer.state[param]['exp_avg'].abs().sum()>0) for name,param in names.items()}
        if not all(moments.values()):raise AssertionError('Every expected native LoRA must acquire a real Adam gradient moment')
        dev_token=protocol['smoke']['DEV_SELECT'][0]
        dev_condition=environment.condition(dev_token,'DEV_SELECT')
        root=torch.randn((1,8,3),generator=torch.Generator().manual_seed(2101))
        reference_calls=environment.actor._reference_calls
        trajectory=environment.sample_ode(dev_condition,root)
        evaluation=environment.score_actions(dev_token,trajectory,'DEV_SELECT',proposal_source='qualification_single_ODE')
        if environment.actor._reference_calls!=reference_calls:raise AssertionError('ODE deployment consulted the reference')
        report=dict(status='PASS',qualification_only=True,formal_initialization_eligible=False,
            common_protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],
            B=args.B,G=args.G,logical_optimizer_updates=4,physical_optimizer_updates=7,
            archive_profile_only=True,archive_covered_scenes=len(archive.scenes),extra_replay_probe_scenes=extra,
            budgets_profile_only_not_formal=budgets,all_native_LoRA_Adam_moments_nonzero=len(moments),
            frozen_parameters_and_buffers_before=before,frozen_parameters_and_buffers_after=after,
            mid_inner_exact_resume=True,subsequent_chain_old_probability_reward_exact=True,
            actual_source_inner_steps=4,initial_chain_kept_fixed=True,
            current_source_single_ODE_works_without_archive_reference_or_teacher=True,
            one_DEV_qualification_PDMS=evaluation[0]['raw_reward'],not_deployment_efficacy_evidence=True,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.monotonic()-start,
            continuous_logs=continuous_logs,resumed_logs=resumed_logs,physical_ledger=environment.ledger.snapshot())
        atomic_json(out/'qualification.json',report)
        print(json.dumps({k:report[k] for k in ['status','wall_seconds','peak_GPU_allocated_bytes','all_native_LoRA_Adam_moments_nonzero','mid_inner_exact_resume']}))
    finally:
        environment.close()


if __name__=='__main__':main()
