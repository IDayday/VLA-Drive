"""Finite P0-P5 native source training with shared runtime and checkpoint writer."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import signal
import time
import traceback
import uuid
import numpy as np
import torch
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol, require_scene, require_dispatch
from starVLA.rl.qwen_flow_exploration.checkpoint import save_state, load_state
from starVLA.rl.qwen_flow_exploration.preservation.archive import FrozenArchive
from starVLA.rl.qwen_flow_exploration.preservation.update import PreservationUpdate
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state, restore_adapters
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch


class NativeStop(Exception):
    pass


def sealed(path):
    value=json.loads(Path(path).read_text());body=dict(value)
    if body.pop('sha256')!=fingerprint(body):raise ValueError('Unsealed registration/calibration')
    return value


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p.add_argument('--arm',required=True,choices=['P0','P1','P2','P3','P4','P5']);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--calibration');p.add_argument('--registration');p.add_argument('--resume')
    p.add_argument('--qualification',action='store_true')
    p.add_argument('--rounds',type=int);p.add_argument('--B',type=int);p.add_argument('--G',type=int)
    args=p.parse_args();protocol=read_protocol(args.protocol);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'failure_evidence').mkdir(parents=True,exist_ok=True)
    owner=(out/'owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inv=uuid.uuid4().hex;start=time.monotonic()
    stopping=[False]
    for signum in (signal.SIGUSR1,signal.SIGTERM):signal.signal(signum,lambda *_:stopping.__setitem__(0,True))
    calibration=sealed(args.calibration) if args.calibration else None
    if args.arm!='P0' and calibration is None:raise ValueError('Nonbaseline arms require a locked TRAIN calibration')
    if calibration is not None:
        if (calibration['protocol_sha256']!=protocol['sha256'] or calibration['base_model_sha256']!=protocol['base_model_sha256'] or
            calibration['TRAIN_token_hash']!=fingerprint(protocol['splits']['TRAIN'])):
            raise ValueError('Calibration does not bind the common TRAIN population/base')
    if args.qualification:
        rounds,B,G=args.rounds or 1,args.B or 2,args.G or 4
        if rounds>2 or B>4 or G not in (4,8):raise ValueError('Finite qualification limits; use formal registration for screening')
        registration=None
    else:
        if any(v is not None for v in [args.rounds,args.B,args.G]):raise ValueError('Formal budget cannot be overridden by CLI')
        if args.registration is None:raise ValueError('Profile first and lock one finite formal registration')
        registration=sealed(args.registration)
        if registration['protocol_sha256']!=protocol['sha256']:raise ValueError('Wrong formal common protocol')
        r=registration.get('preservation',registration)
        if args.arm not in r['arms'] or args.seed not in r['training_seeds']:raise ValueError('Unregistered method/seed')
        rounds,B,G=int(r['rounds']),int(r['B']),int(r['G'])
        if rounds<1 or rounds>128 or B!=16 or G!=16:raise ValueError('Finite matched study budget required')
        if calibration is not None and calibration.get('planned_optimizer_updates')!=rounds*4:
            raise ValueError('P1/scale calibration and formal optimizer budget differ')
    replay=args.arm in ('P2','P3','P5');constrained=args.arm in ('P4','P5')
    archive=None
    if replay:
        archive_path=Path(calibration['archive_path'])
        if file_sha(archive_path)!=calibration['archive_file_sha256']:raise ValueError('Fixed initial archive changed')
        archive=FrozenArchive.from_state_dict(json.loads(archive_path.read_text()))
        if not set(archive.scenes)<=set(protocol['splits']['TRAIN']):raise ValueError('Archive exceeds the registered TRAIN study population')
    multiplier=float(calibration['P1_learning_rate_multiplier']) if args.arm=='P1' else 1.
    if not 0<multiplier<=1:raise ValueError('P1 must be a locked conservative TRAIN control')
    env=NativeEnvironment(protocol,trainable='lora',ledger_path=out/'physical_invocations'/f'{inv}.json',experiment_id=f'{args.arm}_seed{args.seed}')
    updater=PreservationUpdate(args.arm,seed=args.seed,archive=archive,
        replay_coefficient=float(calibration['replay_coefficient']) if replay else 0.,
        budgets=calibration['cost_budgets'] if constrained else None,metrics=protocol['safety_metrics'])
    optimizer=torch.optim.AdamW([x for x in env.actor.parameters() if x.requires_grad],lr=5e-5*multiplier,
                               betas=(.9,.95),eps=1e-8,weight_decay=.01)
    env.actor._loss.optimizer=optimizer;env.actor._loss.learning_rate=5e-5*multiplier
    warmup=max(1,round(rounds*4*.03));scheduler=env.actor._loss._build_scheduler('constant',rounds*4,warmup)
    rollout_rng=torch.Generator(device='cpu').manual_seed(seed_for('preservation_fresh_source_SDE',args.seed,'shared_population'))
    identity=dict(schema='preservation_native_run_v1',arm=args.arm,seed=args.seed,qualification=args.qualification,
        common_protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],
        calibration_sha256=calibration['sha256'] if calibration else None,registration_sha256=registration['sha256'] if registration else None,
        B=B,G=G,rounds=rounds,inner_epochs=4,learning_rate=5e-5*multiplier,warmup_updates=warmup,
        reference='permanent common complete merged mature canonical50k',
        source_ratio='relative-clamped dimension-mean stochastic-transition surrogate, not final-trajectory ratio')
    source_root=Path(__file__).resolve().parents[3]
    paths=[Path(__file__),*(source_root/'starVLA/rl/qwen_flow_exploration/preservation').glob('*.py')]
    paths += [source_root/p for p in ['starVLA/rl/qwen_flow_exploration/native.py','starVLA/rl/qwen_flow_exploration/protocol.py',
        'starVLA/rl/qwen_flow_exploration/checkpoint.py','starVLA/rl/qwen_sourceflow/actor.py',
        'starVLA/rl/simwam_recipe/vendor/trainer_math.py','starVLA/rl/simwam_recipe/vendor/action_math.py']]
    identity['execution_source_files']={str(p.relative_to(source_root)):file_sha(p) for p in sorted(paths)}
    identity['sha256']=fingerprint(identity)
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())!=identity or args.resume is None:raise ValueError('Run identity changed or duplicate launch')
    else:
        if args.resume:raise ValueError('Resume must preserve original run directory')
        atomic_json(out/'identity.json',identity)
    state=dict(phase='READY',round=0,updates=0,next_inner=0,arm=args.arm,seed=args.seed,scene_cursor=0,
               logical_scored_candidates=0,logical_scene_exposures=0)
    batch=None
    def payload():
        return dict(identity_sha256=identity['sha256'],adapters=adapter_state(env.actor),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    updater=updater.state_dict(),rollout_rng=rollout_rng.get_state(),batch=batch)
    if args.resume:
        state,saved=load_state(args.resume,protocol['sha256'])
        if saved['identity_sha256']!=identity['sha256']:raise ValueError('Resume loaded another scientific run')
        restore_adapters(env.actor,saved['adapters']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        updater.load_state_dict(saved['updater']);rollout_rng.set_state(saved['rollout_rng']);batch=saved['batch']
    metrics=out/'metrics';metrics.mkdir(exist_ok=True)
    # Preserve records from rolled-back physical work under their old attempt.
    for path in metrics.glob('update*.json'):
        if int(path.stem.removeprefix('update'))>state['updates']:
            dest=out/'aborted_attempts'/inv/path.name;dest.parent.mkdir(parents=True,exist_ok=True);path.rename(dest)
    try:
        if not args.resume:save_state(out,state,payload(),protocol['sha256'])
        while state['round']<rounds:
            require_dispatch()
            if stopping[0]:raise NativeStop('Stop requested at legal boundary')
            if batch is None:
                params=adapter_state(env.actor)
                env.set_behavior_identity(dict(experiment_id=identity['sha256'],version=state['updates'],policy_sha256=fingerprint({k:hash_tensor(v) for k,v in params.items()})))
                rollouts,record_groups=[],[]
                try:
                    for local in range(B):
                        position=state['scene_cursor']+local;tokens=protocol['splits']['TRAIN'];epoch,offset=divmod(position,len(tokens))
                        order=torch.randperm(len(tokens),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
                        token=tokens[order[offset]];require_scene(protocol,token,'TRAIN',update=True)
                        condition=env.condition(token)
                        rollout,records=env.sample_source_sde(condition,G,rollout_rng,token=token,proposal_source=args.arm)
                        rollouts.append(rollout);record_groups.append(records);del condition
                    prepared=updater.prepare(rollouts,record_groups,collection=state['round'])
                    batch=dict(rollouts=rollouts,records=record_groups,prepared=prepared)
                    state.update(phase='ROLLOUT_COMMITTED',next_inner=0,scene_cursor=state['scene_cursor']+B,
                        logical_scored_candidates=state['logical_scored_candidates']+B*G,logical_scene_exposures=state['logical_scene_exposures']+B)
                    save_state(out,state,payload(),protocol['sha256'])
                except BaseException:
                    atomic_torch(out/'failure_evidence'/f'collection_{inv}.pt',dict(state=state,uncommitted_rollouts=rollouts,
                        uncommitted_records=record_groups,error=traceback.format_exc(),resume_boundary='last legal latest; no failure publication'))
                    raise
            if stopping[0]:raise NativeStop('Stop requested after committed collection')
            def committed(inner,row,custom):
                state.update(updates=state['updates']+1,next_inner=inner,phase='INNER_COMMITTED')
                env.ledger.add(optimizer_updates=1,update_seconds=row['seconds'])
                atomic_json(metrics/f"update{state['updates']:06d}.json",dict(invocation=inv,identity=identity['sha256'],**row))
                save_state(out,state,payload(),protocol['sha256'])
                if stopping[0]:raise NativeStop('Stop requested after committed optimizer inner')
            if state['next_inner']<4:
                updater.update(env,batch['rollouts'],batch['prepared'],optimizer,scheduler=scheduler,
                               start_inner=state['next_inner'],after_inner=committed)
            batch=None;state.update(round=state['round']+1,next_inner=0,phase='READY')
            save_state(out,state,payload(),protocol['sha256']);env.ledger.save()
            atomic_json(out/'progress.json',dict(**state,invocation=inv,seconds_this_invocation=time.monotonic()-start))
        state.update(phase='COMPLETE');path=save_state(out,state,payload(),protocol['sha256'])
        result=dict(status='COMPLETE',qualification=args.qualification,identity=identity['sha256'],checkpoint=str(path),
                    updates=state['updates'],fresh_collections=state['round'],scene_exposures=state['logical_scene_exposures'],
                    new_candidates=state['logical_scored_candidates'],seconds_this_invocation=time.monotonic()-start,
                    preservation_custom=updater.state_dict(),physical_invocation_ledger=env.ledger.snapshot())
        # Custom RNG tensors live in native checkpoint; public result is JSON.
        for key in ['replay_scene_rng','replay_target_rng','replay_fm_rng']:result['preservation_custom'].pop(key)
        atomic_json(out/'COMPLETE.json',result);print(json.dumps({k:result[k] for k in ['status','updates','new_candidates','checkpoint']}))
    except NativeStop:
        atomic_json(out/'PAUSED.json',dict(status='PAUSED',state=state,invocation=inv,resume_from_latest_committed=True))
    except BaseException:
        atomic_json(out/'failure_evidence'/f'failure_{inv}.json',dict(error=traceback.format_exc(),state=state,
            failure_handler_does_not_publish_latest=True,resume_from_latest_committed=True))
        raise
    finally:
        env.close()


def hash_tensor(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


if __name__=='__main__':main()
