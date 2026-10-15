"""Finite TRAIN-only SFT/RL sampler probe; no optimization or model selection.

Three matched G8 output means, never candidate-max deployment scores. Sampling
and scoring call the existing implementations. FP32 roots deliberately match
the training sampler, not the historical BF16-root single-candidate evaluation.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid
from types import SimpleNamespace

import numpy as np
import torch

from starVLA.rl.simwam_recipe.contracts import (ROOT, POPULATION, RAW_CONDITION,
    SFT_SHA256, CONDITION_SHA, fingerprint, file_sha, seed_for, load_recipe, require_dispatch)
from starVLA.rl.flow_grpo.transactions import atomic_json

MODES = ('NATIVE_ODE10', 'SHIFT5_ODE10', 'SHIFT5_LAST3_SDE10')
MODELS = ('SFT', 'R_CANONICAL_20000')
REPO = Path(__file__).resolve().parents[3]
CHECKPOINT = Path(ROOT)/'runs/R_CANONICAL_seed42_v1/checkpoints/update020000_inner0'
DIAGNOSTIC = Path('/mnt/project/flow-safe-improvement-20261013/evaluations/original_export/SFT_TRAIN_D5_samples/manifest.json')


def grid_audit():
    from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
    from starVLA.rl.simwam_recipe.vendor.action_math import ActionGRPOMixin
    from infer import _X_STD, _Y_STD
    result = {}
    for shift in (1, 5):
        ts, ds = WanContinuousFlowMatchScheduler(1000, shift=shift).build_inference_schedule(10, 'cpu', torch.float32)
        obj = SimpleNamespace(grpo_sigma_max_guard=float(ts[1])/1000,
                              grpo_noise_level=.1, grpo_min_std=.001)
        rows = []
        for k, (t, delta) in enumerate(zip(ts, ds)):
            sigma, delta = float(t)/1000, float(delta)
            _, std = ActionGRPOMixin._sde_mean_std(obj, torch.zeros(1), torch.zeros(1), sigma, delta)
            sig = obj.grpo_sigma_max_guard if sigma >= 1 else sigma
            g2 = .01*sig/(1-sig)
            coeff = -delta*(1+g2*(1-sig)/(2*sig))
            rows.append(dict(k=k, sigma=sigma, ddp_time=1-sigma,
                bucket=int(((1-t/1000)*1000).long()), dt=-delta,
                hypothetical_rigorous_std=std, stochastic_in_formal=(shift==5 and k in (7,8,9)),
                one_step_x_std_m=std*_X_STD, one_step_y_std_m=std*_Y_STD,
                mean_velocity_coefficient=coeff, anchor_velocity_coefficient_squared=coeff**2,
                standardized_mean_velocity_sensitivity=coeff/std))
        result[str(shift)] = rows
    # Actual frozen SFT beta parameters; not an alternative sampler.
    result['SFT_time_formula'] = dict(alpha=1.5, beta=1., noise_s=.999,
        formula='t=(0.999-Beta(1.5,1))/0.999', negative_time_probability=1-.999**1.5,
        interpretation='Inherited continuous-time SFT convention, not evidence of an RL implementation bug')
    result['units'] = dict(x_std_m=float(_X_STD), y_std_m=float(_Y_STD),
        heading='unscaled sin/cos pair; atan2 discards positive radial scaling',
        innovation='Each trajectory waypoint/channel independent; std includes sqrt(dt)',
        caveat='One-step physical std is not the full generated trajectory distribution')
    return result


def register(root):
    from starVLA.rl.safe_flow_improvement.v1_reward import protocol_identity
    root.mkdir(parents=True, exist_ok=True)
    pop = json.loads(Path(POPULATION).read_text())
    base = json.loads(DIAGNOSTIC.read_text())['explicit_tokens']
    if len(base)!=512 or not set(base)<=set(pop['train_tokens']):
        raise ValueError('Existing TRAIN-only diagnostic population changed')
    tokens = sorted(base, key=lambda token: fingerprint(['sampler_transfer_train64_v1',42,token]))[:64]
    if set(tokens)&(set(pop['dev_tokens'])|set(pop['navtest_tokens'])):
        raise ValueError('Evaluation population cannot enter this probe')
    for token in tokens:
        if not (Path(RAW_CONDITION)/token[:2]/(token+'.json')).exists():
            raise FileNotFoundError('Missing canonical condition for fixed token '+token)
    complete = json.loads((CHECKPOINT/'COMPLETE').read_text())
    if file_sha(CHECKPOINT/'files.json')!=complete['files_sha256']:
        raise ValueError('Unsealed checkpoint')
    files = json.loads((CHECKPOINT/'files.json').read_text())
    value = dict(schema='simwam_sampler_transfer_train64_v1', tokens=tokens,
        token_logs={t:pop['token_logs'][t] for t in tokens}, population_identity=pop['identity_sha256'],
        parent_diagnostic_manifest_sha256=file_sha(DIAGNOSTIC),
        population='64 preselected by hash from registered TRAIN512; diagnostic, not population-unbiased',
        models=list(MODELS), modes=list(MODES), checkpoint=str(CHECKPOINT),
        checkpoint_files_sha256=complete['files_sha256'], adapters_file_sha256=files['optimizer.pt'],
        SFT_sha256=SFT_SHA256, condition_identity=CONDITION_SHA,
        G=8, seed=2101, root_dtype='float32', root_rng_backend='source CPU generator then transfer',
        step_rng='source CPU generator, independent from initial root; shared across models',
        candidate_budget=64*8*3*2, optimizer_updates=0, selection='all candidates; score every requested trajectory',
        no_DEV_or_Navtest=True, source_script_sha256=file_sha(__file__),
        scoring_identity=protocol_identity(REPO)['sha256'], grid=grid_audit())
    value['sha256'] = fingerprint(value)
    path = root/'registration.json'
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('Existing registration is immutable')
    atomic_json(path,value)
    return value


def load_registration(root):
    value=json.loads((root/'registration.json').read_text())
    copy=dict(value);sha=copy.pop('sha256')
    if fingerprint(copy)!=sha or value['source_script_sha256']!=file_sha(__file__):
        raise ValueError('Probe source or registration changed')
    return value


def geometry(actions, requested):
    radius=np.linalg.norm(actions[...,2:4],axis=-1)
    xy=requested[...,:2]
    v=np.diff(xy,axis=1)/.5
    a=np.diff(v,axis=1)/.5
    jerk=np.diff(a,axis=1)/.5
    pair=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1).mean(-1)
    return dict(heading_radius_mean=float(radius.mean()), heading_radius_p05=float(np.quantile(radius,.05)),
        heading_radius_p95=float(np.quantile(radius,.95)), heading_radius_below_025=float((radius<.25).mean()),
        mean_abs_heading_radius_error=float(np.abs(radius-1).mean()),
        request_forward_end_m=float(xy[:,-1,0].mean()),
        requested_mean_speed_m_s=float(np.linalg.norm(v,axis=-1).mean()),
        requested_jerk_RMS_m_s3=float(np.sqrt(np.mean(jerk**2))),
        pairwise_ADE_m=float(pair[np.triu_indices(len(xy),1)].mean()),
        caveat='Request finite differences at 0.5s, not official simulated comfort or safety')


def worker(args):
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    from starVLA.rl.simwam_recipe.trainer_adapter import make_actor
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
    from starVLA.rl.simwam_recipe.condition import ProjectedConditions
    from starVLA.rl.simwam_recipe.reward_readouts import RewardService
    from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
    from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request
    host=require_dispatch();configure_numerics();torch.cuda.set_device(0)
    root=Path(args.root);reg=load_registration(root)
    if not 0<=args.shard<args.shards:raise ValueError('Invalid shard')
    output=root/args.model;output.mkdir(exist_ok=True)
    lock=(output/f'rank{args.shard}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    invocation=uuid.uuid4().hex;beg=time.time()
    status=dict(schema='sampler_probe_worker_v1',invocation=invocation,host=host,pid=os.getpid(),
        model=args.model,rank=args.shard,shards=args.shards,status='STARTING',registration_sha256=reg['sha256'],
        start_unix=beg,new_scenes=0,reused_scenes=0,physical_scored_candidates=0,
        generation_seconds=0.,scoring_wait_seconds=0.,scoring_cpu_seconds=0.,velocity_forward_calls=0,
        GPU_ownership='shared resources; worker wall is not exclusive GPU compute time')
    invoke=output/'invocations'/(invocation+'.json')
    invoke.parent.mkdir(parents=True,exist_ok=True)
    status['source_module']=str(Path(__file__).resolve())
    atomic_json(invoke,status)
    service=None
    try:
        cfg=load_recipe(REPO/'configs/flow_rl/simwam_canonical_ddp.yaml')
        actor,_,sft=make_actor(cfg)
        if args.model=='R_CANONICAL_20000':
            ck=Path(reg['checkpoint'])
            if file_sha(ck/'optimizer.pt')!=reg['adapters_file_sha256']:raise ValueError('Changed adapter state')
            saved=torch.load(ck/'optimizer.pt',map_location='cpu',weights_only=False)
            restore_adapters(actor,saved['adapters']);del saved
        actor.eval().requires_grad_(False)
        # Fixed SFT projection only. Any new projection is in this probe's own cache.
        conditions=ProjectedConditions(root/'projected_condition',actor.action_expert)
        service=RewardService(REPO,workers=1)
        if service.protocol['sha256']!=reg['scoring_identity']:raise ValueError('Scoring protocol changed')
        status['model_load_seconds']=time.time()-beg;status['scorer_identity']=service.protocol['sha256']
        status['status']='RUNNING';atomic_json(invoke,status)
        for token in reg['tokens'][args.shard::args.shards]:
            require_dispatch()
            path=output/'scenes'/(token+'.pt');seal=path.with_suffix('.json')
            if seal.exists():
                saved=json.loads(seal.read_text())
                if saved['registration_sha256']!=reg['sha256'] or saved['payload_sha256']!=file_sha(path):
                    raise ValueError('Changed existing probe scene')
                status['reused_scenes']+=1;continue
            start=time.monotonic()
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                condition=conditions.get(token)
                cond={'context':condition.expand(8,-1,-1).contiguous()}
                root_key=seed_for('sampler_transfer_root_fp32',2101,token)
                step_key=seed_for('sampler_transfer_innovations',2101,token)
                roots=actor._randn((8,8,4),actor.device,torch.float32,
                    torch.Generator(device='cpu').manual_seed(root_key))
                native=actor.action_expert._euler_sample(roots.clone(),cond['context'])
                ode,times,deltas,indices=actor.sample_action_chain(cond,8,deterministic=True,init_actions=roots.clone())
                if len(indices):raise ValueError('ODE path became stochastic')
                chain,ts,ds,indices=actor.sample_action_chain(cond,8,init_actions=roots.clone(),
                    generator=torch.Generator(device='cpu').manual_seed(step_key))
                if indices.tolist()!=[7,8,9] or not torch.equal(ts,times) or not torch.equal(ds,deltas):
                    raise ValueError('Changed source sampler')
                logp,terms=actor.action_chain_logprobs(cond,chain,ts,ds,stoch_idx=indices,return_terms=True)
                outputs=torch.stack([native,ode[:,-1],chain[:,-1]]).float().cpu()
                if not torch.isfinite(outputs).all():raise ValueError('Nonfinite sample: abort without dropping candidate')
                act_norm=int(sft.datasets.vla_data.act_norm)
                requested=decode_request(outputs.flatten(0,1).numpy(),act_norm=act_norm).reshape(3,8,8,3)
                endpoint_mean=terms['mean'][:,-1].cpu()
                mean_request=decode_request(endpoint_mean.numpy(),act_norm=act_norm)
                endpoint_noise=outputs[-1]-endpoint_mean
            torch.cuda.synchronize();status['generation_seconds']+=time.monotonic()-start
            status['velocity_forward_calls']+=33
            start=time.monotonic();records=service.score((token,reg['token_logs'][token],requested.reshape(24,8,3)))
            status['scoring_wait_seconds']+=time.monotonic()-start
            status['scoring_cpu_seconds']+=sum(r['scoring_seconds'] for r in records)
            status['physical_scored_candidates']+=len(records)
            grouped={mode:records[i*8:(i+1)*8] for i,mode in enumerate(MODES)}
            stats={mode:geometry(outputs[i].numpy(),requested[i]) for i,mode in enumerate(MODES)}
            angle=np.arctan2(np.sin(requested[-1,...,2]-mean_request[...,2]),np.cos(requested[-1,...,2]-mean_request[...,2]))
            final_noise=dict(normalized_RMS_by_channel=endpoint_noise.square().mean((0,1)).sqrt().tolist(),
                requested_XY_RMS_m=np.sqrt(np.mean((requested[-1,...,:2]-mean_request[...,:2])**2,axis=(0,1))).tolist(),
                requested_heading_RMS_rad=float(np.sqrt(np.mean(angle**2))),
                mean_request_is_unscored_diagnostic=True)
            payload=dict(token=token,log=reg['token_logs'][token],registration_sha256=reg['sha256'],
                model=args.model,root_key=root_key,step_key=step_key,roots=roots.cpu(),
                outputs=outputs,shift5_ODE_chain=ode.cpu(),SDE_chain=chain.cpu(),times=ts.cpu(),deltas=ds.cpu(),
                stochastic_indices=indices.cpu(),SDE_logprob_proxy=logp.cpu(),SDE_terms=terms,
                requested=requested,records=grouped,geometry=stats,final_noise=final_noise)
            atomic_torch(path,payload)
            atomic_json(seal,dict(token=token,log=reg['token_logs'][token],registration_sha256=reg['sha256'],
                model=args.model,payload_sha256=file_sha(path),scorer_identity=service.protocol['sha256'],
                metrics={m:[r['metrics'] for r in rows] for m,rows in grouped.items()},geometry=stats,final_noise=final_noise))
            status['new_scenes']+=1;atomic_json(invoke,status)
        status['status']='ASSIGNED_COMPLETE'
    except BaseException as exc:
        status.update(status='FAILED',error=repr(exc));raise
    finally:
        if service:service.close()
        status.update(end_unix=time.time(),worker_wall_seconds=time.time()-beg,
                      peak_GPU_bytes=torch.cuda.max_memory_allocated())
        atomic_json(invoke,status);atomic_json(output/f'rank{args.shard}.json',status)


def summarize(root):
    reg=load_registration(root);all_rows={};means={};keys=['score','no_at_fault_collisions',
        'drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort']
    for model in MODELS:
        rows=[]
        for token in reg['tokens']:
            path=root/model/'scenes'/(token+'.pt');row=json.loads(path.with_suffix('.json').read_text())
            if (row['registration_sha256']!=reg['sha256'] or row['payload_sha256']!=file_sha(path)
                    or row['scorer_identity']!=reg['scoring_identity']):
                raise ValueError('Changed or incomplete scene')
            rows.append(row)
        all_rows[model]=rows;means[model]={}
        for mode in MODES:
            array={k:np.array([[m[k] for m in r['metrics'][mode]] for r in rows]) for k in keys}
            safe=np.logical_and.reduce([array[k]==1 for k in keys[1:4]])
            means[model][mode]=dict(metrics_0_100={k:float(v.mean()*100) for k,v in array.items()},
                joint_safe_fraction=float(safe.mean()),all_zero_groups=int((array['score']==0).all(1).sum()),
                tied_groups=int((np.ptp(array['score'],axis=1)==0).sum()),
                scene_count=len(rows),candidates_per_scene=8,
                geometry={k:float(np.mean([r['geometry'][mode][k] for r in rows]))
                          for k in rows[0]['geometry'][mode] if k!='caveat'})
    paired={};rng=np.random.default_rng(42);logs=sorted(set(reg['token_logs'].values()))
    cluster=[np.array([i for i,t in enumerate(reg['tokens']) if reg['token_logs'][t]==log]) for log in logs]
    for mode in MODES:
        changes=[]
        for sft,rl in zip(all_rows['SFT'],all_rows['R_CANONICAL_20000']):
            changes.append(np.mean([m['score'] for m in rl['metrics'][mode]])-
                           np.mean([m['score'] for m in sft['metrics'][mode]]))
        changes=np.asarray(changes);boot=[]
        for _ in range(2000):
            ids=np.concatenate([cluster[i] for i in rng.integers(len(cluster),size=len(cluster))])
            boot.append(changes[ids].mean()*100)
        paired[mode]=dict(delta_PDMS_points=float(changes.mean()*100),
            log_cluster_95_percent_interval=np.quantile(boot,[.025,.975]).tolist(),
            G_plus_points=float(np.maximum(changes,0).mean()*100),
            G_minus_points=float(np.maximum(-changes,0).mean()*100))
        safety={}
        for key in keys[1:4]:
            old=np.array([[v[key] for v in r['metrics'][mode]] for r in all_rows['SFT']])
            new=np.array([[v[key] for v in r['metrics'][mode]] for r in all_rows['R_CANONICAL_20000']])
            safety[key]=dict(new_failures=int(((old==1)&(new<1)).sum()),
                recoveries=int(((old<1)&(new==1)).sum()),denominator=64*8)
        paired[mode]['matched_candidate_safety_transitions']=safety
    costs=[]
    for model in MODELS:
        for p in (root/model/'invocations').glob('*.json'):costs.append(json.loads(p.read_text()))
    result=dict(status='COMPLETE',registration_sha256=reg['sha256'],scenes=64,logs=len(logs),G=8,
        population=reg['population'],interpretation='Equal G random-output means, not deployed best-of-N; no independent training repeats',
        root_dtype='FP32 training probe; not the BF16-root formal seed2101 evaluation',
        means=means,paired_RL_minus_SFT=paired,
        SDE_minus_shift5_ODE_points={m:means[m][MODES[2]]['metrics_0_100']['score']-means[m][MODES[1]]['metrics_0_100']['score'] for m in MODELS},
        final_noise={m:{k:np.mean([r['final_noise'][k] for r in all_rows[m]],axis=0).tolist()
                       for k in ('normalized_RMS_by_channel','requested_XY_RMS_m','requested_heading_RMS_rad')} for m in MODELS},
        costs=dict(logical_candidates=reg['candidate_budget'],physical_scored_candidates=sum(c['physical_scored_candidates'] for c in costs),
            cumulative_worker_wall_seconds=sum(c['worker_wall_seconds'] for c in costs),
            generation_worker_seconds=sum(c['generation_seconds'] for c in costs),
            scoring_wait_worker_seconds=sum(c['scoring_wait_seconds'] for c in costs),
            elapsed_job_wall_seconds=max(c['end_unix'] for c in costs)-min(c['start_unix'] for c in costs),
            optimizer_updates=0,exclusive_GPU_hours=None),limitations=['Small fixed TRAIN subset; no population-unbiased claim',
            'Cluster interval excludes training-seed uncertainty and does not promote G8 to eight independent replications',
            'Canonical-only old RL control; X1 still training; unchanged reward and sampler'])
    atomic_json(root/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['register','worker','summarize'])
    p.add_argument('--root',required=True);p.add_argument('--model',choices=MODELS)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=8)
    a=p.parse_args();root=Path(a.root)
    if a.command=='register':print(json.dumps(register(root)))
    elif a.command=='worker':
        if a.model is None:raise ValueError('Worker model required')
        worker(a)
    else:print(json.dumps(summarize(root)))


if __name__=='__main__':main()
