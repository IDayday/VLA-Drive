"""Score saved last-transition means: no new rollout, gradient, or policy.

Removing only the last innovation holds the preceding SDE chain and its
score-corrected mean fixed. These counterfactuals are never on-policy buffers
or deployable single-ODE results.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
import torch

from starVLA.rl.simwam_recipe.contracts import fingerprint,file_sha,require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.simwam_recipe.probe_sampler_transfer import MODELS,load_registration,geometry


def register(root,parent):
    parent=parent.resolve();reg=load_registration(parent)
    summary=json.loads((parent/'summary.json').read_text())
    if summary['status']!='COMPLETE':raise ValueError('Parent must be complete')
    value=dict(schema='saved_last_innovation_counterfactual_v1',parent=str(parent),parent_identity=reg['sha256'],
        parent_summary_sha256=file_sha(parent/'summary.json'),scoring_identity=reg['scoring_identity'],
        source_script_sha256=file_sha(__file__),counterfactual='Use saved last SDE transition mean, remove only final innovation',
        population='All previously fixed TRAIN64, both SFT and R_CANONICAL20k, all G8; no outcome selection',
        additional_candidate_scores=1024,new_actor_rollouts=0,new_velocity_forwards=0,optimizer_updates=0,
        act_norm=1,not_an_on_policy_buffer=True,not_a_deployment_result=True,
        prior_result_observed=True,limits='Does not isolate earlier SDE noise, mean correction, grid or retraining effect')
    value['sha256']=fingerprint(value);root.mkdir(exist_ok=True,parents=True)
    p=root/'registration.json'
    if p.exists() and json.loads(p.read_text())!=value:raise ValueError('Immutable registration changed')
    atomic_json(p,value);return value


def load(root):
    reg=json.loads((root/'registration.json').read_text());v=dict(reg);sha=v.pop('sha256')
    if fingerprint(v)!=sha or reg['source_script_sha256']!=file_sha(__file__):raise ValueError('Counterfactual identity changed')
    parent=load_registration(Path(reg['parent']))
    if parent['sha256']!=reg['parent_identity']:raise ValueError('Parent changed')
    return reg,parent


def worker(a):
    from starVLA.rl.simwam_recipe.reward_readouts import RewardService
    from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
    from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request
    host=require_dispatch();root=Path(a.root);reg,parent=load(root)
    out=root/a.model;out.mkdir(exist_ok=True)
    guard=(out/f'rank{a.shard}.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    invocation=uuid.uuid4().hex;path=out/'invocations'/(invocation+'.json');path.parent.mkdir(exist_ok=True)
    start=time.time();status=dict(status='RUNNING',invocation=invocation,pid=os.getpid(),host=host,
        rank=a.shard,start_unix=start,new_scenes=0,reused=0,actual_scores=0,scoring_wait_seconds=0.,
        actor_forwards=0,GPU_compute=False,registration=reg['sha256'])
    atomic_json(path,status);service=RewardService(Path(__file__).resolve().parents[3],workers=1)
    try:
        if service.protocol['sha256']!=reg['scoring_identity']:raise ValueError('Changed scorer')
        for token in parent['tokens'][a.shard::8]:
            require_dispatch();dest=out/'scenes'/(token+'.pt');seal=dest.with_suffix('.json')
            if seal.exists():
                old=json.loads(seal.read_text())
                if old['registration']!=reg['sha256'] or old['payload_sha256']!=file_sha(dest):raise ValueError('Changed result')
                status['reused']+=1;continue
            original=Path(reg['parent'])/a.model/'scenes'/(token+'.pt')
            parent_seal=json.loads(original.with_suffix('.json').read_text())
            if parent_seal['payload_sha256']!=file_sha(original):raise ValueError('Parent chain modified')
            row=torch.load(original,map_location='cpu',weights_only=False)
            target=row['SDE_terms']['mean'][:,-1].float()
            if target.shape!=(8,8,4) or not torch.isfinite(target).all():raise ValueError('Invalid saved transition mean')
            requested=decode_request(target.numpy(),act_norm=reg['act_norm'])
            begin=time.monotonic();records=service.score((token,parent['token_logs'][token],requested))
            status['scoring_wait_seconds']+=time.monotonic()-begin;status['actual_scores']+=8
            stats=geometry(target.numpy(),requested)
            atomic_torch(dest,dict(token=token,log=parent['token_logs'][token],parent_payload_sha256=parent_seal['payload_sha256'],
                registration=reg['sha256'],normalized_actions=target,requested=requested,records=records,
                source='saved_final_SDE_mean_without_innovation',on_policy=False,geometry=stats))
            atomic_json(seal,dict(token=token,log=parent['token_logs'][token],registration=reg['sha256'],
                payload_sha256=file_sha(dest),metrics=[r['metrics'] for r in records],geometry=stats))
            status['new_scenes']+=1;atomic_json(path,status)
        status['status']='ASSIGNED_COMPLETE'
    except BaseException as exc:
        status.update(status='FAILED',error=repr(exc));raise
    finally:
        service.close();status.update(end_unix=time.time(),worker_wall_seconds=time.time()-start)
        atomic_json(path,status);atomic_json(out/f'rank{a.shard}.json',status)


def summarize(root):
    reg,parent=load(root);out={};cost=[]
    keys=['score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort']
    logs=sorted(set(parent['token_logs'].values()))
    clusters=[np.array([i for i,t in enumerate(parent['tokens']) if parent['token_logs'][t]==log]) for log in logs]
    for model in MODELS:
        rows=[];original=[]
        for token in parent['tokens']:
            path=root/model/'scenes'/(token+'.pt');r=json.loads(path.with_suffix('.json').read_text())
            if r['registration']!=reg['sha256'] or r['payload_sha256']!=file_sha(path):raise ValueError('Incomplete or changed record')
            rows.append(r)
            original.append(json.loads((Path(reg['parent'])/model/'scenes'/(token+'.json')).read_text()))
        new={k:np.array([[m[k] for m in r['metrics']] for r in rows]) for k in keys}
        old={k:np.array([[m[k] for m in r['metrics']['SHIFT5_LAST3_SDE10']] for r in original]) for k in keys}
        delta=(new['score']-old['score']).mean(1)*100;rng=np.random.default_rng(42);boot=[]
        for _ in range(2000):
            ix=np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))]);boot.append(delta[ix].mean())
        out[model]=dict(counterfactual_metrics_0_100={k:float(v.mean()*100) for k,v in new.items()},
            joint_safe_fraction=float(np.logical_and.reduce([new[k]==1 for k in keys[1:4]]).mean()),
            delta_PDMS_vs_actual_SDE_points=float(delta.mean()),log_cluster_95_percent_interval=np.quantile(boot,[.025,.975]).tolist(),
            safety_transitions={k:dict(new_failures=int(((old[k]==1)&(new[k]<1)).sum()),recoveries=int(((old[k]<1)&(new[k]==1)).sum()),denominator=512) for k in keys[1:4]},
            requested_jerk_RMS_m_s3=float(np.mean([r['geometry']['requested_jerk_RMS_m_s3'] for r in rows])))
        cost.extend(json.loads(p.read_text()) for p in (root/model/'invocations').glob('*.json'))
    value=dict(status='COMPLETE',registration_sha256=reg['sha256'],scenes=64,logs=len(logs),G=8,models=out,
        costs=dict(actual_scores=sum(c['actual_scores'] for c in cost),new_actor_rollouts=0,new_velocity_forwards=0,optimizer_updates=0,
            cumulative_worker_wall_seconds=sum(c['worker_wall_seconds'] for c in cost),
            elapsed_job_wall_seconds=max(c['end_unix'] for c in cost)-min(c['start_unix'] for c in cost)),
        interpretation='Conditional counterfactual of only the final innovation on fixed saved chains, not a trained/deployed policy improvement')
    atomic_json(root/'summary.json',value);return value


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['register','worker','summarize']);p.add_argument('--root',required=True)
    p.add_argument('--parent');p.add_argument('--model',choices=MODELS);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args();root=Path(a.root)
    if a.command=='register':print(json.dumps(register(root,Path(a.parent))))
    elif a.command=='worker':
        if a.model is None or a.shard not in range(8):raise ValueError('Valid model/shard required')
        worker(a)
    else:print(json.dumps(summarize(root)))


if __name__=='__main__':main()
