"""Finite CPU-only endpoint proposals from sealed TRAIN SUPPORT ODE16.

No actor, Qwen, optimizer, policy likelihood, clipping or training is executed.
Each novel requested trajectory is officially scored in its own reference pair.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

for _thread_env in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_env]='2'
import numpy as np

from tools.qwen_world_research.probe_zero_rollouts import sha, sealed, read_sealed, atomic
from .probe_support import SCHEMA as SUPPORT_SCHEMA, good, describe, validate_train

SCHEMA='TRAIN_Safety10k_ODE16_structured_terminal_proposal_CPU_v1'
STRATEGIES=('E_terminal_Gaussian','F_terminal_AR1','G_terminal_ramped_AR1')
BASELINES=('A_late','B_mixed','D_ODE')
STD=np.array([1.4864179440431657,.9354143489633349,.07861934885334695],dtype=np.float64)
RHO=.8
TIMES=np.arange(1,9,dtype=np.float64)*.5
SOURCE_FILES=('tools/qwen_world_safety_exploration/probe_structured.py',
 'tools/qwen_world_safety_exploration/probe_support.py',
 'tools/qwen_world_research/probe_zero_rollouts.py',
 'starVLA/rl/qwen_world_safety_exploration/group_runtime.py',
 'starVLA/rl/qwen_world_safety_exploration/contracts.py',
 'starVLA/rl/qwen_world_research/runtime.py',
 'starVLA/rl/simwam_recipe/reward_readouts.py',
 'starVLA/rl/safe_flow_improvement/v1_reward.py',
 'scripts/analysis/constrained_flow_noise/batched_canonical_score.py')


def array_hash(a):
    a=np.ascontiguousarray(a)
    return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def noise_from_innovations(z):
    """Gaussian and stationary AR1 share the same 16x8x3 independent N(0,1) z."""
    z=np.asarray(z)
    if z.dtype!=np.float64 or z.shape!=(16,8,3) or not np.isfinite(z).all():
        raise ValueError('Finite FP64 independent innovations [16,8,3] required')
    ar=np.empty_like(z);ar[:,0]=z[:,0]
    for t in range(1,8):ar[:,t]=RHO*ar[:,t-1]+math.sqrt(1-RHO**2)*z[:,t]
    noise=np.stack((z*STD,ar*STD,ar*STD*(TIMES/4)[None,:,None]))
    return noise,ar


def propose(base,token):
    from starVLA.rl.simwam_recipe.contracts import seed_for
    base=np.asarray(base)
    if base.shape!=(16,8,3) or not np.isfinite(base).all():raise ValueError('Exact ODE16 endpoints required')
    seed=seed_for('TRAIN_Safety10k_structured_terminal_proposal',42,token)
    private=np.random.Generator(np.random.PCG64(seed))
    z=private.standard_normal((16,8,3),dtype=np.float64)
    noise,ar=noise_from_innovations(z)
    requested=(base.astype(np.float64)[None]+noise).reshape(48,8,3)
    return requested,dict(innovations=z,stationary_AR1=ar,noise=noise),dict(
        seed=seed,seed_base=42,generator='private numpy.PCG64, no global RNG mutation',
        independent_innovations_hash=array_hash(z),stationary_AR1_hash=array_hash(ar),
        independent_draw_count=16*8*3,noise_hashes={k:array_hash(noise[i]) for i,k in enumerate(STRATEGIES)},
        requested_hashes={k:array_hash(requested[i*16:(i+1)*16]) for i,k in enumerate(STRATEGIES)},
        realized_noise_squared_energy={k:float((noise[i]**2).sum()) for i,k in enumerate(STRATEGIES)},
        marginal_variance='Gaussian/AR1: STD_axis^2 at every horizon; ramped: STD_axis^2*(t/4)^2',
        same_expected_energy_Gaussian_AR1=True,finite_realization_energy_not_forced_equal=True,
        Gaussian_AR1_only_covariance_changes=True,ramped_also_changes_horizon_variance=True,
        action_or_innovation_clip=False,policy_probability_evaluated=False)


def load_source(case,source):
    folder=Path(source['output']);path=folder/'scenes'/f"{case['token']}.json"
    row=read_sealed(path)
    if (sha(path)!=case['source_scene_sha256'] or row['identity']!=source['identity_sha256']
            or any(row[k]!=case[k] for k in ('token','log','role','pair_token'))
            or row['status']!='OK'):raise ValueError('Changed sealed source case')
    arrays=folder/row['arrays']['path']
    if sha(arrays)!=case['source_arrays_sha256']:raise ValueError('Changed ODE16 source arrays')
    with np.load(arrays,allow_pickle=False) as values:
        base=values['requested'][48:64].copy();roots=values['roots'].copy()
    if base.shape!=(16,8,3) or roots.shape!=(16,8,3) or not np.isfinite(base).all() or not np.isfinite(roots).all():
        raise ValueError('Incomplete ODE16 endpoints/roots')
    root=array_hash(roots)
    old=row['pairing']['per_strategy']['D_ODE']['effective_root']
    if root['sha256']!=old['sha256'] or root['shape']!=old['shape']:
        raise ValueError('ODE16 endpoint root binding changed')
    if root!=case['source_root_hash'] or array_hash(base)!=case['source_ODE_requested_hash']:
        raise ValueError('Registered ODE endpoint/root hash changed')
    for j,record in enumerate(row['strategies']['D_ODE']['records']):
        if (record['status']!='OK' or record['token']!=case['token'] or record['log']!=case['log']
                or record['protocol_sha256']!=source['protocol_sha256']
                or record['cache']['sha256']!=case['cache_sha256']
                or not np.array_equal(np.asarray(record['requested_trajectory']),base[j])):
            raise ValueError('Original official ODE label/endpoint/cache mismatch')
    return row,base,roots


def validate_shards(output,identity):
    tokens=[r['token'] for r in identity['cases']]
    if len(tokens)!=312 or len(set(tokens))!=312 or identity['world']!=8:
        raise ValueError('Only complete312 cases/eight shards')
    for rank in range(8):
        node=json.loads((Path(output)/f'rank{rank}.json').read_text());expected=tokens[rank::8]
        if (node['status']!='COMPLETE' or node['world']!=8 or node['identity']!=identity['sha256']
                or node['expected_tokens']!=expected or node['completed_tokens']!=expected):
            raise ValueError('Incomplete or changed whole39-case shard')
    return tokens


def prepare(args):
    old_path=Path(args.baseline_manifest).resolve();old=read_sealed(old_path)
    folder=Path(old['required_output']);identity=read_sealed(folder/'identity.json');summary=read_sealed(folder/'SUMMARY.json')
    if (old['schema']!=SUPPORT_SCHEMA or identity['manifest_sha256']!=old['sha256']
            or summary['status']!='COMPLETE' or summary['count']!=312 or summary['identity_sha256']!=identity['sha256']
            or identity['cases']!=old['cases'] or identity['world']!=8 or identity['updates']!=10000):
        raise ValueError('Complete fixed Safety10k312-case source SUPPORT required')
    validate_shards(folder,identity)
    if sha(old['inputs']['population']['path'])!=old['inputs']['population']['sha256']:
        raise ValueError('Pinned TRAIN/NAV population changed')
    pop=json.loads(Path(old['inputs']['population']['path']).read_text())
    validate_train([c['token'] for c in old['cases']],pop)
    source=dict(output=str(folder),manifest=str(old_path),manifest_file_sha256=sha(old_path),
        identity_sha256=identity['sha256'],identity_file_sha256=sha(folder/'identity.json'),
        summary_file_sha256=sha(folder/'SUMMARY.json'),protocol_sha256=identity['protocol']['sha256'],
        checkpoint=old['rl_checkpoint'],checkpoint_optimizer_sha256=old['checkpoint_optimizer_sha256'])
    cases=[]
    for old_case in old['cases']:
        token=old_case['token'];row=read_sealed(folder/'scenes'/f'{token}.json')
        cache={r['cache']['sha256'] for r in row['strategies']['D_ODE']['records']}
        if len(cache)!=1:raise ValueError('Source official cache is inconsistent')
        with np.load(folder/row['arrays']['path'],allow_pickle=False) as v:
            base=v['requested'][48:64].copy();roots=v['roots'].copy()
        case={k:old_case[k] for k in ('token','log','role','pair_token')}
        case.update(source_scene_sha256=sha(folder/'scenes'/f'{token}.json'),
            source_arrays_sha256=row['arrays']['sha256'],source_root_hash=array_hash(roots),
            source_ODE_requested_hash=array_hash(base),cache_sha256=cache.pop())
        _,_,case['expected_noise_audit']=propose(base,token)
        load_source(case,source);cases.append(case)
    destination=Path(args.output).resolve()
    if destination==folder or folder in destination.parents:raise ValueError('Never write into old SUPPORT output')
    cfg=old['inputs']['config']
    if sha(cfg['path'])!=cfg['sha256']:raise ValueError('Pinned service configuration changed')
    body=dict(schema=SCHEMA,status='PREPARED_NOT_DISPATCHED',purpose='selected TRAIN endpoint proposal quality only; not StructRL training or NAV improvement',
        source_probe=source,config=cfg,source_population=old['inputs']['population'],cases=cases,count=312,world=8,
        required_output=str(destination/'results'),strategies=list(STRATEGIES),candidates_per_strategy=16,
        reference_candidate_independent_scoring=True,queries=14976,queries_per_shard=1872,cases_per_shard=39,
        physical_std=STD.tolist(),physical_units=['meters','meters','radians'],rho=RHO,horizon_seconds=TIMES.tolist(),
        ramped_expected_noise_energy_ratio_vs_stationary=float(np.mean((TIMES/4)**2)),
        noise_protocol='stationary AR1 first sample z0, following rho*previous+sqrt(1-rho^2)*zt; E/F share identical z; G=F*(t/4)',
        pairing='candidate index0..15 inherits the exact existing D_ODE root and clean endpoint; private token42 independent terminal innovation reused across all3 proposals',
        no_actor_or_optimizer=True,no_model_checkpoint_load=True,no_logp_or_probability_claim=True,
        original_canonical_single_ODE_not_called=True,source_B1='old private42 D_ODE[0], not formal seed2101 deployment',
        scientific_sources={p:sha(p) for p in SOURCE_FILES})
    manifest=sealed(body);path=destination/'MANIFEST.json'
    if path.exists() and read_sealed(path)!=manifest:raise FileExistsError('Immutable structured proposal manifest differs')
    atomic(path,manifest)
    atomic(destination/'LAUNCH_PLAN.json',sealed(dict(schema=SCHEMA+'_launch',status='READY_NOT_DISPATCHED',manifest=str(path),
        manifest_sha256=manifest['sha256'],output=body['required_output'],CPU_only=True,world=8,CPU_threads_per_rank_max=2,
        scoring_workers_per_rank=1,cases_per_rank=39,new_official_queries=14976,no_GPU_dispatched=True,
        command_template=['python','-u','-m','tools.qwen_world_safety_exploration.probe_structured','run','--manifest',str(path),
            '--output',body['required_output'],'--rank','RANK0..7','--world','8','--cpu-threads','2'],
        summarize=['python','-m','tools.qwen_world_safety_exploration.probe_structured','summarize','--output',body['required_output']])) )
    return manifest


def build_service(manifest):
    """Source GroupRuntime service hook only; never invoke make_actor/load_recipe."""
    import yaml
    from starVLA.rl.qwen_world_safety_exploration.group_runtime import GroupRuntime,validate_recipe
    cfg=manifest['config']
    if sha(cfg['path'])!=cfg['sha256']:raise ValueError('Service config changed')
    runtime=GroupRuntime();runtime.cfg=validate_recipe(yaml.safe_load(Path(cfg['path']).read_text()))
    return runtime.make_reward_service()


def validate_scene(row,case,identity,output):
    from tools.qwen_world_postcot.evaluate import validate_feedback
    if (row['schema']!=SCHEMA or row['identity']!=identity['sha256'] or row['status']!='OK'
            or any(row[k]!=case[k] for k in ('token','log','role','pair_token','source_root_hash'))
            or set(row['strategies'])!=set(STRATEGIES)):
        raise ValueError('Changed structured scene identity/root/strategies')
    path=Path(output)/row['arrays']['path']
    if sha(path)!=row['arrays']['sha256']:raise ValueError('Changed structured proposal arrays')
    with np.load(path,allow_pickle=False) as a:
        base=a['source_ODE_requested'].copy();roots=a['source_roots'].copy();requested=a['requested'].copy()
        want,noise,audit=propose(base,case['token'])
        if (array_hash(roots)!=case['source_root_hash'] or array_hash(base)!=case['source_ODE_requested_hash']
                or not np.array_equal(requested,want) or row['noise_audit']!=audit
                or audit!=case['expected_noise_audit']
                or not np.array_equal(a['innovations'],noise['innovations'])
                or not np.array_equal(a['stationary_AR1'],noise['stationary_AR1'])
                or not np.array_equal(a['noise'],noise['noise'])):
            raise ValueError('Changed base/root/private innovations or proposal kernel')
    for k,name in enumerate(STRATEGIES):
        records=row['strategies'][name]['records']
        if len(records)!=16:raise ValueError('All16 labels required for each proposal')
        for j,r in enumerate(records):
            validate_feedback(r,token=case['token'],log=case['log'],requested=requested[16*k+j],protocol_sha256=identity['protocol']['sha256'])
            if (r['cache']['sha256']!=case['cache_sha256']
                    or not np.array_equal(np.asarray(r['requested_trajectory']),requested[16*k+j])):
                raise ValueError('Changed official cache or exact FP64 scored request')
    return row


def run(args):
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]=str(args.cpu_threads)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from threadpoolctl import threadpool_limits
    from starVLA.rl.qwen_world_safety_exploration.contracts import require_dispatch,boundary_stop_due
    from scripts.analysis.qwen_sourceflow.evaluate_policy import json_feedback
    from tools.qwen_world_postcot.evaluate import validate_feedback
    host=require_dispatch();m=read_sealed(args.manifest)
    if (m['schema']!=SCHEMA or args.world!=8 or not 0<=args.rank<8 or not 1<=args.cpu_threads<=2
            or str(Path(args.output).resolve())!=m['required_output'] or m['queries']!=14976):
        raise ValueError('Wrong immutable CPU proposal shard/output/budget')
    for path,h in m['scientific_sources'].items():
        if sha(path)!=h:raise ValueError('Changed proposal/scorer source '+path)
    torch.set_num_threads(args.cpu_threads)
    _native_thread_limit=threadpool_limits(limits=args.cpu_threads)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'scenes').mkdir(exist_ok=True)
    guard=(out/f'rank{args.rank}.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    service=build_service(m)
    if service.protocol['sha256']!=m['source_probe']['protocol_sha256']:
        service.close();raise ValueError('Official reference-candidate scorer protocol changed')
    identity=sealed(dict(schema=SCHEMA,manifest_sha256=m['sha256'],manifest_path=str(Path(args.manifest).resolve()),
        cases=m['cases'],count=312,world=8,source_probe=m['source_probe'],protocol=service.protocol,
        scientific_sources=m['scientific_sources'],strategies=m['strategies'],physical_std=m['physical_std'],rho=RHO,
        no_actor_or_optimizer=True,CPU_only=True,proposal_quality_not_policy_rollout=True))
    with (out/'identity.lock').open('a') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX)
        if (out/'identity.json').exists() and read_sealed(out/'identity.json')!=identity:
            service.close();raise ValueError('Changed structured proposal identity')
        atomic(out/'identity.json',identity)
    cases=m['cases'][args.rank::8]
    state=dict(status='RUNNING',identity=identity['sha256'],rank=args.rank,world=8,host=host,
        expected_tokens=[c['token'] for c in cases],completed_tokens=[],new=0,reused=0,start_unix=time.time())
    try:
        for case in cases:
            if boundary_stop_due(host):state['status']='STOPPED_AT_SCENE_BOUNDARY';break
            token=case['token'];dest=out/'scenes'/f'{token}.json'
            if dest.exists():
                validate_scene(read_sealed(dest),case,identity,out)
                state['completed_tokens'].append(token);state['reused']+=1;continue
            tick=time.monotonic();source,base,roots=load_source(case,m['source_probe'])
            requested,arrays,audit=propose(base,token)
            if audit!=case['expected_noise_audit']:raise ValueError('Presealed private noise/hash changed')
            feedback=service.score((token,case['log'],requested))
            if len(feedback)!=48:raise ValueError('Partial labels cannot count as complete evidence')
            for j,r in enumerate(feedback):
                validate_feedback(r,token=token,log=case['log'],requested=requested[j],protocol_sha256=service.protocol['sha256'])
                if (r['cache']['sha256']!=case['cache_sha256']
                        or not np.array_equal(np.asarray(r['requested_trajectory']),requested[j])):
                    raise ValueError('Source/proposal context or exact scored FP64 request differs')
            npz=out/'scenes'/f'{token}.npz';temp=npz.with_name(npz.name+'.writing-'+str(os.getpid()))
            with temp.open('xb') as f:
                np.savez_compressed(f,source_ODE_requested=base,source_roots=roots,requested=requested,**arrays)
                f.flush();os.fsync(f.fileno())
            os.replace(temp,npz)
            row=sealed(dict(schema=SCHEMA,status='OK',identity=identity['sha256'],**case,
                source_probe_identity=m['source_probe']['identity_sha256'],noise_audit=audit,
                strategies={k:dict(records=[json_feedback(r) for r in feedback[16*i:16*(i+1)]]) for i,k in enumerate(STRATEGIES)},
                arrays=dict(path='scenes/'+npz.name,sha256=sha(npz)),new_official_queries=48,
                actor_forward_calls=0,optimizer_steps=0,seconds=time.monotonic()-tick))
            atomic(dest,row);state['new']+=1;state['completed_tokens'].append(token)
            atomic(out/f'rank{args.rank}.json',state)
        else:state['status']='COMPLETE'
    except BaseException:
        state.update(status='FAILED_REQUIRES_REVIEW',error=traceback.format_exc());raise
    finally:
        state.update(end_unix=time.time(),CUDA_initialized=torch.cuda.is_initialized())
        atomic(out/f'rank{args.rank}.json',state);service.close()
    return state


def summarize(output):
    out=Path(output);identity=read_sealed(out/'identity.json');m=read_sealed(identity['manifest_path'])
    if m['sha256']!=identity['manifest_sha256'] or identity['schema']!=SCHEMA:raise ValueError('Changed proposal manifest binding')
    tokens=validate_shards(out,identity)
    if {p.stem for p in (out/'scenes').glob('*.json')}!=set(tokens):raise ValueError('Missing/foreign scene results')
    rows=[];sources=[];scenes=[]
    for case in identity['cases']:
        row=validate_scene(read_sealed(out/'scenes'/f"{case['token']}.json"),case,identity,out)
        source=read_sealed(Path(m['source_probe']['output'])/'scenes'/f"{case['token']}.json")
        if sha(Path(m['source_probe']['output'])/'scenes'/f"{case['token']}.json")!=case['source_scene_sha256']:
            raise ValueError('Changed old strategy comparison')
        rows.append(row);sources.append(source)
        describe_all={k:describe(row['strategies'][k]['records']) for k in STRATEGIES}
        describe_all.update({k:describe(source['strategies'][k]['records']) for k in BASELINES})
        scenes.append(dict(token=case['token'],log=case['log'],role=case['role'],pair_token=case['pair_token'],
            strategies=describe_all,source_root_hash=case['source_root_hash'],noise_audit=row['noise_audit']))
    cohorts={}
    names=BASELINES+STRATEGIES
    for role in ('historical_failure','matched_high_control'):
        selected=[(r,s) for r,s in zip(rows,sources) if r['role']==role]
        cohort={}
        for name in names:
            recordsets=[(r if name in STRATEGIES else s)['strategies'][name]['records'] for r,s in selected]
            descs=[describe(rs) for rs in recordsets];flat=[v for rs in recordsets for v in rs]
            values=dict(**describe(flat),scenes_with_quality=sum(d['found_strict_quality'] for d in descs),
                scenes_with_feasible=sum(d['found_feasible'] for d in descs),scene_count=len(selected),
                component_means={k:float(np.mean([v['metrics'][k] for v in flat])) for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')})
            if name in STRATEGIES:
                values['vs_old']={}
                for baseline in BASELINES:
                    old=[describe(s['strategies'][baseline]['records']) for r,s in selected]
                    values['vs_old'][baseline]=dict(quality_gained=sum(n['found_strict_quality'] and not o['found_strict_quality'] for n,o in zip(descs,old)),
                        quality_lost=sum(o['found_strict_quality'] and not n['found_strict_quality'] for n,o in zip(descs,old)),
                        feasible_gained=sum(n['found_feasible'] and not o['found_feasible'] for n,o in zip(descs,old)),
                        feasible_lost=sum(o['found_feasible'] and not n['found_feasible'] for n,o in zip(descs,old)))
                oldroot=[v for r,s in selected for v in s['strategies']['D_ODE']['records']]
                values['matched_root_vs_ODE']=dict(quality_repaired=sum(good(n['metrics']) and not good(o['metrics']) for n,o in zip(flat,oldroot)),
                    quality_degraded=sum(good(o['metrics']) and not good(n['metrics']) for n,o in zip(flat,oldroot)))
            cohort[name]=values
        cohorts[role]=cohort
    result=sealed(dict(schema=SCHEMA+'_summary',status='COMPLETE',count=312,world=8,
        identity_sha256=identity['sha256'],cohorts=cohorts,scenes=scenes,
        costs=dict(new_official_queries=sum(r['new_official_queries'] for r in rows),actor_forward_calls=0,optimizer_steps=0,reused_old_candidates=312*16*3),
        interpretation='Selected TRAIN proposal quality only. E/F equal marginal std and expected noise energy, common innovations; G additionally ramps horizon variance. Same-root endpoints reused, no policy logp/training/NAV score.'))
    if result['costs']['new_official_queries']!=14976:raise ValueError('Budget mismatch')
    path=out/'SUMMARY.json'
    if path.exists() and read_sealed(path)!=result:raise ValueError('Immutable complete summary changed')
    atomic(path,result);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','run','summarize'))
    p.add_argument('--output',required=True);p.add_argument('--baseline-manifest');p.add_argument('--manifest')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8);p.add_argument('--cpu-threads',type=int,default=2)
    a=p.parse_args()
    if a.mode=='prepare' and not a.baseline_manifest:p.error('--baseline-manifest required')
    if a.mode=='run' and not a.manifest:p.error('--manifest required')
    result=prepare(a) if a.mode=='prepare' else run(a) if a.mode=='run' else summarize(a.output)
    print(json.dumps(dict(status=result.get('status'),count=result.get('count'),rank=result.get('rank'),output=a.output,sha256=result.get('sha256'))),flush=True)

if __name__=='__main__':main()
