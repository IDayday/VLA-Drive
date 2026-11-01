"""Finite CPU H/I controls for completed structured terminal proposals.

Only new H/I candidates are scored. Frozen v2, original z, roots and labels stay
read-only. No actor/optimizer/policy probability or automatic dispatch exists.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

from . import probe_structured as P
np=P.np
sha,read_sealed,sealed,atomic=P.sha,P.read_sealed,P.sealed,P.atomic
SCHEMA='TRAIN_Safety10k_terminal_noise_profile_controls_CPU_FP32_v1'
STRATEGIES=('H_terminal_ramped_IID','I_terminal_energy_matched_IID')
ENERGY_RATIO=float(np.mean((P.TIMES/4)**2))
AMPLITUDE=float(np.sqrt(ENERGY_RATIO))
SOURCE_FILES=P.SOURCE_FILES+('tools/qwen_world_safety_exploration/probe_noise_profiles.py',)


def propose(base,z):
    base=np.asarray(base);z=np.asarray(z)
    if (base.shape!=(16,8,3) or z.shape!=(16,8,3) or z.dtype!=np.float64
            or not np.isfinite(base).all() or not np.isfinite(z).all()):raise ValueError('Exact16 endpoints and savedFP64 z required')
    noise=np.stack((z*P.STD*(P.TIMES/4)[None,:,None],z*P.STD*AMPLITUDE))
    raw=(base.astype(np.float64)[None]+noise).reshape(32,8,3)
    requested=raw.astype(np.float32)
    audit=dict(source_innovations_hash=P.array_hash(z),new_independent_draws=0,reused_independent_coordinates=384,
        noise_hashes={k:P.array_hash(noise[i]) for i,k in enumerate(STRATEGIES)},
        raw_requested_hashes={k:P.array_hash(raw[16*i:16*(i+1)]) for i,k in enumerate(STRATEGIES)},
        requested_hashes={k:P.array_hash(requested[16*i:16*(i+1)]) for i,k in enumerate(STRATEGIES)},
        quantization_max_abs_per_axis=np.max(np.abs(requested.astype(np.float64)-raw),axis=(0,1)).tolist(),
        realized_noise_squared_energy={k:float((noise[i]**2).sum()) for i,k in enumerate(STRATEGIES)},
        H_profile='IID, rho0, STD_axis*(t/4), t=.5..4',
        I_profile='IID, rho0, STD_axis*sqrt(mean((t/4)^2)) at every horizon',
        H_G_same_analytic_horizon_variance=True,I_H_G_same_analytic_total_trace=True,
        analytic_energy_ratio=ENERGY_RATIO,constant_amplitude_multiplier=AMPLITUDE,
        analytic_variance_before_FP32_rounding=True,realized_noise_energy_not_rescaled=True,
        precision='FP64 saved z/noise/raw -> FP32 BEFORE all score/store/hash',no_clipping_or_policy_logp=True)
    return requested,dict(innovations=z.copy(),noise=noise,raw_requested=raw),audit


def load_source(case,m):
    folder=Path(m['source_profile']['output']);path=folder/'scenes'/f"{case['token']}.json"
    row=read_sealed(path)
    if (sha(path)!=case['profile_scene_sha256'] or row['schema']!=P.SCHEMA
            or row['identity']!=m['source_profile']['identity_sha256'] or row['status']!='OK'
            or any(row[k]!=case[k] for k in ('token','log','role','pair_token','source_root_hash'))):
        raise ValueError('Completed v2 source case changed')
    arrays=folder/row['arrays']['path']
    if sha(arrays)!=case['profile_arrays_sha256']:raise ValueError('Completed v2 source arrays changed')
    with np.load(arrays,allow_pickle=False) as values:
        base=values['source_ODE_requested'].copy();roots=values['source_roots'].copy();z=values['innovations'].copy()
    if (P.array_hash(base)!=case['source_ODE_requested_hash'] or P.array_hash(roots)!=case['source_root_hash']
            or P.array_hash(z)!=case['source_innovations_hash']
            or P.array_hash(z)!=row['noise_audit']['independent_innovations_hash']):
        raise ValueError('Source ODE root/endpoint or exact saved independent innovations changed')
    return row,base,roots,z


def prepare(args):
    path=Path(args.structured_manifest).resolve();old=read_sealed(path);folder=Path(old['required_output'])
    identity=read_sealed(folder/'identity.json');summary=read_sealed(folder/'SUMMARY.json')
    if (old['schema']!=P.SCHEMA or summary['status']!='COMPLETE' or summary['count']!=312
            or identity['manifest_sha256']!=old['sha256'] or summary['identity_sha256']!=identity['sha256']
            or summary['costs']['new_official_queries']!=14976 or summary['costs']['prior_failed_official_queries']!=384
            or identity['cases']!=old['cases']):raise ValueError('Completed312-case FP32v2 required')
    P.validate_shards(folder,identity)
    population=old['source_population']
    if sha(population['path'])!=population['sha256']:raise ValueError('TRAIN population changed')
    P.validate_train([c['token'] for c in old['cases']],json.loads(Path(population['path']).read_text()))
    source=dict(output=str(folder),manifest=str(path),manifest_file_sha256=sha(path),manifest_sha256=old['sha256'],
        identity_sha256=identity['sha256'],summary_path=str(folder/'SUMMARY.json'),summary_file_sha256=sha(folder/'SUMMARY.json'),
        protocol_sha256=identity['protocol']['sha256'])
    cases=[]
    for old_case in old['cases']:
        token=old_case['token'];scene=folder/'scenes'/f'{token}.json';row=read_sealed(scene)
        case={k:old_case[k] for k in ('token','log','role','pair_token','source_root_hash','source_ODE_requested_hash','cache_sha256')}
        case.update(profile_scene_sha256=sha(scene),profile_arrays_sha256=row['arrays']['sha256'],
            source_innovations_hash=old_case['expected_noise_audit']['independent_innovations_hash'])
        _,base,roots,z=load_source(case,dict(source_profile=source))
        _,_,case['expected_noise_audit']=propose(base,z);cases.append(case)
    destination=Path(args.output).resolve()
    if destination==path.parent or path.parent in destination.parents:raise ValueError('Do not write into completed v2')
    m=sealed(dict(schema=SCHEMA,status='PREPARED_NOT_DISPATCHED',count=312,world=8,cases=cases,
        source_profile=source,source_probe=old['source_probe'],config=old['config'],source_population=population,
        required_output=str(destination/'results'),strategies=list(STRATEGIES),physical_std=P.STD.tolist(),
        horizon_seconds=P.TIMES.tolist(),energy_ratio=ENERGY_RATIO,constant_I_amplitude_multiplier=AMPLITUDE,
        requested_dtype='float32 pre-service',source_innovation_dtype='float64 unchanged saved v2',
        candidates_per_strategy=16,queries=9984,queries_per_shard=1248,cases_per_shard=39,
        prior_costs=dict(failed_v1_queries=384,successful_v2_queries=14976),
        contrasts=dict(G_vs_H='same analytic horizon variance/ramp and same z/root, changes AR1rho.8->IIDrho0',
            H_vs_I='IID both, equal expected total energy, changes temporal variance allocation',
            E_vs_I='IID constant horizon variance both, changes amplitude1->sqrt(.3984375)',
            caveat='All analytic controls refer to prequantization noise; FP32 quantization errors separately recorded; old late comparison changes more than profile'),
        no_actor_optimizer_or_policy_logp=True,TRAIN_selected_diagnostic_not_NAV=True,
        scientific_sources={p:sha(p) for p in SOURCE_FILES}))
    target=destination/'MANIFEST.json'
    if target.exists() and read_sealed(target)!=m:raise ValueError('Immutable control manifest differs')
    atomic(target,m)
    atomic(destination/'LAUNCH_PLAN.json',sealed(dict(schema=SCHEMA+'_launch',status='READY_NOT_DISPATCHED',
        manifest=str(target),manifest_sha256=m['sha256'],output=m['required_output'],world=8,cases_per_rank=39,
        CPU_threads_max=2,scorer_processes_per_rank=1,new_queries=9984,prior_failed_v1=384,prior_successful_v2=14976,
        command_template=['python','-u','-m','tools.qwen_world_safety_exploration.probe_noise_profiles','run',
            '--manifest',str(target),'--output',m['required_output'],'--rank','RANK0..7','--world','8','--cpu-threads','2'],
        summarize=['python','-m','tools.qwen_world_safety_exploration.probe_noise_profiles','summarize','--output',m['required_output']],
        no_dispatch_performed=True)))
    return m


def validate_scene(row,case,identity,output):
    from tools.qwen_world_postcot.evaluate import validate_feedback
    from starVLA.model.modules.qwen_world_postcot.candidate_bank import trajectory_hash
    if (row['schema']!=SCHEMA or row['identity']!=identity['sha256'] or row['status']!='OK'
            or set(row['strategies'])!=set(STRATEGIES)
            or any(row[k]!=case[k] for k in ('token','log','role','pair_token','source_root_hash','source_innovations_hash'))):
        raise ValueError('Control scene identity/root/noise mismatch')
    path=Path(output)/row['arrays']['path']
    if sha(path)!=row['arrays']['sha256']:raise ValueError('Control arrays changed')
    with np.load(path,allow_pickle=False) as a:
        base=a['source_ODE_requested'].copy();roots=a['source_roots'].copy();z=a['innovations'].copy();requested=a['requested'].copy()
        wanted,arrays,audit=propose(base,z)
        if (requested.dtype!=np.float32 or not np.array_equal(requested,wanted)
                or audit!=row['noise_audit'] or audit!=case['expected_noise_audit']
                or P.array_hash(base)!=case['source_ODE_requested_hash'] or P.array_hash(roots)!=case['source_root_hash']
                or P.array_hash(z)!=case['source_innovations_hash']
                or not np.array_equal(a['noise'],arrays['noise']) or not np.array_equal(a['raw_requested'],arrays['raw_requested'])):
            raise ValueError('Control proposal or source noise binding changed')
    for i,k in enumerate(STRATEGIES):
        records=row['strategies'][k]['records']
        if len(records)!=16:raise ValueError('Need all16 independent labels')
        for j,r in enumerate(records):
            candidate=requested[16*i+j];trajectory_hash(candidate)
            validate_feedback(r,token=case['token'],log=case['log'],requested=candidate,protocol_sha256=identity['protocol']['sha256'])
            if r['cache']['sha256']!=case['cache_sha256'] or not np.array_equal(r['requested_trajectory'],candidate):
                raise ValueError('Control label not exact same FP32 request/cache')
    return row


def run(args):
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]=str(args.cpu_threads)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from threadpoolctl import threadpool_limits
    from starVLA.rl.qwen_world_safety_exploration.contracts import require_dispatch,boundary_stop_due
    from starVLA.model.modules.qwen_world_postcot.candidate_bank import trajectory_hash
    from tools.qwen_world_postcot.evaluate import validate_feedback
    from scripts.analysis.qwen_sourceflow.evaluate_policy import json_feedback
    host=require_dispatch();m=read_sealed(args.manifest)
    if (m['schema']!=SCHEMA or args.world!=8 or not 0<=args.rank<8 or not 1<=args.cpu_threads<=2
            or m['queries']!=9984 or str(Path(args.output).resolve())!=m['required_output']):raise ValueError('Wrong registered control shard/budget')
    for path,h in m['scientific_sources'].items():
        if sha(path)!=h:raise ValueError('Control/scorer source changed: '+path)
    torch.set_num_threads(args.cpu_threads);_thread_limit=threadpool_limits(limits=args.cpu_threads)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'scenes').mkdir(exist_ok=True)
    guard=(out/f'rank{args.rank}.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    service=P.build_service(m)
    if service.protocol['sha256']!=m['source_profile']['protocol_sha256']:
        service.close();raise ValueError('Official scorer protocol changed')
    identity=sealed(dict(schema=SCHEMA,manifest_sha256=m['sha256'],manifest_path=str(Path(args.manifest).resolve()),
        count=312,world=8,cases=m['cases'],source_profile=m['source_profile'],source_probe=m['source_probe'],
        protocol=service.protocol,scientific_sources=m['scientific_sources'],CPU_only=True,no_actor_optimizer=True))
    with (out/'identity.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'identity.json').exists() and read_sealed(out/'identity.json')!=identity:
            service.close();raise ValueError('Control identity changed')
        atomic(out/'identity.json',identity)
    cases=m['cases'][args.rank::8];state=dict(status='RUNNING',rank=args.rank,world=8,host=host,identity=identity['sha256'],
        expected_tokens=[c['token'] for c in cases],completed_tokens=[],new=0,reused=0,start_unix=time.time())
    try:
        for case in cases:
            if boundary_stop_due(host):state['status']='STOPPED_AT_SCENE_BOUNDARY';break
            dest=out/'scenes'/f"{case['token']}.json"
            if dest.exists():validate_scene(read_sealed(dest),case,identity,out);state['reused']+=1;state['completed_tokens'].append(case['token']);continue
            started=time.monotonic();_,base,roots,z=load_source(case,m);requested,arrays,audit=propose(base,z)
            if audit!=case['expected_noise_audit']:raise ValueError('Control noise plan differs')
            for candidate in requested:trajectory_hash(candidate)
            feedback=service.score((case['token'],case['log'],requested))
            if len(feedback)!=32:raise ValueError('Incomplete controls scoring')
            for j,r in enumerate(feedback):
                validate_feedback(r,token=case['token'],log=case['log'],requested=requested[j],protocol_sha256=service.protocol['sha256'])
                if r['cache']['sha256']!=case['cache_sha256'] or not np.array_equal(r['requested_trajectory'],requested[j]):raise ValueError('Exact FP32 scored request/cache differs')
            target=out/'scenes'/f"{case['token']}.npz";temp=target.with_name(target.name+'.writing-'+str(os.getpid()))
            with temp.open('xb') as f:np.savez_compressed(f,source_ODE_requested=base,source_roots=roots,requested=requested,**arrays);f.flush();os.fsync(f.fileno())
            os.replace(temp,target)
            row=sealed(dict(schema=SCHEMA,status='OK',identity=identity['sha256'],**case,noise_audit=audit,
                strategies={k:dict(records=[json_feedback(r) for r in feedback[16*i:16*(i+1)]]) for i,k in enumerate(STRATEGIES)},
                arrays=dict(path='scenes/'+target.name,sha256=sha(target)),new_official_queries=32,seconds=time.monotonic()-started))
            atomic(dest,row);state['new']+=1;state['completed_tokens'].append(case['token']);atomic(out/f'rank{args.rank}.json',state)
        else:state['status']='COMPLETE'
    except BaseException:state.update(status='FAILED_REQUIRES_REVIEW',error=traceback.format_exc());raise
    finally:state.update(end_unix=time.time(),CUDA_initialized=torch.cuda.is_initialized());atomic(out/f'rank{args.rank}.json',state);service.close()
    return state


def summarize(output):
    out=Path(output);identity=read_sealed(out/'identity.json');m=read_sealed(identity['manifest_path'])
    if identity['schema']!=SCHEMA or identity['manifest_sha256']!=m['sha256']:raise ValueError('Changed control manifest')
    tokens=P.validate_shards(out,identity)
    if {p.stem for p in (out/'scenes').glob('*.json')}!=set(tokens):raise ValueError('Partial/foreign controls scenes')
    previous=read_sealed(m['source_profile']['summary_path'])
    if sha(m['source_profile']['summary_path'])!=m['source_profile']['summary_file_sha256']:raise ValueError('Completed comparison summary changed')
    reference={r['token']:r for r in previous['scenes']};rows=[];sources=[];scenes=[]
    for case in m['cases']:
        row=validate_scene(read_sealed(out/'scenes'/f"{case['token']}.json"),case,identity,out)
        source,_,_,_=load_source(case,m);rows.append(row);sources.append(source)
        desc=dict(reference[case['token']]['strategies']);desc.update({k:P.describe(row['strategies'][k]['records']) for k in STRATEGIES})
        scenes.append(dict(token=case['token'],log=case['log'],role=case['role'],pair_token=case['pair_token'],strategies=desc,
            source_root_hash=case['source_root_hash'],source_innovations_hash=case['source_innovations_hash']))
    cohorts={role:dict(values) for role,values in previous['cohorts'].items()}
    for role in ('historical_failure','matched_high_control'):
        selected=[(r,s) for r,s in zip(rows,sources) if r['role']==role]
        for name in STRATEGIES:
            sets=[r['strategies'][name]['records'] for r,s in selected];flat=[v for rs in sets for v in rs];descs=[P.describe(rs) for rs in sets]
            values=dict(**P.describe(flat),scene_count=len(selected),scenes_with_quality=sum(d['found_strict_quality'] for d in descs),
                scenes_with_feasible=sum(d['found_feasible'] for d in descs),
                component_means={k:float(np.mean([v['metrics'][k] for v in flat])) for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')},vs_completed={})
            for baseline in P.BASELINES+P.STRATEGIES:
                older=[reference[r['token']]['strategies'][baseline] for r,s in selected]
                values['vs_completed'][baseline]=dict(quality_gained=sum(n['found_strict_quality'] and not o['found_strict_quality'] for n,o in zip(descs,older)),quality_lost=sum(o['found_strict_quality'] and not n['found_strict_quality'] for n,o in zip(descs,older)),feasible_gained=sum(n['found_feasible'] and not o['found_feasible'] for n,o in zip(descs,older)),feasible_lost=sum(o['found_feasible'] and not n['found_feasible'] for n,o in zip(descs,older)))
            cohorts[role][name]=values
    queries=sum(r['new_official_queries'] for r in rows)
    if queries!=9984:raise ValueError('Control query budget differs')
    result=sealed(dict(schema=SCHEMA+'_summary',status='COMPLETE',count=312,world=8,identity_sha256=identity['sha256'],
        cohorts=cohorts,scenes=scenes,contrasts=m['contrasts'],
        costs=dict(prior_failed_v1_queries=384,prior_successful_v2_queries=14976,new_successful_control_queries=queries,total_noise_family_queries=384+14976+queries,actor_forward_calls=0,optimizer_steps=0),
        interpretation='H/G isolate rho under identical prequantization horizon variance; H/I isolate time allocation under same expected trace; E/I isolate IID amplitude. Same saved z/ODE16 roots; selected TRAIN proposals, not policy training/NAV improvement.'))
    target=out/'SUMMARY.json'
    if target.exists() and read_sealed(target)!=result:raise ValueError('Immutable controls summary differs')
    atomic(target,result);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','run','summarize'));p.add_argument('--output',required=True)
    p.add_argument('--structured-manifest');p.add_argument('--manifest');p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8);p.add_argument('--cpu-threads',type=int,default=2)
    a=p.parse_args()
    if a.mode=='prepare' and not a.structured_manifest:p.error('--structured-manifest required')
    if a.mode=='run' and not a.manifest:p.error('--manifest required')
    v=prepare(a) if a.mode=='prepare' else run(a) if a.mode=='run' else summarize(a.output)
    print(json.dumps(dict(status=v.get('status'),count=v.get('count'),rank=v.get('rank'),output=a.output,sha256=v.get('sha256'))),flush=True)

if __name__=='__main__':main()
