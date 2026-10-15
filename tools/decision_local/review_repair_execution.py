"""Independent execution review of every strict REPAIR, without changing labels.

prepare/validate use the geometry environment; canonical uses the pinned NAVSIM
environment. The immutable result is a loss-only eligibility overlay. Rejected
repairs retain their local facts, scene membership and original failure record.
No solver retry, PDMS maximization or online output selection occurs here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import importlib.metadata
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import resource
import sys
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,read,digest,identity,source_sha,check_host


def case_key(token,anchor,plan):
    return f'{token}:{anchor}:{plan}'


def relative_execution_states(states,origin):
    """PDM stores global poses but instantaneous-body velocity/acceleration.

    BatchKinematicBicycleModel.get_state_dot multiplies VELOCITY_X by the
    heading to integrate global position. Rotating those body components as
    if they were global would create a false heading/velocity disagreement.
    """
    states=np.asarray(states).copy();x,y,yaw=origin;c,s=np.cos(yaw),np.sin(yaw)
    states[:,:2]=(states[:,:2]-[x,y])@np.array([[c,-s],[s,c]])
    states[:,2]=np.arctan2(np.sin(states[:,2]-yaw),np.cos(states[:,2]-yaw))
    return states


def prepare(a):
    import torch
    from safetensors import safe_open
    from starVLA.dataloader.foresight_dataset import decode_ego
    from starVLA.model.modules.decision_local.natural_error_contract import SCHEMA
    labels=read(a.labels/'identity.json');features=read(a.features/'identity.json');anchor=read(a.anchor)
    if labels['schema']!=SCHEMA or labels['feature_identity']!=features['identity'] or labels['anchor_identity']!=anchor['identity']:
        raise ValueError('Foreign local labels, feature bank or anchor')
    for bank,meta in ((a.labels,labels),(a.features,features)):
        if read(bank/'COMPLETE.json')['identity']!=meta['identity']:raise ValueError('Complete immutable bank required')
    if labels.get('action_codec'):
        from starVLA.dataloader.a_only_codec import install_legacy_codec
        codec=read(labels['action_codec'])
        if codec['identity']!=labels['action_codec_identity']:raise ValueError('Wrong action codec')
        install_legacy_codec(codec)
    declaration=dict(schema='natural_repair_execution_input_v1',labels_identity=labels['identity'],
        features_identity=features['identity'],anchor_identity=anchor['identity'],anchor_record=str(a.anchor),
        labels=str(a.labels),features=str(a.features),split=features['split'],constraints=labels['constraints'],
        source_sha=source_sha(ROOT),tool_sha256=digest(__file__),scope=labels['scope'])
    ident=identity(declaration)
    if (a.output/'identity.json').exists() and read(a.output/'identity.json')['identity']!=ident:
        raise ValueError('Preserve the prior review version')
    atomic_json(a.output/'identity.json',dict(identity=ident,**declaration));(a.output/'inputs').mkdir(exist_ok=True)
    chunks={}
    for path in sorted((a.features/'records').glob('*.json')):
        receipt=read(path)
        if receipt['identity']!=features['identity']:raise ValueError('Foreign feature chunk')
        chunks[str(a.features/'chunks'/(path.stem+'.safetensors'))]=receipt
    verified=set();cases=[]
    for row in labels['index']:
        receipt=read(a.labels/'records'/(row['token']+'.json'));path=a.labels/'scenes'/(row['token']+'.npz')
        if receipt['identity']!=labels['identity'] or digest(path)!=receipt['sha256']:raise ValueError('Changed label bytes')
        with np.load(path) as bank:
            pairs=np.argwhere((bank['case_status']==1)&bank['target_eligible'])
            if not len(pairs):continue
            chunk=receipt['source_feature_chunk'];offset=receipt['source_feature_offset']
            if chunks[chunk]['tokens'][offset]!=row['token']:raise ValueError('Wrong legal current state')
            if chunk not in verified:
                if digest(chunk)!=chunks[chunk]['sha256']:raise ValueError('Changed feature bytes')
                verified.add(chunk)
            with safe_open(chunk,framework='pt',device='cpu') as f:state=f.get_slice('state')[offset]
            velocity=decode_ego(state[None])[0,:2].numpy()/.5
            for anchor_index,plan_index in pairs:
                key=case_key(row['token'],int(anchor_index),int(plan_index));filename=key.replace(':','_')+'.npz'
                target=decode_ego(torch.from_numpy(bank['repair_target'][anchor_index,plan_index])).numpy()
                if not np.isfinite(target).all():raise ValueError('Invalid strict REPAIR')
                destination=a.output/'inputs'/filename;tmp=destination.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,plan=bank['plan_in_physical'][anchor_index,plan_index],target=target,velocity=velocity)
                tmp.replace(destination)
                cases.append(dict(key=key,token=row['token'],log=row['log'],anchor_index=int(anchor_index),
                    plan_index=int(plan_index),input=str(destination),input_sha256=digest(destination),label_sha256=receipt['sha256']))
    atomic_json(a.output/'PREPARED.json',dict(identity=ident,cases=cases,strict_REPAIR_cases=len(cases),
        scenes=len({c['token'] for c in cases}),logs=len({c['log'] for c in cases}),selection='Every original eligible REPAIR; no score or outcome selection'))


def canonical_case(task):
    case,metric,output=task;start=time.monotonic();cpu=time.process_time()
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    if digest(case['input'])!=case['input_sha256']:raise ValueError('Changed target before execution')
    with np.load(case['input']) as f:plans={k:f[k].copy() for k in ('plan','target')}
    with lzma.open(metric['cache_path'],'rb') as f:cache=pickle.load(f)
    sampling=TrajectorySampling(num_poses=40,interval_length=.1);results={};arrays={'times':np.arange(41)*.1}
    origin=cache.ego_state.rear_axle
    for kind,plan in plans.items():
        scorer=PDMScorer(sampling);simulator=PDMSimulator(sampling)
        score=asdict(pdm_score(cache,Trajectory(plan.astype(np.float64),TrajectorySampling(num_poses=8,interval_length=.5)),sampling,simulator,scorer))
        states=relative_execution_states(scorer._states[1],[origin.x,origin.y,origin.heading])
        arrays[kind]=states;results[kind]=score
    stem=case['key'].replace(':','_');path=Path(output)/'executed'/(stem+'.npz');tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)
    receipt=dict(key=case['key'],input_sha256=case['input_sha256'],metric_cache_sha256=digest(metric['cache_path']),
        executed=str(path),executed_sha256=digest(path),canonical_scores=results,
        state_coordinates='pose_ego0_velocity_acceleration_instantaneous_body_v1',
        wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu)
    atomic_json(Path(output)/'executed'/(stem+'.json'),receipt);return receipt


def canonical(a):
    from tools.local_interaction_mask_v2.score_async import initialize,python_tree_digest
    initialize(str(a.devkit));import nuplan
    versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','shapely')}
    if versions!={'numpy':'1.26.4','scipy':'1.13.1','shapely':'2.0.7'}:raise ValueError('Use the locked canonical environment')
    if python_tree_digest(a.devkit/'navsim')!='1468974c9af4405597c9dd3e8db1b68e7d7ab337327a7595fb57c2b3dcd4a84f':raise ValueError('Changed canonical NAVSIM')
    if python_tree_digest(Path(nuplan.__file__).parent)!='0e83aab8d5b92bf5792d815a1f25686088d1f4b8db70ea604331cd98f1606849':raise ValueError('Changed canonical nuPlan')
    prepared=read(a.output/'PREPARED.json');lookup={r['token']:r for r in read(a.metric_index)}
    (a.output/'executed').mkdir(exist_ok=True);tasks=[];done=[]
    for case in prepared['cases']:
        metric=lookup[case['token']]
        if metric['log']!=case['log']:raise ValueError('Wrong metric-cache scene')
        receipt_path=a.output/'executed'/(case['key'].replace(':','_')+'.json')
        if receipt_path.exists():
            receipt=read(receipt_path)
            if (receipt['input_sha256']!=case['input_sha256'] or digest(receipt['executed'])!=receipt['executed_sha256'] or
                digest(metric['cache_path'])!=receipt['metric_cache_sha256']):raise ValueError('Changed review during resume')
            done.append(receipt)
        else:tasks.append((case,metric,str(a.output)))
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(a.devkit),)) as pool:
        done.extend(pool.map(canonical_case,tasks))
    atomic_json(a.output/'CANONICAL_COMPLETE.json',dict(identity=prepared['identity'],cases=len(done),
        metric_index_sha256=digest(a.metric_index),runtime_versions=versions,canonical_score_used_for_target_selection=False,
        canonical_sampling_s=.1,worker_CPU_seconds=sum(r['CPU_seconds'] for r in done)))


def validate_case(task):
    case,meta,output=task;start=time.monotonic();cpu=time.process_time()
    from starVLA.dataloader.local_evidence_scene import load_local_evidence_scene
    from starVLA.model.modules.decision_local.local_constraint_checker import LocalConstraintChecker,ConstraintConfig
    receipt=read(Path(output)/'executed'/(case['key'].replace(':','_')+'.json'))
    if receipt.get('state_coordinates')!='pose_ego0_velocity_acceleration_instantaneous_body_v1':raise ValueError('Unknown simulator state coordinates')
    if receipt['input_sha256']!=case['input_sha256'] or digest(receipt['executed'])!=receipt['executed_sha256']:
        raise ValueError('Changed executed states')
    scene=load_local_evidence_scene(read(meta['anchor_record']),meta['split'],case)
    cfg=ConstraintConfig(**meta['constraints']);checker=LocalConstraintChecker(scene,cfg)
    with np.load(case['input']) as f:plan=f['plan'].copy();target=f['target'].copy();velocity=f['velocity'].copy()
    direct=checker.check(target,velocity,reference=plan)
    with np.load(receipt['executed']) as f:states=f['target'].copy();base_times=f['times'].copy()
    # Inspect the actual 0.1 s simulation and its linear half steps. This is a
    # sampled execution check, not a continuous-time safety proof.
    times=np.linspace(0,4.,81)
    poses=np.column_stack([np.interp(times,base_times,states[:,i] if i<2 else np.unwrap(states[:,2])) for i in range(3)])
    values,valid,support=checker.relations(poses,times)
    speed=np.linalg.norm(states[:,3:5],axis=-1)
    c,s=np.cos(states[:,2]),np.sin(states[:,2]);vx,vy=states[:,3],states[:,4]
    ego0_velocity=np.column_stack((vx*c-vy*s,vx*s+vy*c))
    # Numerical inertial acceleration includes turning; PDM's stored body
    # command acceleration alone omits this centripetal component.
    acceleration=np.linalg.norm(np.gradient(ego0_velocity,base_times,axis=0,edge_order=2),axis=-1)
    yaw_rate=np.abs(states[:,9]);segments=np.diff(states[:,:2],axis=0)
    tangent=np.arctan2(segments[:,1],segments[:,0]);heading=states[:-1,2]
    error=np.abs(np.arctan2(np.sin(heading-tangent),np.cos(heading-tangent)))
    error=np.where(np.linalg.norm(segments,axis=-1)/np.diff(base_times)>=.5,error,0.)
    checks=dict(strict_returned_target=bool(direct['feasible']),full_road_support=bool(valid[:,0].all()),
        full_annotated_object_support=bool(support['constraint_object_support'].all()),
        road_clearance=bool((values[:,0]>=cfg.road_margin_m).all()),
        annotated_object_clearance=bool((values[:,1]>=cfg.obstacle_clearance_m).all()),
        speed=bool(speed.max()<=cfg.max_speed_mps),acceleration=bool(acceleration.max()<=cfg.max_acceleration_mps2),
        yaw_rate=bool(yaw_rate.max()<=cfg.max_yaw_rate_rps),heading_tangent=bool(error.max()<=cfg.max_heading_tangent_error_rad))
    return dict(key=case['key'],token=case['token'],log=case['log'],anchor_index=case['anchor_index'],plan_index=case['plan_index'],
        admitted=all(checks.values()),checks=checks,rejection_reasons=[k for k,v in checks.items() if not v],
        road_margin_min_m=float(values[:,0].min()),annotated_clearance_min_m=float(values[:,1].min()),
        speed_max_mps=float(speed.max()),acceleration_max_mps2=float(acceleration.max()),yaw_rate_max_rps=float(yaw_rate.max()),
        heading_tangent_error_max_rad=float(error.max()),
        strict_target_recheck=direct,canonical_scores=receipt['canonical_scores'],
        input_sha256=case['input_sha256'],label_sha256=case['label_sha256'],executed_sha256=receipt['executed_sha256'],
        CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start)


def validate(a):
    prepared=read(a.output/'PREPARED.json');meta=read(a.output/'identity.json');complete=read(a.output/'CANONICAL_COMPLETE.json')
    if complete['identity']!=meta['identity'] or complete['cases']!=len(prepared['cases']):raise ValueError('Incomplete independent execution')
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        cases=list(pool.map(validate_case,[(case,meta,str(a.output)) for case in prepared['cases']]))
    result=dict(schema='natural_repair_execution_review_v1',labels_identity=meta['labels_identity'],
        source_sha=source_sha(ROOT),tool_sha256=digest(__file__),execution_input_identity=meta['identity'],
        canonical_receipt_sha256=digest(a.output/'CANONICAL_COMPLETE.json'),strict_REPAIR_cases=len(cases),
        accepted_REPAIR_cases=sum(r['admitted'] for r in cases),rejected_REPAIR_cases=sum(not r['admitted'] for r in cases),
        all_label_repairs_checked=True,all_admitted_repairs_independently_checked=True,
        cases={r['key']:r for r in cases},selection='Original fixed-budget target; no retry, replacement or PDMS selection',
        on_rejection='Full target loss disabled; original scene, status and local facts retained',
        KEEP_policy='Original annotated-geometry fixed-point targets unchanged; not a guarantee about tracked execution',
        scope='4 s annotated local constraints, canonical 0.1 s states plus linear 0.05 s checks; no continuous or real-world safety proof')
    result['identity']=identity(result);atomic_json(a.output/'REVIEW.json',result)
    print({k:v for k,v in result.items() if k!='cases'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','canonical','validate'))
    p.add_argument('--output',type=Path,required=True)
    for name in ('labels','features','anchor','metric-index','devkit'):p.add_argument('--'+name,type=Path)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if not 1<=a.workers<=16:raise ValueError('Finite CPU concurrency required')
    check_host();a.output.mkdir(parents=True,exist_ok=True)
    required={'prepare':('labels','features','anchor'),'canonical':('metric_index','devkit'),'validate':()}
    if any(getattr(a,key) is None for key in required[a.phase]):p.error('Missing phase input')
    start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=time.process_time()
    {'prepare':prepare,'canonical':canonical,'validate':validate}[a.phase](a)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    atomic_json(a.output/(a.phase+'_CPU.json'),dict(wall_seconds=time.monotonic()-start,parent_CPU_seconds=time.process_time()-cpu,
        waited_child_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,resource_check=check_host()))


if __name__=='__main__':main()
