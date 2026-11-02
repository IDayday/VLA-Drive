"""Explicit synthetic TRAIN-only progress proposals,112 CPU queries maximum."""
from concurrent.futures import as_completed
from collections import Counter
from datetime import datetime,timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
P=Path('/mnt/project/qwen-world-performance-rl-20261103')
SOURCE=P/'runtime/formal_v1'
S6=P/'runs/S6_VERIFIED_QUALITY'
PREVIOUS=P/'analysis/S6_FIRST_WINDOW_GT_FALLBACK_AUDIT'
OUT=P/'analysis/S6_FIRST_WINDOW_SYNTHETIC_PROGRESS_AUDIT'
sys.path.insert(0,str(P/'analysis/VERIFIED_TRAIN_GT_TEACHER_AUDIT'))
import audit_verified_gt as A
import numpy as np
import torch
from scipy.interpolate import CubicSpline,CubicHermiteSpline,PchipInterpolator
from starVLA.rl.qwen_world_research_data.manifest import atomic_json,file_sha,fingerprint,read_sealed
from starVLA.rl.qwen_world_research_data.data import ResearchSceneData
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.simwam_recipe.reward_readouts import RewardService

SCHEMA='world_S6_TRAIN7_explicit_synthetic_progress_teacher_diagnostic_v1'
DELTAS=(.5,1.,1.5,2.,3.,4.,5.,6.)
T=4.
TIMES=np.arange(1,9,dtype=np.float64)*.5


def seal(x):
    x=A.native(x);return dict(x,sha256=fingerprint(x))


def path_and_time(base,current_vx):
    points=np.vstack((np.zeros((1,2)),np.asarray(base,dtype=np.float64)[:,:2]))
    chords=np.linalg.norm(np.diff(points,axis=0),axis=1)
    s=np.r_[0.,np.cumsum(chords)]
    # Stationary knots have no new path geometry. Keep their timing knots in
    # s_base(t), but do not feed duplicate abscissae into the spatial spline.
    keep=np.r_[True,np.diff(s)>1e-6]
    if keep.sum()<3:raise ValueError('Insufficient distinct path geometry; no synthetic label guessing')
    xy=points[keep];ss=s[keep];yaw=float(np.asarray(base)[-1,2])
    spatial=CubicSpline(ss,xy,axis=0,bc_type=((1,np.array([1.,0.])),(1,np.array([np.cos(yaw),np.sin(yaw)]))))
    knots=np.arange(9)*.5
    derivative=PchipInterpolator(knots,s).derivative()(knots)
    derivative[0]=float(current_vx)
    temporal=CubicHermiteSpline(knots,s,derivative)
    dense=np.linspace(0,T,801)
    if current_vx<0 or temporal(dense,1).min() < -1e-6:raise ValueError('Cannot satisfy monotone forward progress and current initial vx with this base')
    length=float(ss[-1]);endpoint=spatial(length);tangent=spatial(length,1);second=spatial(length,2)
    theta=float(np.arctan2(tangent[1],tangent[0]))
    curvature=float((tangent[0]*second[1]-tangent[1]*second[0])/np.linalg.norm(tangent)**3)
    def geometry(progress):
        progress=np.asarray(progress);inside=progress<=length
        xy=spatial(np.minimum(progress,length));direction=spatial(np.minimum(progress,length),1)
        angle=np.arctan2(direction[:,1],direction[:,0])
        u=progress[~inside]-length
        if len(u):
            a=theta+curvature*u
            if abs(curvature)<1e-9:
                xy[~inside]=endpoint+u[:,None]*np.array([np.cos(theta),np.sin(theta)])
            else:
                xy[~inside]=endpoint+np.column_stack(((np.sin(a)-np.sin(theta))/curvature,(np.cos(theta)-np.cos(a))/curvature))
            angle[~inside]=a
        return xy,np.arctan2(np.sin(angle),np.cos(angle))
    initial_velocity=spatial(0.,1)*temporal(0.,1)
    if not np.allclose(initial_velocity,[current_vx,0.],rtol=0,atol=1e-9):raise ValueError('Initial bicycle-model velocity continuity failed')
    return temporal,geometry,dict(path_coordinate='cumulative-chord metre parameter; not claimed exact arc length between knots',
        path_length_parameter=length,terminal_heading=theta,terminal_curvature_per_m=curvature,
        initial_velocity_xy=initial_velocity.tolist(),initial_forward_vx_exact=True,
        baseline_initial_path_acceleration=float(temporal(0.,2)),monotone_base_speed_min=float(temporal(dense,1).min()))


def proposals(base,vx,family):
    temporal,geometry,meta=path_and_time(base,vx)
    requested=[];clean=[];details=[];dense=np.linspace(0,T,801)
    for delta in DELTAS:
        progress=temporal(TIMES)+delta*(TIMES/T)**3
        xy,yaw=geometry(progress)
        physical=np.column_stack((xy,yaw)).astype(np.float32)
        normalized=ActionCodec.norm_odo(torch.from_numpy(physical))
        actual=ActionCodec.denorm_odo(normalized).numpy()
        if not np.isfinite(actual).all():raise ValueError('Nonfinite synthesized roundtrip action')
        dd=temporal(dense)+delta*(dense/T)**3
        dense_xy,dense_yaw=geometry(dd)
        vel=np.gradient(dense_xy,dense,axis=0);acc=np.gradient(vel,dense,axis=0)
        yawrate=np.gradient(np.unwrap(dense_yaw),dense)
        requested.append(actual);clean.append(normalized.numpy())
        details.append(dict(family=family,synthetic=True,not_logged_GT=True,delta_progress_at4_m=delta,
            formula='s_new(t)=s_base(t)+delta*(t/4)^3, t in [0,4] seconds',
            added_initial_position_velocity_acceleration=[0.,0.,0.],
            added_terminal_path_speed=3*delta/4,added_max_path_acceleration=6*delta/16,
            synthetic_extension_past_last_known_path_m=float(max(0,progress[-1]-meta['path_length_parameter'])),
            dense_speed_max_m_per_s=float(np.linalg.norm(vel,axis=1).max()),
            dense_total_acceleration_max_m_per_s2=float(np.linalg.norm(acc,axis=1).max()),
            dense_yaw_rate_abs_max_rad_per_s=float(np.abs(yawrate).max()),
            dense_kinematics='proposal interpolation diagnostics, not actual simulator vehicle states or official comfort proof',
            physical_before_codec_typed_sha256=A.typed_sha(physical),actual_requested_typed_sha256=A.typed_sha(actual),
            raw_to_roundtrip_max_abs_by_dimension=np.max(np.abs(actual-physical),axis=0).tolist(),**meta))
    return requested,clean,details


def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True);(OUT/'scenes').mkdir(exist_ok=True)
    lock=(OUT/'audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'BUDGET_LEDGER.json').exists():raise RuntimeError('Never repeat an ambiguous reserved/scored budget')
    old=read_sealed(PREVIOUS/'PREPARED.json');oldprotocol=read_sealed(PREVIOUS/'PROTOCOL.json');complete=read_sealed(PREVIOUS/'COMPLETE.json')
    if complete['status']!='COMPLETE':raise ValueError('Sealed GT64 source required')
    targets={t:r for t,r in old['rows'].items() if r['fallback_no_sampled_strict_high']}
    if len(targets)!=7 or any(not r['eligible_for_scoring'] for r in targets.values()):raise ValueError('Exact seven valid TRAIN scenes required')
    data=ResearchSceneData('/mnt/project/qwen-world-rl-optimization-20261030/data/NAVTRAIN',role='NAVTRAIN')
    if data.sha!=oldprotocol['data_manifest_sha256']:raise ValueError('Current-ego input source changed')
    windows={};inputs={};arrays={};behavior_hashes={}
    for t,r in targets.items():
        rank=r['rank'];path=S6/'behavior/collection000001'/f'rank{rank}.pt'
        if rank not in windows:
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if digest!=oldprotocol['behavior_file_sha256'][str(path)]:raise ValueError('Exact initial behavior changed')
            windows[rank]=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);behavior_hashes[str(path)]=digest
        leaf=next(s for s in windows[rank]['samples'] if s['token']==t)
        nominal=leaf['nominal_improvement']['nominal'];extra=leaf['nominal_improvement']['extra']
        if extra is None or len(extra['records'])!=16 or any(A.strict_high(v['metrics']) for v in extra['records']):raise ValueError('Expected same-scene unsuccessful extraG16 pool')
        b1=np.asarray(nominal['feedback']['requested_trajectory'],dtype=np.float32)
        if A.typed_sha(b1)!=r['B1_actual_requested_sha256']:raise ValueError('Actual B1 trajectory differs')
        ego=np.asarray(data.records[t]['proprio'][:4],dtype=np.float64)
        req=[];clean=[];description=[]
        for family,base in [('GT_path',np.asarray(r['raw_GT'],dtype=np.float32)),('B1_path',b1)]:
            rr,cc,dd=proposals(base,float(ego[0]),family);req.extend(rr);clean.extend(cc);description.extend(dd)
        if len({A.typed_sha(x) for x in req})!=16:raise ValueError('Duplicate proposals: do not waste real queries')
        arrays[t]=np.stack(req)
        inputs[t]=dict(token=t,log=r['log'],cache=r['cache'],B1_metrics=r['B1_metrics'],GT_roundtrip_metrics=read_sealed(PREVIOUS/'scenes'/(t+'.json'))['GT_metrics'],
            current_ego_vx_vy_ax_ay=ego.tolist(),current_ego_source='actual NAVTRAIN current proprio; no extra future beyond audited GT8',
            initial_velocity_constraint='official bicycle uses dx/dt=vx*cos(heading),dy/dt=vx*sin(heading); local heading0 gives[vx,0]. Logged vy is recorded, not falsely claimed matched by the no-slip pose dynamics.',
            source_version=r['source_version'],baseline_extra16_records=A.native(extra['records']),
            descriptors=description,clean_normalized=np.stack(clean),actual_requests=arrays[t],
            synthetic_proposals=16,all_training_or_deployment_eligible=False)
    protocol=seal(dict(schema=SCHEMA,status='PREPARED_CPU_DIAGNOSTIC_ONLY',script_sha256=file_sha(__file__),
        preceding_GT64_complete_sha256=file_sha(PREVIOUS/'COMPLETE.json'),preceding_protocol_sha256=oldprotocol['sha256'],behavior_file_sha256=behavior_hashes,
        scoring_source=str(SOURCE),source_files_sha256={n:file_sha(SOURCE/n) for n in ('starVLA/model/modules/qwen_sourceflow/action_codec.py',
            'starVLA/rl/simwam_recipe/reward_readouts.py','scripts/analysis/constrained_flow_noise/batched_canonical_score.py',
            'navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/simulation/batch_kinematic_bicycle.py')},
        current_ego_manifest_sha256=data.sha,proposal_families=['GT_path','B1_path'],deltas_at4_m=list(DELTAS),
        units=dict(xy='metre, rear-axle local',yaw='radian',time='second',path_coordinate='metre cumulative-chord coordinate'),
        source_points='only known0..4s GT8/current B1/current ego; no later GT, map, obstacle or signal query for proposal construction',
        geometry='CubicSpline xy(cumulative chord), first tangent[1,0], terminal tangent from existing last yaw; yaw derived from this same xy derivative. Beyond known terminal path use explicit constant terminal-curvature circular extrapolation.',
        timing='CubicHermite s_base(t) through true0,.5..4s knots, PCHIP interior slopes and initial derivative=current vx; s_new=s_base+delta*(t/4)^3.',
        continuity='origin/heading0 and source bicycle initial forward velocity exact; delta leaves interpolated baseline initial acceleration unchanged, not a claim that baseline initial acceleration matches measured ax.',
        added_progress_speed_and_accel_bounds='delta<=6m gives terminal extra progress speed<=4.5m/s and extra path-coordinate acceleration<=2.25m/s²; total vehicle acceleration and comfort are NOT guaranteed.',
        actual_scored_request='CPU FP32 ActionCodec denorm(norm(synthetic_xy_yaw_FP32)),8 poses at fixed0.5..4s',
        teacher='actualNC=DAC=TTC=Comfort1 AND canonical>=.9 AND gain_vs_real_B1>=.02; unchanged',
        input_scenes=7,proposals_per_scene=16,maximum_new_CPU_PDM_queries=112,workers=2,new_GPU=0,training_updates=0,
        comparison='same7 prior independent extraG16 pools have112 scored candidates; synthetic uses privileged offline TRAIN GT and different generation mechanism/cost, not a learned policy or fair observation-only sampler ablation',
        on_policy_advantage_eligible=False,formal_S6_mutations=False))
    atomic_json(OUT/'PROTOCOL.json',protocol)
    atomic_json(OUT/'PREPARED.json',seal(dict(schema=SCHEMA,rows=inputs,protocol_sha256=protocol['sha256'])))
    np.savez_compressed(OUT/'TRAJECTORIES.npz',tokens=np.array(list(inputs)),actual_requested_FP32=np.stack([arrays[t] for t in inputs]),
        clean_normalized_FP32=np.stack([inputs[t]['clean_normalized'] for t in inputs]))
    ledger=dict(schema=SCHEMA,maximum=112,reserved=0,completed=0,scenes={});atomic_json(OUT/'BUDGET_LEDGER.json',ledger)
    service=RewardService(SOURCE,workers=2,cache_roots=data.manifest['canonical_metric_cache_roots']);outputs=[]
    try:
        futures={}
        for t,r in inputs.items():
            if ledger['reserved']+16>112:raise ValueError('Synthetic query cap')
            ledger['reserved']+=16;ledger['scenes'][t]=dict(status='RESERVED',queries=16);atomic_json(OUT/'BUDGET_LEDGER.json',ledger)
            futures[service.submit((t,r['log'],arrays[t]))]=t
        for future in as_completed(futures):
            t=futures[future];r=inputs[t];records=future.result()
            if len(records)!=16:raise ValueError('Lost proposal')
            for i,record in enumerate(records):
                if record['token']!=t or record['log']!=r['log'] or record['cache']['sha256']!=r['cache']['sha256'] or A.typed_sha(record['requested_trajectory'])!=r['descriptors'][i]['actual_requested_typed_sha256']:raise ValueError('Actual scored synthetic provenance differs')
                if record['protocol_sha256']!=oldprotocol['scoring_protocol']['sha256']:raise ValueError('Scorer protocol changed')
            def counts(pool):
                return dict(candidates=len(pool),strict4=sum(A.strict4(q['metrics']) for q in pool),strict_high=sum(A.strict_high(q['metrics']) for q in pool),
                    qualifying_teacher=sum(A.strict_high(q['metrics']) and q['metrics']['score']-r['B1_metrics']['score']>=.02 for q in pool),
                    NC_failed=sum(q['metrics']['no_at_fault_collisions']!=1 for q in pool),DAC_failed=sum(q['metrics']['drivable_area_compliance']!=1 for q in pool),
                    TTC_failed=sum(q['metrics']['time_to_collision_within_bound']!=1 for q in pool),Comfort_failed=sum(q['metrics']['comfort']!=1 for q in pool),
                    best_score=max(q['metrics']['score'] for q in pool),best_strict4_score=max((q['metrics']['score'] for q in pool if A.strict4(q['metrics'])),default=None))
            stats=counts(records);qualified=[i for i,q in enumerate(records) if A.strict_high(q['metrics']) and q['metrics']['score']-r['B1_metrics']['score']>=.02]
            row=seal(dict(r,records=records,statistics=stats,GT_path_statistics=counts(records[:8]),B1_path_statistics=counts(records[8:]),
                old_extra_G16_statistics=counts(r['baseline_extra16_records']),qualifying_indices=qualified,
                best_qualified_index=max(qualified,key=lambda i:records[i]['metrics']['score'],default=None)))
            atomic_json(OUT/'scenes'/(t+'.json'),row);outputs.append(row)
            ledger['completed']+=16;ledger['scenes'][t]['status']='COMPLETE';atomic_json(OUT/'BUDGET_LEDGER.json',ledger)
    except Exception as exc:
        atomic_json(OUT/'FAILED.json',seal(dict(schema=SCHEMA,type=type(exc).__name__,message=str(exc),budget=ledger)));raise
    finally:service.close()
    outputs.sort(key=lambda r:list(inputs).index(r['token']))
    metrics={}
    for field in ('statistics','GT_path_statistics','B1_path_statistics','old_extra_G16_statistics'):
        metrics[field]={k:sum(r[field][k] for r in outputs) for k in ('candidates','strict4','strict_high','qualifying_teacher','NC_failed','DAC_failed','TTC_failed','Comfort_failed')}
        metrics[field]['scenes_with_qualifying_teacher']=sum(r[field]['qualifying_teacher']>0 for r in outputs)
    summary=seal(dict(schema=SCHEMA,status='COMPLETE_CPU_SYNTHETIC_DIAGNOSTIC_ONLY',protocol_sha256=protocol['sha256'],
        cohorts=metrics,cases=[dict(token=r['token'],log=r['log'],B1_metrics=r['B1_metrics'],GT_metrics=r['GT_roundtrip_metrics'],
            statistics=r['statistics'],GT_path=r['GT_path_statistics'],B1_path=r['B1_path_statistics'],old_extra_G16=r['old_extra_G16_statistics'],
            qualifying_indices=r['qualifying_indices'],best_qualified_index=r['best_qualified_index'],
            best_qualified_descriptor=None if r['best_qualified_index'] is None else r['descriptors'][r['best_qualified_index']]) for r in outputs],
        budget=ledger,no_training_or_native_mutation=True,
        limitations=['Synthetic retiming/extrapolation is not actual GT or a model-policy sample; saved teacher may only be considered offline in a separately registered method.',
            'No map/obstacle/traffic input was used to construct proposals; safety/comfort determined only by actual official scorer.',
            'Best-of16 diagnostic is not deployment or learned policy PDMS; this fixed7 set has no population/sampling-seed uncertainty claim.']))
    atomic_json(OUT/'SUMMARY.json',summary)
    atomic_json(OUT/'COMPLETE.json',seal(dict(schema=SCHEMA,status='COMPLETE',actual_CPU_PDM_queries=112,maximum=112,
        files_sha256={n:file_sha(OUT/n) for n in ('audit_synthetic_progress.py','PROTOCOL.json','PREPARED.json','TRAJECTORIES.npz','SUMMARY.json','BUDGET_LEDGER.json')})))
    print(json.dumps(dict(status='COMPLETE',cohorts=metrics,actual_queries=112)),flush=True)


if __name__=='__main__':main()
