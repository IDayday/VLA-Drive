"""Read-only requested geometry and actual controller-executed behavior.

NAVSIM v1 StateIndex: global rear-axle x/y/heading, global vx/vy and ax/ay.
The requested eight poses are already in the CURRENT ego rear-axle frame.
These summaries are diagnostics, never rewards, constraints or actor inputs.
"""
import json
from pathlib import Path
import numpy as np
from .feedback.process import process_summary
from .common_analysis import load_evaluation
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha


def local_positions(states):
    states=np.asarray(states,dtype=np.float64)
    if states.shape!=(41,11) or not np.isfinite(states).all():
        raise ValueError('Expected canonical41x11 states, t0 then40 future0.1s states')
    xy=states[:,:2]-states[0,:2]
    c,s=np.cos(states[0,2]),np.sin(states[0,2])
    return xy@np.array([[c,-s],[s,c]])


def time_to_distance(states, distance):
    if not np.isfinite(distance) or distance<.1:return None
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(np.asarray(states)[:,:2],axis=0),axis=1))]
    reached=np.flatnonzero(arc>=distance)
    if not len(reached):return None
    i=int(reached[0])
    return float(.1*(i-1+(distance-arc[i-1])/(arc[i]-arc[i-1])))


def features(row):
    result=process_summary(row['feedback'])['physical_feedback']
    states=np.asarray(row['simulated_trajectory'],dtype=np.float64)
    local=local_positions(states);plan=np.asarray(row['planned_trajectory'],dtype=np.float64)
    if plan.shape!=(8,3):raise ValueError('Requested poses must remain8x3 local rear-axle poses')
    result=dict(result)
    # Do not duplicate a time grid in every aggregate feature vector.
    result.pop('driving_time_seconds')
    result.update(simulated_arc_length_m=float(np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1).sum()),
                  simulated_final_forward_m=float(local[-1,0]),simulated_final_lateral_m=float(local[-1,1]),
                  requested_final_forward_m=float(plan[-1,0]),requested_final_lateral_m=float(plan[-1,1]),
                  requested_mean_abs_lateral_m=float(np.abs(plan[:,1]).mean()))
    return result


def pair_features(current, reference):
    a=np.asarray(current['simulated_trajectory'],dtype=np.float64)
    b=np.asarray(reference['simulated_trajectory'],dtype=np.float64)
    if not np.allclose(a[0,:3],b[0,:3],atol=1e-7,rtol=0):
        raise ValueError('Paired scenes have different immutable initial poses')
    ac,bc=features(current),features(reference)
    pa=np.asarray(current['planned_trajectory'],dtype=np.float64)
    pb=np.asarray(reference['planned_trajectory'],dtype=np.float64)
    distance=np.linalg.norm(pa[:,:2]-pb[:,:2],axis=-1)
    angle=(pa[:,2]-pb[:,2]+np.pi)%(2*np.pi)-np.pi
    target=.9*bc['simulated_arc_length_m']
    return dict(current=ac,reference=bc,
        request_ADE_m=float(distance.mean()),request_FDE_m=float(distance[-1]),
        request_RMS_heading_rad=float(np.sqrt(np.mean(angle**2))),
        current_time_to_reference_90percent_path_m=time_to_distance(a,target),
        reference_time_to_reference_90percent_path_m=time_to_distance(b,target),
        distance_target_m=target if target>=.1 else None)


def summarize_vectors(vectors):
    keys=sorted(set().union(*(v.keys() for v in vectors)))
    result={}
    for key in keys:
        values=[v[key] for v in vectors if v.get(key) is not None]
        if not values:
            result[key]=dict(available_scenes=0,mean=None);continue
        result[key]=dict(available_scenes=len(values),mean=float(np.mean(values)),
                         quantiles=dict(zip(['p05','p50','p95'],np.quantile(values,[.05,.5,.95]).tolist())))
    return result


def analyze_behavior(current_path, reference_path, protocol):
    current=load_evaluation(current_path,protocol);reference=load_evaluation(reference_path,protocol)
    for key in ['tokens','seeds','logs','scorer_sha256','sampler_sha256']:
        if getattr(current,key)!=getattr(reference,key):raise ValueError('Unmatched behavior population: '+key)
    def records(path):
        return {(r['scene_token'],r['seed']):r for r in
                (json.loads(p.read_text()) for p in (Path(path)/'scenes').glob('*.json'))}
    left,right=records(current_path),records(reference_path)
    scene_rows=[]
    for token,log in zip(current.tokens,current.logs):
        pairs=[pair_features(left[(token,seed)],right[(token,seed)]) for seed in current.seeds]
        # Average seeds within a scene; undefined/censored times stay explicit.
        row=dict(scene_token=token,log_id=log)
        for side in ['current','reference']:
            for name in pairs[0][side]:
                values=[p[side][name] for p in pairs if p[side][name] is not None]
                row[side+'_'+name]=float(np.mean(values)) if values else None
        for name in ['request_ADE_m','request_FDE_m','request_RMS_heading_rad']:
            row[name]=float(np.mean([p[name] for p in pairs]))
        valid=[p for p in pairs if p['distance_target_m'] is not None]
        reached=[p for p in valid if p['current_time_to_reference_90percent_path_m'] is not None]
        row['current_reached_reference_distance_fraction']=len(reached)/len(valid) if valid else None
        row['reference_distance_defined_fraction']=len(valid)/len(pairs)
        row['current_time_to_reference_distance_given_reached_seconds']=(float(np.mean([p['current_time_to_reference_90percent_path_m'] for p in reached])) if reached else None)
        row['canonical_score_delta']=float(np.mean([left[(token,s)]['raw_reward']-right[(token,s)]['raw_reward'] for s in current.seeds]))
        scene_rows.append(row)
    summary=dict(schema='three_route_readonly_physical_behavior_v1',status='COMPLETE',
        protocol_sha256=protocol['sha256'],current_identity=current.identity['sha256'],reference_identity=reference.identity['sha256'],
        source_sha256=file_sha(__file__),scenes=len(scene_rows),logs=len(set(current.logs)),seeds=list(current.seeds),
        metrics=summarize_vectors([{k:v for k,v in r.items() if k not in ['scene_token','log_id']} for r in scene_rows]),
        semantics=['Requested8-point local rear-axle geometry and actual41-state global simulated consequences remain distinct',
                   'Executed dynamics use existing process_summary; no request-waypoint jerk as a safety proxy',
                   'Time-to-distance targets90% of reference executed path length, not official arrival or goal completion; unreached/undefined stays censored',
                   'Seeds average inside scenes; conditional time means alone are not an uncensored speed comparison',
                   'ADE/FDE describe policy displacement, not output KL or useful exploration by themselves'])
    summary['sha256']=fingerprint(summary)
    return summary,scene_rows
