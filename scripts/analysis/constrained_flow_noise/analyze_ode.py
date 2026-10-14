"""Single-seed ODE deployment analysis; no candidate selection or policy fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.analysis.flow_safe_improvement.evaluate_policy import load_scores
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, fingerprint, file_sha, SFT_SHA256
from starVLA.rl.flow_grpo.transactions import atomic_json

SAFETY=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')


def motion(root,tokens):
    values={k:[] for k in ('requested_forward_end_m','simulated_forward_end_m','simulated_max_abs_lateral_m',
                          'simulated_mean_speed_mps','simulated_final_speed_mps','request_second_difference_m','official_progress_m')}
    longitudinal_paths=[]
    for token in tokens:
        row=torch.load(Path(root)/'scenes'/token[:2]/(token+'.pt'),weights_only=True,map_location='cpu')
        feedback=row['feedback'][0]
        request=np.asarray(feedback['requested_trajectory']);states=np.asarray(feedback['simulated_states'])
        if states.shape!=(41,11):raise ValueError('Locked official state format changed')
        x,y,heading=states[0,:3];c,s=np.cos(heading),np.sin(heading)
        dx,dy=states[:,0]-x,states[:,1]-y
        longitudinal,lateral=c*dx+s*dy,-s*dx+c*dy
        longitudinal_paths.append(longitudinal)
        speed=np.linalg.norm(states[:,3:5],axis=-1)
        for key,value in [('requested_forward_end_m',request[-1,0]),('simulated_forward_end_m',longitudinal[-1]),
                          ('simulated_max_abs_lateral_m',np.abs(lateral[1:]).max()),('simulated_mean_speed_mps',speed[1:].mean()),
                          ('simulated_final_speed_mps',speed[-1]),('request_second_difference_m',np.linalg.norm(np.diff(request[:,:2],n=2,axis=0),axis=-1).mean()),
                          ('official_progress_m',feedback['raw_progress_meters'])]:values[key].append(float(value))
    return {k:np.array(v) for k,v in values.items()},np.asarray(longitudinal_paths)


def arrival_at_distance(path, distance):
    """First crossing on physical .1s grid; horizon-censored is a separate event."""
    if distance<=0 or np.asarray(path).shape!=(41,):raise ValueError('Positive fixed distance and 41 simulated states required')
    crossed=np.flatnonzero(np.asarray(path)>=distance)
    if not len(crossed):return 4.,True
    i=int(crossed[0])
    if i==0:return 0.,False
    return (i-1+(distance-path[i-1])/(path[i]-path[i-1]))*.1,False


def analyze(scores,baseline_scores,pop,previous=None):
    for root in (scores,baseline_scores):
        if not (Path(root)/'COMPLETE').is_file():raise ValueError('Full evaluation required; failed/missing scenes stay in denominator')
    manifest,current,current_sha=load_scores(scores,pop,'evaluation',[2101])
    base_manifest,baseline,base_sha=load_scores(baseline_scores,pop,'evaluation',[2101])
    if base_manifest['sample_manifest']['behavior_sha256']!=SFT_SHA256:raise ValueError('Fixed SFT deployment reference required')
    if (manifest['protocol_sha256']!=base_manifest['protocol_sha256'] or
        manifest['sample_manifest']['execution_profile']!=base_manifest['sample_manifest']['execution_profile'] or set(current)!=set(baseline)):
        raise ValueError('Unmatched model execution, scorer or scene population')
    tokens=list(current);logs=[pop['token_logs'][t] for t in tokens]
    metrics={};safety={}
    for key in current[tokens[0]]:
        c=np.array([current[t][key][0] for t in tokens]);b=np.array([baseline[t][key][0] for t in tokens])
        metrics[key]=dict(current=float(c.mean()),SFT=float(b.mean()),paired=log_bootstrap_difference(c-b,logs),
                          current_quantiles_0_5_10_50_90_100=np.quantile(c,[0,.05,.1,.5,.9,1]).tolist(),
                          any_negative_difference=int((c<b).sum()),decline_at_least_0_005=int((c-b<=-.005).sum()))
    for key in (*SAFETY,'joint'):
        fields=SAFETY if key=='joint' else (key,)
        c=np.array([all(current[t][f][0]==1 for f in fields) for t in tokens]);b=np.array([all(baseline[t][f][0]==1 for f in fields) for t in tokens])
        safety[key]=dict(current_violations=int((~c).sum()),SFT_violations=int((~b).sum()),
                         new_violations_in_fixed_SFT_pass_population=int((b&~c).sum()),recoveries=int((~b&c).sum()),
                         fixed_SFT_pass_population=int(b.sum()),violation_difference=log_bootstrap_difference((~c).astype(float)-(~b).astype(float),logs))
    c=np.array([current[t]['score'][0]==0 for t in tokens]);b=np.array([baseline[t]['score'][0]==0 for t in tokens])
    zeros=dict(current=int(c.sum()),SFT=int(b.sum()),nonzero_to_zero=int((~b&c).sum()),zero_to_nonzero=int((b&~c).sum()),
               seed_semantics='one fixed seed: any/all/scene-mean-zero coincide; no multi-seed robustness claim')
    (cm,cp),(bm,bp)=motion(scores,tokens),motion(baseline_scores,tokens)
    motions={k:dict(current=float(cm[k].mean()),SFT=float(bm[k].mean()),paired=log_bootstrap_difference(cm[k]-bm[k],logs)) for k in cm}
    eligible_arrival=np.flatnonzero(bp[:,-1]>=1.)
    arrival=[]
    for i in eligible_arrival:
        threshold=.5*bp[i,-1]
        current_time,current_censored=arrival_at_distance(cp[i],threshold)
        base_time,base_censored=arrival_at_distance(bp[i],threshold)
        arrival.append((current_time,base_time,current_censored,base_censored))
    times=np.asarray(arrival)
    arrival_report=dict(definition='Time to first cross half the fixed SFT final forward displacement; include SFT forward end >=1m; physical .1s interpolation',
                        eligible_scenes=len(arrival),ineligible_scenes=len(tokens)-len(arrival),
                        censoring='Unreached crossing is flagged and capped at horizon4s; capped time is not an observed arrival')
    if len(times):
        arrival_report.update(current_capped_mean_seconds=float(times[:,0].mean()),SFT_capped_mean_seconds=float(times[:,1].mean()),
                              current_not_reached=int(times[:,2].sum()),SFT_not_reached=int(times[:,3].sum()),
                              paired_capped_seconds=log_bootstrap_difference(times[:,0]-times[:,1],[logs[i] for i in eligible_arrival]))
    pause=[]
    for key in SAFETY:
        d=safety[key]['violation_difference'];prior=previous['safety'][key]['violation_difference'] if previous else None
        if d['mean']>=.02 or (prior and d['mean']>=.005 and d['CI95'][0]>0 and prior['mean']>=.005 and prior['CI95'][0]>0):pause.append(key)
    eligible=all(safety[k]['current_violations']<=safety[k]['SFT_violations'] for k in SAFETY) and zeros['current']<=zeros['SFT']
    result=dict(schema='cfn_single_ODE_report_v1',scenes=len(tokens),logs=len(set(logs)),seed=2101,training_seed=42,
                population_identity=pop['identity_sha256'],protocol_sha256=manifest['protocol_sha256'],
                behavior_sha256=manifest['sample_manifest']['behavior_sha256'],score_payloads_sha256=current_sha,baseline_payloads_sha256=base_sha,
                deployment='original_VLAAgent_current_observation_single_10_step_ODE',metrics=metrics,safety=safety,zeros=zeros,motion=motions,
                arrival=arrival_report,
                safe_DEV_selection_eligible=eligible,severe_safety_pause_components=pause,
                uncertainty='log-cluster bootstrap with scene weights; no training/inference-seed uncertainty',
                reference_safety='fixed original SFT seed2101 population; not all-noise or base-pretraining-unseen safety')
    result['sha256']=fingerprint(result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--scores',required=True);p.add_argument('--baseline-scores',required=True)
    p.add_argument('--population',required=True);p.add_argument('--output',required=True);p.add_argument('--previous-report')
    a=p.parse_args();require_dispatch()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    previous=json.loads(Path(a.previous_report).read_text()) if a.previous_report else None
    report=analyze(a.scores,a.baseline_scores,json.loads(Path(a.population).read_text()),previous)
    atomic_json(out,report)
    print(json.dumps(dict(PDMS=100*report['metrics']['score']['current'],scenes=report['scenes'],
                         eligible=report['safe_DEV_selection_eligible'],pause=report['severe_safety_pause_components'])),flush=True)


if __name__=='__main__':main()
