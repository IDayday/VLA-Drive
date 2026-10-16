"""Complete-population single-ODE comparison; no best-of-two grid selection."""
import argparse
from scripts.analysis.simwam_recipe.cost_ledger import evaluation_cost
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.rl.simwam_recipe.contracts import POPULATION,file_sha,fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.simwam_recipe.evaluation_compatibility import match_protocol

METRICS=['score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance']
SAFETY=METRICS[1:4]


def tokens_for(identity,pop):
    if identity['cohort']=='DEV':return pop['dev_tokens']
    if identity['cohort']=='NAVTEST':return pop['navtest_tokens']
    path=Path('/mnt/project/flow-safe-improvement-20261013/evaluations/original_export/SFT_TRAIN_D5_samples/manifest.json')
    return json.loads(path.read_text())['explicit_tokens']


def motion(record):
    request=np.asarray(record['requested_trajectory'],dtype=np.float64)
    states=np.asarray(record['simulated_states'],dtype=np.float64)
    if states.shape!=(41,11) or request.shape!=(8,3):raise ValueError('Changed actual simulator/request contract')
    # Only the locked simulator's SE2 columns [x,y,heading] are needed.
    offset=states[:,0:2]-states[:1,0:2];heading=states[0,2]
    forward=offset@np.array([np.cos(heading),np.sin(heading)])
    lateral=offset@np.array([-np.sin(heading),np.cos(heading)])
    path=np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1)
    cumulative=np.cumsum(path)
    t90=float((np.searchsorted(cumulative,.9*cumulative[-1])+1)*.1) if cumulative[-1]>1e-6 else None
    req_v=np.diff(np.vstack([np.zeros((1,2)),request[:,:2]]),axis=0)/.5
    req_a=np.diff(req_v,axis=0)/.5;req_jerk=np.diff(req_a,axis=0)/.5
    return dict(request_final_forward_m=float(request[-1,0]),request_final_lateral_m=float(request[-1,1]),
                request_mean_segment_speed_mps=float(np.linalg.norm(req_v,axis=1).mean()),
                request_segment_jerk_RMS_mps3=float(np.sqrt(np.mean(req_jerk**2))),
                simulated_final_forward_m=float(forward[-1]),simulated_abs_final_lateral_m=float(abs(lateral[-1])),
                simulated_mean_segment_speed_mps=float(path.mean()/.1),
                simulated_time_to_90percent_own_path_s=t90,raw_progress_meters=float(record['raw_progress_meters']))


def read(root):
    root=Path(root);identity=json.loads((root/'manifest.json').read_text())
    if identity.get('qualification_only'):raise ValueError('Qualification subset cannot become full evaluation')
    pop=json.loads(Path(POPULATION).read_text());tokens=tokens_for(identity,pop)
    if fingerprint(tokens)!=identity['tokens_sha256'] or len(tokens)!=identity['scenes']:raise ValueError('Population mismatch')
    ranks=[json.loads((root/f'rank{i}.json').read_text()) for i in range(identity['shards'])]
    if any(r['status']!='ASSIGNED_COMPLETE' for r in ranks):raise ValueError('Incomplete denominator; no missing-scene removal')
    rows={mode:[] for mode in identity['modes']};motions={mode:[] for mode in identity['modes']}
    for token in tokens:
        path=root/'scenes'/token[:2]/(token+'.pt');seal=json.loads(path.with_suffix('.json').read_text())
        if seal['identity_sha256']!=identity['sha256'] or seal['payload_sha256']!=file_sha(path):raise ValueError('Evaluation record changed')
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['token']!=token or data['log']!=pop['token_logs'][token]:raise ValueError('Scene/log mismatch')
        for mode in identity['modes']:
            record=data['records'][mode]
            if record['status']!='OK' or record['metrics']!=seal['metrics'][mode]:raise ValueError('Invalid actual score')
            rows[mode].append([record['metrics'][key] for key in METRICS]);motions[mode].append(motion(record))
    arrays={mode:np.asarray(v,dtype=np.float64) for mode,v in rows.items()}
    if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite population score')
    return identity,tokens,[pop['token_logs'][t] for t in tokens],arrays,motions,ranks


def main(protocol_matcher=match_protocol):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',required=True);p.add_argument('--reference');p.add_argument('--output',required=True)
    p.add_argument('--source-compatibility',help='Explicit qualified receipt for an exact logging-only evaluator source pair')
    a=p.parse_args();identity,tokens,logs,rows,motions,ranks=read(a.evaluation)
    baseline=None;compatibility=None
    if a.reference:
        ref,rt,rl,baseline,_,_=read(a.reference)
        compatibility=protocol_matcher(identity,ref,a.source_compatibility)
        if rt!=tokens or rl!=logs:raise ValueError('Unpaired scenes')
    result=dict(schema='simwam_evaluation_analysis_v2',analysis_identity=file_sha(__file__),status='COMPLETE_POPULATION',identity=identity,reference=a.reference,scenes=len(tokens),logs=len(set(logs)),seed=2101,
                score_units='raw0to1; mean_percent multiplied100; gain/loss sums remain raw score',modes={},
                cohort_caveat='TRAIN diagnostic contains preselected difficult scenes and is not a population-unbiased estimate' if identity['cohort']=='TRAIN_DIAGNOSTIC' else None,
                cost=evaluation_cost(a.evaluation,identity,ranks),source_compatibility=compatibility)
    for mode,values in rows.items():
        ms=motions[mode];movement={}
        for key in ms[0]:
            valid=[m[key] for m in ms if m[key] is not None]
            movement[key]=dict(mean=float(np.mean(valid)) if valid else None,defined_scenes=len(valid))
        item=dict(mean={k:float(values[:,i].mean()) for i,k in enumerate(METRICS)},
                  mean_percent={k:float(100*values[:,i].mean()) for i,k in enumerate(METRICS)},
                  score_quantiles=dict(zip(['p01','p05','p10','p50'],np.quantile(values[:,0],[.01,.05,.1,.5]).tolist())),
                  zero_scenes=int((values[:,0]==0).sum()),fully_safe_scenes=int((values[:,1:4]==1).all(1).sum()),motion=movement)
        if baseline is not None:
            base=baseline[mode];delta=values[:,0]-base[:,0];safe=(base[:,1:4]==1).all(1)
            comparison=dict(PDMS_log_cluster_bootstrap=log_bootstrap_difference(delta,logs),
                positive_gain_sum=float(delta[delta>0].sum()),negative_degradation_sum=float(-delta[delta<0].sum()),
                any_improved_scenes=int((delta>0).sum()),any_degraded_scenes=int((delta<0).sum()),
                practical_improved_ge005=int((delta>=.005).sum()),practical_degraded_le_minus005=int((delta<=-.005).sum()),
                nonzero_to_zero=int(((base[:,0]>0)&(values[:,0]==0)).sum()),zero_to_nonzero=int(((base[:,0]==0)&(values[:,0]>0)).sum()),
                reference_safe_population=int(safe.sum()),reference_safe_new_joint_violation=int((safe&(values[:,1:4]<1).any(1)).sum()),
                safety_transitions={})
            for i,k in enumerate(SAFETY,1):
                old=base[:,i]<1;new=values[:,i]<1
                comparison['safety_transitions'][k]=dict(new_violation=int((~old&new).sum()),recovery=int((old&~new).sum()),
                                                       mean_delta=float((values[:,i]-base[:,i]).mean()))
            item['paired_comparison']=comparison
        result['modes'][mode]=item
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Preserve the old summary; select a new analysis output')
    atomic_json(output,result)
    print(json.dumps(dict(status=result['status'],scenes=len(tokens),PDMS_percent={m:d['mean_percent']['score'] for m,d in result['modes'].items()},output=str(output))))


if __name__=='__main__':main()
