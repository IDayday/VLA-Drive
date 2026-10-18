"""Complete native single-ODE evaluation, using the existing log bootstrap."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.analysis.simwam_recipe.analyze_evaluation import METRICS,SAFETY,motion
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.qwen_sourceflow.contracts import POPULATION,fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def read(path):
    path=Path(path);identity=json.loads((path/'identity.json').read_text())
    if identity['qualification'] or not identity['full_population']:raise ValueError('No qualification/subset main result')
    pop=json.loads(POPULATION.read_text());tokens=pop[{'DEV':'dev_tokens','NAVTEST':'navtest_tokens','TRAIN':'train_tokens'}[identity['cohort']]]
    if fingerprint(tokens)!=identity['token_sha256']:raise ValueError('Changed evaluation population')
    rows=[]
    for token in tokens:
        row=json.loads((path/'scenes'/f'{token}.json').read_text());sha=row.pop('sha256')
        if (fingerprint(row)!=sha or row['identity']!=identity['sha256'] or row['token']!=token or
            row['log']!=pop['token_logs'][token] or row['status']!='OK'):
            raise ValueError('Changed/missing/invalid scored scene: '+token)
        rows.append(row)
    values=np.array([[row['feedback']['metrics'][k] for k in METRICS] for row in rows],dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('Nonfinite complete population')
    return identity,tokens,rows,values


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    current,tokens,rows,value=read(a.evaluation);reference,rt,_,base=read(a.reference)
    for key in ['schema','cohort','data_sha256','token_sha256','count','metric','protocol','seed','inference','precision','source_files']:
        if current[key]!=reference[key]:raise ValueError('Unmatched native evaluation: '+key)
    if tokens!=rt:raise ValueError('Unpaired population')
    if (reference['rl_export'] is not None or current['rl_export'] is None or
        current['rl_export']['parent_model_sha256']!=reference['parent_model_sha256']):
        raise ValueError('RL must compare against its exact IL parent')
    delta=value[:,0]-base[:,0];safe=(base[:,1:4]==1).all(1);moves=[motion(row['feedback']) for row in rows]
    result=dict(schema='native_Qwen_sourceflow_paired_v1',status='COMPLETE',scenes=len(tokens),
        logs=len({r['log'] for r in rows}),seed=2101,training_seed_repeats=1,inference_seeds=1,
        evaluation=a.evaluation,reference=a.reference,analysis_sha256=file_sha(__file__),
        mean_percent={k:float(100*value[:,i].mean()) for i,k in enumerate(METRICS)},
        own_IL_PDMS=float(100*base[:,0].mean()),RL_PDMS=float(100*value[:,0].mean()),
        RL_minus_own_IL_PDMS_points=float(100*delta.mean()),reached92=bool(100*value[:,0].mean()>=92),
        PDMS_log_cluster_bootstrap=log_bootstrap_difference(delta,[r['log'] for r in rows]),
        uncertainty='Scene-weighted log-cluster resampling only; not training/inference seed uncertainty',
        G_plus_raw_score_sum=float(delta[delta>0].sum()),G_minus_raw_score_sum=float(-delta[delta<0].sum()),
        net_raw_score_sum=float(delta.sum()),practical_improved_ge005=int((delta>=.005).sum()),
        practical_degraded_le_minus005=int((delta<=-.005).sum()),any_improved=int((delta>0).sum()),any_degraded=int((delta<0).sum()),
        nonzero_to_zero=int(((base[:,0]>0)&(value[:,0]==0)).sum()),zero_to_nonzero=int(((base[:,0]==0)&(value[:,0]>0)).sum()),
        fixed_IL_safe_scenes=int(safe.sum()),fixed_IL_safe_new_joint_violation=int((safe&(value[:,1:4]<1).any(1)).sum()),
        score_quantiles=dict(zip(['p01','p05','p10','p50'],np.quantile(value[:,0],[.01,.05,.1,.5]).tolist())),safety_transitions={})
    for i,key in enumerate(SAFETY,1):
        old=base[:,i]<1;new=value[:,i]<1
        result['safety_transitions'][key]=dict(new_violation=int((~old&new).sum()),recovery=int((old&~new).sum()),mean_delta=float((value[:,i]-base[:,i]).mean()))
    result['motion']={k:dict(mean=float(np.mean(v)) if v else None,defined_scenes=len(v)) for k in moves[0] for v in [[m[k] for m in moves if m[k] is not None]]}
    if Path(a.output).exists():raise FileExistsError(a.output)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    result['sha256']=fingerprint(result);atomic_json(a.output,result)
    print(json.dumps({k:result[k] for k in ['status','scenes','own_IL_PDMS','RL_PDMS','RL_minus_own_IL_PDMS_points','reached92']}))


if __name__=='__main__':main()
