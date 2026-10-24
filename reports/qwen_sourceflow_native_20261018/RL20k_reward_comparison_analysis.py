"""Read-only matched20k SOURCE minus CANONICAL; not checkpoint selection."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'/mnt/project/qwen-sourceflow-native-20261018/snapshots/RL_training_review_20261024')
import numpy as np
from scripts.analysis.qwen_sourceflow.analyze_paired import read
from scripts.analysis.simwam_recipe.analyze_evaluation import METRICS,SAFETY
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json

ROOT=Path('/mnt/project/qwen-sourceflow-native-20261018')
C=ROOT/'RL_interim20k_20261024'
reg=json.loads((C/'registration.json').read_text())
current,tokens,rows,value=read(reg['variants']['source']['output'])
reference,rt,_,base=read(reg['variants']['canonical']['output'])
il_identity,it,il_rows,il=read(reg['baseline_evaluation'])
for key in ['schema','cohort','data_sha256','token_sha256','count','metric','protocol','seed','inference','precision','source_files','parent_model_sha256']:
    assert current[key]==reference[key],key
assert tokens==rt and len(tokens)==12146
assert tokens==it and il_identity['parent_model_sha256']==current['parent_model_sha256']
assert current['rl_export']['updates']==reference['rl_export']['updates']==20000
delta=value[:,0]-base[:,0];boot=log_bootstrap_difference(delta,[r['log'] for r in rows])
old_safe=(base[:,1:4]==1).all(1);new_safe=(value[:,1:4]==1).all(1)
result=dict(schema='native_matched20k_reward_comparison_v1',status='COMPLETE',analysis_sha256=file_sha(__file__),
    interpretation='Fixed matched20k reward-version comparison; not final50k or checkpoint selection',
    scenes=len(tokens),logs=len({r['log'] for r in rows}),seed=2101,
    mean_percent={label:{k:float(100*v[:,i].mean()) for i,k in enumerate(METRICS)} for label,v in [('CANONICAL',base),('SOURCE',value)]},
    SOURCE_minus_CANONICAL_PDMS_points=float(100*delta.mean()),
    SOURCE_minus_CANONICAL_metric_points={k:float(100*(value[:,i]-base[:,i]).mean()) for i,k in enumerate(METRICS)},
    PDMS_log_cluster_bootstrap=dict(**boot,CI95_PDMS_points=[100*x for x in boot['CI95']]),
    SOURCE_new_joint_violation_relative_CANONICAL=int((old_safe&~new_safe).sum()),
    SOURCE_joint_recovery_relative_CANONICAL=int((~old_safe&new_safe).sum()),
    SOURCE_new_zero_relative_CANONICAL=int(((base[:,0]>0)&(value[:,0]==0)).sum()),
    SOURCE_zero_recovery_relative_CANONICAL=int(((base[:,0]==0)&(value[:,0]>0)).sum()),
    original_IL_parent_sha256=current['parent_model_sha256'],safety_transitions={},
    policies={},own_IL_transitions={},
    uncertainty='Scene-weighted log bootstrap only; not training/inference seed uncertainty')
for i,key in enumerate(SAFETY,1):
    old=base[:,i]<1;new=value[:,i]<1
    result['safety_transitions'][key]=dict(CANONICAL_violations=int(old.sum()),SOURCE_violations=int(new.sum()),new_in_SOURCE=int((~old&new).sum()),recovered_in_SOURCE=int((old&~new).sum()))
il_safe=(il[:,1:4]==1).all(1)
for label,values in [('IL53k',il),('CANONICAL',base),('SOURCE',value)]:
    safe=(values[:,1:4]==1).all(1)
    result['policies'][label]=dict(PDMS=float(100*values[:,0].mean()),zero_scenes=int((values[:,0]==0).sum()),
        joint_safe_scenes=int(safe.sum()),full_score_scenes=int((values[:,0]==1).sum()),
        means={k:float(100*values[:,i].mean()) for i,k in enumerate(METRICS)},
        violations={key:int((values[:,i]<1).sum()) for i,key in enumerate(SAFETY,1)})
    if label!='IL53k':
        result['own_IL_transitions'][label]=dict(new_joint_violation=int((il_safe&~safe).sum()),
            joint_recovery=int((~il_safe&safe).sum()),new_zero=int(((il[:,0]>0)&(values[:,0]==0)).sum()),
            zero_recovery=int(((il[:,0]==0)&(values[:,0]>0)).sum()))
result['sha256']=fingerprint(result)
path=ROOT/'analysis/RL20k_SOURCE_minus_CANONICAL_NAVTEST_20261024.json'
if path.exists():raise FileExistsError(path)
atomic_json(path,result)
print(json.dumps(dict(status='COMPLETE',SOURCE_minus_CANONICAL_PDMS_points=result['SOURCE_minus_CANONICAL_PDMS_points'],CI95=result['PDMS_log_cluster_bootstrap']['CI95_PDMS_points'])))
