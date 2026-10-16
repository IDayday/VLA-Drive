"""Compare the two registered ODE grids at fixed weights, scenes and root seeds.

Uses only existing complete evaluations. This is an inference-grid contrast,
never another training run, an RL gain, or a best-of-two policy score.
"""
import argparse
from pathlib import Path
import json
from scripts.analysis.simwam_recipe.analyze_evaluation import read,METRICS,SAFETY
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve existing grid analysis')
    identity,tokens,logs,values,_,_=read(a.evaluation)
    if identity['seed']!=2101 or identity['modes']!=['E_NATIVE','E_SHIFT5']:
        raise ValueError('Only registered matched single-candidate grids')
    native,shift=values['E_NATIVE'],values['E_SHIFT5'];delta=native[:,0]-shift[:,0]
    result=dict(schema='fm_transfer_fixed_weight_grid_contrast_v1',evaluation=str(a.evaluation),
        evaluation_identity=identity,analysis_sha256=file_sha(__file__),scenes=len(tokens),logs=len(set(logs)),
        contrast='E_NATIVE minus E_SHIFT5 at the same weight and fixed root',
        PDMS_raw_difference=log_bootstrap_difference(delta,logs),
        metric_mean_raw_difference={key:float((native[:,i]-shift[:,i]).mean()) for i,key in enumerate(METRICS)},
        native_nonzero_shift_zero=int(((native[:,0]>0)&(shift[:,0]==0)).sum()),
        native_zero_shift_nonzero=int(((native[:,0]==0)&(shift[:,0]>0)).sum()),
        safety={key:dict(native_pass_shift_fail=int(((native[:,i]==1)&(shift[:,i]<1)).sum()),
            native_fail_shift_pass=int(((native[:,i]<1)&(shift[:,i]==1)).sum())) for i,key in enumerate(SAFETY,1)},
        additional_model_inferences=0,additional_scores=0,best_of_two_used=False,
        interpretation='Grid/time-bucket implementation contrast at fixed weights; no claim of an isolated SFT time-weighting or RL effect.')
    atomic_json(output,result)
    print(json.dumps(dict(output=str(output),difference_percentage_points=100*result['PDMS_raw_difference']['mean'],
        CI95_percentage_points=[100*x for x in result['PDMS_raw_difference']['CI95']])))


if __name__=='__main__':main()
