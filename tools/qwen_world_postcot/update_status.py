"""Write compact evidence-backed campaign status without inventing results."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
from starVLA.rl.qwen_world_postcot.contracts import ROOT

def read(path):return json.loads(path.read_text()) if path.exists() else None

def completed_scores(comparison):
    """Expose only the controller's sealed, complete primary single-track result."""
    if not comparison or comparison.get('status')!='COMPLETE':return None
    primary=comparison['primary_seed2101']
    if primary.get('status')!='COMPLETE' or primary.get('primary_sampling_seed')!=2101:
        raise ValueError('Complete campaign has no fixed primary seed result')
    values={}
    for kind in ('IL','RL'):
        result=primary[kind]
        score=result['PDMS']
        if (result.get('status')!='COMPLETE' or result.get('qualification')is not False
                or result.get('cohort')!='NAVTEST' or result.get('seed')!=2101
                or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=100):
            raise ValueError('Incomplete/diagnostic result cannot establish a method score')
        values[kind]=score
    return values

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',default='reports/qwen_world_postcot_20261025/RUN_STATUS.json');a=p.parse_args()
    result=dict(schema='world_postcot_run_status_v1',observed_UTC=datetime.now(timezone.utc).isoformat(),
        starting_commit='425eb5420ce1b592786f455ceea9bd6d124d2c54',
        branch='feature/qwen-flow-world-postcot-500m-20261025',
        uses_metric_supervision=True,IL_uses_rl=False,
        original_native_IL_and_RL_preserved=True,old_IQE_22k_identity_preserved=True,
        resource_authorization=read(ROOT/'RESOURCE_AUTHORIZATION.json'),
        REAL_CURRENT_IMAGE_INTEGRATION=read(ROOT/'audits/REAL_QWEN_AND_DEPLOYMENT.json'),
        REAL_ALL_LOSS_GRADIENTS=read(ROOT/'audits/REAL_GRADIENTS.json'),
        ENGINEERING_ACCEPTANCE=read(ROOT/'audits/ENGINEERING_ACCEPTANCE.json'),
        native_profile=read(ROOT/'profiles/native32/SUMMARY.json'),
        full_model_profile=read(ROOT/'audits/FINAL_V3_TRAINING_PROFILE.json'),
        deployment_latency=read(ROOT/'audits/DEPLOYMENT_LATENCY.json'),
        campaign=read(ROOT/'campaign/STATE.json'),
        formal_IL=read(ROOT/'runs/IL_full_seed42/progress.json'),
        formal_RL=read(ROOT/'runs/RL_full_CANONICAL_seed42/progress.json'),
        scientific_conclusion='UNTESTED',new_method_IL_PDMS=None,new_method_RL_PDMS=None,
        note='Only sealed complete evaluations establish scores; engineering PASS is not scientific gain.')
    if result['deployment_latency']:
        result['deployment_latency'].pop('rows',None)
    completed=read(ROOT/'campaign/FINAL_COMPARISON.json')
    scores=completed_scores(completed)
    if scores is not None:
        result['scientific_conclusion']='EVALUATED_SEE_MATCHED_RESULTS_AND_RISK'
        result['new_method_IL_PDMS']=scores['IL'];result['new_method_RL_PDMS']=scores['RL']
        result['final_comparison']=str(ROOT/'campaign/FINAL_COMPARISON.json')
    # Keep large per-module source inventories in their original sealed audit.
    for key in ['REAL_CURRENT_IMAGE_INTEGRATION','REAL_ALL_LOSS_GRADIENTS']:
        if result[key]:result[key].pop('source_files',None)
    result['targets']={}
    for role in ['TRAIN_resolved_v3','DEV_resolved_v3']:
        root=ROOT/'teachers'/role;manifest=read(root/'MANIFEST.json')
        identity=read(root/'IDENTITY.json');physical=payload=0
        if identity:
            record_bytes=identity['layout']['record_bytes']
            for blob in (root/'shards').glob('*.bin'):
                if blob.with_suffix('.json').exists():
                    size=blob.stat().st_size
                    if size%record_bytes:raise ValueError('Committed shard has a nonintegral record payload')
                    payload+=size;physical+=size//record_bytes
        result['targets'][role]=dict(status='COMPLETE' if manifest and manifest.get('status')=='COMPLETE' else 'BUILDING' if identity else 'NOT_RUN',
            manifest=str(root/'MANIFEST.json'),committed_physical_records=physical,payload_bytes=payload,
            count_scope='Committed blob sizes only while building; unique scene coverage is established only by the final sealed manifest.',
            coverage=read(ROOT/'runtime'/('teacher_'+role)/'COVERAGE_NOW.json') or read(ROOT/'runtime'/('teacher_'+role)/'PROFILE1024_READY.json'))
    result['candidate_bank']={}
    for role in ['pilot32','performance1024','full_train_initial_v1']:
        root=ROOT/'candidates'/role
        result['candidate_bank'][role]=dict(status='COMPLETE' if (root/'COMPLETE.json').exists() else 'BUILDING' if root.exists() else 'WAITING_INPUTS',root=str(root))
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(dest),formal_IL=result['formal_IL'],formal_RL=result['formal_RL'],scientific_conclusion=result['scientific_conclusion'])))
if __name__=='__main__':main()
