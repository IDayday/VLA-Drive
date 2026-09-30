"""Explicit fixed-milestone Navtest request, separate from final model selection."""
import argparse
import json
from pathlib import Path
import subprocess

from tools.foresight.checkpoints import checkpoint_identity
from tools.foresight.score_pdms import atomic_json, identity_hash, digest


def validate_probe(models, development, update, allow_prior_development_evidence=False):
    if (update<=0 or development.get('schema')!='ddp_full_foresight_planning_v1' or
            not development['valid'] or development['split']!='dev' or development['diagnostic']):
        raise ValueError('A real development report and positive fixed update are required')
    common=[];records=[];keys=set();steps=set()
    for identity,checkpoint in models:
        key=(checkpoint['arm'],checkpoint['training_seed'])
        if key in keys or identity['scope']!='formal' or identity['schema']!='ddp_full_foresight_student_v1' or checkpoint['completed']!=update:
            raise ValueError('Duplicate, foreign, or wrong-milestone model')
        group=development['groups'].get(f'{key[0]}_train{key[1]}')
        evidence=group.get('checkpoint',{}) if group else {}
        prior_matches=(allow_prior_development_evidence and
                       0<evidence.get('completed',0)<=update and
                       all(checkpoint.get(k) is not None and evidence.get(k)==checkpoint[k]
                           for k in ('run_identity','training_source_sha','arm','training_seed','scope','model_class')) and
                       evidence.get('scope')=='formal')
        if not group or not group['valid'] or (evidence!=checkpoint and not prior_matches):
            raise ValueError('Matched checkpoint development evidence required')
        keys.add(key);records.append(checkpoint)
        common.append({k:identity[k] for k in ('source_sha','data','ego','selected_index_hash','scene_count','global_batch','updates','schedule','precision','world_size','micro_batch')})
        steps.add(identity['config']['framework']['action_model']['num_inference_timesteps'])
    if not records or len(steps)!=1 or any(x!=common[0] for x in common):
        raise ValueError('Requested models need matching common training and inference protocols')
    return records,steps.pop(),common[0]


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('models','development-report','current-root','metric-index','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--update',type=int,required=True)
    p.add_argument('--sampling-seed',type=int,default=42)
    p.add_argument('--allow-prior-development-evidence',action='store_true',
                   help='Explicit user checkpoint probe only: accept earlier dev evidence from the identical training run')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Immutable evaluation lock already exists')
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Commit evaluation source first')
    models=[checkpoint_identity(r['training_run'],r['checkpoint_tag']) for r in json.loads(Path(a.models).read_text())]
    development=json.loads(Path(a.development_report).read_text())
    records,steps,contract=validate_probe(models,development,a.update,a.allow_prior_development_evidence)
    evidence={f'{r["arm"]}_train{r["training_seed"]}':development['groups'][f'{r["arm"]}_train{r["training_seed"]}']['checkpoint'] for r in records}
    matched=all(evidence[f'{r["arm"]}_train{r["training_seed"]}']==r for r in records)
    root=Path(a.current_root);current=json.loads((root/'identity.json').read_text());index=json.loads((root/'index.json').read_text())
    metric=json.loads(Path(a.metric_index).read_text())
    if (current['schema']!='ddpolicy_current_cameras_v1' or current['split']!='navtest' or
            identity_hash(index)!=current['index_sha256'] or len(index)!=12146 or
            len({r['token'] for r in index})!=12146 or len({r['log'] for r in index})!=136 or
            len(metric)!=12146 or {(r['token'],r['log']) for r in metric}!={(r['token'],r['log']) for r in index}):
        raise ValueError('Complete official current-only Navtest population required')
    lock={'schema':'foresight_navtest_checkpoint_probe_lock_v1',
          'evaluation_purpose':'user_requested_fixed_checkpoint','requested_update':a.update,
          'authorization':f'Explicit user request for fixed checkpoint evaluation at {a.update} completed updates',
          'evaluation_source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'checkpoint_selection':f'Common {a.update}-update checkpoints; no test-driven selection or recipe changes',
          'checkpoints':[r['sha256'] for r in records],'checkpoint_records':{r['sha256']:r for r in records},
          'training_contract':contract,'models_registry_sha256':digest(a.models),
          'development_report_sha256':digest(a.development_report),
          'development_evidence_scope':'matched_checkpoint' if matched else 'prior_checkpoint_same_training_run',
          'development_evidence_checkpoints':evidence,
          'current_data_identity':current['identity'],'current_index_identity':current['index_sha256'],
          'scene_count':12146,'log_count':136,'official_metric_index_sha256':digest(a.metric_index),
          'sampling_seeds':[a.sampling_seed],'inference_steps':steps,'candidates_per_scene':1,
          'learned_scorer':None,'precision':'FP32','final_endpoint_comparison':False}
    lock['identity']=identity_hash(lock);atomic_json(out,lock)


if __name__=='__main__':main()
