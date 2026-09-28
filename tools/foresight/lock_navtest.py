"""Lock common-endpoint formal models after complete, matched five-run development evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
from .checkpoints import checkpoint_identity
from .score_pdms import atomic_json,identity_hash,digest

SEEDS=[42,43,44,45,46]


def validate_models(models,development):
    if development.get('schema')!='foresight_paired_planning_v1' or development['split']!='dev' or development['diagnostic'] or not development['valid'] or not development['complete_primary_matrix']:
        raise ValueError('Complete formal development results are required before locking Navtest')
    keys=set();common=[];steps=set();records=[]
    for identity,checkpoint,status in models:
        arm,seed=checkpoint['arm'],checkpoint['training_seed'];key=(arm,seed)
        if key in keys:raise ValueError('Duplicate formal model');
        keys.add(key)
        if identity['scope']!='formal' or checkpoint['scope']!='formal' or status['status']!='COMPLETE':raise ValueError('Only completed formal training may enter Navtest')
        if status['identity']!=identity['identity'] or checkpoint['run_identity']!=identity['identity']:
            raise ValueError('Model/status training identity differs')
        if status['completed']!=identity['updates'] or checkpoint['completed']!=identity['updates']:
            raise ValueError('Main comparison uses the same registered complete endpoint, not arm-specific earlier checkpoints')
        group=development['groups'].get(f'{arm}_train{seed}')
        if group is None or not group['valid'] or group['sampling_seeds']!=SEEDS or group['checkpoint']!=checkpoint:
            raise ValueError('Model lacks matching complete five-run development evidence')
        common.append({k:identity[k] for k in ('source_sha','data','ego','selected_index_hash','scene_count','global_batch','updates','schedule','precision','world_size','micro_batch')})
        common[-1]['exposure']=status['exposure']
        steps.add(identity['config']['framework']['action_model']['num_inference_timesteps'])
        records.append(checkpoint)
    if not {('R',42),('A',42),('B',42),('C',42),('D',42)}.issubset(keys):raise ValueError('Primary R/A/B/C/D seed42 matrix incomplete')
    if len(steps)!=1 or any(c!=common[0] for c in common):raise ValueError('Formal training exposure/data/schedule or inference steps differ')
    for seed in {seed for _,seed in keys}-{42}:
        if not {('B',seed),('D',seed)}.issubset(keys):raise ValueError('Additional training seed requires matched B/D key pair')
    return records,steps.pop(),common[0]


def validate_lock(lock,checkpoint,source,current,scene_count,sampling_seed,steps):
    if lock.get('schema')!='foresight_navtest_lock_v1' or identity_hash({k:v for k,v in lock.items() if k!='identity'})!=lock['identity']:
        raise ValueError('Invalid final Navtest lock')
    if checkpoint['sha256'] not in lock['checkpoints'] or lock['checkpoint_records'][checkpoint['sha256']]!=checkpoint:
        raise ValueError('Final checkpoint differs from completed development selection')
    if source!=lock['evaluation_source_sha']:raise ValueError('Final evaluation source changed')
    if current['identity']!=lock['current_data_identity'] or current['index_sha256']!=lock['current_index_identity'] or scene_count!=lock['scene_count']:
        raise ValueError('Final Navtest population changed')
    if sampling_seed not in lock['sampling_seeds'] or steps!=lock['inference_steps']:raise ValueError('Final inference protocol changed')


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('models','development-report','current-root','metric-index','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Final selection lock must not be overwritten')
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze clean inference source first')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    registry=json.loads(Path(a.models).read_text());models=[]
    for row in registry:
        identity,checkpoint=checkpoint_identity(row['training_run'],row['checkpoint_tag'])
        status=json.loads((Path(row['training_run'])/'status.json').read_text());models.append((identity,checkpoint,status))
    report=json.loads(Path(a.development_report).read_text());records,steps,training=validate_models(models,report)
    root=Path(a.current_root);current=json.loads((root/'identity.json').read_text());index=json.loads((root/'index.json').read_text())
    metric=json.loads(Path(a.metric_index).read_text())
    if current['schema']!='ddpolicy_current_cameras_v1' or current['split']!='navtest' or identity_hash(index)!=current['index_sha256']:
        raise ValueError('Wrong final current-data identity')
    if not index or len({r['token'] for r in index})!=len(index) or len({r['token'] for r in metric})!=len(metric):raise ValueError('Empty/duplicate final population')
    if {(r['token'],r['log']) for r in index}!={(r['token'],r['log']) for r in metric}:raise ValueError('Official cache/current population mismatch')
    lock={'schema':'foresight_navtest_lock_v1','evaluation_source_sha':source,
          'checkpoint_selection':'common registered complete training endpoint; no per-arm best-epoch or Navtest selection',
          'checkpoints':[r['sha256'] for r in records],'checkpoint_records':{r['sha256']:r for r in records},
          'training_contract':training,'models_registry_sha256':digest(a.models),'development_report_sha256':digest(a.development_report),
          'current_data_identity':current['identity'],'current_index_identity':current['index_sha256'],
          'scene_count':len(index),'log_count':len({r['log'] for r in index}),'official_metric_index_sha256':digest(a.metric_index),
          'sampling_seeds':SEEDS,'inference_steps':steps,'candidates_per_scene':1,'learned_scorer':None,'precision':'FP32'}
    lock['identity']=identity_hash(lock);atomic_json(out,lock)


if __name__=='__main__':main()
