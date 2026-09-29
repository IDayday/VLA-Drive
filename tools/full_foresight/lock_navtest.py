"""Lock registered full-method finalists after completed five-run development."""
import argparse
import json
from pathlib import Path
import subprocess
from tools.foresight.checkpoints import checkpoint_identity
from tools.foresight.lock_navtest import SEEDS
from tools.foresight.score_pdms import atomic_json, identity_hash, digest


def validate_models(models, development, finalists):
    if development.get('schema')!='ddp_full_foresight_planning_v1' or development['split']!='dev' or development.get('screen') or development['diagnostic'] or not development['valid'] or not development['complete_primary_matrix']:
        raise ValueError('Complete five-run formal development results required')
    if set(development['expected_arms'])!=set(finalists):raise ValueError('Registered finalists differ from development report')
    records=[];keys=set();common=[];steps=set()
    for identity,checkpoint,status in models:
        key=(checkpoint['arm'],checkpoint['training_seed'])
        if key in keys or key[0] not in finalists:raise ValueError('Duplicate/unregistered final model')
        keys.add(key)
        if identity['schema']!='ddp_full_foresight_student_v1' or identity['scope']!='formal' or status['status']!='COMPLETE':
            raise ValueError('Completed full-method formal training required')
        if status['identity']!=identity['identity'] or checkpoint['run_identity']!=identity['identity'] or status['completed']!=identity['updates'] or checkpoint['completed']!=identity['updates']:
            raise ValueError('Main final comparison requires the common registered endpoint')
        group=development['groups'].get(f'{key[0]}_train{key[1]}')
        if not group or not group['valid'] or group['sampling_seeds']!=SEEDS or group['checkpoint']!=checkpoint:
            raise ValueError('Missing matched five-run development model')
        contract={k:identity[k] for k in ('source_sha','data','ego','selected_index_hash','scene_count','global_batch','updates','schedule','precision','world_size','micro_batch')}
        contract['exposure']=status['exposure'];common.append(contract)
        steps.add(identity['config']['framework']['action_model']['num_inference_timesteps']);records.append(checkpoint)
    if not all((arm,42) in keys for arm in finalists) or len(steps)!=1 or any(c!=common[0] for c in common):
        raise ValueError('Final training endpoints/protocols or primary population differ')
    return records,steps.pop(),common[0]


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('models','development-report','selection-registration','current-root','metric-index','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Final lock is immutable')
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Lock clean evaluation source')
    selection=json.loads(Path(a.selection_registration).read_text())
    if selection.get('selected_on')!='dev' or selection.get('navtest_seen') is not False:
        raise ValueError('Finalists must be selected and frozen using development only')
    models=[]
    for item in json.loads(Path(a.models).read_text()):
        identity,checkpoint=checkpoint_identity(item['training_run'],item['checkpoint_tag'])
        models.append((identity,checkpoint,json.loads((Path(item['training_run'])/'status.json').read_text())))
    records,steps,contract=validate_models(models,json.loads(Path(a.development_report).read_text()),selection['finalists'])
    root=Path(a.current_root);current=json.loads((root/'identity.json').read_text());index=json.loads((root/'index.json').read_text())
    metric=json.loads(Path(a.metric_index).read_text())
    if current['schema']!='ddpolicy_current_cameras_v1' or current['split']!='navtest' or identity_hash(index)!=current['index_sha256']:
        raise ValueError('Wrong final current observation population')
    if not index or len({r['token'] for r in index})!=len(index) or len({r['token'] for r in metric})!=len(metric):
        raise ValueError('Empty/duplicate final population')
    if {(r['token'],r['log']) for r in index}!={(r['token'],r['log']) for r in metric}:raise ValueError('Official environment population mismatch')
    # Reuse the strict existing inference lock format, never its old R/A/B/C/D selector.
    lock={'schema':'foresight_navtest_lock_v1','method':'ddp_full_foresight',
          'evaluation_source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'checkpoint_selection':'registered complete common endpoint, selected using development only',
          'checkpoints':[r['sha256'] for r in records],'checkpoint_records':{r['sha256']:r for r in records},
          'training_contract':contract,'selection_sha256':digest(a.selection_registration),
          'development_report_sha256':digest(a.development_report),'models_registry_sha256':digest(a.models),
          'current_data_identity':current['identity'],'current_index_identity':current['index_sha256'],
          'scene_count':len(index),'log_count':len({r['log'] for r in index}),'official_metric_index_sha256':digest(a.metric_index),
          'sampling_seeds':SEEDS,'inference_steps':steps,'candidates_per_scene':1,'learned_scorer':None,'precision':'FP32'}
    lock['identity']=identity_hash(lock);atomic_json(out,lock)

if __name__=='__main__':main()
