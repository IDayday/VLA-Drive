"""Fixed 20k/170k Navtest locks; reuse the unchanged qualified GPU worker."""
import argparse
import json
from pathlib import Path
import time
import torch
from scripts.analysis.fm_transfer.final_lock import authenticated, validate_export
from scripts.analysis.fm_transfer.source_process_controller import ProcessEvaluations
from starVLA.rl.fm_transfer.source_process import ROOT, MODES, B_SFT, expected_config
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha, fingerprint
from starVLA.rl.simwam_recipe.recovery_state import validate_boundary, validate_ranks
from starVLA.rl.flow_grpo.transactions import atomic_json

STAGES=(20000,170000)
OLD_ASSETS=Path('/mnt/project/fm-transfer-20261016/final_navtest/LOCK.json')


def checkpoint_for(mode, update):
    if mode not in MODES or update not in STAGES:raise ValueError('Unregistered Navtest observation')
    return ROOT/'runs'/f'{mode}_seed42_v1'/'checkpoints'/f'update{update:06d}_inner0'


def ready(update):
    missing=[]
    for mode in MODES:
        missing += [str(p) for p in (checkpoint_for(mode,update)/'COMPLETE',
            ROOT/'exports'/mode/f'RL_update{update:06d}'/'COMPLETE') if not p.exists()]
        for cohort in ('DEV','TRAIN_DIAGNOSTIC'):
            name=f'{mode}_RL_update{update:06d}_{cohort}'
            for suffix in ('.json','_paired_own_SFT.json'):
                path=ROOT/'analysis'/(name+suffix)
                if not path.exists():missing.append(str(path))
    return missing


def verify_checkpoint(mode,update):
    path=checkpoint_for(mode,update);identity=authenticated(path/'identity.json')
    seal=json.loads((path/'COMPLETE').read_text());files=json.loads((path/'files.json').read_text())
    recipe=dict(identity['recipe']);parent=recipe.pop('supervised_transfer')['SFT_identity']
    phase='COMPLETE' if update==170000 else 'READY'
    if (identity['purpose']!='formal' or identity['world']!=8 or recipe!=expected_config(mode) or
            Path(parent['checkpoint']).resolve()!=B_SFT.resolve() or parent['qualification'] or
            parent['model_sha256']!='4e71b1fcf963d38456174de1f8d17762a9a4ccf1597c22b7cec262026d9201b7' or
            seal['phase']!=phase or seal['updates']!=update or seal['next_inner']!=0 or seal['world']!=8 or
            file_sha(path/'files.json')!=seal['files_sha256']):
        raise ValueError('Wrong registered committed milestone')
    required={'identity.json','optimizer.pt','transaction.json',*[f'rank{i}.pt' for i in range(8)]}
    if not required<=files.keys():raise ValueError('Missing native state')
    for name,digest in files.items():
        if Path(name).name!=name or file_sha(path/name)!=digest:raise ValueError('Changed native state')
    common=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False)
    transaction=json.loads((path/'transaction.json').read_text());summaries=[]
    for rank in range(8):
        state=torch.load(path/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        summary=validate_boundary(state['state'],state['rollout'],identity,common['optimizer'],common['scheduler'])
        if summary!=transaction['ranks'][rank]:raise ValueError('Wrong cross-rank transaction')
        summaries.append(summary)
    validate_ranks(summaries)
    if (summaries[0]['collections']!=update//4 or summaries[0]['scene_cursor']!=2*update or
            transaction['execution_identity']!=identity['execution_identity']):
        raise ValueError('Wrong effective budget')
    return identity,parent


def make_lock(update,source,output):
    require_dispatch();output=Path(output);source=Path(source).resolve()
    if output.exists():raise FileExistsError('Preserve the fixed checkpoint selection')
    missing=ready(update)
    if missing:raise ValueError('Await fixed milestones and both complete DEV evaluations: '+repr(missing))
    assets=authenticated(OLD_ASSETS);pop=json.loads(Path(POPULATION).read_text())
    registration=authenticated(ROOT/'registration.json')
    if (registration['Navtest_fixed_observations']!=list(STAGES) or
            registration['Navtest_checkpoint_selection'] or assets['population_identity']!=pop['identity_sha256']):
        raise ValueError('Changed pre-registered observations or assets')
    for key,rel in [('sampler_sha256','scripts/analysis/simwam_recipe/evaluate_policy.py'),
                    ('transfer_evaluation_sha256','scripts/analysis/fm_transfer/evaluate_policy.py')]:
        if assets[key]!=file_sha(source/rel):raise ValueError('Changed qualified inference source')
    if not json.loads((source/'SOURCE_COMPLETE.json').read_text())['git_archive_extraction_complete']:
        raise ValueError('Incomplete evaluation source')
    models={};evidence={}
    for mode in MODES:
        identity,parent=verify_checkpoint(mode,update)
        exp=ROOT/'exports'/mode/f'RL_update{update:06d}'
        models[mode]=validate_export(exp,checkpoint_for(mode,update),identity,'RL',update,parent)
        for cohort in ('DEV','TRAIN_DIAGNOSTIC'):
            name=f'{mode}_RL_update{update:06d}_{cohort}'
            manifest=authenticated(ROOT/'evaluations'/name/'manifest.json')
            if (manifest['cohort']!=cohort or manifest['seed']!=2101 or manifest['qualification_only'] or
                    manifest['weight_sha256']!=models[mode]['weight_sha256']):
                raise ValueError('Milestone population/model mismatch')
            for suffix in ('.json','_paired_own_SFT.json'):
                p=ROOT/'analysis'/(name+suffix);report=json.loads(p.read_text())
                if report['status']!='COMPLETE_POPULATION' or report['identity']!=manifest:
                    raise ValueError('Incomplete development population')
                evidence[str(p)]=file_sha(p)
    value={key:assets[key] for key in ('population_identity','inference_seeds','training_seeds','modes',
        'sampler_sha256','transfer_evaluation_sha256','asset_identity','data_root','navtest_datalist','v1_cache')}
    value.update(schema='source_process_fixed_milestone_navtest_v1',status='LOCKED_BEFORE_FINAL_NAVTEST',
        observed_update=update,evaluation_budget_complete=True,formal_budget_complete=update==170000,
        selection_rule='Pre-registered fixed20k observation and fixed170k endpoint; no score selection',
        models=models,weight_sha256=[m['weight_sha256'] for m in models.values()],
        evaluation_source=str(source),registration_sha256=file_sha(ROOT/'registration.json'),
        old_asset_lock_sha256=file_sha(OLD_ASSETS),milestone_evidence=evidence,
        Navtest_scores_read_for_selection=False,historically_observed_Navtest_not_fully_blind=True,
        unix=time.time(),lock_implementation_sha256=file_sha(__file__))
    value['sha256']=fingerprint(value);atomic_json(output,value);return value


class SourceNavtestDispatch:
    models=MODES

    @staticmethod
    def validate_selection(path,model,source):
        selection=authenticated(path);source=Path(source).resolve();update=selection['observed_update']
        if (selection['schema']!='source_process_fixed_milestone_navtest_v1' or update not in STAGES or
                selection['status']!='LOCKED_BEFORE_FINAL_NAVTEST' or not selection['evaluation_budget_complete'] or
                selection['formal_budget_complete']!=(update==170000) or set(selection['models'])!=set(MODES) or
                selection['inference_seeds']!=[2101] or selection['training_seeds']!=[42] or
                selection['modes']!=['E_NATIVE','E_SHIFT5'] or selection['Navtest_scores_read_for_selection'] or
                selection['registration_sha256']!=file_sha(ROOT/'registration.json') or
                selection['evaluation_source']!=str(source) or
                selection['population_identity']!=json.loads(Path(POPULATION).read_text())['identity_sha256']):
            raise ValueError('Invalid fixed-stage lock')
        for key,rel in [('sampler_sha256','scripts/analysis/simwam_recipe/evaluate_policy.py'),
                        ('transfer_evaluation_sha256','scripts/analysis/fm_transfer/evaluate_policy.py')]:
            if selection[key]!=file_sha(source/rel):raise ValueError('Changed worker')
        chosen=selection['models'][model]
        identity=authenticated(checkpoint_for(model,update)/'identity.json')
        current=validate_export(chosen['export'],checkpoint_for(model,update),identity,'RL',update,chosen['new_SFT'])
        if current!=chosen:raise ValueError('Changed locked export')
        return selection,chosen


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--update',type=int,choices=STAGES,required=True)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if a.check:print(json.dumps(dict(missing=ready(a.update),new_predictions=0)));return
    value=make_lock(a.update,a.source,a.output);print(json.dumps(dict(status=value['status'],sha256=value['sha256'])))

if __name__=='__main__':main()
