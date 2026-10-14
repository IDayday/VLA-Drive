"""Fixed-budget endpoint selection and read-only follow-up evaluation."""
import argparse
import io
import json
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import atomic_json, atomic_bytes, read_json, file_hash, digest, lock
from iqe.evaluation.retention import module_hash
from iqe.robust_scorer.evaluation import statistics
from iqe.specialist_study.bank import load_bank, pool
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.metrics import report_metrics, select
from iqe.specialist_study.scorer_training import predict as diagnose
from iqe.specialist_study.training import runtime
from .experiment import bind, create, predict, recipe
from .config import load


def endpoint(config, device='cpu', *, terminal=False):
    folder=config.root/'scorer'; result=read_json(folder/'result.json')
    require(result['status']=='COMPLETE' and result['optimizer_steps']==config.updates
            and result['termination']['terminal'], 'formal fixed budget incomplete')
    model,initial=create(config)
    require(initial==read_json(folder/'SCORER_INITIALIZATION.json'), 'initialization lineage changed')
    if terminal:
        receipt=read_json(result['checkpoint']+'.COMPLETE.json')
        path=result['checkpoint']
        require(file_hash(path)==receipt['checksum'], 'terminal checkpoint corrupt')
        state=torch.load(path,map_location='cpu',weights_only=False)['module']
    else:
        require(file_hash(folder/'best.pt')==result['best_checkpoint_hash'], 'best checkpoint corrupt')
        state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval().requires_grad_(False),result


def lock_choice(config):
    binding=bind(config);folder=config.root/'scorer'
    with lock(config.root/'MODEL_CHOICE.lock'):
        model,result=endpoint(config)
        rows=[(step,read_json(folder/f'validation_{step:06d}.json'))
              for step in recipe(config)['validation_steps']]
        step,stats=max(rows,key=lambda row:(row[1]['selection_key'],-row[0]))
        require(stats['scorer_state_hash']==module_hash(model)
                and stats['selection_key']==result['best_selection_key'], 'best validation mismatch')
        choice=dict(experiment_hash=digest(binding),selection_role='stage_val',
            selected=dict(step=step,checkpoint=str(folder/'best.pt'),checkpoint_hash=result['best_checkpoint_hash'],
                          state_hash=module_hash(model),metrics=stats),
            pool_hash=pool(config.source())['hash'],dev_used_for_selection=False,
            tie_break='fewer new NC+DAC; earlier endpoint',
            validation_hashes={str(s):file_hash(folder/f'validation_{s:06d}.json') for s,_ in rows})
        atomic_json(config.root/'MODEL_CHOICE.json',choice,immutable=True)
        return choice


def predictions(config,role,device,*,terminal=False):
    require(role in ('scorer_fit','stage_val','selector_cal','dev_report'), 'final_test/navtest forbidden')
    choice=read_json(config.root/'MODEL_CHOICE.json')
    require(choice['experiment_hash']==digest(bind(config)), 'stale model choice')
    source=config.source();meta,bank=load_bank(source,'deploy',role)
    scenes=scenes_by_role(source)[role]
    require(meta['scene_ids']==[s.scene_id for s in scenes], 'bank/scene order changed')
    name='terminal_6000' if terminal else 'selected'
    folder=config.root/'evaluation'/name/role
    model,result=endpoint(config,device,terminal=terminal)
    identity=dict(choice_hash=file_hash(config.root/'MODEL_CHOICE.json'),bank_hash=meta['hash'],
        state_hash=module_hash(model),role=role,endpoint=name,requested=len(scenes))
    with lock(folder/'PREDICT.lock'):
        receipt=folder/'COMPLETE.json';path=folder/'predictions.npz'
        if receipt.exists():
            old=read_json(receipt)
            require(old['identity']==identity and file_hash(path)==old['checksum'], 'stale prediction cache')
            with np.load(path) as f:values=f['values'].copy()
        else:
            values=predict(config,model,scenes,bank,device)
            require(values.shape==bank['scores'].shape and np.isfinite(values).all(), 'incomplete predictions')
            stream=io.BytesIO();np.savez(stream,values=values)
            atomic_bytes(path,stream.getvalue(),immutable=True)
            atomic_json(receipt,dict(identity=identity,checksum=file_hash(path)),immutable=True)
    return meta,bank,dict(values=values)


def calibrate(config,device):
    rules=dict(grid=[0.,.01,.02,.05,.1],role='selector_cal',new_NC=0,new_DAC=0,high_to_zero=0,
        fallback='always Base is not success',choice_hash=file_hash(config.root/'MODEL_CHOICE.json'))
    atomic_json(config.root/'MARGIN_RULES.json',rules,immutable=True)
    meta,bank,pred=predictions(config,'selector_cal',device);rows=[]
    for delta in rules['grid']:
        stats=statistics(config,meta,bank,pred,delta)
        feasible=stats['high_to_zero']==0 and all(stats['safety'][key]['new_failures']==0 for key in
            ('no_at_fault_collisions','drivable_area_compliance'))
        rows.append(dict(delta=delta,feasible=feasible,metrics=stats))
    eligible=[r for r in rows if r['feasible']]
    best=max(eligible,key=lambda r:(r['metrics']['selected_01'],-r['metrics']['switch_count'],r['delta'])) if eligible else None
    result=dict(rules_hash=file_hash(config.root/'MARGIN_RULES.json'),rows=rows,
        selected_delta=best['delta'] if best else None,always_Base=best is None or best['metrics']['switch_count']==0)
    atomic_json(config.root/'SELECTOR_MARGIN.json',result,immutable=True)
    return result


def evaluate(config,device):
    device=runtime(device);choice=lock_choice(config);margin=calibrate(config,device)
    source=config.source()
    atomic_json(config.root/'FINAL_EVALUATION_LOCK.json',dict(
        choice_hash=file_hash(config.root/'MODEL_CHOICE.json'),margin_hash=file_hash(config.root/'SELECTOR_MARGIN.json'),
        dev_scene_ids_hash=digest([s.scene_id for s in scenes_by_role(source)['dev_report']]),
        prior_development_observations_disclosed=True,test_access=False),immutable=True)
    report=dict(units='[0,1]',rows={},fixed_pool_hash=pool(source)['hash'],new_navtest=False,ACTIVE_updated=False)
    for role in ('stage_val','scorer_fit','dev_report'):
        meta,bank,pred=predictions(config,role,device)
        report['rows'][role]=statistics(config,meta,bank,pred)
        atomic_json(config.root/'evaluation'/'selected'/role/'RESULT.json',report['rows'][role],immutable=True)
    meta,bank,pred=predictions(config,'dev_report',device,terminal=True)
    report['terminal_6000_dev']=statistics(config,meta,bank,pred)
    meta,bank,pred=predictions(config,'dev_report',device)
    indices=np.zeros(len(bank['scores']),dtype=int) if margin['selected_delta'] is None else select(
        pred['values'],bank['valid'],margin['selected_delta'],expert_ids=meta['provenance']['sources'])
    report['optional_margin']=report_metrics(bank['scores'],bank['components'],indices,meta['source_logs'],
        seed=source.bootstrap_seed,repetitions=source.bootstrap_repetitions)
    model,_=endpoint(config,device);diagnostics={}
    for kind in ('shuffle_V','mask_V','scene_mismatch','trajectory_time_shuffle'):
        meta,bank,values=diagnose(source,model,'stage_val',device,diagnostic=kind)
        diagnostics[kind]=statistics(config,meta,bank,dict(values=values))
    report['readonly_stage_val_diagnostics']=diagnostics
    atomic_json(config.root/'FINAL_RESULTS.json',report,immutable=True)
    stats=report['rows']['dev_report']
    risk=stats['high_to_zero']>0 or any(x['new_failures']>0 for x in stats['safety'].values())
    verdict=dict(IMPLEMENTATION='PASS',TRAINING_COMPLETE=True,external_scorer_weights_loaded=False,
        selected_step=choice['selected']['step'],heldout_gain_01=stats['selected_gain_01'],
        gain_over_best_fixed_01=stats['selected_over_best_fixed_01'],safety_regression=risk,
        gain_CI_excludes_zero=stats['gain_ci']['lower_95']>0,
        empirical_heldout_qualification=stats['gain_ci']['lower_95']>0 and
            stats['gain_over_best_fixed_ci']['lower_95']>0 and not risk,
        universal_robustness_claim=False,multi_seed='UNTESTED',new_navtest=False,ACTIVE_updated=False)
    atomic_json(config.root/'VERDICT.json',verdict,immutable=True)
    return verdict


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('lock-choice','evaluate'))
    parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv);config=load(args.config)
    from iqe.specialist_study.resource_guard import check
    check(config,'evaluate-scorer-matrix')
    result=lock_choice(config) if args.command=='lock-choice' else evaluate(config,args.device)
    print(json.dumps(dict(status='COMPLETE',result=result),allow_nan=False))


if __name__=='__main__':main()
