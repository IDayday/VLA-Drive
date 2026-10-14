"""Lock the geometric trial endpoint on stage_val, then report fixed populations."""
import argparse
from pathlib import Path
import io
import json
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import atomic_bytes, atomic_json, digest, file_hash, lock, read_json
from iqe.evaluation.retention import module_hash
from iqe.specialist_study.metrics import report_metrics, select
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.bank import load_bank, pool
from iqe.specialist_study.training import runtime
from iqe.robust_scorer.evaluation import score_diagnostics
from .full import load_config, binding, create, predict, resource_check, training_config


def selected(config,device='cpu'):
    folder=config.root/'scorer';result=read_json(folder/'result.json')
    require(result['status']=='COMPLETE' and result['optimizer_steps']==6000 and
            result['termination']['terminal'], 'full fixed-budget Scorer incomplete')
    require(file_hash(folder/'best.pt')==result['best_checkpoint_hash'],'best snapshot changed')
    model,identity=create(config)
    require(identity==read_json(folder/'SCORER_INITIALIZATION.json'),'initializer/lineage changed')
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True),strict=True)
    return model.to(device).eval().requires_grad_(False),result


def lock_choice(config):
    bound=binding(config);source=config.source();folder=config.root/'scorer'
    model,result=selected(config)
    rows=[(s,read_json(folder/f'validation_{s:06d}.json')) for s in training_config(source)['validation_steps']]
    step,stats=max(rows,key=lambda x:(x[1]['selection_key'],-x[0]))
    require(module_hash(model)==stats['scorer_state_hash'] and
            result['best_selection_key']==stats['selection_key'],'best snapshot/validation mismatch')
    choice=dict(experiment_hash=digest(bound),step=step,checkpoint=str(folder/'best.pt'),
        checkpoint_hash=result['best_checkpoint_hash'],state_hash=module_hash(model),
        validation_hash=file_hash(folder/f'validation_{step:06d}.json'),metrics=stats,
        pool_hash=pool(source)['hash'],selection_role='stage_val',dev_used_for_this_choice=False)
    atomic_json(config.root/'MODEL_CHOICE.json',choice,immutable=True)
    return choice


def statistics(config,meta,bank,pred,margin=0.):
    source=config.source()
    chosen=select(pred,bank['valid'],margin,expert_ids=meta['provenance']['sources'])
    result=report_metrics(bank['scores'],bank['components'],chosen,meta['source_logs'],
        seed=source.bootstrap_seed,repetitions=source.bootstrap_repetitions)
    result['prediction_diagnostics']=score_diagnostics({'values':pred},bank)
    return result


def predictions(config,role,device,diagnostic=None):
    require(role in ('scorer_fit','stage_val','selector_cal','dev_report'),'no test access')
    choice=read_json(config.root/'MODEL_CHOICE.json');source=config.source()
    meta,bank=load_bank(source,'deploy',role)
    identity=dict(choice_hash=file_hash(config.root/'MODEL_CHOICE.json'),
        bank_hash=meta['hash'],role=role,diagnostic=diagnostic,
        code=file_hash(__file__),calibration_hash=file_hash(Path(config.calibration_bank)/'MANIFEST.json'))
    folder=config.root/'evaluation'/role/(diagnostic or 'correct')
    with lock(folder/'PREDICT.lock'):
        done=folder/'COMPLETE.json';path=folder/'predictions.npz'
        if done.exists():
            receipt=read_json(done)
            require(receipt['identity']==identity and file_hash(path)==receipt['checksum'],'stale prediction')
            with np.load(path) as f:return meta,bank,f['values'].copy()
        model,result=selected(config,device)
        require(result['best_checkpoint_hash']==choice['checkpoint_hash'],'model choice changed')
        _,_,p=predict(config,model,role,device,diagnostic=diagnostic)
        require(p.shape==bank['scores'].shape,'incomplete population prediction')
        buf=io.BytesIO();np.savez(buf,values=p);atomic_bytes(path,buf.getvalue(),immutable=True)
        atomic_json(done,dict(identity=identity,checksum=file_hash(path),scenes=len(p)),immutable=True)
        return meta,bank,p


def calibrate(config,device):
    rules=dict(role='selector_cal',grid=[0.,.01,.02,.05,.1],new_NC=0,new_DAC=0,high_to_zero=0,
        tie_break='fewer switches, then larger margin',fallback='always Base if no nontrivial feasible margin',
        guarantee_for_unseen_scenes=False,choice_hash=file_hash(config.root/'MODEL_CHOICE.json'))
    atomic_json(config.root/'MARGIN_RULES.json',rules,immutable=True)
    meta,bank,p=predictions(config,'selector_cal',device);rows=[]
    for delta in rules['grid']:
        r=statistics(config,meta,bank,p,delta)
        feasible=r['high_to_zero']==0 and all(r['safety'][k]['new_failures']==0 for k in
            ('no_at_fault_collisions','drivable_area_compliance'))
        rows.append(dict(delta=delta,feasible=feasible,metrics=r))
    feasible=[r for r in rows if r['feasible']]
    best=max(feasible,key=lambda r:(r['metrics']['selected_01'],-r['metrics']['switch_count'],r['delta'])) if feasible else None
    result=dict(rows=rows,selected_delta=best['delta'] if best else None,
        always_Base=best is None or best['metrics']['switch_count']==0,rules_hash=file_hash(config.root/'MARGIN_RULES.json'))
    atomic_json(config.root/'SELECTOR_MARGIN.json',result,immutable=True);return result


def evaluate(config,device='cuda:0'):
    resource_check(config);device=runtime(device);lock_choice(config);margin=calibrate(config,device)
    source=config.source()
    lock_info=dict(choice_hash=file_hash(config.root/'MODEL_CHOICE.json'),
        margin_hash=file_hash(config.root/'SELECTOR_MARGIN.json'),
        dev_scenes=digest([s.scene_id for s in scenes_by_role(source)['dev_report']]),
        previous_dev_experiments_disclosed=True,exploratory_followup=True,new_navtest=False)
    atomic_json(config.root/'FINAL_EVALUATION_LOCK.json',lock_info,immutable=True)
    reports={}
    for role in ('scorer_fit','stage_val','dev_report'):
        meta,bank,p=predictions(config,role,device)
        report=statistics(config,meta,bank,p)
        reports[role]=report
        atomic_json(config.root/'evaluation'/role/'REPORT.json',report,immutable=True)
        if role=='dev_report':
            if margin['always_Base']:
                indices=np.zeros(len(p),dtype=int)
                result=report_metrics(bank['scores'],bank['components'],indices,meta['source_logs'],
                    seed=source.bootstrap_seed,repetitions=source.bootstrap_repetitions)
            else:result=statistics(config,meta,bank,p,margin['selected_delta'])
            result['always_Base']=margin['always_Base'];reports['dev_margin']=result
    diagnostics={}
    for kind in ('wrong_visual','mask_visual'):
        meta,bank,p=predictions(config,'stage_val',device,kind)
        diagnostics[kind]=statistics(config,meta,bank,p)
    atomic_json(config.root/'INFORMATION_DIAGNOSTICS.json',diagnostics,immutable=True)
    result=dict(status='COMPLETE',role_reports=reports,choice=read_json(config.root/'MODEL_CHOICE.json'),
        external_scorer_weights_loaded=False,new_navtest=False,ACTIVE_updated=False,
        interpretation='single-seed exploratory evidence; training fit is not heldout gain')
    atomic_json(config.root/'FINAL_RESULTS.json',result,immutable=True)
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    result=evaluate(load_config(a.config),a.device)
    print(json.dumps({k:v for k,v in result.items() if k!='choice'},allow_nan=False))


if __name__=='__main__':main()
