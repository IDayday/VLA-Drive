"""Fixed-last refit evaluation. No external role may select a checkpoint."""
from pathlib import Path
import io
import numpy as np
import torch

from iqe.contracts import require
from iqe.io import atomic_json, atomic_bytes, read_json, digest, file_hash, complete, reusable, lock
from iqe.evaluation.retention import module_hash, module_hash_parameters
from iqe.specialist_study.metrics import report_metrics, select
from iqe.internal_baseline.evaluation import paired_bootstrap
from .binding import verify_binding
from .data import canonical_bank, FrozenContexts, read_split
from .training import create, predict, resource, KINDS
from .models import choose
from .audit import read_reference, csv_bytes


def final_endpoint(config, kind, device='cpu'):
    require(kind in KINDS, 'only the two registered decision heads can be exported/evaluated')
    binding = verify_binding(config)
    plan = read_json(config.root/'REFIT_PLAN.json')
    require(plan['hash'] == digest({k:v for k,v in plan.items() if k != 'hash'}), 'refit plan checksum')
    require(plan['binding_hash'] == binding['hash'] and plan['split_hash'] == read_split(config)['hash'], 'refit ancestry')
    folder = config.root/'refit'/kind
    run = read_json(folder/'RUN_IDENTITY.json')
    result = read_json(folder/'result.json')
    init = read_json(folder/'INITIALIZATION.json')
    expected = plan['models'][kind]
    require(result['status'] == 'COMPLETE' and result['optimizer_steps'] == expected['updates'], 'fixed last refit is incomplete')
    require(result['termination']['terminal'] and result['termination']['termination_reason'] == 'MAX_STEPS', 'refit must reach registered endpoint')
    require(result['best_selection_key'] is None, 'refit must not use validation checkpoint selection')
    require(run['dependencies']['refit_plan'] == plan['hash'] and run['dependencies']['phase'] == 'refit', 'wrong refit lineage')
    require(init['initial_state_hash'] == expected['initialization_hash'] and not init['external_scorer_weights_loaded']
            and not init['own_pretrained_scorer_weights_loaded'], 'refit must start from original random initialization')
    checkpoint = Path(result['checkpoint']); receipt = read_json(str(checkpoint)+'.COMPLETE.json')
    require(file_hash(checkpoint) == receipt['checksum'], 'incomplete refit checkpoint')
    path = folder/'best.pt'
    require(file_hash(path) == result['best_checkpoint_hash'], 'fixed last weights changed')
    model, identity = create(config,kind,'refit')
    require(identity['initial_state_hash'] == init['initial_state_hash'], 'constructor/initialization changed')
    state = torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    require(module_hash_parameters(model) == result['parameter_hash_after'], 'best.pt is not the last refit model')
    last=torch.load(checkpoint,map_location='cpu',weights_only=False)
    require(set(last['module'])==set(state) and all(torch.equal(state[k],last['module'][k]) for k in state),
            'fixed last parameters/buffers differ from complete training checkpoint')
    return model.to(device).eval().requires_grad_(False), dict(kind=kind,weights=str(path),weights_hash=file_hash(path),
        state_hash=module_hash(model),checkpoint=str(checkpoint),checkpoint_hash=receipt['checksum'],
        updates=expected['updates'],initialization_hash=init['initial_state_hash'],refit_plan_hash=plan['hash'],
        initialization_file_hash=file_hash(folder/'INITIALIZATION.json'))


def lock_final(config):
    binding = verify_binding(config); endpoints = {}
    for kind in KINDS:
        model, endpoints[kind] = final_endpoint(config,kind)
        del model
    rules = dict(schema='two_fixed_refit_external_evaluation_v1',binding_hash=binding['hash'],models=endpoints,
        refit_plan_hash=file_hash(config.root/'REFIT_PLAN.json'),pool_hash=binding['pool_hash'],
        primary_rule='raw decision_logits argmax; exact ties Base then stable expert_id',
        external_roles=['stage_val','selector_cal','dev_report'],dev_status='repeatedly observed exploratory development, not blind test',
        external_checkpoint_selection=False,external_training_feedback=False,final_test_access=False,
        margin=dict(grid=[0.,.25,.5,1.,2.],units='decision_logit_difference',role='selector_cal',
                    new_NC_budget=0,new_DAC_budget=0,high_to_zero_reported_not_an_added_gate=True,
                    selection='max selected mean among feasible; ties fewer switches then larger gamma'),
        bootstrap_repetitions=config.bootstrap_repetitions,bootstrap_seed=config.bootstrap_seed,
        primary_comparisons=['M_scene-R00','M_set-R00','M_set-M_scene'],
        single_seed=True,ACTIVE_updated=False)
    rules['hash']=digest(rules)
    atomic_json(config.root/'FINAL_EVALUATION_LOCK.json',rules,immutable=True)
    return rules


def _final_lock(config):
    p=config.root/'FINAL_EVALUATION_LOCK.json'
    r=read_json(p) if p.exists() else lock_final(config)
    require(r['hash']==digest({k:v for k,v in r.items() if k!='hash'}) and r['binding_hash']==verify_binding(config)['hash'],'external lock corrupted')
    for kind,row in r['models'].items():
        require(file_hash(row['weights'])==row['weights_hash'],'final selected model changed')
    return r


def predictions(config,kind,role,device):
    require(role in ('stage_val','selector_cal','dev_report'),'external roles only; final_test prohibited')
    frozen=_final_lock(config)
    if role=='dev_report':
        require((config.root/'MARGIN_CALIBRATION.json').is_file(),'lock optional margin before dev')
        require((config.root/'evaluation/stage_val/COMPLETE.json').is_file(),'external stage_val precedes dev')
    meta,bank,scenes=canonical_bank(config,role)
    folder=config.root/'evaluation'/role/kind
    identity=dict(final_lock=frozen['hash'],bank_hash=meta['hash'],scene_ids_hash=digest(meta['scene_ids']),
                  kind=kind,role=role,weights_hash=frozen['models'][kind]['weights_hash'],
                  prediction_code=file_hash(Path(__file__).with_name('training.py')))
    with lock(folder/'PREDICT.lock'):
        path=folder/'predictions.npz';receipt=folder/'COMPLETE.json'
        if reusable(receipt,identity):
            with np.load(path,allow_pickle=False) as f: logits=f['decision_logits'].copy()
        else:
            module,endpoint=final_endpoint(config,kind,device)
            cache=FrozenContexts(config,scenes)
            logits=predict(module,kind,scenes,bank,cache,device)
            require(np.isfinite(logits).all(),'nonfinite decision logits cannot be excluded')
            b=io.BytesIO();np.savez(b,decision_logits=logits,scene_ids=np.asarray(meta['scene_ids']),expert_ids=np.asarray(meta['expert_ids']))
            atomic_bytes(path,b.getvalue(),immutable=True)
            complete(receipt,identity,[path],requested_scenes=len(scenes),processed_scenes=len(scenes),model_state_hash=endpoint['state_hash'])
    return meta,bank,logits


def deployment_choices(kind,logits,bank,expert_ids,gamma=None,always_base=False):
    """Router chooses before discovering invalid outputs; failures are accounted separately."""
    finite=np.isfinite(bank['trajectories']).all((-1,-2))
    require(finite[:,0].all(),'invalid Base requires an explicit failed evaluation, not hidden filtering')
    if always_base: primary=np.zeros(len(logits),dtype=int)
    else:
        available=None if kind=='M_scene' else torch.from_numpy(finite)
        primary=choose(torch.as_tensor(logits),expert_ids,available,gamma).cpu().numpy()
    invalid=~finite[np.arange(len(primary)),primary]
    selected=np.where(invalid,0,primary)
    return selected,dict(primary_selected=primary.tolist(),invalid_selected_fallback_count=int(invalid.sum()),
                         extra_Base_decoder_calls=int(invalid.sum()),actual_selected=selected.tolist())


def statistics(config,meta,bank,selected,*,kind,rule,failures=None):
    require(bank['valid'].all(),'full fixed population requires complete true labels')
    c=np.where(bank['component_valid'],bank['components'],np.nan)
    out=report_metrics(bank['scores'],c,selected,meta['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
    out.update(model=kind,rule=rule,requested_scenes=len(meta['scene_ids']),invalid_score_count=0,
               pool_hash=verify_binding(config)['pool_hash'],bank_hash=meta['hash'],
               selection_counts={eid:int((selected==i).sum()) for i,eid in enumerate(meta['expert_ids'])})
    if failures: out.update({k:v for k,v in failures.items() if k not in ('primary_selected','actual_selected')})
    return out


def evaluate_role(config,role,device='cuda:0'):
    require(role in ('stage_val','dev_report'),'only fixed external/dev role')
    resource(config,'evaluate-decision-'+role)
    frozen=_final_lock(config);folder=config.root/'evaluation'/role
    identity=dict(final_lock=frozen['hash'],role=role)
    if role=='dev_report':identity['margin_hash']=file_hash(config.root/'MARGIN_CALIBRATION.json')
    if reusable(folder/'COMPLETE.json',identity):return read_json(folder/'RESULTS.json')
    meta,bank,_=canonical_bank(config,role);rows=[];choices={};n=len(meta['scene_ids'])
    for i,eid in enumerate(meta['expert_ids']):
        rows.append(statistics(config,meta,bank,np.full(n,i),kind=eid,rule='fixed_policy'))
    rows.append(statistics(config,meta,bank,bank['scores'].argmax(1),kind='Oracle',rule='offline_true_reward_only'))
    reference=read_reference(config,role,meta)
    choices['R00']=select(reference,np.isfinite(bank['trajectories']).all((-1,-2)),expert_ids=meta['expert_ids'])
    rows.append(statistics(config,meta,bank,choices['R00'],kind='R00',rule='existing_raw_argmax'))
    for kind in KINDS:
        _,_,logits=predictions(config,kind,role,device)
        selected,failure=deployment_choices(kind,logits,bank,meta['expert_ids'])
        choices[kind]=selected
        rows.append(statistics(config,meta,bank,selected,kind=kind,rule='raw_argmax',failures=failure))
        if role=='dev_report':
            calibration=read_json(config.root/'MARGIN_CALIBRATION.json')['models'][kind]
            selected,failure=deployment_choices(kind,logits,bank,meta['expert_ids'],calibration['gamma'],calibration['always_Base'])
            choices[kind+'_margin']=selected
            rows.append(statistics(config,meta,bank,selected,kind=kind+'_margin',rule='selector_cal_logit_margin',failures=failure))
    comparisons=[];truth=bank['scores'];ix=np.arange(n)
    for a,b in [('M_scene','R00'),('M_set','R00'),('M_set','M_scene')]:
        delta=truth[ix,choices[a]].astype(float)-truth[ix,choices[b]].astype(float)
        comparisons.append(dict(first=a,second=b,mean_difference_01=float(delta.mean()),
            log_paired_CI=paired_bootstrap(delta,meta['source_logs'],config.bootstrap_repetitions,config.bootstrap_seed)))
    result=dict(role=role,scope='entire_fixed_manifest',requested_scenes=n,results=rows,comparisons=comparisons,
                identities=identity,units='[0,1]',checkpoint_selection=False,training_feedback=False,
                dev_exploratory=role=='dev_report',refit_durations_may_differ=True,
                R00_historical_selection_protocol_differs=True)
    atomic_json(folder/'RESULTS.json',result,immutable=True)
    fields=['model','rule','scenes','logs','base_01','oracle_01','selected_01','selected_gain_01','selected_over_best_fixed_01',
            'regret_01','oracle_capture','switch_rate','high_to_zero','selected_zero','large_gap_regret_0_1','large_gap_regret_0_5']
    atomic_bytes(folder/'RESULTS.csv',csv_bytes([{k:r[k] for k in fields} for r in rows]),immutable=True)
    # Persist every selected ID on the original common denominator for reproducible tail analysis.
    b=io.BytesIO();np.savez(b,scene_ids=np.asarray(meta['scene_ids']),source_logs=np.asarray(meta['source_logs']),
        expert_ids=np.asarray(meta['expert_ids']),**choices)
    atomic_bytes(folder/'SELECTIONS.npz',b.getvalue(),immutable=True)
    complete(folder/'COMPLETE.json',identity,[folder/'RESULTS.json',folder/'RESULTS.csv',folder/'SELECTIONS.npz'],new_scoring_calls=0)
    return result


def calibrate(config,device='cuda:0'):
    resource(config,'calibrate-decision')
    frozen=_final_lock(config);rule=frozen['margin'];out={}
    require((config.root/'evaluation/stage_val/COMPLETE.json').exists(),'external check before optional calibration')
    path=config.root/'MARGIN_CALIBRATION.json'
    for kind in KINDS:
        meta,bank,logits=predictions(config,kind,'selector_cal',device);rows=[]
        for gamma in rule['grid']:
            selected,failure=deployment_choices(kind,logits,bank,meta['expert_ids'],gamma)
            r=statistics(config,meta,bank,selected,kind=kind,rule='logit_margin',failures=failure)
            feasible=all(r['safety'][name]['new_failures']==0 for name in ('no_at_fault_collisions','drivable_area_compliance'))
            rows.append(dict(gamma=gamma,feasible=feasible,metrics=r))
        eligible=[r for r in rows if r['feasible']]
        best=max(eligible,key=lambda r:(r['metrics']['selected_01'],-r['metrics']['switch_count'],r['gamma'])) if eligible else None
        nontrivial=best is not None and best['metrics']['switch_count']>0
        out[kind]=dict(rows=rows,gamma=best['gamma'] if best else None,always_Base=not nontrivial,
            status='NONTRIVIAL_EMPIRICAL_RULE' if nontrivial else 'NO_NONTRIVIAL_RULE',
            rule_is_safety_probability=False,unknown_scene_safety_guarantee=False,
            checkpoint_hash=frozen['models'][kind]['weights_hash'],bank_hash=meta['hash'])
    r=dict(schema='decision_logit_margin_v1',models=out,rules=rule,final_lock_hash=frozen['hash'],
           role='selector_cal',dev_used=False,ACTIVE_updated=False)
    r['hash']=digest(r);atomic_json(path,r,immutable=True)
    return r
