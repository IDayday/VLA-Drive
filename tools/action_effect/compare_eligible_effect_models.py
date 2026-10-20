"""Versioned, read-only G1 comparison with explicit no-intervention admission.

Original evaluation receipts retain their full requested scene population. This
report admits only the dataset-defined held-out pair task after verifying *all*
native validation pair IDs; it never repairs a genuinely missing prediction by
intersecting available outputs. No forward, training or simulation is executed.
"""
from __future__ import annotations
import argparse
import csv,io,json
from pathlib import Path
from .common import artifact,atomic_bytes,digest,read_json,verify_artifact,verify_identity,write_json
from .compare_effect_models import _load_evaluation,checkpoint_state,require_primary_loss_recipe,_source
from .compare_temporal_ablation import _matched as matched_temporal
from .g1_metrics import summarize_records,paired_group_bootstrap
from .train_effect_model import loss_weights
from starVLA.model.modules.action_effect.effect_model import MODES


def admit_population(original,dataset,records):
    """Pure admission: structural no-action is decided by source, never outputs."""
    verify_identity(dataset)
    if (dataset.get('schema')!='action_effect_pair_dataset_v1' or dataset.get('status')!='COMPLETE'
            or dataset.get('split')!='train' or dataset.get('deployment_inputs') is not False):
        raise ValueError('Complete pinned TRAIN native pair dataset required')
    if dataset.get('invalid_scene_evidence'):
        raise ValueError('Native invalid-scene evidence requires separate admission; not structural no-action')
    index=dataset['scene_index'];by_id={r['scene_id']:r for r in index}
    if len(index)!=len(by_id) or len(index)!=dataset['requested_scenes']:
        raise ValueError('Source scene population is missing or duplicated')
    requested=[r for r in index if r['role']=='validation'];identity=original['identity']
    if requested!=identity['requested_scenes']:
        raise ValueError('Original full requested validation population differs from native source')
    valid={r['scene_id']:r for r in requested};native=[r for r in dataset['entries'] if r['role']=='validation']
    native_ids=[r['identity'] for r in native]
    if not native or len(native_ids)!=len(set(native_ids)):
        raise ValueError('Native validation pairs must be nonempty and unique')
    for row in native:
        if row['scene_id'] not in valid or row['source_log']!=valid[row['scene_id']]['source_log']:
            raise ValueError('Native validation scene/log is inconsistent with the source index')
    predicted=[r['pair_identity'] for r in records]
    if (len(predicted)!=len(set(predicted)) or set(predicted)!=set(native_ids)
            or predicted!=identity['heldout_pair_identities']):
        raise ValueError('Every native validation pair must have exactly its locked prediction; no omissions/duplicates')
    native_by_id={r['identity']:r for r in native}
    for row in records:
        if row.get('role')!='validation' or row['scene_id'] not in valid or row['source_log']!=valid[row['scene_id']]['source_log']:
            raise ValueError('Prediction scene/log identity differs from fixed validation population')
        if any(row[key]!=native_by_id[row['pair_identity']][key] for key in ('scene_id','source_log','role','candidate_id','kind')):
            raise ValueError('Prediction metadata differs from its exact native pair')
    eligible={r['scene_id'] for r in native};covered={r['scene_id'] for r in records}
    if covered!=eligible:
        raise ValueError('Missing prediction for a scene that has native intervention pairs')
    structural=[r for r in requested if r['scene_id'] not in eligible]
    missing=sorted(set(valid)-covered)
    old=original['metrics']
    if (old['requested_scenes']!=len(requested) or old['covered_scenes']!=len(eligible)
            or old['heldout_pairs']!=len(native) or old['missing_scene_ids']!=missing
            or missing!=sorted(r['scene_id'] for r in structural)):
        raise ValueError('Original missing/covered accounting does not equal source-defined structural no-pair scenes')
    training_identity,_,_=checkpoint_state(verify_artifact(identity['checkpoint']))
    metrics=summarize_records(records,**identity['label_spec'],requested_scenes=[r for r in requested if r['scene_id'] in eligible],
        mode=identity['arm'],temporal_supervised=False if training_identity.get('loss_weights',{}).get('temporal')==0. else None)
    if metrics['missing_scene_ids'] or not metrics['selection']['coverage_complete']:
        raise ValueError('All eligible native-pair scenes require predictions and valid scores; no masked omission')
    population={'original_requested_scenes':len(requested),'original_requested_logs':len({r['source_log'] for r in requested}),
        'eligible_native_pair_scenes':len(eligible),'eligible_native_pair_logs':len({valid[s]['source_log'] for s in eligible}),
        'complete_native_validation_pairs':len(native),'structural_no_available_intervention_scenes':len(structural),
        'structural_scene_inventory':[{**r,'reason':'zero valid native intervention pairs in the complete locked bank'} for r in structural],
        'eligible_scene_inventory':[r for r in requested if r['scene_id'] in eligible],
        'eligible_pair_ids_hash':digest(sorted(native_ids)),
        'admission_rule':'source-defined all held-out native pairs, never a common intersection of model outputs',
        'eligible_metrics_denominator':'all eligible native-pair validation scenes; not the original complete requested policy population'}
    # This extension defines only an increment for forced identity/no-action.
    # We have no admitted absolute labels for those no-pair scenes here.
    delta_sum=sum(s['selection']['actual_selected_delta'] for s in metrics['scenes'])
    extension={'status':'DEFINED_INCREMENT_ONLY_NOT_MODEL_RECOMMENDATION_SUCCESS',
        'original_requested_scenes':len(requested),'forced_identity_structural_scenes':len(structural),
        'forced_identity_delta':0.,'mean_increment_over_original_requested_scenes':delta_sum/len(requested),
        'absolute_Base_PDMS':None,'absolute_selected_PDMS':None,
        'interpretation':'structural scenes receive a defined identity increment zero; no absolute score, repair success or model decision is fabricated'}
    return {**original,'metrics':metrics,'population_admission':population,'structural_identity_delta_extension':extension,
            'original_metrics_artifact':original['complete']['metrics'],'training_identity':training_identity}


def derive(path):
    # Existing strict loader first validates original COMPLETE, checkpoint,
    # immutable labels, full native pair membership and original metrics exactly.
    original=_load_evaluation(path);identity=original['identity']
    dataset=verify_identity(read_json(verify_artifact(identity['data_artifact'])))
    predictions=read_json(verify_artifact(original['complete']['predictions']))
    return admit_population(original,dataset,predictions['entries'])


def match_controls(loaded,*,step):
    if not loaded:return
    reference=next(iter(loaded.values()));fingerprints=set()
    keys=('base_binding_hash','context_artifact','data_artifact','label_spec','requested_scenes','heldout_pair_identities')
    for arm,x in loaded.items():
        if x['identity']['arm']!=arm or x['identity']['optimizer_step']!=step:
            raise ValueError('Explicit G1 arm or predeclared fixed endpoint differs')
        if any(x['identity'][k]!=reference['identity'][k] for k in keys) or x['population_admission']!=reference['population_admission']:
            raise ValueError('Different source/eligible/requested populations across controls')
        ti=x['training_identity'];require_primary_loss_recipe(ti)
        runtime={k:v for k,v in ti.get('runtime',{}).items() if k!='device'}
        if 'device' in ti.get('runtime',{}):runtime['device_type']=ti['runtime']['device'].split(':')[0]
        fingerprints.add(digest({'model':{k:v for k,v in ti['model_config'].items() if k!='mode'},
            'recipe':ti['recipe'],'seed':ti['seed'],'capacity':x['training']['private_parameters'],'runtime':runtime,
            'fit_recipe':{k:ti.get(k) for k in ('dataset_size','max_updates','sampling','scene_shuffle_map_hash','withheld_training_interventions')}}))
    if len(fingerprints)>1:raise ValueError('Unmatched model capacity, initialization, numerical runtime or optimizer/sampler recipe')


def _row(arm,x):
    m=x['metrics'];sel=m['selection'];pop=x['population_admission']
    return {'arm':arm,'original_requested_scenes':pop['original_requested_scenes'],
        'eligible_native_pair_scenes':pop['eligible_native_pair_scenes'],
        'structural_no_available_intervention_scenes':pop['structural_no_available_intervention_scenes'],
        'native_validation_pairs':pop['complete_native_validation_pairs'],
        'direction_accuracy_scene_weighted':m['direction_accuracy_scene_weighted'],
        'score_delta_scene_rmse':m['label_metrics']['score']['scene_weighted_rmse'],
        'eligible_native_selected_delta':sel['selected_delta'],'eligible_native_regret':sel['regret'],
        'eligible_high_score_to_zero_count':sel['high_score_to_zero_count'],
        'eligible_introduced_collision_count':sel['introduced_events']['collision'],
        'eligible_introduced_road_count':sel['introduced_events']['road'],
        'eligible_introduced_ttc_count':sel['introduced_events']['ttc'],
        'forced_identity_extended_increment_only':x['structural_identity_delta_extension']['mean_increment_over_original_requested_scenes'],
        'original_full_population_PDMS':None}


def compare(evaluations,*,kind,step,output,bootstrap=2000,seed=42):
    expected=set(MODES) if kind=='six' else {'full','no_temporal'} if kind=='temporal' else set()
    if not expected or set(evaluations)!=expected:raise ValueError('Explicit complete six-arm or two-temporal mapping required')
    if type(step) is not int or step<1 or not 1<=bootstrap<=10000:raise ValueError('Explicit fixed endpoint and bounded bootstrap required')
    loaded={};states={}
    for arm,path in evaluations.items():
        if path is None or not Path(path).is_file():states[arm]={'status':'NOT_RUN','path':str(path) if path is not None else None}
        else:
            loaded[arm]=derive(path);states[arm]={'status':'COMPLETE','source_evaluation':artifact(path)}
            if kind=='temporal' and loaded[arm]['training_identity'].get('loss_weights')!=loss_weights(.2 if arm=='full' else 0.):
                raise ValueError('Temporal comparison requires its exact named full/zero objective even before the other run completes')
    if kind=='six':match_controls(loaded,step=step)
    elif len(loaded)==2:
        matched_temporal(loaded['full'],loaded['no_temporal'],step=step)
        if loaded['full']['population_admission']!=loaded['no_temporal']['population_admission']:
            raise ValueError('Temporal arms have different structural/eligible scene populations')
    elif loaded:
        x=next(iter(loaded.values()))
        if x['identity']['arm']!='delta_feedback' or x['identity']['optimizer_step']!=step:
            raise ValueError('Temporal arm/endpoint differs')
    intervals={}
    ref='delta_feedback' if kind=='six' else 'full'
    if ref in loaded:
        for arm,x in loaded.items():
            if arm!=ref:intervals[ref+'_minus_'+arm]=paired_group_bootstrap(loaded[ref]['metrics'],x['metrics'],iterations=bootstrap,seed=seed)
    report={'schema':'action_effect_derived_eligible_comparison_v1','comparison':kind,
        'status':'COMPLETE' if len(loaded)==len(expected) else 'NOT_RUN','completion':'COMPLETE_DESCRIPTIVE' if len(loaded)==len(expected) else 'PENDING',
        'source_commit':_source(),'fixed_optimizer_step':step,'evaluations':states,'rows':[_row(a,x) for a,x in loaded.items()],
        'population_admission':next(iter(loaded.values()))['population_admission'] if loaded else None,
        'original_metrics_artifacts':{a:x['original_metrics_artifact'] for a,x in loaded.items()},
        'eligible_pair_population_metrics':{a:x['metrics'] for a,x in loaded.items()},
        'structural_identity_delta_extension':{a:x['structural_identity_delta_extension'] for a,x in loaded.items()},
        'paired_log_intervals_on_complete_eligible_population':intervals,
        'training_sources':{a:x['training_identity']['source_commit'] for a,x in loaded.items()},
        'scientific_conclusion':'UNTESTED','policy_temporal_credit_claim':'NOT_ESTABLISHED',
        'not_full_original_population_PDMS':True,'original_evaluations_modified':False,
        'new_model_forwards':0,'new_optimizer_updates':0,'new_simulator_calls':0,
        'checkpoint_selection':'explicit same predeclared endpoint; no DEV/navtest model or threshold selection'}
    report['identity']=digest(report);folder=Path(output)/report['identity'];folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'DERIVED_COMPARISON.json',report)
    rows=report['rows'];keys=sorted(set().union(*(r.keys() for r in rows))) if rows else ['arm','completion']
    text=io.StringIO();writer=csv.DictWriter(text,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    atomic_bytes(folder/'ELIGIBLE_PAIR_RESULTS.csv',text.getvalue().encode())
    lines=['# Derived G1 eligible-pair comparison','',f"State: {report['completion']}; endpoint {step}.",'',
        'Original full requested-population COMPLETE/METRICS/predictions are unchanged. Every native validation pair must have exactly its original locked prediction; a genuinely missing prediction or inconsistent scene/log is rejected.',
        '','Eligible admission is defined by the complete immutable native pair bank, never by model outputs or a common subset. The report separately inventories all structural no-intervention scenes. All displayed prediction/ranking/CI denominators refer to the complete eligible-pair population.',
        '','The optional requested-population extension assigns only a zero increment to structural forced identity. It does not invent absolute scores or call no-action a model repair/selection success. No complete-original-population PDMS is reported.',
        '','Full physical errors, native events, uncertainty, temporal-grid diagnostics and risk remain in the JSON. This is held-out TRAIN bank recommendation analysis, not full DEV policy evaluation or proof of policy temporal-credit benefit. Missing experiment receipts remain PENDING.']
    atomic_bytes(folder/'DERIVED_COMPARISON.md',('\n'.join(lines)+'\n').encode())
    return {**report,'report_directory':str(folder)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',choices=('six','temporal'),required=True)
    p.add_argument('--evaluations-json',required=True,help='Exactly six primary arm paths or full/no_temporal paths to original COMPLETE.json; null retains PENDING')
    p.add_argument('--step',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=42);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    try:
        if a.dry_run:print(json.dumps({'status':'NOT_RUN','new_model_forwards':0,'new_simulator_calls':0}));return 0
        result=compare(read_json(a.evaluations_json),kind=a.comparison,step=a.step,output=a.output,bootstrap=a.bootstrap,seed=a.seed)
        print(json.dumps({'status':result['status'],'identity':result['identity'],'report_directory':result['report_directory']}))
        return 0 if result['status']=='COMPLETE' else 2
    except Exception as e:
        print(json.dumps({'status':'FAIL','error_type':type(e).__name__,'error':str(e)}));return 1


if __name__=='__main__':raise SystemExit(main())
