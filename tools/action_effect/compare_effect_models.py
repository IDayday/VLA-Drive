"""CPU-only held-out G1 evaluation and immutable six-control comparison.

Evaluation reads an explicit trained snapshot and all registered held-out TRAIN
pairs. Comparison reads receipts/histories only; it never refreshes a Base,
launches training, simulates a trajectory, or selects a checkpoint using DEV.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import csv
import io
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import torch

from .common import artifact,atomic_bytes,digest,read_json,verify_artifact,verify_identity,write_json
from .g1_metrics import average,paired_group_bootstrap,summarize_records
from starVLA.model.modules.action_effect.effect_model import MODES


def _source():
    root=Path(__file__).resolve().parents[2]
    return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()


def require_primary_loss_recipe(identity):
    # Legacy reporting fixtures may lack this historical field. Every current
    # real G1 identity binds it; the new ablation must not replace a primary arm.
    from .train_effect_model import loss_weights
    if 'loss_weights' in identity and identity['loss_weights']!=loss_weights():
        raise ValueError('Six primary controls require their unchanged loss recipe; compare the temporal ablation separately')


def checkpoint_state(checkpoint):
    path=Path(checkpoint);identity=read_json(path.parent/'IDENTITY.json')
    receipt=read_json(path.with_suffix('.COMPLETE.json'));ih=digest(identity)
    if verify_artifact(receipt['checkpoint']).resolve()!=path.resolve() or receipt['run_identity']!=ih:
        raise ValueError('Checkpoint receipt does not bind the immutable G1 identity')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if (state['schema']!='action_effect_private_checkpoint_v1' or state['run_identity']!=ih
            or state['optimizer_step']!=receipt['optimizer_step'] or state['optimizer_step']<1):
        raise ValueError('Missing/foreign/untrained G1 checkpoint state')
    if identity.get('arm') not in MODES or identity['model_config']['mode']!=identity['arm']:
        raise ValueError('Not one of the six explicitly named G1 controls')
    if set(state['model'])!=set(identity['private_names']):raise ValueError('Private parameter inventory changed')
    if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state['model'].values()):
        raise ValueError('G1 private state must contain finite FP32 parameters')
    return identity,state,receipt


def training_summary(identity,state,result_path=None):
    history=state['history'];steps=[r['optimizer_step'] for r in history]
    if steps!=list(range(1,state['optimizer_step']+1)):
        raise ValueError('Training history must cover each completed optimizer step exactly once')
    seconds=[r['seconds'] for r in history]
    if any(not np.isfinite(v) or v<0 for v in seconds):raise ValueError('Nonfinite/negative training timing')
    out={'optimizer_step':state['optimizer_step'],'terminal':bool(state['terminal']),
        'termination_reason':state['termination_reason'],'consumed_scene_presentations':state['consumed_scenes'],
        'private_parameters':sum(v.numel() for v in state['model'].values()),
        'private_dtype_counts':{'float32':sum(v.numel() for v in state['model'].values())},
        'update_timing':{'count':len(seconds),'mean_seconds':average(seconds),
            'p50_seconds':float(np.median(seconds)) if seconds else None,
            'p95_seconds':float(np.quantile(seconds,.95)) if seconds else None,
            'sum_measured_update_seconds':float(sum(seconds)),
            'includes':'recorded optimizer update wall including original step_fn and synchronization',
            'checkpoint_and_validation_cost':'UNAVAILABLE in per-update timings; not silently amortized'},
        'validation_history':[{'optimizer_step':r['optimizer_step'],'metrics':r['validation']} for r in history if 'validation' in r],
        'runtime':identity.get('runtime',{'status':'UNAVAILABLE_LEGACY_IDENTITY'}),'training_source_commit':identity['source_commit']}
    if result_path is not None and Path(result_path).is_file():
        p=Path(result_path);before=artifact(p);result=read_json(p)
        if result['run_identity']!=state['run_identity'] or result['optimizer_step']<state['optimizer_step']:
            raise ValueError('RESULT is foreign to, or behind, the evaluated snapshot')
        if before!=artifact(p):raise ValueError('RESULT changed while snapshotting; retry at a stable boundary')
        out['result_snapshot']={'artifact':before,'status':result['status'],'optimizer_step':result['optimizer_step'],
            'last_segment_elapsed_seconds':result['elapsed_seconds'],
            'captured_content':result,
            'scope':'last invocation only; not cumulative training cost or a terminal-completion certificate'}
    else:out['result_snapshot']={'status':'UNAVAILABLE'}
    return out


def evaluate(checkpoint,*,context_manifest,pair_dataset,campaign_binding,output,max_pairs,microbatch=8,threads=2,resume=False):
    if not 1<=microbatch<=64 or not 1<=threads<=8 or type(max_pairs) is not int or max_pairs<1:
        raise ValueError('Explicit small CPU microbatch/threads and full-pair budget required')
    from .context_data import ContextReader
    from .train_effect_model import PairData,load_effect_teacher
    started=time.perf_counter();cpu_started=time.process_time();torch.set_num_threads(threads)
    identity,state,_=checkpoint_state(checkpoint)
    if artifact(context_manifest)!=identity['context_artifact'] or artifact(pair_dataset)!=identity['data_artifact']:
        raise ValueError('G1 evaluation must use this snapshot exact frozen context and pair population')
    reader=ContextReader(context_manifest,campaign_binding);data=PairData(pair_dataset,reader)
    if len(data.validation)>max_pairs:raise ValueError('BLOCKED_BUDGET: cannot truncate the fixed held-out pair population')
    if asdict(data.spec)!=identity['label_spec']:
        # JSON stores tuples as lists; canonical digest preserves this equivalence.
        if digest(asdict(data.spec))!=digest(identity['label_spec']):raise ValueError('Physical label schema differs')
    requested=[r for r in data.manifest['scene_index'] if r['role']=='validation']
    declared={'schema':'action_effect_g1_evaluation_identity_v1','checkpoint':artifact(checkpoint),
        'checkpoint_identity':artifact(Path(checkpoint).parent/'IDENTITY.json'),
        'context_artifact':artifact(context_manifest),'data_artifact':artifact(pair_dataset),
        'campaign_binding':artifact(campaign_binding),'base_binding_hash':identity['base_binding_hash'],
        'arm':identity['arm'],'optimizer_step':state['optimizer_step'],'source_commit':_source(),
        'label_spec':identity['label_spec'],'requested_scenes':requested,
        'heldout_pair_identities':[r['identity'] for r in data.validation],
        'runtime':{'device':'cpu','torch':str(torch.__version__),'microbatch':microbatch,'threads':threads,
                   'inference_dtype':'float32','autocast':False},
        'data_role':'held-out TRAIN logs only','checkpoint_selection':'explicit; report does not choose a new snapshot',
        'full_population_required':True}
    declared['identity']=digest(declared);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/'IDENTITY.json',declared)
    if (output/'COMPLETE.json').is_file():
        if not resume:raise FileExistsError('Evaluation already exists; use --resume for verified immutable reuse')
        completion=verify_identity(read_json(output/'COMPLETE.json'))
        if completion['evaluation_identity']!=declared['identity']:raise ValueError('Foreign evaluation completion')
        for key in ('predictions','metrics','training_snapshot'):verify_artifact(completion[key])
        return {**completion,'reused':True}
    model=load_effect_teacher(checkpoint,expected_base_binding_hash=reader.binding['base_checkpoint_binding_hash'],allowed_modes=(identity['arm'],))
    loaded_wall=time.perf_counter()-started;loaded_cpu=time.process_time()-cpu_started
    records=[];forward_started=time.perf_counter();forward_cpu=time.process_time()
    with torch.no_grad(),torch.autocast(device_type='cpu',enabled=False):
        for at in range(0,len(data.validation),microbatch):
            rows=data.validation[at:at+microbatch];p=data.predict(model,rows,'cpu')
            ep=p['event_logits'].softmax(-1);rp=p['repair_logit'].sigmoid()
            for i,row in enumerate(rows):
                before=row['base_labels'];after=row['modified_labels']
                records.append({'pair_identity':row['identity'],'native_pair_identity':row['pair_identity'],
                    'scene_id':row['scene_id'],'source_log':row['source_log'],'role':row['role'],
                    'candidate_id':row['candidate_id'],'kind':row['kind'],
                    'before':before['values'],'after':after['values'],'before_valid':before['valid'],'after_valid':after['valid'],
                    'valid':(np.asarray(before['valid'])&np.asarray(after['valid'])).tolist(),
                    'mean_delta':p['mean_delta'][i].tolist(),'logvar_delta':p['logvar_delta'][i].tolist(),
                    'event_probabilities':ep[i].tolist(),'event_transition':row['event_transition'],
                    'repair_probability':float(rp[i]),'eligible_repair':row['eligible_repair'],
                    'temporal_delta':p['temporal_delta'][i].tolist(),'temporal_target':row['temporal_event_delta'],
                    'temporal_valid':row['temporal_event_valid']})
    forward_wall=time.perf_counter()-forward_started;forward_cpu=time.process_time()-forward_cpu
    metrics=summarize_records(records,names=data.spec.names,units=data.spec.units,scales=data.spec.scales,
        requested_scenes=requested,mode=identity['arm'],
        temporal_supervised=False if identity.get('loss_weights',{}).get('temporal')==0. else None)
    snapshot=training_summary(identity,state,Path(checkpoint).parent/'RESULT.json')
    predictions={'schema':'action_effect_g1_predictions_v1','evaluation_identity':declared['identity'],'entries':records}
    metrics={'schema':'action_effect_g1_metrics_v1','evaluation_identity':declared['identity'],**metrics}
    write_json(output/'PAIR_PREDICTIONS.json',predictions);write_json(output/'METRICS.json',metrics)
    write_json(output/'TRAINING_SNAPSHOT.json',snapshot)
    result={'schema':'action_effect_g1_evaluation_complete_v1','status':'COMPLETE','evaluation_identity':declared['identity'],
        'identity_artifact':artifact(output/'IDENTITY.json'),'predictions':artifact(output/'PAIR_PREDICTIONS.json'),
        'metrics':artifact(output/'METRICS.json'),'training_snapshot':artifact(output/'TRAINING_SNAPSHOT.json'),
        'cpu_cost':{'load_and_native_admission_wall_seconds':loaded_wall,'load_and_native_admission_process_seconds':loaded_cpu,
            'frozen_head_inference_wall_seconds':forward_wall,'frozen_head_inference_process_seconds':forward_cpu,
            'total_wall_seconds':time.perf_counter()-started,'total_process_seconds':time.process_time()-cpu_started,
            'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'gpu_used':False,'simulator_calls':0,'pairs':len(records)},
        'scientific_status':'UNTESTED','deployment_status':'NOT_RUN'}
    result['identity']=digest(result);write_json(output/'COMPLETE.json',result);return result


def _load_evaluation(path):
    complete=verify_identity(read_json(path));identity=verify_identity(read_json(verify_artifact(complete['identity_artifact'])))
    if (complete['schema']!='action_effect_g1_evaluation_complete_v1' or complete['status']!='COMPLETE'
            or complete['evaluation_identity']!=identity['identity']):raise ValueError('Invalid complete G1 evaluation')
    predictions=read_json(verify_artifact(complete['predictions']));metrics=read_json(verify_artifact(complete['metrics']))
    training=read_json(verify_artifact(complete['training_snapshot']))
    for x in (predictions,metrics):
        if x['evaluation_identity']!=identity['identity']:raise ValueError('Prediction/metric identity differs')
    verify_artifact(identity['checkpoint']);verify_artifact(identity['checkpoint_identity'])
    verify_artifact(identity['context_artifact']);verify_artifact(identity['campaign_binding'])
    dataset=verify_identity(read_json(verify_artifact(identity['data_artifact'])))
    native={r['identity']:r for r in dataset['entries'] if r['role']=='validation'}
    if len(native)!=sum(r['role']=='validation' for r in dataset['entries']):
        raise ValueError('Native held-out pairs are not unique')
    if set(native)!=set(identity['heldout_pair_identities']):raise ValueError('Held-out pair population differs from native source')
    if identity['requested_scenes']!=[r for r in dataset['scene_index'] if r['role']=='validation']:
        raise ValueError('Requested held-out scenes differ from native source')
    for r in predictions['entries']:
        source=native.get(r['pair_identity'])
        if source is None:raise ValueError('Prediction has no native compact pair')
        for key in ('scene_id','source_log','role','candidate_id','kind','event_transition','eligible_repair'):
            if r[key]!=source[key]:raise ValueError('Prediction/native compact metadata differs: '+key)
        for target,source_key in (('before','base_labels'),('after','modified_labels')):
            if r[target]!=source[source_key]['values'] or r[target+'_valid']!=source[source_key]['valid']:
                raise ValueError('Prediction/native physical labels differ')
        if r['temporal_target']!=source['temporal_event_delta'] or r['temporal_valid']!=source['temporal_event_valid']:
            raise ValueError('Prediction/native temporal labels differ')
    original_identity,original_state,_=checkpoint_state(identity['checkpoint']['path'])
    if any(identity[k]!=original_identity[k] for k in ('data_artifact','context_artifact','base_binding_hash','arm','label_spec')):
        raise ValueError('Evaluation artifacts differ from the trained checkpoint binding')
    if identity['optimizer_step']!=original_state['optimizer_step']:
        raise ValueError('Evaluation endpoint differs from the trained checkpoint')
    actual_parameters=sum(v.numel() for v in original_state['model'].values())
    if (training['private_parameters']!=actual_parameters or training['optimizer_step']!=original_state['optimizer_step']
            or training['validation_history']!=[{'optimizer_step':r['optimizer_step'],'metrics':r['validation']}
                for r in original_state['history'] if 'validation' in r]):
        raise ValueError('Training snapshot differs from actual immutable checkpoint')
    # Recompute reporting metrics from immutable per-pair outputs, never accept
    # a self-hashed summary that silently excludes a failed scene.
    spec=identity['label_spec'];recomputed=summarize_records(predictions['entries'],**spec,
        requested_scenes=identity['requested_scenes'],mode=identity['arm'],
        temporal_supervised=False if original_identity.get('loss_weights',{}).get('temporal')==0. else None)
    if digest(recomputed)!=digest({k:v for k,v in metrics.items() if k not in ('schema','evaluation_identity')}):
        raise ValueError('Stored G1 metrics do not match their full pair predictions')
    if [r['pair_identity'] for r in predictions['entries']]!=identity['heldout_pair_identities']:
        raise ValueError('Partial/reordered held-out pair coverage')
    return {'complete':complete,'identity':identity,'metrics':metrics,'training':training,'source':artifact(path)}


def compare(evaluations,*,output,step,bootstrap=2000,seed=42):
    if type(step) is not int or step<1 or not 1<=bootstrap<=10000:raise ValueError('Explicit fixed step/bootstrap budget required')
    if set(evaluations)!=set(MODES):raise ValueError('Exactly all six named controls must be declared, including pending paths')
    loaded={};states={}
    for arm,path in evaluations.items():
        if path is None or not Path(path).is_file():
            states[arm]={'status':'NOT_RUN','completion':'PENDING','path':str(path) if path is not None else None};continue
        x=_load_evaluation(path)
        if x['identity']['arm']!=arm or x['identity']['optimizer_step']!=step:
            raise ValueError('Control arm or predeclared fixed optimizer endpoint differs')
        loaded[arm]=x;states[arm]={'status':'PASS','scope':'artifact integrity only; not a scientific pass','source':x['source']}
    match_keys=('base_binding_hash','context_artifact','data_artifact','label_spec','requested_scenes','heldout_pair_identities')
    reference=next(iter(loaded.values()),None)
    if reference:
        for arm,x in loaded.items():
            if any(x['identity'][k]!=reference['identity'][k] for k in match_keys):raise ValueError('Unmatched Base/data/heldout population across G1 controls')
    # All original model/config/cost information is retained, not just one mean.
    rows=[];matched_counts=set();matched_cfg=set();matched_recipe=set();matched_seeds=set();training_sources=set()
    for arm in MODES:
        if arm not in loaded:rows.append({'arm':arm,'completion':'PENDING','optimizer_step':step});continue
        x=loaded[arm];ti=read_json(verify_artifact(x['identity']['checkpoint_identity']));metrics=x['metrics'];training=x['training']
        require_primary_loss_recipe(ti)
        cfg={k:v for k,v in ti['model_config'].items() if k!='mode'}
        matched_counts.add(training['private_parameters']);matched_cfg.add(digest(cfg));matched_recipe.add(digest(ti['recipe']))
        matched_seeds.add(ti['seed'])
        training_sources.add(ti['source_commit']);selection=metrics['selection']
        rows.append({'arm':arm,'completion':'COMPLETE','optimizer_step':step,'private_parameters':training['private_parameters'],
            'requested_scenes':metrics['requested_scenes'],'covered_scenes':metrics['covered_scenes'],'heldout_pairs':metrics['heldout_pairs'],
            'direction_accuracy_scene_weighted':metrics['direction_accuracy_scene_weighted'],
            'direction_accuracy_pair_weighted':metrics['direction_accuracy_pair_weighted'],
            'score_delta_scene_rmse':metrics['label_metrics']['score']['scene_weighted_rmse'],
            'selected_native_delta':selection['selected_delta'],'oracle_native_delta':selection['oracle_delta'],
            'selection_regret':selection['regret'],'high_score_to_zero_count':selection['high_score_to_zero_count'],
            'introduced_collision_count':selection['introduced_events']['collision'],
            'introduced_road_count':selection['introduced_events']['road'],'introduced_ttc_count':selection['introduced_events']['ttc'],
            'measured_update_seconds_sum':training['update_timing']['sum_measured_update_seconds'],
            'cpu_evaluation_wall_seconds':x['complete']['cpu_cost']['total_wall_seconds']})
    if any(len(x)>1 for x in (matched_counts,matched_cfg,matched_recipe,matched_seeds)):
        raise ValueError('Capacity, initialization recipe or optimizer recipe differs across controls')
    intervals={}
    if 'delta_feedback' in loaded:
        for arm,x in loaded.items():
            if arm!='delta_feedback':
                intervals['delta_feedback_minus_'+arm]=paired_group_bootstrap(loaded['delta_feedback']['metrics'],x['metrics'],iterations=bootstrap,seed=seed)
    report={'schema':'action_effect_g1_six_control_comparison_v1','status':'COMPLETE' if len(loaded)==len(MODES) else 'NOT_RUN',
        'completion':'COMPLETE_DESCRIPTIVE' if len(loaded)==len(MODES) else 'PENDING',
        'fixed_optimizer_step':step,'source_commit':_source(),'controls':states,'rows':rows,
        'matched_parameter_count':next(iter(matched_counts),None),'model_recipe_matched':len(matched_cfg)<=1,
        'optimizer_recipe_matched':len(matched_recipe)<=1,'training_source_commits':sorted(training_sources),
        'source_commit_note':'Source revisions are disclosed; equal dimensions do not prove equal source semantics.',
        'matched_artifacts':{k:reference['identity'][k] for k in match_keys if k not in ('requested_scenes','heldout_pair_identities')} if reference else None,
        'paired_log_intervals':intervals,
        'full_metrics':{a:x['complete']['metrics'] for a,x in loaded.items()},
        'per_scene_metrics':'Each METRICS artifact contains every covered held-out scene and the requested/missing population.',
        'checkpoint_selection':{'status':'NOT_SELECTED_BY_REPORT','rule':'explicit same predeclared fixed step',
            'default_teacher':'delta_feedback is the declared paired-effect method, not selected by DEV/navtest',
            'forbidden':'No automatic best-arm/checkpoint/threshold selection or production activation'},
        'missing_privileged_control':{'status':'BLOCKED','reason':'No legal privileged feature asset is registered; no fake G control.'},
        'scientific_conclusion':'UNTESTED','gpu_or_simulator_started':False}
    report['identity']=digest(report);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/'COMPARISON.json',report)
    keys=sorted(set().union(*(r.keys() for r in rows)));stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    atomic_bytes(output/'SIX_CONTROL_RESULTS.csv',stream.getvalue().encode())
    lines=['# G1 six-control fixed-endpoint comparison','',f"Completion: **{report['completion']}**, endpoint {step}.",
        'Artifact integrity and CPU numerical diagnostics are separate from scientific benefit.','',
        '| Arm | Completion | Scenes | Score Δ RMSE | Native top-1 Δ | Regret |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+ ' | '.join(str(r.get(k,'—')) for k in ('arm','completion','covered_scenes','score_delta_scene_rmse','selected_native_delta','selection_regret'))+' |')
    lines.extend(['','The JSON pins all physical label, event, uncertainty, per-scene and training-cost artifacts. Missing tasks retain their real masks; missing event times are never treated as zero.',
        '','Top-k true gain is an offline upper bound within recommendations. Only top-1 predicted selection uses no oracle. Scene/log-matched native scores belong to held-out TRAIN intervention banks; this is not a DEV/NAVTEST policy result.',
        '','The report does not choose a model, tune a threshold, claim multi-seed stability, or mark a physical research hypothesis PASS.'])
    atomic_bytes(output/'G1_COMPARISON.md',('\n'.join(lines)+'\n').encode());return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    e=sub.add_parser('evaluate',help='CPU-only full held-out prediction receipt for one explicit snapshot')
    for key in ('checkpoint','context-manifest','pair-dataset','campaign-binding','output'):e.add_argument('--'+key,required=True)
    e.add_argument('--max-pairs',type=int,required=True);e.add_argument('--microbatch',type=int,default=8);e.add_argument('--threads',type=int,default=2)
    e.add_argument('--resume',action='store_true');e.add_argument('--dry-run',action='store_true')
    c=sub.add_parser('compare',help='Read six explicitly declared immutable completion paths; absent controls stay pending')
    c.add_argument('--evaluations-json',required=True,help='JSON mapping all six arm names to COMPLETE.json paths or null')
    c.add_argument('--step',type=int,required=True);c.add_argument('--bootstrap',type=int,default=2000);c.add_argument('--seed',type=int,default=42)
    c.add_argument('--output',required=True);c.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    try:
        if a.dry_run:print(json.dumps({'status':'NOT_RUN','phase':a.phase,'device':'cpu','simulator_calls':0}));return 0
        if a.phase=='evaluate':result=evaluate(a.checkpoint,context_manifest=a.context_manifest,pair_dataset=a.pair_dataset,
            campaign_binding=a.campaign_binding,output=a.output,max_pairs=a.max_pairs,microbatch=a.microbatch,threads=a.threads,resume=a.resume)
        else:result=compare(read_json(a.evaluations_json),output=a.output,step=a.step,bootstrap=a.bootstrap,seed=a.seed)
        print(json.dumps({'status':result['status'],'identity':result['identity'],'output':a.output}));return 0 if result['status']=='COMPLETE' else 2
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)}));return 1


if __name__=='__main__':raise SystemExit(main())
