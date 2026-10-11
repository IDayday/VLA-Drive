"""Read-only checkpoint evaluation and quality qualification before formal locking."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from .contracts import require, CandidateRecord
from .io import atomic_json, digest, file_hash, read_json, BlockedError
from .data.manifests import load_scenes, save_scenes
from .data.sources import load_observation, load_target
from .data.candidate_bank import save_trajectory
from .s0_adapter import S0Adapter
from .training.resume import inspect_checkpoint
from .training.randomness import initialization_rng
from .evaluation.retention import module_hash


def binding(contract, checkpoint_hash, manifest_hash, protocol_hash):
    return {'checkpoint_hash':checkpoint_hash, 'framework_hash':contract['framework_contract_hash'],
            'input_hash':digest({k:contract[k] for k in ('tokenizer_hash','prompt_hash','transform_hash','camera_time_hash')}),
            'normalizer_hash':contract['normalizer_hash'], 'validation_manifest_hash':manifest_hash,
            'score_protocol_hash':protocol_hash}


def fixed_manifest(pipeline):
    source = pipeline.config['base_evaluation']['manifest']
    require(source is not None, 'evaluate-base requires a predeclared fixed base_evaluation.manifest')
    scenes = load_scenes(source)
    require(scenes and all(s.split_role == 'stage_val' for s in scenes), 'Base selection only accepts stage_val')
    require(len(scenes) <= pipeline.max_samples, 'fixed Base validation exceeds explicit sample budget; cannot truncate its list')
    from .data.splits import validate_isolation
    validate_isolation(scenes)
    return scenes, file_hash(source)


def evaluate_base(pipeline, checkpoint):
    start = time.time()
    info = inspect_checkpoint(checkpoint)
    require(info['schema'] == 'iqe_query_base', 'evaluate-base requires a complete Query Base checkpoint')
    scenes, manifest_hash = fixed_manifest(pipeline)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    require(state['framework_contract_hash'] == pipeline.contract['framework_contract_hash'], 'Base evaluation framework mismatch')
    require(not state.get('learned_driving_weights_loaded', True), 'unexpected old learned driving weights')
    require(not set(state['fit_scene_ids']) & {s.scene_id for s in scenes}, 'Base validation was seen by Query base training')
    source_index = read_json(Path(pipeline.config['data']['train_root'])/'index.json')
    fit_ids = set(state['fit_scene_ids'])
    fit_logs = {row['log'] for row in source_index if row['token'] in fit_ids}
    require(fit_logs and not fit_logs & {s.source_log_id for s in scenes}, 'Base validation log leakage or missing training source evidence')
    key = binding(pipeline.contract, info['checksum'], manifest_hash, pipeline.backend().protocol_hash)
    rules_path = pipeline.config['base_evaluation']['rules']
    rules_hash = file_hash(rules_path) if rules_path else None
    identity = key | {'rules_hash':rules_hash, 'official':pipeline.config['base_evaluation']['official']}
    folder = pipeline.root/'base_evaluation'/digest(identity)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/'IDENTITY.json',identity,immutable=True)
    save_scenes(folder/'stage_val.json',scenes)
    # Preserve legacy artifacts. New attempts use a separate schema identity.
    legacy_folder = folder
    identity = identity | {'evaluation_schema': 2, 'eval_seed': pipeline.config['metric']['eval_seed']}
    folder = pipeline.root/'base_evaluation'/digest(identity)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/'IDENTITY.json', identity, immutable=True)
    result_path = folder/'EVALUATION.json'
    from .base_evaluation_cache import candidate_for, verify_row, score_prediction, summarize, verify_completion, complete_evaluation
    from .io import lock
    budget = pipeline.config['base_evaluation']['scoring_attempts_per_scene_per_call']
    require(type(budget) is int and 0 <= budget <= 10, 'invalid scoring retry budget')
    contract = deepcopy(pipeline.contract)
    contract.update(binding_status='LOCKED_QUERY_S0',query_checkpoint=str(Path(checkpoint).resolve()),query_checkpoint_hash=info['checksum'])
    costs={'checkpoint_read_seconds':time.time()-start,'inference_seconds':0.,'official_scoring_seconds':0.}
    # One writer per evaluation. SIGKILL releases the lock; predictions and attempts survive.
    with lock(folder/'evaluation.lock'), initialization_rng(0):
        if result_path.exists():
            report = read_json(result_path)
            require(report['identity'] == identity, 'Base evaluation dependency mismatch')
            require([r['scene_id'] for r in report['rows']] == [s.scene_id for s in scenes], 'evaluation scene list changed')
            for scene, row in zip(scenes, report['rows']):
                verify_row(row, scene, candidate_for(row, scene, contract, pipeline, info['checksum']), pipeline.backend().protocol_hash)
            if not Path(str(result_path)+'.COMPLETE.json').exists():
                require(report['status']=='COMPLETE' and all(r['outcome'] in {'valid_scored','invalid_prediction','prediction_only'} for r in report['rows']),
                        'BASE_EVALUATION_INCOMPLETE')
                complete_evaluation(result_path, report['rows'])
            verify_completion(result_path)
            return report | {'reused': True}
        adapter = None
        rows=[]
        for scene in scenes:
            path=folder/'scenes'/scene.scene_id/'PREDICTION.json'
            legacy_path=legacy_folder/'scenes'/(scene.scene_id+'.json')
            if path.exists():
                row=read_json(path)
            elif legacy_path.exists():
                old=read_json(legacy_path)
                row={k:v for k,v in old.items() if k != 'score'} | {'score': None}
                candidate=candidate_for(row,scene,contract,pipeline,info['checksum'])
                verify_row(old,scene,candidate,pipeline.backend().protocol_hash)
                atomic_json(path,row,immutable=True)
                atomic_json(path.parent/'LEGACY_SOURCE.json', {'path':str(legacy_path),'checksum':file_hash(legacy_path),
                    'status':'legacy_incomplete' if not old.get('score') or not old['score']['score_valid'] else 'valid_scored'},immutable=True)
                if old.get('score') and old['score']['score_valid']:
                    atomic_json(path.parent/'SCORED.json',{'prediction_hash':digest(row),'score':old['score']},immutable=True)
            else:
                if adapter is None:
                    adapter=S0Adapter.load_and_validate_s0(contract,pipeline.device)
                    before=module_hash(adapter)
                t=time.perf_counter()
                with torch.no_grad():
                    features=adapter.encode_scene([load_observation(scene,contract)])
                    prediction=adapter.predict_s0(features)
                raw=prediction.raw.detach().cpu()
                physical=prediction.physical.detach().cpu()
                target=load_target(scene,pipeline.trajectory_contract)[None]
                target_physical=pipeline.trajectory_contract.physical(target)
                finite=bool(torch.isfinite(raw).all() and torch.isfinite(physical).all()
                            and physical.shape == target_physical.shape and raw.shape == target.shape)
                row={'scene_id':scene.scene_id,'log':scene.source_log_id,'observation_hash':scene.observation_hash,
                     'context_hash':scene.metric_context_hash,'finite':finite,'raw_IL_loss':None,'ADE_m':None,'FDE_m':None,'yaw_error_rad':None,'score':None}
                if finite:
                    with torch.no_grad():
                        terms=adapter.compute_expert_il_loss(prediction,target.to(pipeline.device))
                    error=(physical[...,:2]-target_physical[...,:2]).norm(dim=-1)
                    dy=physical[...,2]-target_physical[...,2]
                    row.update(raw_IL_loss=float(terms['il'].numerator/terms['il'].denominator.clamp_min(1)),
                               ADE_m=float(error.mean()),FDE_m=float(error[:,-1].mean()),
                               yaw_error_rad=float(torch.atan2(dy.sin(),dy.cos()).abs().mean()))
                costs['inference_seconds']+=time.perf_counter()-t
                ref=folder/'trajectories'/(scene.scene_id+'.npz')
                row.update(trajectory_ref=str(ref),trajectory_hash=save_trajectory(ref,physical[0].numpy()))
                atomic_json(path,row,immutable=True)
            candidate=candidate_for(row,scene,contract,pipeline,info['checksum'])
            verify_row(row,scene,candidate,pipeline.backend().protocol_hash)
            t=time.perf_counter()
            rows.append(score_prediction(folder,row,scene,candidate,pipeline.backend(),budget=budget,
                official=pipeline.config['base_evaluation']['official']))
            costs['official_scoring_seconds']+=time.perf_counter()-t
        if adapter is not None:
            require(module_hash(adapter)==before,'evaluate-base changed model parameters/buffers')
        summary=summarize(rows,pipeline.contract['metric']['components'])
        complete=summary['unfinished']==0
        attempt=len(list((folder/'attempts').glob('*.json')))
        report_path=result_path if complete else folder/'attempts'/f'{attempt:04d}.json'
        report={'schema_version':2,'status':'COMPLETE' if complete else 'BLOCKED_RETRYABLE','identity':identity,'binding':key,'optimizer_step':info['optimizer_step'],
            'checkpoint':info['path'],'checkpoint_mode':state.get('mode'),'smoke_or_profile':state.get('mode')!='full',
            'model_hash_unchanged':True,'training_state_read_only':True,'started_unix':start,'finished_unix':time.time(),
            'summary':summary,'per_log':{g:summarize([r for r in rows if r['log']==g],pipeline.contract['metric']['components']) for g in sorted({r['log'] for r in rows})},
            'rows':rows,'costs':costs,'result_path':str(report_path.resolve()),'scientific_gain':'UNTESTED'}
        atomic_json(report_path,report,immutable=True)
        if complete:
            complete_evaluation(result_path, rows)
        atomic_json(folder/'LATEST_ATTEMPT.json',{'path':str(report_path.resolve()),'checksum':file_hash(report_path),'status':report['status']})
        return report


def qualify_base(pipeline, checkpoint, evaluation_path, *, select=False):
    info=inspect_checkpoint(checkpoint)
    scenes,manifest_hash=fixed_manifest(pipeline)
    from .base_evaluation_cache import verify_completion
    evaluation=verify_completion(evaluation_path, require_all_valid=False)
    expected=binding(pipeline.contract,info['checksum'],manifest_hash,pipeline.backend().protocol_hash)
    require(evaluation['binding']==expected and evaluation['optimizer_step']==info['optimizer_step'],'qualification checkpoint/manifest/protocol mismatch')
    rules_path=pipeline.config['base_evaluation']['rules']
    rules=read_json(rules_path) if rules_path else None
    rules_hash=file_hash(rules_path) if rules_path else None
    require(evaluation['identity']['rules_hash']==rules_hash,'rules changed after evaluation; cannot tune thresholds on observed results')
    registered=bool(rules and rules.get('status')=='PREREGISTERED' and rules.get('registered_unix',float('inf')) < evaluation['started_unix'])
    summary=evaluation['summary']
    checks={'rules_preregistered':registered,'formal_checkpoint':not evaluation['smoke_or_profile'],
            'complete_official_labels':summary['valid_official_scores']==len(scenes),'all_finite':summary['nonfinite']==0}
    if registered:
        require(rules['validation_manifest_hash']==manifest_hash and rules['score_protocol_hash']==expected['score_protocol_hash'],'registered protocol differs')
        scope=rules.get('qualification_scope','planning_quality')
        require(scope in {'planning_quality','engineering_incremental_experiment'}, 'unknown Base qualification scope')
        checks.update(minimum_scenes=len(scenes)>=rules['minimum_scenes'],minimum_logs=len(evaluation['per_log'])>=rules['minimum_logs'])
        if rules.get('candidate_checkpoint_hashes'):
            checks['predeclared_checkpoint']=info['checksum'] in rules['candidate_checkpoint_hashes']
        if scope=='engineering_incremental_experiment':
            require(rules.get('scientific_baseline_quality')=='UNTESTED' and rules.get('permits_deployment') is False,
                    'engineering qualification cannot assert convergence or authorize deployment')
            checks['model_read_only']=evaluation.get('model_hash_unchanged') is True and evaluation.get('training_state_read_only') is True
            checks['finite_planning_statistics']=all(summary.get(k) is not None and np.isfinite(summary[k])
                for k in ('PDMS_01','raw_IL_loss','ADE_m','FDE_m','yaw_error_rad'))
            for metric in ('no_at_fault_collisions','drivable_area_compliance'):
                checks[metric+'_labels']=all(r['score'] and r['score']['score_valid'] and r['score']['component_valid_masks'].get(metric,False)
                                           for r in evaluation['rows'])
        else:
            for key in ('minimum_mean_PDMS_01','maximum_zero_rate','maximum_NC_violation_rate','maximum_DAC_violation_rate','minimum_scenes','minimum_logs'):
                require(key in rules and isinstance(rules[key],(int,float)), 'qualification rule missing: '+key)
            checks.update(minimum_scenes=len(scenes)>=rules['minimum_scenes'],minimum_logs=len(evaluation['per_log'])>=rules['minimum_logs'],
                          mean_PDMS=summary['PDMS_01'] is not None and summary['PDMS_01']>=rules['minimum_mean_PDMS_01'],
                          zero_rate=summary['zero_rate'] is not None and summary['zero_rate']<=rules['maximum_zero_rate'])
            for name,metric in [('NC','no_at_fault_collisions'),('DAC','drivable_area_compliance')]:
                valid=[r['score'] for r in evaluation['rows'] if r['score'] and r['score']['score_valid'] and r['score']['component_valid_masks'][metric]]
                checks[name]=len(valid)==len(scenes) and sum(s['named_components'][metric]<1-1e-8 for s in valid)/len(scenes)<=rules['maximum_'+name+'_violation_rate']
    status='QUALIFIED' if all(checks.values()) else 'BASE_QUALITY_UNTESTED' if not registered else 'BASE_UNQUALIFIED'
    report={'status':status,'qualification_scope':rules.get('qualification_scope','planning_quality') if rules else 'unregistered',
            'scientific_baseline_quality':rules.get('scientific_baseline_quality','UNTESTED') if rules else 'UNTESTED',
            'binding':expected,'optimizer_step':info['optimizer_step'],'rules':rules,
            'rules_hash':rules_hash,'evaluation_path':str(Path(evaluation_path).resolve()),'evaluation_hash':file_hash(evaluation_path),
            'checks':checks,'smoke_or_profile':evaluation['smoke_or_profile'],
            'baseline_relationship':rules.get('baseline_relationship','NO_BOUND_COMPARABLE_BASELINE') if rules else 'NO_BOUND_COMPARABLE_BASELINE'}
    identity=expected | {'rules_hash':rules_hash,'evaluation_hash':report['evaluation_hash']}
    qualification_id=digest(identity)
    path=pipeline.root/'base_qualification'/qualification_id/'QUALIFICATION.json'
    report.update(qualification_id=qualification_id,qualification_path=str(path.resolve()))
    atomic_json(path,report,immutable=True)
    pointer={'qualification_id':qualification_id,'path':str(path.resolve()),'checksum':file_hash(path),'status':status}
    atomic_json(pipeline.root/'LATEST_QUALIFICATION_ATTEMPT.json',pointer)
    if select:
        require(status=='QUALIFIED','cannot select an unqualified Base')
        locked=pipeline.root/'LOCKED_S0.json'
        if locked.exists() and read_json(locked).get('base_mode')=='full':
            previous=read_json(locked)
            require(previous['query_checkpoint_hash']==info['checksum'] and previous['base_qualification']['checksum']==file_hash(path),
                    'LOCKED_BASE_IMMUTABLE: use a new experiment output namespace')
        atomic_json(pipeline.root/'SELECTED_BASE_QUALIFICATION.json',pointer)
    return report


def require_qualification(pipeline, checkpoint, *, locked_qualification=None):
    explicit=pipeline.config['base_evaluation']['qualification']
    if locked_qualification:
        path=Path(locked_qualification['path'])
        require(file_hash(path)==locked_qualification['checksum'], 'locked qualification checksum mismatch')
        require(not explicit or Path(explicit).resolve()==path.resolve(), 'explicit qualification differs from immutable lock')
    elif explicit:
        path=Path(explicit)
    elif (pipeline.root/'SELECTED_BASE_QUALIFICATION.json').exists():
        pointer=read_json(pipeline.root/'SELECTED_BASE_QUALIFICATION.json')
        path=Path(pointer['path'])
        require(file_hash(path)==pointer['checksum'], 'selected qualification checksum mismatch')
    else:
        path=pipeline.root/'BASE_QUALIFICATION.json'
    if not Path(path).exists():
        raise BlockedError('BASE_QUALITY_UNTESTED: formal freeze requires matching BASE_QUALIFICATION.json')
    report=read_json(path)
    if 'qualification_id' not in report:
        atomic_json(pipeline.root/'base_qualification'/'legacy'/file_hash(path)/'MIGRATION.json',
            {'source_path':str(Path(path).resolve()),'checksum':file_hash(path),'status':report['status'],'action':'read_only_legacy_reference'},immutable=True)
    require(report['status']=='QUALIFIED' and all(report['checks'].values()) and not report['smoke_or_profile'], 'BASE_UNQUALIFIED')
    _,manifest_hash=fixed_manifest(pipeline)
    expected=binding(pipeline.contract,file_hash(checkpoint),manifest_hash,pipeline.backend().protocol_hash)
    require(report['binding']==expected,'BASE_UNQUALIFIED: checkpoint/framework/input/normalizer/validation/protocol mismatch')
    require(file_hash(report['evaluation_path'])==report['evaluation_hash'],'BASE_UNQUALIFIED: evaluation changed')
    from .base_evaluation_cache import verify_completion
    evaluation=verify_completion(report['evaluation_path'],require_all_valid=True)
    require(evaluation['binding']==expected, 'BASE_UNQUALIFIED: evaluation binding mismatch')
    if 'qualification_id' in report:
        require(report['qualification_id']==digest(expected | {'rules_hash':report['rules_hash'],'evaluation_hash':report['evaluation_hash']}), 'qualification identity mismatch')
    rules_path=pipeline.config['base_evaluation']['rules']
    require(rules_path and file_hash(rules_path)==report['rules_hash'],'BASE_UNQUALIFIED: preregistered rules changed')
    return {'path':str(Path(path).resolve()),'checksum':file_hash(path),'binding':expected}
