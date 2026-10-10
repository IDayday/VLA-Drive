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
    result_path = folder/'EVALUATION.json'
    if result_path.exists():
        report=read_json(result_path)
        require(report['identity'] == identity, 'Base evaluation dependency mismatch')
        return report | {'reused':True}
    # Evaluation never writes LOCKED_S0 and never accesses training optimizer state.
    contract = deepcopy(pipeline.contract)
    contract.update(binding_status='LOCKED_QUERY_S0',query_checkpoint=str(Path(checkpoint).resolve()),query_checkpoint_hash=info['checksum'])
    costs={'checkpoint_read_seconds':time.time()-start,'inference_seconds':0.,'official_scoring_seconds':0.}
    with initialization_rng(0):
        adapter = S0Adapter.load_and_validate_s0(contract,pipeline.device)
        before = module_hash(adapter)
        rows=[]
        for scene in scenes:
            path=folder/'scenes'/(scene.scene_id+'.json')
            if path.exists():
                row=read_json(path)
                require(row['observation_hash']==scene.observation_hash and row['context_hash']==scene.metric_context_hash, 'Base evaluation scene changed')
                rows.append(row);continue
            t=time.perf_counter()
            with torch.no_grad():
                features=adapter.encode_scene([load_observation(scene,contract)])
                prediction=adapter.predict_s0(features)
            raw=prediction.raw.detach().cpu()
            physical=prediction.physical.detach().cpu()
            target=load_target(scene,pipeline.trajectory_contract)[None]
            target_physical=pipeline.trajectory_contract.physical(target)
            finite=bool(torch.isfinite(raw).all() and torch.isfinite(physical).all())
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
            trajectory_hash=save_trajectory(ref,physical[0].numpy())
            row['trajectory_ref']=str(ref);row['trajectory_hash']=trajectory_hash
            if pipeline.config['base_evaluation']['official']:
                candidate=CandidateRecord(1,scene.scene_id,'expert_0',info['checksum'],digest(contract),trajectory_hash,str(ref),
                    pipeline.trajectory_contract.raw_representation,'ego_relative_rear_axle',pipeline.trajectory_contract.horizon,
                    pipeline.trajectory_contract.dt,finite,pipeline.contract['iqe_code_hash'],pipeline.config['config_hash'],'base_independent_evaluation')
                t=time.perf_counter();score=pipeline.backend().score(candidate,scene)
                costs['official_scoring_seconds']+=time.perf_counter()-t
                row['score']=asdict(score)
            atomic_json(path,row,immutable=True);rows.append(row)
        require(module_hash(adapter)==before,'evaluate-base changed model parameters/buffers')
    valid_scores=[r['score'] for r in rows if r['score'] and r['score']['score_valid']]
    def summarize(group):
        scores=[r['score'] for r in group if r['score'] and r['score']['score_valid']]
        out={'scenes':len(group),'nonfinite':sum(not r['finite'] for r in group),'valid_official_scores':len(scores),
             'PDMS_01':float(np.mean([s['total_score_01'] for s in scores])) if scores else None,
             'zero_rate':sum(s['total_score_01']<=1e-8 for s in scores)/len(scores) if scores else None}
        for k in ('raw_IL_loss','ADE_m','FDE_m','yaw_error_rad'):
            values=[r[k] for r in group if r[k] is not None]
            out[k]=float(np.mean(values)) if values else None
        out['components']={c:float(np.mean([s['named_components'][c] for s in scores if s['component_valid_masks'][c]]))
            for c in pipeline.contract['metric']['components'] if any(s['component_valid_masks'][c] for s in scores)}
        return out
    report={'status':'COMPLETE','identity':identity,'binding':key,'optimizer_step':info['optimizer_step'],
        'checkpoint':info['path'],'checkpoint_mode':state.get('mode'),'smoke_or_profile':state.get('mode')!='full',
        'model_hash_unchanged':True,'training_state_read_only':True,'started_unix':start,'finished_unix':time.time(),
        'summary':summarize(rows),'per_log':{g:summarize([r for r in rows if r['log']==g]) for g in sorted({r['log'] for r in rows})},
        'rows':rows,'costs':costs,'result_path':str(result_path.resolve()),'scientific_gain':'UNTESTED'}
    atomic_json(result_path,report,immutable=True)
    return report


def qualify_base(pipeline, checkpoint, evaluation_path):
    info=inspect_checkpoint(checkpoint)
    scenes,manifest_hash=fixed_manifest(pipeline)
    evaluation=read_json(evaluation_path)
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
        for key in ('minimum_mean_PDMS_01','maximum_zero_rate','maximum_NC_violation_rate','maximum_DAC_violation_rate','minimum_scenes','minimum_logs'):
            require(key in rules and isinstance(rules[key],(int,float)), 'qualification rule missing: '+key)
        checks.update(minimum_scenes=len(scenes)>=rules['minimum_scenes'],minimum_logs=len(evaluation['per_log'])>=rules['minimum_logs'],
                      mean_PDMS=summary['PDMS_01'] is not None and summary['PDMS_01']>=rules['minimum_mean_PDMS_01'],
                      zero_rate=summary['zero_rate'] is not None and summary['zero_rate']<=rules['maximum_zero_rate'])
        for name,metric in [('NC','no_at_fault_collisions'),('DAC','drivable_area_compliance')]:
            valid=[r['score'] for r in evaluation['rows'] if r['score'] and r['score']['score_valid'] and r['score']['component_valid_masks'][metric]]
            checks[name]=len(valid)==len(scenes) and sum(s['named_components'][metric]<1-1e-8 for s in valid)/len(scenes)<=rules['maximum_'+name+'_violation_rate']
    status='QUALIFIED' if all(checks.values()) else 'BASE_QUALITY_UNTESTED' if not registered else 'BASE_UNQUALIFIED'
    report={'status':status,'binding':expected,'optimizer_step':info['optimizer_step'],'rules':rules,
            'rules_hash':rules_hash,'evaluation_path':str(Path(evaluation_path).resolve()),'evaluation_hash':file_hash(evaluation_path),
            'checks':checks,'smoke_or_profile':evaluation['smoke_or_profile'],
            'baseline_relationship':rules.get('baseline_relationship','NO_BOUND_COMPARABLE_BASELINE') if rules else 'NO_BOUND_COMPARABLE_BASELINE'}
    atomic_json(pipeline.root/'BASE_QUALIFICATION.json',report,immutable=True)
    return report


def require_qualification(pipeline, checkpoint):
    path=pipeline.config['base_evaluation']['qualification'] or pipeline.root/'BASE_QUALIFICATION.json'
    if not Path(path).exists():
        raise BlockedError('BASE_QUALITY_UNTESTED: formal freeze requires matching BASE_QUALIFICATION.json')
    report=read_json(path)
    require(report['status']=='QUALIFIED' and all(report['checks'].values()) and not report['smoke_or_profile'], 'BASE_UNQUALIFIED')
    _,manifest_hash=fixed_manifest(pipeline)
    expected=binding(pipeline.contract,file_hash(checkpoint),manifest_hash,pipeline.backend().protocol_hash)
    require(report['binding']==expected,'BASE_UNQUALIFIED: checkpoint/framework/input/normalizer/validation/protocol mismatch')
    require(file_hash(report['evaluation_path'])==report['evaluation_hash'],'BASE_UNQUALIFIED: evaluation changed')
    rules_path=pipeline.config['base_evaluation']['rules']
    require(rules_path and file_hash(rules_path)==report['rules_hash'],'BASE_UNQUALIFIED: preregistered rules changed')
    return {'path':str(Path(path).resolve()),'checksum':file_hash(path),'binding':expected}
