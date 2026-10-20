"""Exactly two prespecified, untrained G2 controls; no simulator or best-of-N.

Each scene produces one proposal from original S0 physical trajectory + original
noise. The model function receives neither scene identity nor labels/context.
Offline prediction registration reuses the existing S0 population and scorer.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np

from .common import artifact,digest,load_config,lock,read_json,sha256,verify_artifact,verify_identity,write_json
from starVLA.model.modules.action_effect.intervention_basis import BasisSpec,reconstruct,validate_trajectory
from starVLA.model.modules.action_effect.trajectory_constraints import ConstraintSpec,geometric_screen

CONTROLS=('RandomLocal','FixedLateBrake')


def validate_request(cfg,role,max_samples,seed,dataset_path):
    if type(max_samples) is not int or max_samples<1 or seed!=cfg['seed']:
        raise ValueError('Explicit fixed population budget and registered campaign seed required')
    if role=='dev_report':
        from .evaluation_context_data import FULL_SCENES
        if dataset_path is not None or max_samples!=FULL_SCENES:
            raise ValueError('DEV control uses the fixed full population and no training repair data')
    elif role in ('fit','validation'):
        if dataset_path is None or max_samples>cfg['cohort_size']:raise ValueError('TRAIN control needs its fixed registered repair population')
    else:raise ValueError('Controls never access final_test/navtest')


def primitive_coefficients(basis=BasisSpec()):
    basis.validate()
    if basis!=BasisSpec():raise ValueError('Controls use the exact registered original four-window basis')
    values=[]
    for index,radius in ((0,.5),(2,.5),(4,.15),(6,.15)):
        for sign in (-1.,1.):
            c=np.zeros(basis.size,dtype=np.float64);c[index]=sign*radius;values.append(c)
    return values


def apply_control_coefficients(base,coefficients,*,basis=BasisSpec(),limits=ConstraintSpec()):
    base=validate_trajectory(base,basis);limits.validate()
    proposed=reconstruct(base,coefficients,basis)
    screen=geometric_screen(proposed,basis,limits) # No observed velocity or native feedback.
    if not all(np.isfinite(x) for x in screen['maxima'].values()):raise ValueError('Nonfinite preliminary geometric diagnostics')
    accepted=bool(screen['geometrically_admissible'])
    return {'trajectory':proposed if accepted else base.copy(),'proposed_trajectory':proposed,
        'coefficients':np.asarray(coefficients,dtype=np.float64),'geometry_accepted':accepted,
        'geometry_screen':screen,'fallback_to_Base':not accepted,'native_safety_verified':False}


def generate_control(base_physical,initial_noise,*,control,seed=42):
    """Pure deploy function: only original Base trajectory and FP32 noise."""
    if control not in CONTROLS or type(seed) is not int or not 0<=seed<2**63:
        raise ValueError('Only the two registered controls and a fixed nonnegative seed are supported')
    noise=np.asarray(initial_noise)
    if noise.shape!=(8,4) or noise.dtype!=np.float32 or not np.isfinite(noise).all():
        raise ValueError('Original finite FP32 raw8x4 noise required; never resample or quantize')
    little=np.ascontiguousarray(noise,dtype='<f4').tobytes()
    key=hashlib.sha256(b'action_effect_random_local_v1\0'+seed.to_bytes(8,'little')+little).digest()
    choices=primitive_coefficients()
    # One deterministic hash choice, never score all alternatives or select best.
    index=int.from_bytes(key[:8],'little')%len(choices) if control=='RandomLocal' else 2
    result=apply_control_coefficients(base_physical,choices[index])
    return {**result,'primitive_index':index,'noise_sha256':hashlib.sha256(little).hexdigest(),
        'control':control,'seed':seed,'deploy_candidate_count':1,'oracle_selection_used':False}


def generator_spec(control,seed):
    if control not in CONTROLS or type(seed) is not int or not 0<=seed<2**63:raise ValueError('Unregistered control/seed')
    root=Path(__file__).resolve().parents[2]
    paths=['tools/action_effect/predict_control.py','tools/action_effect/common.py',
        'starVLA/model/modules/action_effect/intervention_basis.py',
        'starVLA/model/modules/action_effect/trajectory_constraints.py']
    value={'schema':'action_effect_engineering_control_generator_v1','control':control,'seed':seed,
        'basis':asdict(BasisSpec()),'constraints':asdict(ConstraintSpec()),
        'primitive_coefficients':[x.tolist() for x in primitive_coefficients()],
        'selection':'SHA256(domain, uint64_le_seed, original_noise_float32_le_bytes) first64bits modulo8'
                    if control=='RandomLocal' else 'fixed coefficient[2] = -0.5 metres; late longitudinal displacement ramp',
        'noise_hash_domain':'action_effect_random_local_v1\\0','geometry':'same preliminary screen as A deploy; no native velocity',
        'fallback':'unchanged original Base on preliminary geometric failure',
        'input_fields':['base_physical','initial_noise'],'one_output_per_scene':True,'score_or_label_inputs':False,
        'trained_parameters':0,'new_VLM_calls':0,'new_simulation_calls':0,'is_native_safety_certificate':False,
        'physical_reconstruction_dtype':'original numpy float64 basis from exact cached FP32 Base values',
        'source':{p:sha256(root/p) for p in paths},
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
    value['identity']=digest(value);return value


def read_source_predictions(path,*,reader,context_manifest,role,max_samples):
    m=verify_identity(read_json(path));reg=verify_identity(read_json(verify_artifact(m['registration'])))
    binding=reader.binding
    selected=[r for r in reader.rows.values() if r['role']==role][:max_samples]
    expected_count=len(selected)
    if (m.get('schema')!='action_effect_refiner_predictions_v1' or m.get('status')!='COMPLETE'
        or m.get('registration_identity')!=reg['identity'] or reg.get('arm')!='S0'
        or reg.get('checkpoint_kind')!='immutable_original_Base_binding_not_private_head'
        or reg['checkpoint']!=binding['base_binding'] or reg['base_checkpoint']!=binding['base_checkpoint']
        or reg['campaign_binding']!=binding['identity'] or reg['context_manifest']!=artifact(context_manifest)
        or reg['role']!=role or expected_count<1 or m['requested_scenes']!=expected_count or len(m['entries'])!=expected_count
        or reg['scene_ids']!=[r['scene_id'] for r in m['entries']] or len(set(reg['scene_ids']))!=expected_count):
        raise ValueError('Control requires the exact completed original S0 population/context/checkpoint')
    source=verify_identity(read_json(verify_artifact(reg['g0_registration'])))
    if (source['base_checkpoint']!=reg['base_checkpoint'] or source['numeric_runtime_hash']!=reg['numeric_runtime_hash']
        or source['spec']['score_protocol_hash']!=reg['score_protocol_hash']):
        raise ValueError('Control cannot change the locked official scoring context/protocol')
    if [r['scene_id'] for r in selected]!=reg['scene_ids']:raise ValueError('Control cannot select scenes by labels or results')
    for r,expected in zip(m['entries'],selected):
        verify_identity(r)
        if any(r[k]!=expected[k] for k in ('scene_id','source_log','role','observation_hash')):
            raise ValueError('Control source observation/role join mismatch')
    return m,reg


def predict(cfg,*,context_manifest,output,control,role,max_samples,seed=42,
            dataset_path=None,campaign_binding=None,s0_predictions=None,resume=False):
    validate_request(cfg,role,max_samples,seed,dataset_path)
    if role=='dev_report':
        from .evaluation_context_data import EvaluationContextReader
        reader=EvaluationContextReader(context_manifest,binding_path=campaign_binding)
    elif role in ('fit','validation'):
        from .train_refiner import prepare_dataset
        reader=prepare_dataset(cfg,dataset_path,context_manifest,campaign_binding).reader
    from tools.st_world_policy.base_binding import read_binding
    if reader.binding['base_checkpoint_binding_hash']!=read_binding(cfg['base_binding'])['binding_sha256']:
        raise ValueError('Control Base is not the campaign original S0')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with lock(output/'PREDICT.lock'):
        if (output/'PREDICTIONS.json').exists() and not resume:raise ValueError('Existing control prediction requires explicit --resume')
        source_path=Path(s0_predictions) if s0_predictions else output/'source_s0'/'PREDICTIONS.json'
        if s0_predictions is None:
            from .evaluate_effect_flow import predict_baseline
            predict_baseline(cfg,context_manifest=context_manifest,output=source_path.parent,role=role,
                max_samples=max_samples,dataset_path=dataset_path,campaign_binding=campaign_binding,
                resume=resume or source_path.exists())
        source,base_reg=read_source_predictions(source_path,reader=reader,context_manifest=context_manifest,role=role,max_samples=max_samples)
        spec=generator_spec(control,seed);write_json(output/'GENERATOR_SPEC.json',spec)
        registration={k:v for k,v in base_reg.items() if k!='identity'}
        registration.update(arm=control,checkpoint=artifact(output/'GENERATOR_SPEC.json'),
            checkpoint_kind='engineering_control_generator_spec_not_trained_checkpoint',
            source_s0_predictions=artifact(source_path),source_s0_registration=source['registration'],
            generator_spec=artifact(output/'GENERATOR_SPEC.json'),generator_kind='engineering_control',
            baseline_definition='one prespecified primitive on exact original S0; no oracle/search/score input',
            construction_dtype='numpy_float64_physical',seed=seed)
        registration['identity']=digest(registration);write_json(output/'PREDICTION_REGISTRATION.json',registration)
        rows=[]
        for row in source['entries']:
            current=reader[row['scene_id']]
            base=current['physical'].double().numpy()
            if np.asarray(row['physical'],dtype=np.float64).tobytes()!=base.tobytes():
                raise ValueError('Cached S0 prediction does not equal the original frozen context')
            # This is the entire model input boundary: no IDs, labels, score or GT.
            proposal=generate_control(base,current['noise'].numpy(),control=control,seed=seed)
            result={'scene_id':row['scene_id'],'source_log':row['source_log'],'role':role,'observation_hash':row['observation_hash'],
                'physical':proposal['trajectory'].tolist(),'proposed_physical':proposal['proposed_trajectory'].tolist(),
                'base_physical':base.tolist(),'base_metrics':row.get('base_metrics'),'base_feedback':row.get('base_feedback'),
                'base_canonical_receipt':row.get('base_canonical_receipt'),'coefficients':proposal['coefficients'].tolist(),
                **{k:proposal[k] for k in ('geometry_accepted','geometry_screen','fallback_to_Base','native_safety_verified',
                    'primitive_index','noise_sha256','control','seed','deploy_candidate_count','oracle_selection_used')},
                'teacher_forward_calls':0,'VLM_forward_calls':0,'real_feedback_used_for_prediction':False}
            result['identity']=digest(result);rows.append(result)
        result={'schema':'action_effect_refiner_predictions_v1','status':'COMPLETE',
            'registration':artifact(output/'PREDICTION_REGISTRATION.json'),'registration_identity':registration['identity'],
            'requested_scenes':len(rows),'entries':rows,'device':'cpu_untrained_engineering_control'}
        result['identity']=digest(result);write_json(output/'PREDICTIONS.json',result)
    return {'status':'COMPLETE','predictions':artifact(output/'PREDICTIONS.json'),'generator_spec':artifact(output/'GENERATOR_SPEC.json'),
        'requested_scenes':len(rows),'new_optimizer_updates':0,'new_VLM_calls':0,'new_simulation_calls':0,
        'official_scoring':'NOT_RUN; reuse evaluate_refiner.score only after explicit budget registration',
        'scientific_gain':'UNTESTED'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--context-manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--control',required=True,choices=CONTROLS);p.add_argument('--role',choices=('fit','validation','dev_report'),required=True)
    p.add_argument('--max-samples',required=True,type=int);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--repair-dataset');p.add_argument('--campaign-binding');p.add_argument('--s0-predictions')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    try:
        cfg=load_config(a.config)
        validate_request(cfg,a.role,a.max_samples,a.seed,a.repair_dataset)
        if a.dry_run:print(json.dumps({'status':'NOT_RUN','control':a.control,'max_samples':a.max_samples,'new_training':False,'new_scoring':False}));return 0
        result=predict(cfg,context_manifest=a.context_manifest,output=a.output,control=a.control,role=a.role,
            max_samples=a.max_samples,seed=a.seed,dataset_path=a.repair_dataset,campaign_binding=a.campaign_binding,
            s0_predictions=a.s0_predictions,resume=a.resume)
        print(json.dumps(result));return 0
    except Exception as e:
        print(json.dumps({'status':'FAIL','error_type':type(e).__name__,'error':str(e),'scientific_gain':'UNTESTED'}));return 1


if __name__=='__main__':raise SystemExit(main())
