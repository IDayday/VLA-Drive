"""One Euler10 IEG prediction, followed separately by shared official scoring.

Cached-context evaluation measures the actual original action head and private
adapter. It is not image-to-action latency or proof of held-out DEV improvement.
"""
from pathlib import Path
import argparse
import json
import os
import time
import torch

from .common import artifact,digest,load_config,lock,read_json,verify_artifact,verify_identity,write_json
from .training import load_private
from .train_effect_flow import load_original_action_head
from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter,EffectFlowAdapterConfig
from starVLA.model.framework.effect_internalized_flow import EffectInternalizedFlow


def canonical_inference_backend():
    """Match the bound original-master replay, independent of parent flags."""
    if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Start a fresh canonical inference process with CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    return {'deterministic_algorithms':True,'matmul_allow_tf32':False,'cudnn_allow_tf32':False,
        'cudnn_benchmark':False,'autocast':False,'parameter_dtype':'float32','CUBLAS_WORKSPACE_CONFIG':':4096:8'}


def load_flow_policy(checkpoint,*,base_binding,expected_base_binding_hash,device,
                     expected_repair_artifact=None,expected_context_artifact=None):
    backend=canonical_inference_backend()
    checkpoint=Path(checkpoint).resolve()
    identity=read_json(checkpoint.parent/'IDENTITY.json');receipt=read_json(checkpoint.with_suffix('.COMPLETE.json'))
    if (identity.get('schema')!='action_effect_small_training_v1' or identity.get('arm') not in EffectInternalizedFlow.ARMS
            or identity['base_binding_hash']!=expected_base_binding_hash or receipt['run_identity']!=digest(identity)
            or verify_artifact(receipt['checkpoint']).resolve()!=checkpoint
            or (expected_repair_artifact is not None and identity['repair_data']['dataset']!=expected_repair_artifact)
            or (expected_context_artifact is not None and identity['context_artifact']!=expected_context_artifact)):
        raise ValueError('IEG checkpoint/receipt/Base/data/context identity mismatch')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (state['run_identity']!=receipt['run_identity'] or state['optimizer_step']!=receipt['optimizer_step']
            or state['optimizer_step']<1):raise ValueError('No complete actually updated IEG checkpoint')
    head,pin=load_original_action_head(verify_artifact(identity['head']['head_receipt']),base_binding,device=device)
    if pin!=identity['head']:raise ValueError('Canonical head dependency changed')
    model=EffectInternalizedFlow(head,EffectFlowAdapter(EffectFlowAdapterConfig(**identity['adapter_config'])),arm=identity['arm']).to(device)
    load_private(model,state['model']);model.requires_grad_(False).eval()
    model.provenance={'checkpoint':artifact(checkpoint),'identity':artifact(checkpoint.parent/'IDENTITY.json'),
        'optimizer_step':state['optimizer_step'],'run_identity':state['run_identity'],'arm':identity['arm'],
        'base_binding_hash':expected_base_binding_hash,'teacher_deployed':False,'generation':'one original Euler10',
        'inference_backend':backend}
    return model


def predict_baseline(cfg,*,context_manifest,output,role,max_samples,dataset_path=None,campaign_binding=None,resume=False):
    """Register the exact cached S0 trajectory; zero model or RNG execution."""
    if type(max_samples) is not int or max_samples<1:raise ValueError('Explicit bounded baseline population required')
    if role=='dev_report':
        if dataset_path is not None:raise ValueError('DEV S0 prediction reads no training repair dataset')
        from .evaluation_context_data import EvaluationContextReader
        reader=EvaluationContextReader(context_manifest,binding_path=campaign_binding);source=reader.binding
        if source.get('split')!='dev' or source.get('for_training') is not False:raise ValueError('Independent DEV-only registration required')
        if max_samples>len(reader.rows):raise ValueError('Baseline budget exceeds registered DEV population')
        selected=list(reader.rows.values())[:max_samples]
        protocol=source['score_protocol_hash'];numeric=source['numeric_runtime_hash'];base=source['base_checkpoint']
        registration_pin=reader.manifest['registration'];checkpoint_pin=source['base_binding']
        population={'population_artifact':artifact(context_manifest),'population_artifact_kind':'DEV_context_only_no_repairs'}
    elif role in ('fit','validation'):
        from .train_refiner import prepare_dataset
        if dataset_path is None or max_samples>cfg['cohort_size']:raise ValueError('Registered TRAIN baseline dataset required')
        dataset=prepare_dataset(cfg,dataset_path,context_manifest,campaign_binding);reader=dataset.reader;source=dataset.manifest
        selected=[r for r in dataset.entries if r['role']==role][:max_samples]
        protocol=source['score_protocol_hash'];numeric=source['numeric_runtime_hash'];base=source['base_checkpoint']
        registration_pin=source['registration'];checkpoint_pin=reader.binding['base_binding']
        population={'repair_dataset':artifact(dataset_path)}
    else:raise ValueError('Only original TRAIN roles or independent DEV are supported')
    if not selected:raise ValueError('No registered baseline scenes')
    registration={'schema':'action_effect_refiner_prediction_registration_v1','arm':'S0',
        'checkpoint':checkpoint_pin,'checkpoint_kind':'immutable_original_Base_binding_not_private_head',
        'base_checkpoint':base,'context_manifest':artifact(context_manifest),'g0_registration':registration_pin,
        'campaign_binding':reader.binding['identity'],'role':role,'scene_ids':[r['scene_id'] for r in selected],
        'score_protocol_hash':protocol,'numeric_runtime_hash':numeric,'requested_max_samples':max_samples,
        'baseline_definition':'exact canonical cached S0 physical trajectory re-scored in this identical reference context',
        'head_forward_calls':0,'VLM_forward_calls':0,'new_noise_draws':0,**population}
    registration['identity']=digest(registration)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with lock(output/'PREDICT.lock'):
        if (output/'PREDICTIONS.json').exists() and not resume:raise ValueError('Prior S0 predictions require --resume')
        write_json(output/'PREDICTION_REGISTRATION.json',registration)
        rows=[]
        for row in selected:
            current=reader[row['scene_id']];physical=current['physical'].double().tolist()
            record={'scene_id':row['scene_id'],'source_log':row['source_log'],'role':role,
                'observation_hash':row['observation_hash'],'physical':physical,'base_physical':physical,
                'base_metrics':row.get('base_metrics'),'base_feedback':row.get('base_feedback'),
                'base_canonical_receipt':row.get('base_canonical_receipt'),'teacher_forward_calls':0,
                'head_forward_calls':0,'VLM_forward_calls':0,'real_feedback_used_for_prediction':False}
            record['identity']=digest(record);rows.append(record)
        result={'schema':'action_effect_refiner_predictions_v1','status':'COMPLETE',
            'registration':artifact(output/'PREDICTION_REGISTRATION.json'),'registration_identity':registration['identity'],
            'requested_scenes':len(rows),'entries':rows,'device':'cpu_cached_original_physical'}
        result['identity']=digest(result);write_json(output/'PREDICTIONS.json',result)
    return {'status':'COMPLETE','requested_scenes':len(rows),'predictions':artifact(output/'PREDICTIONS.json'),
        'model_forward_calls':0,'simulator_calls':0,'official_scoring':'separate bounded score phase required'}


def predict(cfg,*,dataset_path,context_manifest,checkpoint,output,role,max_samples,device,
            campaign_binding=None,resume=False):
    from starVLA.dataloader.foresight_dataset import decode_ego
    if type(max_samples) is not int or max_samples<1:raise ValueError('Explicit positive population budget required')
    if role=='dev_report':
        # Separate schema/reader: DEV can never be admitted by a TRAIN parser.
        from .evaluation_context_data import EvaluationContextReader
        reader=EvaluationContextReader(context_manifest,binding_path=campaign_binding)
        source=reader.binding
        if source.get('split')!='dev' or source.get('for_training') is not False:
            raise ValueError('DEV context must explicitly prohibit backward/training')
        if max_samples>len(reader.rows):raise ValueError('Requested DEV budget exceeds complete fixed population')
        selected=[{**r,'base_physical':None,
            'base_metrics':None,'base_feedback':None,'base_canonical_receipt':None,
            'target_kind':'not_read_for_dev'} for r in list(reader.rows.values())[:max_samples]]
        if dataset_path is not None:raise ValueError('DEV prediction reads no training repair dataset; omit --repair-dataset')
        repair_artifact=None
        registration_artifact=reader.manifest['registration']
        protocol=source['score_protocol_hash'];numeric_runtime=source['numeric_runtime_hash'];base_checkpoint=source['base_checkpoint']
        expected_context=None
    elif role in ('fit','validation'):
        from .train_refiner import prepare_dataset
        if dataset_path is None or max_samples>cfg['cohort_size']:raise ValueError('Registered TRAIN dataset/sample budget required')
        dataset=prepare_dataset(cfg,dataset_path,context_manifest,campaign_binding);reader=dataset.reader
        selected=[r for r in dataset.entries if r['role']==role][:max_samples]
        repair_artifact=artifact(dataset_path);registration_artifact=dataset.manifest['registration']
        protocol=dataset.manifest['score_protocol_hash'];numeric_runtime=dataset.manifest['numeric_runtime_hash']
        base_checkpoint=dataset.manifest['base_checkpoint'];expected_context=artifact(context_manifest)
    else:raise ValueError('Only registered TRAIN fit/validation or independent dev_report is allowed')
    if not selected:raise ValueError('Requested role has no registered scenes')
    model=load_flow_policy(checkpoint,base_binding=cfg['base_binding'],
        expected_base_binding_hash=reader.binding['base_checkpoint_binding_hash'],device=device,
        expected_repair_artifact=repair_artifact,expected_context_artifact=expected_context)
    registration={'schema':'action_effect_refiner_prediction_registration_v1','arm':model.provenance['arm'],
        'checkpoint':model.provenance['checkpoint'],'model_provenance':model.provenance,
        'context_manifest':artifact(context_manifest),
        'campaign_binding':reader.binding['identity'],'g0_registration':registration_artifact,
        'role':role,'requested_max_samples':max_samples,'scene_ids':[r['scene_id'] for r in selected],
        'base_checkpoint':base_checkpoint,'score_protocol_hash':protocol,
        'numeric_runtime_hash':numeric_runtime,'effect_teacher_deployed':False,
        'evaluation_split':'dev' if role=='dev_report' else 'train',
        'baseline_definition':'same immutable S0 scene noise, original Euler10 and G0 reference context'}
    if role=='dev_report':
        registration.update(population_artifact=artifact(context_manifest),population_artifact_kind='DEV_context_only_no_repairs')
    else:registration['repair_dataset']=repair_artifact
    registration['identity']=digest(registration)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with lock(output/'PREDICT.lock'):
        if (output/'PREDICTIONS.json').exists() and not resume:raise ValueError('Prior IEG predictions require explicit --resume')
        write_json(output/'PREDICTION_REGISTRATION.json',registration)
        rows=[];started=time.perf_counter()
        with torch.no_grad():
            for row in selected:
                current=reader[row['scene_id']]
                base_physical=(row['base_physical'] if row['base_physical'] is not None else current['physical'].double().tolist())
                with torch.autocast(device_type=torch.device(device).type,enabled=False):
                    raw=model.predict_action(current['condition'][None].to(device),initial_noise=current['noise'][None].to(device))
                physical=decode_ego(raw.float())[0].cpu().double()
                if not torch.isfinite(raw).all() or not torch.isfinite(physical).all():
                    raise FloatingPointError('Nonfinite IEG prediction; never replace with zero or silently discard')
                record={'scene_id':row['scene_id'],'source_log':row['source_log'],'role':row['role'],
                    'observation_hash':row['observation_hash'],'raw':raw[0].cpu().tolist(),'physical':physical.tolist(),
                    'base_physical':base_physical,'base_metrics':row['base_metrics'],
                    'base_feedback':row['base_feedback'],'base_canonical_receipt':row['base_canonical_receipt'],
                    'target_kind':row['target_kind'],'teacher_forward_calls':0,'refiner_forward_calls':0,
                    'real_feedback_used_for_prediction':False,'Euler_steps':10,'complete_generations':1}
                record['identity']=digest(record);rows.append(record)
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-started
        result={'schema':'action_effect_refiner_predictions_v1','status':'COMPLETE',
            'registration':artifact(output/'PREDICTION_REGISTRATION.json'),'registration_identity':registration['identity'],
            'requested_scenes':len(selected),'entries':rows,'device':str(device)}
        result['identity']=digest(result)
        write_json(output/'PREDICTIONS.json',result)
        write_json(output/f'PREDICT_INVOCATION_{time.time_ns()}.json',{'status':'COMPLETE',
            'elapsed_seconds':elapsed,'predictions':artifact(output/'PREDICTIONS.json'),'cached_context_only':True})
        return {'status':'COMPLETE','requested_scenes':len(selected),'predictions':artifact(output/'PREDICTIONS.json'),
            'elapsed_seconds':elapsed,'scientific_gain':'UNTESTED until separate complete official scoring'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--phase',choices=('predict','score'),required=True)
    p.add_argument('--repair-dataset');p.add_argument('--context-manifest');p.add_argument('--campaign-binding')
    p.add_argument('--checkpoint');p.add_argument('--predictions');p.add_argument('--output',required=True)
    p.add_argument('--role',choices=('fit','validation','dev_report'),default='validation');p.add_argument('--max-samples',type=int)
    p.add_argument('--max-simulation-budget',type=int);p.add_argument('--max-seconds',type=float,default=3600.)
    p.add_argument('--device',default='cpu');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--baseline-only',action='store_true',help='Register cached original S0 physical trajectories; no new model/noise execution.')
    a=p.parse_args(argv)
    try:
        cfg=load_config(a.config)
        if a.phase=='predict' and (not all((a.context_manifest,a.max_samples)) or
                (not a.baseline_only and a.checkpoint is None) or
                (a.role!='dev_report' and a.repair_dataset is None)):
            raise ValueError('Prediction requires registered repair/context/checkpoint/sample budget')
        if a.phase=='score' and (not a.predictions or a.max_simulation_budget is None):
            raise ValueError('Official scoring requires predictions and explicit call budget')
        if a.phase=='score' and a.baseline_only:raise ValueError('--baseline-only is a prediction operation; score its explicit predictions')
        if a.dry_run:print(json.dumps({'status':'NOT_RUN','phase':a.phase,'simulation_calls':0}));return 0
        if a.phase=='predict':
            if a.baseline_only:
                if a.checkpoint is not None:raise ValueError('S0 baseline uses immutable context Base, not another checkpoint')
                result=predict_baseline(cfg,dataset_path=a.repair_dataset,context_manifest=a.context_manifest,
                    output=a.output,role=a.role,max_samples=a.max_samples,campaign_binding=a.campaign_binding,resume=a.resume)
            else:
                result=predict(cfg,dataset_path=a.repair_dataset,context_manifest=a.context_manifest,
                    checkpoint=a.checkpoint,output=a.output,role=a.role,max_samples=a.max_samples,device=a.device,
                    campaign_binding=a.campaign_binding,resume=a.resume)
        else:
            from .evaluate_refiner import score
            result=score(cfg,predictions_path=a.predictions,output=a.output,
                max_simulation_budget=a.max_simulation_budget,max_seconds=a.max_seconds,resume=a.resume)
        print(json.dumps(result));return 0 if result['status']=='COMPLETE' else 1
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)}));return 1


if __name__=='__main__':raise SystemExit(main())
