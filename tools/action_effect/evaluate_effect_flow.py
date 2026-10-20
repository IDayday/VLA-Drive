"""One Euler10 IEG prediction, followed separately by shared official scoring.

Cached-context evaluation measures the actual original action head and private
adapter. It is not image-to-action latency or proof of held-out DEV improvement.
"""
from pathlib import Path
import argparse
import json
import time
import torch

from .common import artifact,digest,load_config,lock,read_json,verify_artifact,verify_identity,write_json
from .training import load_private
from .train_effect_flow import load_original_action_head
from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter,EffectFlowAdapterConfig
from starVLA.model.framework.effect_internalized_flow import EffectInternalizedFlow


def canonical_inference_backend():
    """Match the bound original-master replay, independent of parent flags."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    return {'deterministic_algorithms':True,'matmul_allow_tf32':False,'cudnn_allow_tf32':False,
        'cudnn_benchmark':False,'autocast':False,'parameter_dtype':'float32'}


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


def predict(cfg,*,dataset_path,context_manifest,checkpoint,output,role,max_samples,device,
            campaign_binding=None,resume=False):
    from .train_refiner import prepare_dataset
    from starVLA.dataloader.foresight_dataset import decode_ego
    if role not in ('fit','validation') or type(max_samples) is not int or not 1<=max_samples<=cfg['cohort_size']:
        raise ValueError('Explicit bounded registered TRAIN fit/validation population required')
    dataset=prepare_dataset(cfg,dataset_path,context_manifest,campaign_binding)
    selected=[r for r in dataset.entries if r['role']==role][:max_samples]
    if not selected:raise ValueError('Requested role has no registered scenes')
    model=load_flow_policy(checkpoint,base_binding=cfg['base_binding'],
        expected_base_binding_hash=dataset.reader.binding['base_checkpoint_binding_hash'],device=device,
        expected_repair_artifact=artifact(dataset_path),expected_context_artifact=artifact(context_manifest))
    registration={'schema':'action_effect_refiner_prediction_registration_v1','arm':model.provenance['arm'],
        'checkpoint':model.provenance['checkpoint'],'model_provenance':model.provenance,
        'repair_dataset':artifact(dataset_path),'context_manifest':artifact(context_manifest),
        'campaign_binding':dataset.reader.binding['identity'],'g0_registration':dataset.manifest['registration'],
        'role':role,'requested_max_samples':max_samples,'scene_ids':[r['scene_id'] for r in selected],
        'base_checkpoint':dataset.manifest['base_checkpoint'],'score_protocol_hash':dataset.manifest['score_protocol_hash'],
        'numeric_runtime_hash':dataset.manifest['numeric_runtime_hash'],'effect_teacher_deployed':False,
        'baseline_definition':'same immutable S0 scene noise, original Euler10 and G0 reference context'}
    registration['identity']=digest(registration)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with lock(output/'PREDICT.lock'):
        if (output/'PREDICTIONS.json').exists() and not resume:raise ValueError('Prior IEG predictions require explicit --resume')
        write_json(output/'PREDICTION_REGISTRATION.json',registration)
        rows=[];started=time.perf_counter()
        with torch.no_grad():
            for row in selected:
                current=dataset.reader[row['scene_id']]
                with torch.autocast(device_type=torch.device(device).type,enabled=False):
                    raw=model.predict_action(current['condition'][None].to(device),initial_noise=current['noise'][None].to(device))
                physical=decode_ego(raw.float())[0].cpu().double()
                if not torch.isfinite(raw).all() or not torch.isfinite(physical).all():
                    raise FloatingPointError('Nonfinite IEG prediction; never replace with zero or silently discard')
                record={'scene_id':row['scene_id'],'source_log':row['source_log'],'role':row['role'],
                    'observation_hash':row['observation_hash'],'raw':raw[0].cpu().tolist(),'physical':physical.tolist(),
                    'base_physical':row['base_physical'],'base_metrics':row['base_metrics'],
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
    p.add_argument('--role',choices=('fit','validation'),default='validation');p.add_argument('--max-samples',type=int)
    p.add_argument('--max-simulation-budget',type=int);p.add_argument('--max-seconds',type=float,default=3600.)
    p.add_argument('--device',default='cpu');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    try:
        cfg=load_config(a.config)
        if a.phase=='predict' and not all((a.repair_dataset,a.context_manifest,a.checkpoint,a.max_samples)):
            raise ValueError('Prediction requires registered repair/context/checkpoint/sample budget')
        if a.phase=='score' and (not a.predictions or a.max_simulation_budget is None):
            raise ValueError('Official scoring requires predictions and explicit call budget')
        if a.dry_run:print(json.dumps({'status':'NOT_RUN','phase':a.phase,'simulation_calls':0}));return 0
        if a.phase=='predict':
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
