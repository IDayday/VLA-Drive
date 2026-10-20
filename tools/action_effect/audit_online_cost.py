"""CPU-only small-head operator FLOPs and supplemental online/cache checks.

No complete VLA, DiT, teacher, simulator, training or GPU is instantiated. The
original shared projection is extracted with mmap after checksum verification;
only its small MLP is rebuilt. FLOPs are profiler-covered operator lower bounds.
"""
from pathlib import Path
import argparse
import importlib
import json
import os
import subprocess
import sys

import torch

from .common import artifact,digest,read_json,sha256,verify_artifact,verify_identity,write_json


def exact_tensor(left,right):
    if left.dtype!=right.dtype or left.shape!=right.shape:return False
    if left.is_floating_point() and (not torch.isfinite(left).all() or not torch.isfinite(right).all()):return False
    return torch.equal(left.contiguous().reshape(-1).view(torch.uint8),right.contiguous().reshape(-1).view(torch.uint8))


def event_summary(events):
    counted=[];attention=[];unaccounted=[]
    for event in events:
        row={'operator':event.key,'calls':event.count,'input_shapes':event.input_shapes,'reported_FLOPs':int(event.flops or 0)}
        if row['reported_FLOPs']>0:counted.append(row)
        if any(x in event.key.lower() for x in ('attention','scaled_dot','softmax')):attention.append(row)
        if not row['reported_FLOPs'] and event.key.startswith(('aten::','egr_','ieg_')):unaccounted.append(row)
    total=sum(x['reported_FLOPs'] for x in counted)
    return {'operator_covered_FLOPs_lower_bound':total,'operator_covered_GFLOPs_lower_bound':total/1e9,
        'counted_operators':counted,'attention_operators':attention,'zero_reported_operators':unaccounted,
        'coverage':'only installed torch.profiler with_flops supported operators; zero means unreported, not zero work',
        'not_included':'full Qwen/visual/DiT, unsupported/fused attention arithmetic, geometric NumPy screen, image IO, deployment wall time'}


def profile_call(fn,*,trace):
    with torch.no_grad(),torch.autocast('cpu',enabled=False),torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,with_flops=True) as prof:
        fn()
    result=event_summary(prof.key_averages(group_by_input_shape=True))
    prof.export_chrome_trace(str(trace));result['trace']=artifact(trace)
    return result


def validate_source(root,manifest):
    root=Path(root).resolve()
    if not manifest.get('source_inventory'):raise ValueError('Missing actual deployment source inventory')
    for relative,want in manifest['source_inventory'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or sha256(root/relative)!=want:
            raise ValueError('Use the unchanged qualified deployment source: '+relative)


def private_module(manifest,root):
    validate_source(root,manifest)
    payload=torch.load(verify_artifact(manifest['private_head']),map_location='cpu',weights_only=True)
    if payload['route']!=manifest['route'] or payload['arm']!=manifest['arm']:
        raise ValueError('Actual private payload route/arm differs')
    for value in payload['state_dict'].values():
        if value.is_floating_point() and (value.dtype!=torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('Noncanonical private tensor')
    if payload['route']=='A':
        from starVLA.model.modules.action_effect.effect_model import EffectModelConfig
        from starVLA.model.modules.action_effect.intervention_basis import BasisSpec
        from starVLA.model.modules.action_effect.refinement_policy import CoefficientRefiner,RefinerConfig
        cfg=dict(payload['config']);cfg['model']=EffectModelConfig(**cfg['model'])
        b=dict(cfg['basis']);b['windows']=tuple(tuple(x) for x in b['windows']);cfg['basis']=BasisSpec(**b)
        for key in ('coefficient_scales','active_coefficients'):cfg[key]=tuple(cfg[key])
        model=CoefficientRefiner(RefinerConfig(**cfg))
    else:
        from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter,EffectFlowAdapterConfig
        if payload['config']['effect_dim']!=0:raise ValueError('Do not profile/deploy training-only B heads')
        model=EffectFlowAdapter(EffectFlowAdapterConfig(**payload['config']))
    module=importlib.import_module(type(model).__module__)
    if not Path(module.__file__).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Module came from another source worktree')
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.requires_grad_(False).eval()


def original_projection(receipt_pin,expected_base):
    receipt=read_json(verify_artifact(receipt_pin))
    if (receipt.get('schema')!='action_effect_original_action_head_v1' or receipt.get('strict_original_master') is not True
        or receipt.get('base_binding_hash')!=expected_base):raise ValueError('Foreign original-head receipt')
    # One bounded sequential checksum read, then mmap only the four MLP tensors.
    # Never construct the original819M DiT or load all its weights into RAM.
    path=verify_artifact(receipt['artifact'])
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if state.get('schema')!=receipt['schema'] or state.get('base_binding_hash')!=expected_base:
        raise ValueError('Original-head body and receipt disagree')
    cfg=state['config']['framework'];weights={k.removeprefix('qwen_proj.'):v.clone()
        for k,v in state['state_dict'].items() if k.startswith('qwen_proj.')}
    if set(weights)!={'layer1.weight','layer1.bias','layer2.weight','layer2.bias'}:
        raise ValueError('Unexpected original shared projection structure')
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError('Original projection must retain finite FP32 master values')
    from starVLA.model.modules.action_model.GR00T_ActionHeader import MLP
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        torch.random.default_generator.manual_seed(0)
        model=MLP(cfg['qwenvl']['vl_hidden_dim'],cfg['action_model']['hidden_size'],cfg['action_model']['hidden_size'])
    model.load_state_dict(weights,strict=True);model.requires_grad_(False).eval()
    return model,{'receipt':receipt_pin,'head_artifact':receipt['artifact'],'original_full_head_instantiated':False,
        'full_blob_checksum_read_bytes':path.stat().st_size,'mmap_projection_tensor_bytes':sum(v.numel()*v.element_size() for v in weights.values()),
        'projection_tensor_sha256':{k:digest([list(v.shape),str(v.dtype),__import__('hashlib').sha256(v.numpy().tobytes()).hexdigest()]) for k,v in weights.items()}}


def audit(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('Run with CUDA_VISIBLE_DEVICES empty; this audit is CPU-only')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(args.output)
    if out.exists():raise FileExistsError('Preserve supplemental evidence; choose a new output')
    online=Path(args.online_root);old_complete=artifact(online/'COMPLETE.json')
    done=read_json(online/'COMPLETE.json');worker=read_json(verify_artifact(done['worker_result']))
    if done.get('status')!='PASS' or worker.get('status')!='PASS' or set(worker['modes'])!={'S0','A','B'}:
        raise ValueError('Actual complete common-load S0/A/B online qualification required')
    manifests={mode:verify_identity(read_json(verify_artifact(worker['bundles'][mode]))) for mode in ('A','B')}
    root=Path(args.deployment_source).resolve()
    for m in manifests.values():validate_source(root,m)
    if any(name=='starVLA' or name.startswith('starVLA.') for name in sys.modules):
        raise ValueError('Start a fresh audit process before importing model source')
    sys.path.insert(0,str(root))
    a=private_module(manifests['A'],root);b=private_module(manifests['B'],root)
    from .context_data import ContextReader,batch_context
    reader=ContextReader(args.context_manifest,args.campaign_binding);record=reader[worker['scene_id']]
    if (record['observation_hash']!=worker['observation_hash']
        or record['base_binding_hash']!=worker['base_binding_hash']):raise ValueError('Online/cache observation or original Base changed')
    context=batch_context([record],'cpu');condition=record['condition'][None];noise=record['noise'][None]
    physical=record['physical'][None]
    s0={'physical_exact':exact_tensor(record['physical'],torch.tensor(worker['modes']['S0']['physical'],dtype=torch.float32)),
        'raw_exact':exact_tensor(record['raw'],torch.tensor(worker['modes']['S0']['raw'],dtype=torch.float32))}
    if not all(s0.values()):raise ValueError('Actual original S0 online/cache raw or physical drift')
    from .qualify_online_deployment import compare_prediction
    comparisons={'S0':s0}
    for mode,path in [('A',args.a_predictions),('B',args.b_predictions)]:
        comparisons[mode]=(compare_prediction(path,mode,worker['modes'][mode],worker) if path else {'status':'WAITING_INPUTS'})
        if path and not comparisons[mode]['exact_physical']:raise ValueError('Actual private online/cache physical drift: '+mode)
    counts={'A':sum(p.numel() for p in a.parameters()),'B':sum(p.numel() for p in b.parameters())}
    if counts!={'A':worker['parameters']['A_private'],'B':worker['parameters']['B_private_velocity']}:
        raise ValueError('Private parameter count differs from actual online qualification')
    out.mkdir(parents=True)
    profiles={'A_refiner_once':profile_call(lambda:a(context,physical),trace=out/'A_refiner_once.trace.json')}
    def b_ten():
        for i in range(10):b(condition,noise,torch.tensor([i/10],dtype=torch.float32))
    profiles['B_velocity_adapter_ten']=profile_call(b_ten,trace=out/'B_velocity_adapter_ten.trace.json')
    projection,pin=original_projection(manifests['B']['original_action_head_receipt'],worker['base_binding_hash'])
    profiles['shared_original_qwen_projection_once']=profile_call(lambda:projection(condition),trace=out/'original_projection_once.trace.json')
    if artifact(online/'COMPLETE.json')!=old_complete:raise ValueError('Original online completion was modified during audit')
    if torch.cuda.is_initialized():raise ValueError('CPU audit unexpectedly initialized CUDA')
    result={'schema':'action_effect_cpu_operator_cost_and_cache_audit_v1','status':'PASS',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        'deployment_source_commit':worker['source_commit'],'tool':artifact(__file__),
        'original_online_complete_unchanged':old_complete,'online_worker':done['worker_result'],
        'context_manifest':artifact(args.context_manifest),'campaign_binding':artifact(args.campaign_binding),
        'scene_id':worker['scene_id'],'observation_hash':worker['observation_hash'],'bundle_pins':worker['bundles'],
        'CPU_profile_runtime':{'torch':torch.__version__,'threads':1,'dtype':'float32','device':'cpu','autocast':False},
        'original_online_parameters':worker['parameters'],'actual_small_module_parameters':counts,
        'original_projection_parameters':sum(p.numel() for p in projection.parameters()),'original_projection':pin,
        'actual_online_latency':{mode:{key:value[key] for key in ('n','warmup','mean_seconds','p50_seconds','p95_seconds',
            'peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','hook_calls_per_prediction')} for mode,value in worker['modes'].items()},
        'supplemental_online_cache_comparisons':comparisons,'profiles':profiles,
        'profile_shape_definition':'batch1 actual frozen TRAIN context; B uses actual initial noise replayed at ten time inputs for shape-based cost, not actual ODE states',
        'shared_projection_accounting':'one original qwen_proj is shared S0/A/B work; B adapter.context runs inside each of ten full adapter calls, not once',
        'not_claimed':'complete VLA FLOPs, full-DiT FLOPs, CPU latency as deployment latency, population throughput, calibrated confidence or scientific planning gain',
        'new_VLM_calls':0,'new_simulator_calls':0,'new_optimizer_updates':0,'new_GPU_model_calls':0,
        'GPU_initialized':False,'original_online_receipt_rewritten':False}
    result['identity']=digest(result);write_json(out/'AUDIT.json',result)
    write_json(out/'COMPLETE.json',{'status':'PASS','audit':artifact(out/'AUDIT.json'),'scope':'supplemental CPU operators/parameters/cache agreement only'})
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('online-root','deployment-source','context-manifest','campaign-binding','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--a-predictions');p.add_argument('--b-predictions');args=p.parse_args(argv)
    try:
        result=audit(args);print(json.dumps({'status':result['status'],'output':args.output,'identity':result['identity']}));return 0
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)}));return 1


if __name__=='__main__':raise SystemExit(main())
