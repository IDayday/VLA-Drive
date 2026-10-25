"""Paired real-image single-trajectory latency; same A800 and fixed scene list.

Alternating execution order controls drift. All timings include synchronization;
cache-free current-image decode/processor and GPU forward are reported separately.
This diagnostic never calls a scorer and never claims trained-model PDMS.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from starVLA.rl.qwen_world_postcot.contracts import require_dispatch,file_sha
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_postcot.deployment import load_export
from starVLA.model.modules.qwen_sourceflow.model import QwenSourceFlowVLA
from starVLA.model.modules.qwen_sourceflow.source_initialization import load_action_initialization
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.qwen_sourceflow.contracts import configure_native_numerics
from starVLA.rl.simwam_recipe.contracts import seed_for

def stats(values):return dict(mean=float(np.mean(values)),p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)),samples=len(values))

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--native-checkpoint',required=True);p.add_argument('--export',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--scenes',type=int,default=48);p.add_argument('--warmup',type=int,default=8);a=p.parse_args();host=require_dispatch()
    if not 32<=a.scenes<=128 or not 4<=a.warmup<a.scenes//2:raise ValueError('Meaningful bounded scene timing window required')
    configure_native_numerics();torch.set_num_threads(4);torch.cuda.set_device(0)
    data=WorldSceneData(a.data);tokens=[data.tokens[data.index(i)] for i in range(a.scenes)]
    qwen='/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct';processor=QwenObservationProcessor(qwen)
    parent=Path(a.native_checkpoint);files=json.loads((parent/'files.json').read_text());seal=json.loads((parent/'COMPLETE').read_text())
    if file_sha(parent/'files.json')!=seal['files_sha256'] or file_sha(parent/'model.pt')!=files['model.pt']:raise ValueError('Unsealed baseline checkpoint')
    action=load_action_initialization('/mnt/project/qwen-sourceflow-native-20261018/assets/action_generic.pt',verify=True)
    baseline=QwenSourceFlowVLA(qwen,action).cuda();baseline.load_state_dict(torch.load(parent/'model.pt',map_location='cpu',weights_only=False),strict=True);baseline.eval().requires_grad_(False)
    obs=data.observation(tokens[0]);probe_inputs=processor([obs]);probe_ego=torch.tensor([obs['proprio']],device='cuda',dtype=torch.float32)
    probe_noise=torch.zeros(1,8,3,device='cuda');torch.cuda.reset_peak_memory_stats();baseline.predict(probe_inputs,probe_ego,probe_noise);torch.cuda.synchronize()
    native_peak=torch.cuda.max_memory_allocated()
    model,export=load_export(a.export,'cuda');models=dict(native=baseline,complete_framework=model);rows=[]
    # Both resident models share only a device, never parameters or contexts.
    allocated_resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    for i,token in enumerate(tokens):
        for name in (list(models) if i%2==0 else list(reversed(models))):
            current=models[name];before=current.qwen.forward_calls
            start=time.monotonic();obs=data.observation(token);batch=processor([obs]);ego=torch.tensor([obs['proprio']],device='cuda',dtype=torch.float32)
            noise=torch.randn(1,8,3,generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,token)),dtype=torch.float32).cuda()
            torch.cuda.synchronize();prepared=time.monotonic();output=current.predict(batch,ego,noise);torch.cuda.synchronize();end=time.monotonic()
            if current.qwen.forward_calls-before!=1 or output['requested_trajectory'].shape!=(1,8,3) or not torch.isfinite(output['requested_trajectory']).all():raise ValueError('Single-current-pass finite native8x3 inference failed')
            rows.append(dict(model=name,token=token,warmup=i<a.warmup,seconds=end-start,
                seconds_current_decode_processor_H2D=prepared-start,seconds_gpu_forward=end-prepared,
                Qwen_calls=1,K=1,ODE_steps=10,shift=5))
    summaries={name:dict(end_to_end=stats([r['seconds'] for r in rows if r['model']==name and not r['warmup']]),
        current_decode_processor_H2D=stats([r['seconds_current_decode_processor_H2D'] for r in rows if r['model']==name and not r['warmup']]),
        forward=stats([r['seconds_gpu_forward'] for r in rows if r['model']==name and not r['warmup']])) for name in models}
    combined_peak=torch.cuda.max_memory_allocated()
    models.pop('native');baseline=None;action=None;current=None;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model.predict(probe_inputs,probe_ego,probe_noise);torch.cuda.synchronize();new_peak=torch.cuda.max_memory_allocated()
    result=dict(status='COMPLETE',host=host,device=torch.cuda.get_device_name(),batch=1,seed=2101,paired_scenes=len(tokens),
        baseline_checkpoint=str(parent),baseline_model_sha256=files['model.pt'],new_model_sha256=export['model_sha256'],new_model_qualification=export['qualification'],
        summaries=summaries,peak_allocated_both_models_resident=combined_peak,resident_parameters_and_buffers_bytes=allocated_resident,
        single_model_peak_allocated=dict(native=native_peak,complete_framework=new_peak),
        memory_scope='Timing has both models resident; standalone peaks separately measured before loading the second model and after deleting the first.',
        numerical_scope='Original native FP32 action ODE versus new clean-x BF16 actor/FP32 state; architecture and execution both differ.',rows=rows)
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2));print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
