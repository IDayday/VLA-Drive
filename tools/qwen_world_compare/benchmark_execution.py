"""Disposable real-checkpoint rank-slice F/B benchmark; no optimizer updates.

Same native microbatch/accumulation, all seven losses, real TRAIN targets,
FP32 parameters/gradients and BF16 forward. It is not an eight-rank step ETA.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
from starVLA.model.modules.qwen_world_compare.factory import build_model
from starVLA.model.modules.qwen_world_compare.execution import checkpoint_flags,apply_checkpoint_policy
from starVLA.model.modules.qwen_world_postcot.teacher_targets import TeacherCacheReader
from starVLA.model.modules.qwen_world_postcot.candidate_bank import CandidateBank
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_compare.batching import BatchBuilder,to_device,update_denominators
from starVLA.rl.qwen_world_compare.track_targets import FutureTrackStore
from starVLA.rl.qwen_world_compare.evidence import EvidenceBank
from starVLA.rl.flow_grpo.reproducibility import configure_numerics


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)


def compare_gradients(model,reference):
    squared=error=0.;maximum=0.;none=[];count=0
    for name,p in model.named_parameters():
        old=reference[name]
        if old is None or p.grad is None:
            if(old is None)!=(p.grad is None):none.append(name)
            continue
        actual=p.grad.detach().cpu();delta=(actual-old).double()
        maximum=max(maximum,float(delta.abs().max()));error+=float(delta.square().sum())
        squared+=float(old.double().square().sum());count+=1
    return dict(compared_parameter_tensors=count,missing_gradient_mismatch=none,
                max_absolute_error=maximum,relative_L2=(error/max(squared,1e-300))**.5)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--snapshot',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--checkpoint-policy',choices=('short_off','all_off'),default='short_off');args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if(root/'RESULT.json').exists():raise FileExistsError('Completed qualification cannot be repeated in place')
    snapshot=json.loads(Path(args.snapshot).read_text());identity=snapshot['identity'];cfg=identity['config'];arch=cfg['architecture_id']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');configure_numerics();torch.set_num_threads(4);torch.cuda.set_device(0)
    device=torch.device('cuda');stage_times={}
    data=WorldSceneData(Path(cfg['data_root'])/'TRAIN',role='TRAIN');processor=QwenObservationProcessor(cfg['qwen_path'])
    teachers=TeacherCacheReader(cfg['teacher_cache'],expected_cohort='TRAIN');bank=CandidateBank(cfg['candidate_bank'],allowed_splits=('TRAIN',))
    tracks=FutureTrackStore(Path(cfg['target_root'])/'TRAIN');evidence=EvidenceBank(cfg['evidence_root'])
    builder=BatchBuilder(data,processor,teachers,bank,tracks,evidence)
    # Instrument only CPU readers in this disposable process; no actor method
    # or production source is patched, and return values/RNG are unaltered.
    def instrument(obj,name,label):
        original=getattr(obj,name)
        def timed(*a,**kw):
            start=time.perf_counter()
            try:return original(*a,**kw)
            finally:stage_times[label]=stage_times.get(label,0)+time.perf_counter()-start
        setattr(obj,name,timed)
    for obj,name,label in((data,'observation','observation_IO_decode'),(teachers,'get','teacher_IO_validate'),
                         (data,'target','GT_encoding'),(tracks,'get','track_cache'),(bank,'sample_pairs','candidate_pair'),(evidence,'get','evidence_cache')):
        instrument(obj,name,label)
    micro=cfg['execution']['micro_batch'];update=snapshot['updates'];cursor=update*32
    tick=time.perf_counter();all_batches=[]
    for rank in range(8):
        for m in range(4//micro):
            positions=[rank*4+m*micro+j for j in range(micro)]
            tokens=[data.tokens[data.index(cursor+i)]for i in positions]
            all_batches.append(builder.build(tokens,positions,update))
    data_seconds=time.perf_counter()-tick
    denominators=update_denominators(all_batches,torch.device('cpu'))
    # World=1 times den/8 reproduces the source rank's world=8 numerator scale.
    denominators={k:torch.as_tensor(v,device=device)/8 for k,v in denominators.items()}
    batches=[to_device(b,device)for b in all_batches[:4//micro]]
    del all_batches
    write(root/'DATA.json',dict(full_global32_build_seconds=data_seconds,stage_seconds=stage_times,
        actual_micro_batch=micro,rank0_microbatches=len(batches),tokens=[b['tokens']for b in batches],
        sampled_global_update=update,padding_built_at_native_microbatch=True))
    print(json.dumps(dict(stage='data_ready',architecture=arch,data_seconds=data_seconds)),flush=True)
    tick=time.perf_counter();raw=torch.load(snapshot['model'],map_location='cpu',weights_only=False)
    model=build_model(cfg,checkpoint_config=copy.deepcopy(identity['qwen_config']),train_heads=True)
    model.load_state_dict(raw['module'],strict=True);del raw
    model.to(device).train();model.configure_stage(False)
    if arch!='D':model.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise TypeError('Parameter storage precision changed')
    source_flags=checkpoint_flags(model)
    version={name:p._version for name,p in model.named_parameters()}
    setup=time.perf_counter()-tick
    rows=[];reference=None;losses={};grad=None;reference_rng=None;rng_ok=True
    # Warm both paths. Alternate subsequent windows to limit occupancy drift.
    order=[True,False]+[enabled for _ in range(args.repeats)for enabled in(True,False)]
    for index,enabled in enumerate(order):
        execution=apply_checkpoint_policy(model,'source'if enabled else args.checkpoint_policy,source_flags)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        start=time.perf_counter();sums={}
        for batch in batches:
            with torch.autocast('cuda',dtype=torch.bfloat16):result=model(batch,denominators=denominators,weights=cfg['il']['loss_weights'])
            result['loss'].backward()
            for name,value in result['losses'].items():sums[name]=sums.get(name,0)+value
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        unchanged=torch.equal(cpu_rng,torch.get_rng_state())and torch.equal(cuda_rng,torch.cuda.get_rng_state());rng_ok&=unchanged
        values={k:float(v)for k,v in sums.items()}
        row=dict(iteration=index,variant='source_checkpoint_all'if enabled else'retain_short_graphs'if args.checkpoint_policy=='short_off'else'retain_all_graphs',warmup=index<2,
            forward_backward_wall_seconds=elapsed,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            loss_components=values,RNG_unchanged=unchanged)
        rows.append(row);print(json.dumps(row),flush=True)
        if index==2:
            reference={name:None if p.grad is None else p.grad.detach().cpu()for name,p in model.named_parameters()};losses=values
        elif index==3:
            grad=compare_gradients(model,reference);del reference;reference=None
            if grad['missing_gradient_mismatch']or grad['relative_L2']>1e-5:raise ValueError('Full real-model gradients are not equivalent: '+str(grad))
            if any(abs(values[k]-losses[k])>1e-6 for k in losses):raise ValueError('Loss component changed')
    if not rng_ok or any(p._version!=version[name]for name,p in model.named_parameters()):raise ValueError('RNG or parameter state mutated')
    target_variant='retain_short_graphs'if args.checkpoint_policy=='short_off'else'retain_all_graphs'
    means={variant:statistics.median(r['forward_backward_wall_seconds']for r in rows if not r['warmup']and r['variant']==variant)
           for variant in('source_checkpoint_all',target_variant)}
    write(root/'RESULT.json',dict(status='PASS',architecture_id=arch,model_sha256=snapshot['model_sha256'],
        snapshot=str(args.snapshot),model_setup_seconds=setup,rows=rows,medians=means,
        observed_rank_slice_speedup=means['source_checkpoint_all']/means[target_variant],
        checkpoint_policy=execution,
        gradient_comparison=grad,parameter_and_gradient_dtype='float32',forward_autocast='bfloat16',
        global32_denominators_scaled_for_one_rank=True,native_micro_accum_preserved=True,
        optimizer_updates=0,scorer_calls=0,formal_training_source_modified=False,
        caveat='Shared-GPU rank0 F/B slice; excludes data/optimizer/collectives. Not an eight-rank end-to-end speedup or ETA.'))
    bank.close()


if __name__=='__main__':main()
