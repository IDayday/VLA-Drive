"""Bounded real-pair NCCL gradient check (component, not a DDP trainer claim).

Run with torchrun --nproc-per-node=2 on two allocated devices. Uses actual
registered TRAIN contexts/native labels; no simulator or optimizer is run.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .common import artifact, load_config, write_json
from .context_data import ContextReader, batch_context
from .train_effect_model import PairData, WEIGHTS
from starVLA.model.modules.action_effect.effect_model import ActionEffectModel, EffectModelConfig
from starVLA.model.modules.action_effect.effect_losses import (
    effect_loss_contributions, combine_contributions, reduce_contributions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--context-manifest',required=True)
    p.add_argument('--pair-dataset',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Qualification requires exactly two allocated CUDA ranks')
    torch.cuda.set_device(local);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    dist.init_process_group('nccl')
    try:
        cfg=load_config(a.config);reader=ContextReader(a.context_manifest,Path(cfg['artifact_root'])/'CAMPAIGN_BINDING.json')
        data=PairData(a.pair_dataset,reader)
        if len(data.scenes)<8:raise ValueError('Need eight unique real fitting scenes')
        rows=[data.by_scene[s][0] for s in data.scenes[:8]]
        mc=EffectModelConfig(mode='delta_feedback',hidden_dim=cfg['training']['hidden'],
            layers=cfg['training']['layers'],heads=cfg['training']['heads'],init_seed=42)
        model=ActionEffectModel(mc,data.spec).cuda(local)
        ddp=DistributedDataParallel(model,device_ids=[local],find_unused_parameters=True)
        def contribution(net,batch,empty):
            context=batch_context([reader[r['scene_id']] for r in batch],f'cuda:{local}')
            before=torch.tensor([r['base_physical'] for r in batch],device=local,dtype=torch.float32)
            after=torch.tensor([r['modified_physical'] for r in batch],device=local,dtype=torch.float32)
            prediction=net(context,before,after);targets=data.targets(batch,f'cuda:{local}')
            if empty:
                for key in ('before_valid','after_valid','event_valid','temporal_valid','repair_valid'):
                    targets[key].zero_()
            return effect_loss_contributions(prediction,mode='delta_feedback',**targets)
        records=[]
        for case in ('all_valid','empty_rank'):
            ddp.zero_grad(set_to_none=True)
            own=rows[rank*4:(rank+1)*4]
            parts=[contribution(ddp,own[i:i+2],case=='empty_rank' and rank==1) for i in (0,2)]
            result=reduce_contributions(combine_contributions(parts),WEIGHTS,distributed=True)
            result.total.backward()
            actual={n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.named_parameters()}
            reference=ActionEffectModel(mc,data.spec).cuda(local)
            full=[contribution(reference,rows[:4],False),contribution(reference,rows[4:],case=='empty_rank')]
            reduce_contributions(combine_contributions(full),WEIGHTS).total.backward()
            worst_abs=0.;worst_rel=0.;used=0
            for name,param in reference.named_parameters():
                g=actual[name]
                if param.grad is None:
                    if g is not None and torch.count_nonzero(g):raise AssertionError('Unexpected nonzero gradient '+name)
                    continue
                if g is None:raise AssertionError('Missing distributed gradient '+name)
                expected=param.grad;err=(g-expected).abs().max().item()
                rel=err/max(expected.abs().max().item(),1e-12)
                worst_abs=max(worst_abs,err);worst_rel=max(worst_rel,rel);used+=1
                # Declared before GPU execution: FP32 collective/shape reduction
                # tolerance, never bitwise or an altered training loss.
                torch.testing.assert_close(g,expected,rtol=2e-4,atol=2e-6)
            records.append({'case':case,'gradient_tensors':used,'max_abs_error':worst_abs,
                'max_tensor_relative_error':worst_rel,'microbatch':2,'local_scenes':4,'global_scenes':8})
            del reference
        gathered=[None,None];dist.all_gather_object(gathered,records)
        if rank==0:
            output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
            write_json(output/'RESULT.json',{'status':'PASS','scope':'NCCL real-pair effect loss/gradient component only',
                'production_DDP_trainer_qualified':False,'data':artifact(a.pair_dataset),'context':artifact(a.context_manifest),
                'model_config':asdict(mc),'ranks':gathered,'optimizer_updates':0,'simulation_calls':0,
                'declared_tolerance':{'rtol':2e-4,'atol':2e-6},'torch':torch.__version__})
            print(json.dumps({'status':'PASS','scope':'actual two-GPU NCCL gradient component'}),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
