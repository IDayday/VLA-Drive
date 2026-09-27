"""Two real GPU graph loss/accumulation equivalence and optimizer resume on real scenes."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from starVLA.model.modules.joint_world.flow import JointTrajectoryFlow, training_loss_sums
from starVLA.model.modules.structured_world.losses import global_mean
from tools.joint_world.train_graph import current_batch, load_samples
from tools.structured_world.runtime import seed_all
from tools.structured_world_v1p1.budget import start, record


class LossModule(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.graph=JointTrajectoryFlow(dim,**{k:v for k,v in cfg.items() if k in ['dim','heads','layers','scale_m','trajectory_mode','agent_scale_m']})

    def forward(self, xy, valid, noise, time, current):
        return training_loss_sums(self.graph,xy,valid,torch.ones_like(valid[:,:,0]),noise,time,**current)


def exact(x,y):
    if torch.is_tensor(x):return torch.equal(x,y)
    if isinstance(x,dict):return x.keys()==y.keys() and all(exact(x[k],y[k]) for k in x)
    if isinstance(x,(tuple,list)):return len(x)==len(y) and all(exact(a,b) for a,b in zip(x,y))
    return x==y


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','cache','targets','data-root','output','ledger','run-id']:p.add_argument('--'+key,required=True)
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Exactly two real GPUs required')
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl',timeout=timedelta(minutes=3))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    run=a.run_id+'_rank'+str(rank);completed=0
    identity={'arguments':vars(a),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'rank':rank,'gradient_atol':1e-5,'gradient_rtol':2e-5,'microbatches_per_rank':2,
              'logical_optimizer_updates_charged_to_rank0':3,'rank1_agent_labels':'intentionally empty, ego retained'}
    start(a.ledger,run,3 if rank==0 else 0,identity)
    try:
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=saved['identity']['config']
        manifest=json.loads((Path(a.cache)/'manifest.json').read_text())
        samples,_,_=load_samples(a.cache,a.targets,a.data_root,8,records=manifest['records'][:4])
        current=current_batch(samples);xy=torch.cat([s['xy'] for s in samples]).cuda();valid=torch.cat([s['valid'] for s in samples]).cuda()
        valid=valid.clone();valid[2:,1:]=False;valid[0,1:,::2]=False
        module=LossModule(current['context'].shape[-1],cfg).cuda();module.graph.load_state_dict(saved['model'],strict=True)
        reference=copy.deepcopy(module);ddp=DDP(module,device_ids=[rank],broadcast_buffers=False)
        seed_all(739);noise=torch.randn_like(xy);time=torch.rand(4,device='cuda')
        sums,counts=reference(xy,valid,noise,time,current)
        sum(sums[k]/max(counts[k],1) for k in sums).backward()
        local=slice(rank*2,rank*2+2)
        denominators={'ego':int(valid[local,0].sum())*2,'agents':int(valid[local,1:].sum())*2}
        def backward(model,noise,time):
            for j in range(2):
                index=rank*2+j;sl=slice(index,index+1)
                context=model.no_sync() if j==0 else __import__('contextlib').nullcontext()
                with context:
                    sums,_=model(xy[sl],valid[sl],noise[sl],time[sl],{k:v[sl] for k,v in current.items()})
                    sum(global_mean(sums[k],denominators[k]) for k in sums).backward()
        backward(ddp,noise,time)
        gradients={n:float((p.grad-dict(reference.named_parameters())[n].grad).abs().max()) for n,p in module.named_parameters()}
        close=all(torch.allclose(p.grad,dict(reference.named_parameters())[n].grad,atol=1e-5,rtol=2e-5) for n,p in module.named_parameters())
        if not close:raise AssertionError('DDP/accumulation gradient mismatch: '+str(max(gradients.values())))
        opt=torch.optim.AdamW(module.parameters(),lr=1e-4);opt.step();completed=1 if rank==0 else 0
        record(a.ledger,run,completed)
        state={'model':copy.deepcopy(module.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),
               'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
        torch.save(state,out/f'rank{rank}_checkpoint.pt')
        opt.zero_grad(set_to_none=True);backward(ddp,torch.randn_like(xy),torch.rand(4,device='cuda'));opt.step()
        completed=2 if rank==0 else 0;record(a.ledger,run,completed)
        expected_rng=(torch.get_rng_state(),torch.cuda.get_rng_state())
        clone=LossModule(current['context'].shape[-1],cfg).cuda()
        restored=torch.load(out/f'rank{rank}_checkpoint.pt',map_location='cpu',weights_only=False)
        clone.load_state_dict(restored['model'],strict=True);other=DDP(clone,device_ids=[rank],broadcast_buffers=False)
        other_opt=torch.optim.AdamW(clone.parameters(),lr=1e-4);other_opt.load_state_dict(restored['optimizer'])
        torch.set_rng_state(restored['cpu_rng'].cpu());torch.cuda.set_rng_state(restored['cuda_rng'].cpu())
        backward(other,torch.randn_like(xy),torch.rand(4,device='cuda'));other_opt.step()
        completed=3 if rank==0 else 0;record(a.ledger,run,completed)
        checks={'gradient_equivalence':close,'model_resume_exact':exact(module.state_dict(),clone.state_dict()),
                'optimizer_resume_exact':exact(opt.state_dict(),other_opt.state_dict()),
                'rng_resume_exact':exact(expected_rng,(torch.get_rng_state(),torch.cuda.get_rng_state()))}
        report={'identity':identity,'checks':checks,'gradient_max_absolute_delta':max(gradients.values()),
                'global_counts':counts,'rank_counts':denominators,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                'scope':'Real cached image conditions and real future labels; graph module only, not full distributed Qwen/DiT training.',
                'status':'PASS' if all(checks.values()) else 'FAIL'}
        (out/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        if not all(checks.values()):raise AssertionError(checks)
        record(a.ledger,run,completed,'complete')
    except BaseException:record(a.ledger,run,completed,'failed');raise
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
