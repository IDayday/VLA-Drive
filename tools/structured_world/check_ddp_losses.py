"""Actual two-GPU global-loss gradient equivalence, including an empty-object rank."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from starVLA.model.modules.structured_world.agent_heads import AgentHeads
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.losses import world_losses, normalize_accumulated_world_losses


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    dist.init_process_group('nccl');rank=dist.get_rank();torch.cuda.set_device(rank)
    torch.manual_seed(19)
    head=AgentHeads(16).cuda();ddp=DistributedDataParallel(head,device_ids=[rank])
    values=torch.load(a.target,map_location=f'cuda:{rank}',weights_only=True)
    target=WorldTargets(**values)
    empty_values={k:(v[:0] if torch.is_tensor(v) and k in ['current_boxes','current_classes','future_xy_in_ego_t0','future_valid_mask','current_supervision_mask','box_valid_mask'] else v) for k,v in values.items()}
    empty_values['track_ids']=();empty_values['overflow']=0
    empty=WorldTargets(**empty_values)
    torch.manual_seed(23);features=torch.randn(2,32,16,device='cuda')
    local=empty if rank==0 else target
    losses,_=world_losses(ddp(features[rank:rank+1]),[local]);sum(losses.values()).backward()
    ddp_grad=torch.cat([p.grad.flatten() for p in head.parameters()]).detach().clone()
    head.zero_grad(set_to_none=True)
    # Two retained microbatch graphs, deliberately different target counts across ranks.
    # Use the underlying head and explicitly average gradients, matching DDP's reducer.
    micros=[]
    for j in range(2):
        local_target=empty if rank==0 or j==0 else target
        sums,_=world_losses(head(features[rank:rank+1]+j*.1),[local_target],return_sums=True)
        micros.append(sums)
    sum(normalize_accumulated_world_losses(micros).values()).backward()
    for param in head.parameters():dist.all_reduce(param.grad);param.grad.div_(dist.get_world_size())
    accumulated=torch.cat([p.grad.flatten() for p in head.parameters()]).detach().clone()
    # Reference with both samples; temporarily disable collectives after finishing DDP.
    dist.barrier();dist.destroy_process_group()
    head.zero_grad(set_to_none=True)
    losses,_=world_losses(head(features),[empty,target]);sum(losses.values()).backward()
    reference=torch.cat([p.grad.flatten() for p in head.parameters()])
    error=float((ddp_grad-reference).abs().max())
    torch.testing.assert_close(ddp_grad,reference,rtol=2e-4,atol=2e-6)
    head.zero_grad(set_to_none=True)
    inputs=torch.cat([features[0:1],features[0:1]+.1,features[1:2],features[1:2]+.1])
    losses,_=world_losses(head(inputs),[empty,empty,empty,target]);sum(losses.values()).backward()
    reference_accum=torch.cat([p.grad.flatten() for p in head.parameters()])
    accum_error=float((accumulated-reference_accum).abs().max())
    torch.testing.assert_close(accumulated,reference_accum,rtol=2e-4,atol=2e-6)
    if rank==0:
        Path(a.output).write_text(json.dumps({'status':'PASS','device':'2 real CUDA GPUs NCCL','empty_rank':0,'max_abs_gradient_difference':error,'accumulation_max_abs_gradient_difference':accum_error,'scope':'world heads/loss reduction; not full Qwen DDP or exact resume'},indent=2))

if __name__=='__main__':main()
