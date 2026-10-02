"""Real NCCL gradient equivalence, including an empty-label GPU rank."""
import json,os
import torch
import torch.distributed as dist
from torch import nn
from starVLA.model.modules.foresight.future_spatiotemporal_head import normalized_clip_loss


def main():
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2:raise ValueError('Two actual GPU processes required')
    torch.cuda.set_device(rank);dist.init_process_group('nccl');device='cuda'
    try:
        torch.manual_seed(42);layer=nn.Linear(5,6,bias=False).cuda()
        x=torch.randn(4,3,2,2,2,5,device=device);t=torch.randn(4,3,2,2,2,6,device=device)
        valid=torch.ones(4,3,2,2,2,dtype=torch.bool,device=device);valid[1,1:]=False;valid[2:]=False
        ids=slice(2*rank,2*(rank+1));loss,_=normalized_clip_loss(layer(x[ids]),t[ids],valid[ids],global_count=2)
        loss.backward();actual=layer.weight.grad.clone();dist.all_reduce(actual);actual/=world
        layer.zero_grad();reference,_=normalized_clip_loss(layer(x),t,valid);reference.backward()
        error=(actual-layer.weight.grad).abs().max();torch.testing.assert_close(actual,layer.weight.grad,rtol=2e-5,atol=2e-6)
        if rank==0:print(json.dumps({'passed':True,'world_size':world,'backend':'actual NCCL','empty_label_rank':1,'maximum_gradient_error':float(error)}),flush=True)
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
