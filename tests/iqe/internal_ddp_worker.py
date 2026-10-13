"""Two-process Gloo/NCCL proof for the new stream-specific objective."""
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from iqe.internal_baseline.objectives import terms
from iqe.losses import reduce_terms


def setup():
    torch.manual_seed(91)
    m = torch.nn.Linear(3, 14)
    x = torch.randn(8, 3)
    b = {"scores": torch.tensor([[.0,.95],[.5,.52],[.4,.4],[.7,.2],[.8,.4],[.1,.7],[.3,.8],[.9,.8]]),
         "valid": torch.tensor([[False,False],[True,True],[True,True],[True,True],[False,True],[True,True],[False,False],[True,True]]),
         "trajectories": torch.zeros(8,2,8,3), "components": torch.full((8,2,6),.5),
         "component_valid": torch.ones(8,2,6,dtype=torch.bool),
         "natural": torch.tensor([True,True,True,False,False,True,False,False]),
         "comparison": torch.tensor([False,False,False,True,True,False,True,True])}
    return m,x,b


def loss(m,x,b):
    p=m(x)
    return terms({"values":p[:,:2].sigmoid(),"component_logits":p[:,2:].reshape(-1,2,6)},b)


def main(output):
    backend=os.environ.get('IQE_INTERNAL_DDP_BACKEND','gloo')
    index=int(os.environ.get('LOCAL_RANK','0'))
    if backend=='nccl':torch.cuda.set_device(index)
    dist.init_process_group(backend)
    rank=dist.get_rank(); m,x,b=setup()
    device=f'cuda:{index}' if backend=='nccl' else 'cpu'
    m=m.to(device);x=x.to(device);b={k:v.to(device) for k,v in b.items()}
    wrapped=DDP(m,device_ids=[index] if backend=='nccl' else None)
    rows=torch.arange(rank,8,2,device=device)
    local={k:v[rows] for k,v in b.items()}
    objective,logs=reduce_terms(loss(wrapped,x[rows],local),{'value':1.,'components':.2,'ranking':.2})
    objective.backward()
    gradients=[p.grad.clone().cpu() for p in m.parameters()]
    all_grad=[None,None];dist.all_gather_object(all_grad,gradients)
    # Check a fully empty-label rank still participates and receives global grad.
    m.zero_grad(set_to_none=True)
    if rank==0:
        local['valid'][:]=False
    empty_loss,_=reduce_terms(loss(wrapped,x[rows],local),{'value':1.,'components':.2,'ranking':.2})
    empty_loss.backward()
    empty=[p.grad.clone().cpu() for p in m.parameters()]
    dist.barrier();dist.destroy_process_group()
    if rank==0:
        ref,x,b=setup()
        expected,_=reduce_terms(loss(ref,x,b),{'value':1.,'components':.2,'ranking':.2});expected.backward()
        for a in all_grad:
            for actual,p in zip(a,ref.parameters()):torch.testing.assert_close(actual,p.grad,rtol=1e-5,atol=1e-7)
        ref.zero_grad(set_to_none=True);b['valid'][::2]=False
        expected,_=reduce_terms(loss(ref,x,b),{'value':1.,'components':.2,'ranking':.2});expected.backward()
        for actual,p in zip(empty,ref.parameters()):torch.testing.assert_close(actual,p.grad,rtol=1e-5,atol=1e-7)
        torch.save({'status':'PASS','backend':backend,'world_size':2,'unequal_denominators':True,'empty_rank':True,'logs':logs},output)


if __name__=='__main__':main(sys.argv[1])
