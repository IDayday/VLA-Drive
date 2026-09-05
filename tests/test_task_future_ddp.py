import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from navsim.agents.EpisodeDrive.layers.world_model.task_future_loss import global_task_mean


def _worker(rank, init):
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=40))
    try:
        weight=torch.tensor([1.,2.,3.],requires_grad=True)
        elements=weight.square()[None].expand(4,3)*(rank+1)
        mask=torch.tensor([[True,False,False]]*4) if rank==0 else torch.tensor([[False,True,False]]*4)
        loss,_,counts=global_task_mean(elements,mask)
        loss.backward()
        dist.all_reduce(weight.grad);weight.grad/=2
        # Single-process global reference: (1^2 + 2 * 2^2)/2.
        torch.testing.assert_close(loss,torch.tensor(4.5))
        torch.testing.assert_close(weight.grad,torch.tensor([1.,4.,0.]))
        assert counts.tolist()==[4,4,0]
    finally:
        dist.destroy_process_group()


def test_different_rank_masks_equal_single_process(tmp_path):
    mp.spawn(_worker,args=('file://'+str(tmp_path/'gloo'),),nprocs=2,join=True)
