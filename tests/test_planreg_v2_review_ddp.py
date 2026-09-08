"""Actual production loss + accumulation counts, whole logical batch reference."""
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from navsim.agents.EpisodeDrive.planreg_v2.losses import world_model_loss
from navsim.agents.EpisodeDrive.planreg_v2.runtime import accumulated_batches,validate_ttc_reduction


def fixtures(case):
    torch.manual_seed(992)
    x=torch.randn(8,3,16,4);target=torch.randn_like(x)
    future=torch.tensor([[1,1,1],[1,0,1],[0,1,1],[1,1,0],[1,0,0],[1,1,1],[0,0,0],[1,1,1]]).bool()
    if case=='rank_empty':future[::2]=False
    if case=='all_empty':future[:]=False
    valid=torch.ones(8,16,dtype=torch.bool);valid[1,8:]=False
    return x,target,future,valid


def make_target(future):
    b=len(future)
    return dict(motion_timestamps=torch.arange(1,9).expand(b,8).float()*.5,
        motion_valid=torch.ones(b,8,dtype=torch.bool),future_valid_mask=future,
        trajectory_valid=torch.ones(b,8,dtype=torch.bool),trajectory_long_valid=torch.ones(b,dtype=torch.bool))


def reference_values(case):
    x,target,future,valid=fixtures(case)
    layer=torch.nn.Linear(4,4,bias=False)
    with torch.no_grad():layer.weight.copy_(torch.eye(4))
    prediction=layer(x)
    # Independent full logical batch call, not a rewritten reduction algorithm.
    loss=world_model_loss(prediction,prediction*.7,target,valid,future,torch.ones(8,3,dtype=torch.bool))['wm_loss']
    loss.backward()
    return loss.detach(),layer.weight.grad


def _worker(rank,path,accumulate,case):
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    x,target,future,valid=fixtures(case)
    layer=torch.nn.Linear(4,4,bias=False)
    with torch.no_grad():layer.weight.copy_(torch.eye(4))
    model=DDP(layer)
    ids=torch.arange(rank,8,2)
    group=[({'ids':part},make_target(future[part])) for part in ids.chunk(accumulate)]
    value=0.
    for i,(features,targets,counts) in enumerate(accumulated_batches(iter(group),accumulate)):
        part=features['ids']
        with (model.no_sync() if i+1<accumulate else nullcontext()):
            pred=model(x[part])
            result=world_model_loss(pred,pred*.7,target[part],valid[part],future[part],
                                    torch.ones(len(part),3,dtype=torch.bool),counts)['wm_loss']
            # Production runner multiplies WM by accum then divides total loss
            # by accum. These cancel; counts already cover the whole optimizer batch.
            result.backward();value=value+result.detach()
    actual_grad=layer.weight.grad.detach().clone()
    scores=torch.zeros(2,64,7)
    if rank==1:scores[0,0,3]=2
    try:validate_ttc_reduction(scores,accumulate)
    except ValueError:pass
    else:raise AssertionError('Every rank must reject sentinel2 collectively')
    dist.destroy_process_group()
    expected,grad=reference_values(case)
    torch.testing.assert_close(value,expected,atol=2e-7,rtol=1e-6)
    torch.testing.assert_close(actual_grad,grad,atol=2e-7,rtol=1e-6)


def test_world2_accumulations_unequal_rank_empty_and_global_empty(tmp_path):
    for accumulate in (1,2):
        for case in ('unequal','rank_empty','all_empty'):
            mp.spawn(_worker,args=(str(tmp_path/(case+str(accumulate))),accumulate,case),nprocs=2,join=True)


def test_world1_accumulations_equal_one_large_batch():
    for case in ('unequal','rank_empty','all_empty'):
        expected,grad=reference_values(case)
        x,target,future,valid=fixtures(case)
        layer=torch.nn.Linear(4,4,bias=False)
        with torch.no_grad():layer.weight.copy_(torch.eye(4))
        groups=[({'ids':part},make_target(future[part])) for part in torch.arange(8).chunk(4)]
        value=0.
        for feature,_,counts in accumulated_batches(iter(groups),4):
            part=feature['ids'];pred=layer(x[part])
            loss=world_model_loss(pred,pred*.7,target[part],valid[part],future[part],torch.ones(len(part),3,dtype=torch.bool),counts)['wm_loss']
            loss.backward();value+=loss.detach()
        torch.testing.assert_close(value,expected,atol=2e-7,rtol=1e-6)
        torch.testing.assert_close(layer.weight.grad,grad,atol=2e-7,rtol=1e-6)
