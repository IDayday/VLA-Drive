from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from navsim.agents.EpisodeDrive.planreg_v2.losses import global_valid_mean
from navsim.agents.EpisodeDrive.planreg_v2.ema import FP32MasterEMA


def _uneven_worker(rank,path):
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    from navsim.agents.EpisodeDrive.planreg_v2.runtime import prepare_run_directory
    prepare_run_directory(path+'.run')
    prepare_run_directory(path+'.run',resume=True)
    try:prepare_run_directory(path+'.run')
    except FileExistsError:pass
    else:raise AssertionError('Existing run must be rejected by every rank')
    layer=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad(): layer.weight.fill_(1.)
    model=DistributedDataParallel(layer)
    x=torch.tensor([[1.],[2.]]) if rank==0 else torch.tensor([[3.],[4.]])
    mask=torch.tensor([True,False]) if rank==0 else torch.tensor([True,True])
    loss=global_valid_mean(model(x).flatten(),mask)
    loss.backward()
    torch.testing.assert_close(loss,torch.tensor(8/3))
    torch.testing.assert_close(layer.weight.grad,torch.tensor([[8/3]]))
    model.zero_grad(set_to_none=True)
    zero=global_valid_mean(model(x).flatten(),torch.zeros(2,dtype=torch.bool))
    zero.backward()
    assert zero==0 and layer.weight.grad.count_nonzero()==0
    dist.destroy_process_group()


def test_uneven_global_valid_counts_and_all_invalid_no_deadlock(tmp_path):
    mp.spawn(_uneven_worker,args=(str(tmp_path/'gloo_init'),),nprocs=2,join=True)


def test_actual_gradscaler_skips_optimizer_and_ema_and_accumulation():
    vision=torch.nn.Linear(2,2)
    adapter=torch.nn.Linear(2,2)
    student=SimpleNamespace(model=SimpleNamespace(vision_model=vision),planning_register_adapter=adapter)
    teacher=FP32MasterEMA(student,10,32)
    optimizer=torch.optim.AdamW(list(vision.parameters())+list(adapter.parameters()),lr=.001)
    optimizer.register_step_post_hook(lambda *_:teacher.update(student))
    scaler=torch.amp.GradScaler('cpu')
    for _ in range(2): scaler.scale(adapter(vision(torch.ones(2,2))).sum()/2).backward()
    assert int(teacher.updates)==0
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
    assert int(teacher.updates)==1
    scaler.scale(adapter(vision(torch.ones(2,2))).sum()*float('inf')).backward()
    scaler.step(optimizer);scaler.update()
    assert int(teacher.updates)==1


def test_accumulated_unequal_valid_masks_match_one_large_batch():
    weight=torch.nn.Parameter(torch.tensor(2.))
    x=torch.tensor([1.,2.,3.,4.]);valid=torch.tensor([1,0,1,1]).bool()
    reference=global_valid_mean(weight*x,valid)
    expected=torch.autograd.grad(reference,weight)[0]
    # Whole optimizer-batch count, not the average of two microbatch means.
    loss=sum(global_valid_mean(weight*part,mask,total_count=valid.sum())
             for part,mask in zip(x.chunk(2),valid.chunk(2)))
    actual=torch.autograd.grad(loss,weight)[0]
    torch.testing.assert_close(actual,expected)


def test_accumulated_runtime_counts_match_tf_ro_missing_prefix():
    from navsim.agents.EpisodeDrive.planreg_v2.runtime import accumulated_batches
    group=[]
    for future in ([True,False,True],[True,True,True]):
        target=dict(motion_timestamps=torch.arange(1,9)[None]*.5-.001,
            motion_valid=torch.ones(1,8,dtype=torch.bool),future_valid_mask=torch.tensor([future]),
            trajectory_valid=torch.ones(1,8,dtype=torch.bool),trajectory_long_valid=torch.tensor([True]))
        group.append(({},target))
    batches=list(accumulated_batches(iter(group),2))
    assert len(batches)==2
    counts=batches[0][2]
    torch.testing.assert_close(counts['tf'],torch.tensor([2.,1.,1.]))
    torch.testing.assert_close(counts['ro'],torch.tensor([2.,1.,2.]))
    assert counts['trajectory']==2 and counts['long']==2
