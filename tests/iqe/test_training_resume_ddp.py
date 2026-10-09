from copy import deepcopy
from pathlib import Path
import random
import os
import numpy as np
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from iqe.losses import LossTerm, scorer_terms, reduce_terms, collective_error
from iqe.scorer import ScorePrediction
from iqe.training.trainer import train
from iqe.training.checkpoint import capture_rng
from iqe.data.sampler import ConsumedSampler
from iqe.io import digest, read_json, atomic_torch, atomic_json, file_hash


def cfg(micro=2):
    return dict(global_batch_size=4,micro_batch_size=micro,max_optimizer_steps=4,warmup_steps=1,lr=.01,weight_decay=.01,
                lr_min_ratio=.1,grad_clip_norm=1.,validation_every_steps=2,early_stop_patience=4)


def plan():
    entries=[{"scene_id":str(i),"index":i,"bucket":"unit","source_group_id":str(i%3)} for i in range(16)]
    return {"entries":entries,"plan_hash":digest(entries)}


def fetch(entries,device):
    ids=torch.tensor([e["index"] for e in entries],device=device).float()
    return {"x":torch.stack([ids/16,torch.ones_like(ids)],-1),"y":ids[:,None]/32}


def loss_fn(module,batch):
    error=(module(batch["x"])-batch["y"]).square()
    return {"mse":LossTerm(error.sum(),error.new_tensor(len(error)))}


def denominator(batch):return {"mse":batch["y"].new_tensor(len(batch["y"]))}


def network(dropout=.0):return nn.Sequential(nn.Linear(2,4),nn.Tanh(),nn.Dropout(dropout),nn.Linear(4,1))


def test_microbatch_loss_updates_match_full_batch(tmp_path):
    torch.manual_seed(9); full=network();micro=deepcopy(full)
    a=train(full,loss_fn,fetch,ConsumedSampler(plan()),cfg(4),{"data":"unit"},tmp_path/"full",seed=6,denominator_function=denominator)
    b=train(micro,loss_fn,fetch,ConsumedSampler(plan()),cfg(1),{"data":"unit"},tmp_path/"micro",seed=6,denominator_function=denominator)
    for x,y in zip(full.parameters(),micro.parameters()):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
    for x,y in zip(a,b):assert abs(x["loss"]-y["loss"])<1e-7


def test_exact_resume_rng_next_ids_loss_parameters(tmp_path):
    torch.manual_seed(4); full=network(.3); interrupted=deepcopy(full)
    validation=lambda m,s:{"selection_key":[-s],"role":"stage_val"}
    whole=train(full,loss_fn,fetch,ConsumedSampler(plan()),cfg(),{"data":"unit"},tmp_path/"whole",seed=11,denominator_function=denominator,validate=validation)
    rng_whole=capture_rng()
    first=train(interrupted,loss_fn,fetch,ConsumedSampler(plan()),cfg(),{"data":"unit"},tmp_path/"interrupted",seed=11,
                denominator_function=denominator,stop_after=2,validate=validation)
    checkpoint=tmp_path/"interrupted/step_000002.pt"
    assert checkpoint.is_file()
    resumed=network(.3)
    second=train(resumed,loss_fn,fetch,ConsumedSampler(plan()),cfg(),{"data":"unit"},tmp_path/"interrupted",seed=11,
                 denominator_function=denominator,resume=checkpoint,validate=validation)
    assert whole[2]["sample_ids"]==second[0]["sample_ids"] and whole[2]["loss"]==second[0]["loss"]
    for a,b in zip(full.parameters(),resumed.parameters()):assert torch.equal(a,b)
    assert torch.equal(rng_whole["cpu"],capture_rng()["cpu"])
    assert rng_whole["python"]==capture_rng()["python"]
    a=read_json(tmp_path/"whole/result.json");b=read_json(tmp_path/"interrupted/result.json")
    assert a["best_selection_key"]==b["best_selection_key"]
    state=torch.load(tmp_path/"interrupted/step_000004.pt",weights_only=False)
    assert state["sampler"]["consumed_cursor"]==16 and state["trainer_state"]["bad_cycles"]==1


def test_exact_resume_world_change_rejected(tmp_path):
    torch.manual_seed(3)
    train(network(),loss_fn,fetch,ConsumedSampler(plan()),cfg(),{"data":"unit"},tmp_path,stop_after=2,denominator_function=denominator)
    path=tmp_path/"step_000002.pt";state=torch.load(path,weights_only=False);state["world_size"]=2
    atomic_torch(path,state); receipt=read_json(str(path)+".COMPLETE.json");receipt["checksum"]=file_hash(path);atomic_json(str(path)+".COMPLETE.json",receipt)
    with pytest.raises(ValueError,match="topology"):
        train(network(),loss_fn,fetch,ConsumedSampler(plan()),cfg(),{"data":"unit"},tmp_path,resume=path,denominator_function=denominator)


def test_completed_trainer_reuses_verified_weights_and_rejects_changed_config(tmp_path):
    torch.manual_seed(2);module=network()
    train(module,loss_fn,fetch,ConsumedSampler(plan()),cfg(),{'data':'unit'},tmp_path,denominator_function=denominator)
    before=file_hash(tmp_path/'step_000004.pt')
    restored=network()
    assert train(restored,loss_fn,fetch,ConsumedSampler(plan()),cfg(),{'data':'unit'},tmp_path,denominator_function=denominator)==[]
    assert file_hash(tmp_path/'step_000004.pt')==before
    assert all(torch.equal(a,b) for a,b in zip(module.parameters(),restored.parameters()))
    with pytest.raises(ValueError,match='config'):
        train(network(),loss_fn,fetch,ConsumedSampler(plan()),cfg()|{'lr':.03},{'data':'unit'},tmp_path,denominator_function=denominator)


def ddp_inputs():
    x=torch.tensor([[[1.,0.],[0.,1.]],[[2.,1.],[1.,2.]]])
    labels=torch.tensor([[.9,.1],[.2,.8]])
    mask=torch.tensor([[False,False],[True,True]])
    cm=torch.tensor([[False,False],[True,False]])
    return x,labels,mask,cm


def ddp_worker(rank,rendezvous,folder,backend="gloo"):
    torch.set_num_threads(1)
    device=f'cuda:{rank}' if backend=='nccl' else 'cpu'
    if backend=='nccl':torch.cuda.set_device(rank)
    dist.init_process_group(backend,init_method="file://"+rendezvous,rank=rank,world_size=2)
    torch.manual_seed(2);module=DDP(nn.Linear(2,1).to(device),device_ids=[rank] if backend=='nccl' else None)
    x,truth,mask,cm=[t.to(device) for t in ddp_inputs()];x=x[rank:rank+1]
    logits=module(x).squeeze(-1)
    pred=ScorePrediction(logits.sigmoid(),{"nc":logits},torch.ones_like(logits,dtype=torch.bool))
    terms=scorer_terms(pred,truth[rank:rank+1],mask[rank:rank+1],{"nc":torch.ones_like(logits)*.5},{"nc":cm[rank:rank+1]})
    loss,logs=reduce_terms(terms,{"value":1.,"components":1.,"ranking":.2});loss.backward()
    torch.save({"gradient":module.module.weight.grad.cpu(),"logs":logs},Path(folder)/f"rank{rank}.pt")
    try:collective_error("missing_cache" if rank==0 else None)
    except ValueError as e:assert "missing_cache" in str(e)
    else:raise AssertionError("not all ranks exited")
    dist.destroy_process_group()


def test_real_two_process_gloo_denominators_gradients_and_collective_exit(tmp_path):
    mp.spawn(ddp_worker,args=(str(tmp_path/"rdzv"),str(tmp_path)),nprocs=2,join=True)
    torch.manual_seed(2);module=nn.Linear(2,1)
    x,truth,mask,cm=ddp_inputs();logits=module(x).squeeze(-1)
    pred=ScorePrediction(logits.sigmoid(),{"nc":logits},torch.ones_like(logits,dtype=torch.bool))
    terms=scorer_terms(pred,truth,mask,{"nc":torch.ones_like(logits)*.5},{"nc":cm})
    loss,logs=reduce_terms(terms,{"value":1.,"components":1.,"ranking":.2});loss.backward()
    for rank in (0,1):
        result=torch.load(tmp_path/f"rank{rank}.pt",weights_only=True)
        torch.testing.assert_close(result["gradient"],module.weight.grad,atol=1e-7,rtol=1e-6)
        assert result["logs"]==logs


def _ddp_resume_worker(rank, init_file, root, backend='gloo'):
    torch.set_num_threads(1)
    device='cuda' if backend=='nccl' else 'cpu'
    os.environ['LOCAL_RANK']=str(rank)
    if backend=='nccl':torch.cuda.set_device(rank)
    dist.init_process_group(backend,init_method='file://'+init_file,rank=rank,world_size=2)
    try:
        root=Path(root)
        torch.manual_seed(83);full=network(.3);partial=deepcopy(full)
        config=cfg(micro=1)
        complete=train(full,loss_fn,fetch,ConsumedSampler(plan()),config,{'test':'two_rank_resume'},root/'whole',seed=59,denominator_function=denominator,device=device)
        full_rng=capture_rng()
        train(partial,loss_fn,fetch,ConsumedSampler(plan()),config,{'test':'two_rank_resume'},root/'resume',seed=59,denominator_function=denominator,stop_after=2,device=device)
        resumed=network(.3)
        tail=train(resumed,loss_fn,fetch,ConsumedSampler(plan()),config,{'test':'two_rank_resume'},root/'resume',seed=59,denominator_function=denominator,resume=root/'resume/step_000002.pt',device=device)
        assert complete[2]['sample_ids']==tail[0]['sample_ids'] and complete[2]['loss']==tail[0]['loss']
        assert torch.equal(full_rng['cpu'],capture_rng()['cpu'])
        assert all(torch.equal(a,b) for a,b in zip(full_rng['cuda'],capture_rng()['cuda']))
        assert all(torch.equal(a,b) for a,b in zip(full.parameters(),resumed.parameters()))
        atomic_json(root/f'rank_{rank}.json',{'exact_resume':True,'next_ids_match':True,'loss_and_parameters_match':True})
    finally:
        dist.destroy_process_group()


def test_two_process_ddp_optimizer_boundary_exact_resume(tmp_path):
    mp.spawn(_ddp_resume_worker,args=(str(tmp_path/'init'),str(tmp_path)),nprocs=2,join=True)
    assert all(read_json(tmp_path/f'rank_{i}.json')['exact_resume'] for i in range(2))


@pytest.mark.skipif(os.environ.get('IQE_RUN_GPU_TESTS')!='1',reason='GPU test requires explicit qualified resources')
def test_nccl_uneven_labels_and_exact_resume(tmp_path):
    from iqe.resources import qualify
    qualify('cuda')
    assert torch.cuda.device_count()>=2
    mp.spawn(ddp_worker,args=(str(tmp_path/'nccl_grad'),str(tmp_path),'nccl'),nprocs=2,join=True)
    torch.manual_seed(2);module=nn.Linear(2,1)
    x,truth,mask,cm=ddp_inputs();logits=module(x).squeeze(-1)
    pred=ScorePrediction(logits.sigmoid(),{'nc':logits},torch.ones_like(logits,dtype=torch.bool))
    loss,logs=reduce_terms(scorer_terms(pred,truth,mask,{'nc':torch.ones_like(logits)*.5},{'nc':cm}),{'value':1.,'components':1.,'ranking':.2})
    loss.backward()
    for rank in (0,1):
        actual=torch.load(tmp_path/f'rank{rank}.pt',weights_only=True)
        torch.testing.assert_close(actual['gradient'],module.weight.grad,atol=1e-6,rtol=1e-5)
        for key in logs:
            for name,value in logs[key].items():assert actual['logs'][key][name]==pytest.approx(value,abs=1e-6)
    mp.spawn(_ddp_resume_worker,args=(str(tmp_path/'nccl_resume'),str(tmp_path),'nccl'),nprocs=2,join=True)
