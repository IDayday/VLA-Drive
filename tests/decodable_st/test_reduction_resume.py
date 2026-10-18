import json
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from starVLA.model.modules.decodable_st.losses import scene_mean,feature_loss
from tools.decodable_st.runtime import save_checkpoint,load_checkpoint


def _rank(rank,init,out):
    dist.init_process_group("gloo",init_method=init,rank=rank,world_size=2)
    model=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2,1,bias=False))
    with torch.no_grad():model.module.weight.fill_(.2)
    x=torch.tensor([[1.,2.],[3.,4.]])+rank
    target=torch.tensor([[.5],[1.]])
    # rank 0 wholly empty, rank 1 valid. The global task still has weight one.
    prediction=model(x)
    loss,_=scene_mean((prediction-target).square(),torch.full_like(target,rank==1,dtype=torch.bool))
    loss.backward()
    torch.save(model.module.weight.grad,Path(out)/f"rank{rank}.pt")
    dist.destroy_process_group()


def test_two_rank_empty_matches_global_batch(tmp_path):
    init="file://"+str(tmp_path/"rendezvous")
    mp.spawn(_rank,args=(init,str(tmp_path)),nprocs=2,join=True)
    model=torch.nn.Linear(2,1,bias=False)
    with torch.no_grad():model.weight.fill_(.2)
    x=torch.tensor([[2.,3.],[4.,5.]])
    ((model(x)-torch.tensor([[.5],[1.]])).square().mean()).backward()
    for rank in range(2):torch.testing.assert_close(torch.load(tmp_path/f"rank{rank}.pt",weights_only=True),model.weight.grad,rtol=0,atol=0)


def test_accumulation_full_optimizer_count():
    x=torch.tensor([[.3,.9],[1.,2.],[3.,4.]],requires_grad=True)
    valid=torch.tensor([[False,False],[True,False],[True,True]])
    loss,_=scene_mean(x.square(),valid);loss.backward();ref=x.grad.clone();x.grad=None
    for i in range(3):
        loss,_=scene_mean(x[i:i+1].square(),valid[i:i+1],global_valid_scenes=2);loss.backward()
    torch.testing.assert_close(x.grad,ref,rtol=0,atol=0)


def test_unknown_feature_nan_no_gradient_contamination():
    x=torch.ones(2,3,4,requires_grad=True);target=x.detach().clone();target[0]=float("nan")
    valid=torch.ones(2,3,1,dtype=torch.bool);valid[0]=False
    loss,count=feature_loss(x,target,valid,0.,1.);loss.backward()
    assert count==1 and torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert x.grad[0].count_nonzero()==0


def test_native_resume_includes_rng_optimizer_scheduler(tmp_path):
    torch.manual_seed(42);model=torch.nn.Linear(4,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,.9)
    def step(m,o,s):
        o.zero_grad();m(torch.randn(3,4)).square().mean().backward();o.step();s.step()
    step(model,optimizer,scheduler)
    save_checkpoint(tmp_path/"latest.pt",model=model,optimizer=optimizer,scheduler=scheduler,
         identity="example_strict_identity",progress={"update":1},sampler={"order":[2,1,0],"offset":1})
    step(model,optimizer,scheduler);expected={k:v.clone() for k,v in model.state_dict().items()}
    other=torch.nn.Linear(4,2);opt=torch.optim.AdamW(other.parameters(),lr=.01);sch=torch.optim.lr_scheduler.ExponentialLR(opt,.9)
    state=load_checkpoint(tmp_path/"latest.pt",model=other,optimizer=opt,scheduler=sch,identity="example_strict_identity")
    assert state["sampler"]["offset"]==1 and state["progress"]["update"]==1
    step(other,opt,sch)
    for k,v in other.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    assert sch.state_dict()==scheduler.state_dict()
