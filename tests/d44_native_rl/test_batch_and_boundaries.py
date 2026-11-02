"""Compare the real accumulation weights with a global-batch gradient oracle."""
from contextlib import nullcontext
from pathlib import Path
import json
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from starVLA.rl.d44_native_rl.gaussian_objective import gaussian_terms,clipped_loss
from starVLA.rl.d44_native_rl.safety_credit import raw_credit
from starVLA.rl.d44_native_rl.recovery import Boundary,checkpoint
from starVLA.rl.d44_native_rl.adaptive_kl import AdaptiveKL


class GaussianProbe(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.linspace(-.002,.002,24).reshape(8,3))
    def forward(self,old,ref,nxt,adv):
        mean=old+self.weight
        logr,kl=gaussian_terms(mean,old,ref,torch.tensor([.1,.2,.3]),nxt)
        return clipped_loss(logr,adv)[0]+.001*kl.mean()


def distributed_gradient(rank,path,out):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    torch.manual_seed(51)
    model=GaussianProbe();ddp=DistributedDataParallel(model)
    old=torch.randn(4,16,3,8,3);ref=old+.01
    nxt=old+torch.randn_like(old)*torch.tensor([.1,.2,.3])[:,None,None]
    factors=torch.ones(4,16,6);factors[:,0,0]=0;factors[1,:,0]=0
    raw,_=raw_credit(factors);valid=torch.tensor([True,True,True,False])
    adv=(raw/(raw[valid].std(unbiased=False)+1e-8)).float()
    for j in range(2):
        i=j*2+rank
        with ddp.no_sync()if j==0 else nullcontext():
            loss=ddp(old[i],ref[i],nxt[i],adv[i])
            (loss*(2/3 if valid[i]else 0)).backward()
    torch.save(model.weight.grad,Path(out)/f'grad{rank}.pt')
    if rank==0:
        baseline=GaussianProbe()
        torch.stack([baseline(old[i],ref[i],nxt[i],adv[i])for i in range(3)]).mean().backward()
        torch.save(baseline.weight.grad,Path(out)/'reference.pt')
    dist.destroy_process_group()


def test_two_rank_accumulation_matches_explicit_global_batch(tmp_path):
    mp.spawn(distributed_gradient,args=(str(tmp_path/'gloo'),str(tmp_path)),nprocs=2,join=True)
    expected=torch.load(tmp_path/'reference.pt',weights_only=True)
    for rank in range(2):torch.testing.assert_close(torch.load(tmp_path/f'grad{rank}.pt',weights_only=True),expected,rtol=1e-5,atol=1e-7)


def state():
    return dict(updates=0,collections=0,next_inner=0,phase='READY',scene_cursor=0,valid_scene_exposures=0,
        new_candidates=0,adaptive_kl=AdaptiveKL().state_dict())


def test_collecting_never_publishes_latest(tmp_path):
    identity=dict(world=1,sha256='own',recipe=dict(inner_updates=2,global_scenes=32))
    actor=nn.Linear(1,1);optimizer=torch.optim.AdamW(actor.parameters());generator=torch.Generator()
    class Scheduler:
        def state_dict(self):return dict(last_epoch=0)
    s=state();s['phase']='COLLECTING'
    with pytest.raises(RuntimeError,match='Uncommitted'):
        checkpoint.save(tmp_path,actor,optimizer,Scheduler(),generator,s,None,identity,runtime=Boundary())
    assert not(tmp_path/'latest.json').exists()and not list(tmp_path.rglob('COMPLETE'))
    checkpoint.failure_evidence(tmp_path,s,'scoring failed','own-attempt')
    evidence=list(tmp_path.rglob('evidence.json'));assert len(evidence)==1
    assert json.loads(evidence[0].read_text())['recoverable']is False


def test_tied_decimal_cost_has_no_fake_signal():
    f=torch.ones(3,32,6);f[...,:3]=torch.tensor([0.,.5,0.])
    a,_=raw_credit(f);assert torch.equal(a,torch.zeros_like(a))


@pytest.mark.parametrize('failure',['rank_write','before_complete'])
def test_failed_publication_does_not_advance_latest_and_new_attempt_works(tmp_path,failure):
    identity=dict(world=1,sha256='own',recipe=dict(inner_updates=2,global_scenes=32))
    actor=nn.Linear(1,1);optimizer=torch.optim.AdamW(actor.parameters());generator=torch.Generator()
    class Scheduler:
        def state_dict(self):return dict(last_epoch=0)
    def inject(where,rank):
        if where==failure:raise OSError('injected native publication failure')
    with pytest.raises(RuntimeError,match='Uncommitted'):
        checkpoint.save(tmp_path,actor,optimizer,Scheduler(),generator,state(),None,identity,fault=inject,runtime=Boundary())
    assert not(tmp_path/'latest.json').exists()
    incomplete=list((tmp_path/'checkpoints').glob('*.incomplete.*'));assert incomplete
    assert not any((p/'COMPLETE').exists()for p in incomplete)
    published=Path(checkpoint.save(tmp_path,actor,optimizer,Scheduler(),generator,state(),None,identity,runtime=Boundary()))
    assert json.loads((tmp_path/'latest.json').read_text())['checkpoint']==str(published)
    assert len(list((tmp_path/'checkpoints').glob('*.incomplete.*')))==len(incomplete)
