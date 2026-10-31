"""Three narrow checks: exact gates, both global G scales, native eight-rank IO."""
import copy
from datetime import timedelta
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_world_safety_exploration_groups import (
    cfg as base_cfg, components, records, rollout, identity as base_identity, state, ToyModel,
)
from starVLA.rl.qwen_world_safety_exploration import dense_credit as D
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit, canonical_components, raw_advantage_parts
from starVLA.rl.qwen_world_safety_exploration.recovery import tensor_digest
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe import checkpoint


def cfg(g):
    out=base_cfg(g);out['research']['structured_advantage']=D.SCHEMA
    return out


def identity(g):
    out=base_identity(g);out['recipe']=cfg(g)
    return out


def flags(g, *, initial=False, tied=False):
    out=torch.zeros(g,41,dtype=torch.bool)
    for j in range(g):out[j,1:1+(10 if tied else j+1)]=True
    out[:,0]=initial
    return out


def population(g, *, fallback=False):
    x=torch.ones(8,g,4,dtype=torch.float64) if fallback else components(g)
    if fallback:x[:,:,3]=1
    x[0,:,0]=1;x[0,:,1]=0;x[0,:,2]=1
    recs=[records(v) for v in x]
    for rec, value in zip(recs[0],flags(g)):
        rec[D.FIELD]=value.numpy().copy()
    rewards=torch.cat([canonical_components(r,g)['canonical_rewards'] for r in recs])
    material=torch.cat((rewards.double().unsqueeze(-1),x),-1)
    raw=raw_advantage_parts(x)['A_raw'].clone()
    info=D.decision(recs[0],x[0:1]);raw[0:1]+=info['delta']
    return x,recs,rewards,material,raw


def install_gather(monkeypatch,material,all_new):
    calls=[]
    monkeypatch.setattr(D.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(D.dist,'get_world_size',lambda:8)
    def gather(out,value):
        complete=material if value.ndim==3 else all_new
        calls.append(tuple(value.shape))
        for rank in range(8):out[rank].copy_(complete[rank:rank+1])
    monkeypatch.setattr(D.dist,'all_gather',gather)
    return calls


def test_eligible_duration_ordering_initial_gate_and_unrelated_A_bit_exact():
    for g in (8,16):
        x,recs,rewards,material,all_new=population(g)
        dense=D.DenseCredit(g);base=GroupCredit(g)
        for rank in range(8):
            info=D.decision(recs[rank],x[rank:rank+1])
            got=dense.advantages(rewards[rank:rank+1],x[rank:rank+1],info=info,all_material=material)
            old=base.advantages(rewards[rank:rank+1],x[rank:rank+1],all_material=material)
            assert torch.equal(got['normalization_std'],old['normalization_std'])
            assert torch.equal(got['global_std'],old['global_std'])
            assert torch.equal(got['A_EP'],old['A_EP'])
            if rank==0:
                assert info['eligible'] and info['flags'].shape==(g,41)
                assert torch.all(torch.diff(got['advantages'])<0)
                assert got['advantages'][0]>0>got['advantages'][-1]
                assert not got['feasible'].any()  # least-cost is still failed
                torch.testing.assert_close(got['A_raw'].sum(),torch.tensor(0.,dtype=torch.float64),atol=1e-14,rtol=0)
            else:
                assert not info['eligible'] and info['flags'] is None
                assert torch.equal(got['A_raw'],old['A_raw'])
                assert torch.equal(got['advantages'],old['advantages'])
                assert torch.equal(got['adv_mask'],old['adv_mask'])
        for mode,status in [('initial','UNCHANGED_INITIALLY_NON_DRIVABLE'),
                            ('tied','UNCHANGED_OFFROAD_DURATION_TIED'),
                            ('missing','UNCHANGED_MISSING_REAL_TIME_FIELD')]:
            bad=copy.deepcopy(recs[0])
            for rec,value in zip(bad,flags(g,initial=mode=='initial',tied=mode=='tied')):rec[D.FIELD]=value.numpy()
            if mode=='missing':del bad[0][D.FIELD]
            info=D.decision(bad,x[:1])
            assert not info['eligible'] and info['status']==status
            assert info['flags'] is info['duration'] is info['delta'] is None
        bad=copy.deepcopy(recs[0]);bad[0][D.FIELD]=np.full(41,2)
        with pytest.raises(ValueError):D.decision(bad,x[:1])


def test_all_eight_rank_normalization_fallback_recompute_and_pollution(monkeypatch):
    for g in (8,16):
        for fallback in (False,True):
            x,recs,rewards,material,all_new=population(g,fallback=fallback)
            calls=install_gather(monkeypatch,material,all_new)
            credit=D.DenseCredit(g)
            rs=[credit.finalize(rollout(rank,g,recs[rank])) for rank in range(8)]
            assert len(calls)==8*(2 if fallback else 1)
            expected_scale=all_new.std(unbiased=False) if fallback else raw_advantage_parts(x)['A_raw'].std(unbiased=False)
            for r in rs:
                assert r['normalization_std']==expected_scale
                assert r['advantage_statistics']['dense_dac_normalization_fallback']==int(fallback)
                assert r['advantage_statistics']['normalization_population_candidates']==8*g
            assert rs[0]['advantage_statistics']['dense_dac_activated_groups']==1
            assert rs[0]['adv_mask'].sum()==g
            summaries=[D.validate_dense_boundary(state(g,1),r,identity(g),credit=credit) for r in rs]
            D.validate_dense_ranks(summaries,credit=credit)
            for u in (0,2,3):
                D.validate_dense_ranks([D.validate_dense_boundary(state(g,u),r,identity(g),credit=credit) for r in rs],credit=credit)
            bad=copy.deepcopy(rs[0]);bad['records'][0][D.FIELD][0]=True
            with pytest.raises(ValueError):D.validate_dense_boundary(state(g,1),bad,identity(g),credit=credit)
            bad=copy.deepcopy(rs[0]);bad['dense_dac']['delta'].requires_grad_(True)
            with pytest.raises(ValueError):D.validate_dense_boundary(state(g,1),bad,identity(g),credit=credit)
            bad=copy.deepcopy(rs[0]);bad['dense_dac']['flags'][0,2]=True
            with pytest.raises(ValueError):D.validate_dense_boundary(state(g,1),bad,identity(g),credit=credit)
            bad=copy.deepcopy(rs[1]);bad['dense_dac']['flags']=torch.zeros(g,41,dtype=torch.bool)
            with pytest.raises(ValueError):D.validate_dense_boundary(state(g,1),bad,identity(g),credit=credit)
            bad=copy.deepcopy(summaries);bad[0]['normalization_std']+=.01
            with pytest.raises(ValueError):D.validate_dense_ranks(bad,credit=credit)
            bad=copy.deepcopy(summaries);bad[0]['advantage_statistics']['dense_dac_activated_groups']=0
            with pytest.raises(ValueError):D.validate_dense_ranks(bad,credit=credit)


def _native_worker(rank,initfile,root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+initfile,rank=rank,world_size=8,timeout=timedelta(seconds=90))
    try:
        g=8;credit=D.DenseCredit(g);ident=identity(g)
        x,recs,rewards,material,all_new=population(g,fallback=True)
        r=credit.finalize(rollout(rank,g,recs[rank]))
        actor=ToyModel(g);actor.device=torch.device('cpu')
        loss=D.DenseDACLoss(cfg(g))
        optimizer=torch.optim.AdamW(actor.parameters(),lr=5e-5)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        generator=torch.Generator().manual_seed(42+rank)
        random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42+rank)
        digest=tensor_digest({k:v for k,v in r.items() if k != 'records'})
        kwargs=dict(boundary_validator=lambda *a,**kw:D.validate_dense_boundary(*a,credit=credit,**kw),
                    rank_validator=lambda ss:D.validate_dense_ranks(ss,credit=credit))
        for u in range(4):
            native_state=state(g,u)
            before=actor.p.detach().clone();expected_rng=generator.get_state().clone()
            with patch.object(torch.cuda,'is_available',lambda:False):
                path=checkpoint.save(root,actor,optimizer,scheduler,generator,native_state,r,ident,**kwargs)
                with torch.no_grad():actor.p.add_(1)
                torch.rand(3,generator=generator);torch.rand(2);random.random();np.random.rand()
                restored,r=checkpoint.restore(path,actor,optimizer,scheduler,generator,ident,**kwargs)
            assert torch.equal(before,actor.p) and torch.equal(expected_rng,generator.get_state())
            assert tensor_digest({k:v for k,v in r.items() if k != 'records'})==digest and restored==native_state
            if rank == 0:
                for rec,original_record in zip(r['records'],recs[rank]):
                    assert np.array_equal(rec[D.FIELD],original_record[D.FIELD])
            baseline=copy.deepcopy(actor)
            total,metrics=loss._policy_loss(actor,r)
            original,_=ResearchLoss(cfg(g))._policy_loss(baseline,r)
            assert torch.equal(total,original)
            assert int(metrics['dense_dac_activated_groups'])==int(rank==0)
            total.backward()
            assert torch.isfinite(actor.p.grad).all()
            dist.all_reduce(actor.p.grad);actor.p.grad.div_(8)
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        ready=state(g,0);ready.update(updates=4,next_inner=0,phase='READY')
        with patch.object(torch.cuda,'is_available',lambda:False):
            path=checkpoint.save(root,actor,optimizer,scheduler,generator,ready,None,ident,**kwargs)
            loaded,none=checkpoint.restore(path,actor,optimizer,scheduler,generator,ident,**kwargs)
        assert loaded==ready and none is None
        if rank==0:
            assert len(list(Path(path).glob('rank*.pt')))==8
            assert (Path(path)/'COMPLETE').is_file()
            (Path(root)/'NATIVE_CPU_PASS').write_text('8 ranks: pending0/1/2/3 + READY4; exact Adam/RNG/rollout restore; no CUDA\n')
    finally:
        dist.destroy_process_group()


def test_actual_native_checkpoint_eight_CPU_ranks_pending_four_inner_roundtrip(tmp_path):
    torch.set_num_threads(1)
    mp.start_processes(_native_worker,args=(str(tmp_path/'gloo'),str(tmp_path/'native')),
                       nprocs=8,join=True,start_method='fork')
    assert (tmp_path/'native/NATIVE_CPU_PASS').is_file()
