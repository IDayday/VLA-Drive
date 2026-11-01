"""Three directed checks for task-gated negative-good suppression and recovery."""
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
from starVLA.rl.qwen_world_safety_exploration import protected_credit as P
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit, canonical_components, raw_advantage_parts
from starVLA.rl.qwen_world_safety_exploration.recovery import tensor_digest
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe import checkpoint


def cfg(g):
    c=base_cfg(g);c['research']['structured_advantage']=P.SCHEMA
    return c


def identity(g):
    out=base_identity(g);out['recipe']=cfg(g)
    return out


def population(g):
    x=components(g)
    x[0,:,:3]=1;x[0,:,3]=1;x[0,0,3]=.76;x[0,1,3]=.9;x[0,2,3]=.8
    x[1,:,:3]=1;x[1,:,3]=1;x[1,0,3]=.76;x[1,-1,2]=0
    recs=[records(v) for v in x]
    for rank in (0,1):
        for j,r in enumerate(recs[rank]):
            comfort=0. if rank==0 and j in (2,3) else 1.
            m=r['metrics'];score=m['no_at_fault_collisions']*m['drivable_area_compliance']*(5*m['ego_progress']+5*m['time_to_collision_within_bound']+2*comfort)/12
            m.update(comfort=comfort,score=score);r['readouts']['canonical_PDMS']=score
    rewards=torch.cat([canonical_components(r,g)['canonical_rewards'] for r in recs])
    material=torch.cat((rewards.double().unsqueeze(-1),x),-1)
    return x,recs,rewards,material,None


def install_gather(monkeypatch,material):
    from starVLA.rl.qwen_world_safety_exploration import credit as C
    calls=[]
    monkeypatch.setattr(C.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(C.dist,'get_world_size',lambda:8)
    def gather(out,value):
        calls.append(value.clone())
        for rank in range(8):out[rank].copy_(material[rank:rank+1])
    monkeypatch.setattr(C.dist,'all_gather',gather)
    return calls


def test_exact_high_negative_gate_every_other_candidate_unchanged_G8_G16():
    for g in (8,16):
        P.validate_protected_recipe(cfg(g));x,recs,rewards,material,_=population(g)
        credit=P.ProtectedCredit(g);base=GroupCredit(g)
        for rank in range(8):
            got=credit.advantages(rewards[rank:rank+1],x[rank:rank+1],records=recs[rank],all_material=material)
            old=base.advantages(rewards[rank:rank+1],x[rank:rank+1],all_material=material)
            info=got['protection'];eligible=info['eligible'];keep=~eligible
            for k in ('A_safe','feasible','cost','global_std','normalization_std','adv_mask'):
                assert torch.equal(got[k],old[k]),k
            assert torch.equal(got['A_raw'][keep],old['A_raw'][keep])
            assert torch.equal(got['advantages'][keep.reshape(-1)],old['advantages'][keep.reshape(-1)])
            assert got['A_raw'][eligible].eq(0).all() and got['advantages'][eligible.reshape(-1)].eq(0).all()
            assert torch.equal(got['A_EP'],old['A_EP']+info['delta'])
            torch.testing.assert_close(got['A_safe']+got['A_EP'],got['A_raw'],atol=1e-14,rtol=0)
            if rank in (0,1):
                assert info['activated'] and info['eligible'][0,0]
                assert recs[rank][0]['metrics']['score']>=.9
                # FP32 reward rounds down: the true official >=.9 gate still protects.
                assert rewards[rank,0]<.9 or float(rewards[rank,0])==pytest.approx(.9)
                assert got['A_raw'].mean()>old['A_raw'].mean()
            else:assert not info['activated']
            if rank==0:
                assert not eligible[0,2] and old['A_raw'][0,2]<0  # Comfort0 negative remains negative
                assert old['A_raw'][0,3]>0 and not eligible[0,3]  # lowR positive remains positive
        # Active good protection cannot invent a fallback if base std is zero.
        info=credit.advantages(rewards[:1],x[:1],records=recs[0],all_material=material)['protection']
        original=GroupCredit.advantages
        def fake_zero(self,*a,**kw):
            out=original(self,*a,**kw);out['normalization_std']=torch.tensor(0.,dtype=torch.float64);return out
        with patch.object(GroupCredit,'advantages',fake_zero):
            with pytest.raises(ValueError,match='no fabricated fallback'):
                credit.advantages(rewards[:1],x[:1],info=info,all_material=material)
        tied=torch.ones(1,g,4,dtype=torch.float64);tr=records(tied[0]);rw=canonical_components(tr,g)['canonical_rewards']
        neutral=credit.advantages(rw,tied,records=tr,all_material=torch.cat((rw.double().unsqueeze(-1),tied),-1).expand(8,-1,-1))
        assert neutral['normalization_std']==0 and neutral['adv_mask'].sum()==0 and not neutral['protection']['activated']


def test_global64_128_and_native_material_pollution_guards(monkeypatch):
    for g in (8,16):
        x,recs,rewards,material,_=population(g);calls=install_gather(monkeypatch,material)
        credit=P.ProtectedCredit(g)
        rs=[credit.finalize(rollout(rank,g,recs[rank])) for rank in range(8)]
        assert len(calls)==8
        for u in range(4):
            summaries=[P.validate_protected_boundary(state(g,u),r,identity(g),credit=credit) for r in rs]
            P.validate_protected_ranks(summaries,credit=credit)
        for change in ('quality','eligible','raw','graph','sat','scale','mask'):
            bad=copy.deepcopy(rs[0])
            if change=='quality':bad['protected_credit']['quality_material'][0,0,0]=0
            if change=='eligible':bad['protected_credit']['eligible'][0,0]=False
            if change=='raw':bad['protected_credit']['original_raw_A'][0,0]+=.01
            if change=='graph':bad['protected_credit']['delta'].requires_grad_(True)
            if change=='sat':bad['saturated_credit']={}
            if change=='scale':bad['normalization_std']+=.01
            if change=='mask':bad['adv_mask'][0]=0
            with pytest.raises(ValueError):
                ss=[P.validate_protected_boundary(state(g,3),bad if rank==0 else r,identity(g),credit=credit) for rank,r in enumerate(rs)]
                P.validate_protected_ranks(ss,credit=credit)
        bad=copy.deepcopy(summaries);bad[0]['advantage_statistics']['protected_credit_activated_candidates']+=1
        with pytest.raises(ValueError):P.validate_protected_ranks(bad,credit=credit)
    from starVLA.rl.qwen_world_safety_exploration.saturated_credit import SaturatedRuntimeMixin
    class Wrong(P.ProtectedRuntimeMixin,SaturatedRuntimeMixin):pass
    with pytest.raises(ValueError):Wrong().make_credit(cfg(8))


def _native_worker(rank, initfile, root, variant="protected"):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+initfile, rank=rank, world_size=8,
                            timeout=timedelta(seconds=90))
    try:
        if variant == 'inversion':
            from starVLA.rl.qwen_world_safety_exploration import inversion_credit as backend
            from test_world_safety_exploration_inversion import population as make_population, cfg as make_cfg, identity as make_identity
            credit_type, loss_type = backend.InversionCredit, backend.InversionLoss
            boundary, ranks = backend.validate_inversion_boundary, backend.validate_inversion_ranks
            metrics_prefix = 'inversion_credit_'
        else:
            credit_type, loss_type = P.ProtectedCredit, P.ProtectedLoss
            boundary, ranks = P.validate_protected_boundary, P.validate_protected_ranks
            make_population, make_cfg, make_identity = population, cfg, identity
            metrics_prefix = 'protected_credit_'
        g = 8; credit = credit_type(g); ident = make_identity(g)
        x, recs, rewards, material, all_new = make_population(g)
        r = credit.finalize(rollout(rank, g, recs[rank]))
        actor = ToyModel(g); actor.device = torch.device('cpu')
        loss = loss_type(make_cfg(g))
        optimizer = torch.optim.AdamW(actor.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        generator = torch.Generator().manual_seed(42+rank)
        random.seed(42+rank); np.random.seed(42+rank); torch.manual_seed(42+rank)
        digest = tensor_digest(r)
        kwargs = dict(boundary_validator=lambda *a, **kw: boundary(*a, credit=credit, **kw),
                      rank_validator=lambda ss: ranks(ss, credit=credit))
        for u in range(4):
            native_state = state(g, u)
            before = actor.p.detach().clone(); expected_rng = generator.get_state().clone()
            with patch.object(torch.cuda, 'is_available', lambda: False):
                path = checkpoint.save(root, actor, optimizer, scheduler, generator, native_state, r, ident, **kwargs)
                with torch.no_grad(): actor.p.add_(1)
                torch.rand(3, generator=generator); torch.rand(2); random.random(); np.random.rand()
                restored, r = checkpoint.restore(path, actor, optimizer, scheduler, generator, ident, **kwargs)
            assert torch.equal(before, actor.p) and torch.equal(expected_rng, generator.get_state())
            assert tensor_digest(r) == digest and restored == native_state
            baseline = copy.deepcopy(actor)
            total, metrics = loss._policy_loss(actor, r)
            original, _ = ResearchLoss(make_cfg(g))._policy_loss(baseline, r)
            assert torch.equal(total, original)
            assert int(metrics[metrics_prefix+'activated_groups']) == int(r[credit.payload_key]['activated'])
            total.backward(); original.backward()
            assert torch.isfinite(actor.p.grad).all() and torch.equal(actor.p.grad, baseline.p.grad)
            dist.all_reduce(actor.p.grad); actor.p.grad.div_(8)
            # CPU Adam must not probe CUDA graph capture on this GPU-equipped host.
            with patch.object(torch.cuda, 'is_available', lambda: False):
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        ready = state(g, 0); ready.update(updates=4, next_inner=0, phase='READY')
        with patch.object(torch.cuda, 'is_available', lambda: False):
            path = checkpoint.save(root, actor, optimizer, scheduler, generator, ready, None, ident, **kwargs)
            loaded, none = checkpoint.restore(path, actor, optimizer, scheduler, generator, ident, **kwargs)
        assert loaded == ready and none is None
        if rank == 0:
            assert len(list(Path(path).glob('rank*.pt'))) == 8 and (Path(path)/'COMPLETE').is_file()
            (Path(root)/'NATIVE_CPU_PASS').write_text('8 CPU ranks: pending0/1/2/3 and READY4; native Adam/RNG/material restore; same source loss+grad\n')
    finally:
        dist.destroy_process_group()


def test_actual_native_eight_CPU_rank_four_inner_roundtrip(tmp_path):
    torch.set_num_threads(1)
    mp.start_processes(_native_worker, args=(str(tmp_path/'gloo'), str(tmp_path/'native')),
                       nprocs=8, join=True, start_method='fork')
    assert (tmp_path/'native/NATIVE_CPU_PASS').is_file()
