"""Three directed CPU checks for saturation-only canonical credit and recovery."""
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
from starVLA.rl.qwen_world_safety_exploration import saturated_credit as S
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit, canonical_components, raw_advantage_parts
from starVLA.rl.qwen_world_safety_exploration.recovery import tensor_digest
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe import checkpoint


def cfg(g):
    out = base_cfg(g); out['research']['structured_advantage'] = S.SCHEMA
    return out


def identity(g):
    out = base_identity(g); out['recipe'] = cfg(g)
    return out


def population(g, *, fallback=False):
    x = torch.ones(8, g, 4, dtype=torch.float64) if fallback else components(g)
    recs = [records(v) for v in x]
    if fallback:
        # One Comfort-variable saturated group, seven genuinely canonical-full ties.
        for rec in (r for group in recs[1:] for r in group):
            rec['metrics'].update(comfort=1., score=1.)
            rec['readouts']['canonical_PDMS'] = 1.
    rewards = torch.cat([canonical_components(r, g)['canonical_rewards'] for r in recs])
    material = torch.cat((rewards.double().unsqueeze(-1), x), -1)
    raw = raw_advantage_parts(x)['A_raw'].clone()
    for rank in range(8):
        info = S.decision(rewards[rank:rank+1], x[rank:rank+1])
        if info['eligible']: raw[rank:rank+1] += info['delta']
    return x, recs, rewards, material, raw


def install_gather(monkeypatch, material):
    calls = []
    monkeypatch.setattr(S.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(S.dist, 'get_world_size', lambda: 8)
    def gather(out, value):
        calls.append(tuple(value.shape))
        for rank in range(8): out[rank].copy_(material[rank:rank+1])
    monkeypatch.setattr(S.dist, 'all_gather', gather)
    return calls


def test_only_strict_EP1_canonical_variable_groups_change_unrelated_groups_bitexact():
    for g in (8, 16):
        S.validate_saturated_recipe(cfg(g))
        x, recs, rewards, material, all_new = population(g)
        credit = S.SaturatedCredit(g); base = GroupCredit(g)
        for rank in range(8):
            info = S.decision(rewards[rank:rank+1], x[rank:rank+1])
            got = credit.advantages(rewards[rank:rank+1], x[rank:rank+1], all_material=material)
            old = base.advantages(rewards[rank:rank+1], x[rank:rank+1], all_material=material)
            for k in ('A_safe', 'feasible', 'cost', 'global_std', 'normalization_std'):
                assert torch.equal(got[k], old[k]), k
            if rank == 4:
                assert info['eligible'] and got['adv_mask'].sum() == g
                expected = rewards[rank:rank+1].double()-rewards[rank:rank+1].double().mean(1, keepdim=True)
                assert torch.equal(info['delta'], expected)
                assert torch.equal(got['A_raw'], expected) and torch.equal(got['A_EP'], expected)
                assert got['advantages'][1] > 0 > got['advantages'][0]
            else:
                assert not info['eligible']
                for k in ('A_EP', 'A_raw', 'advantages', 'adv_mask'):
                    assert torch.equal(got[k], old[k]), k
        # Canonical-full and all-Comfort0 ties stay PG0; no failed/EP-headroom group activates.
        strict = torch.ones(1, g, 4, dtype=torch.float64)
        for reward in (torch.ones(1, g), torch.full((1, g), 10/12)):
            info = S.decision(reward, strict)
            assert not info['eligible'] and info['status'] == 'UNCHANGED_CANONICAL_TIED'
        for component, value in ((0, .5), (1, 0.), (2, 0.), (3, .9)):
            bad = strict.clone(); bad[0, 0, component] = value
            info = S.decision(rewards[4:5], bad)
            assert not info['eligible'] and info['delta'] is None
        with pytest.raises(ValueError):
            wrong = cfg(g); wrong['research']['structured_advantage'] = 'baseline'; S.validate_saturated_recipe(wrong)


def test_global_scale_fallback_pending_four_inners_and_exact_material_guards(monkeypatch):
    for g in (8, 16):
        for fallback in (False, True):
            x, recs, rewards, material, all_new = population(g, fallback=fallback)
            calls = install_gather(monkeypatch, material)
            credit = S.SaturatedCredit(g)
            rs = [credit.finalize(rollout(rank, g, recs[rank])) for rank in range(8)]
            assert len(calls) == 8  # One original material gather suffices even for fallback.
            scale = all_new.std(unbiased=False) if fallback else raw_advantage_parts(x)['A_raw'].std(unbiased=False)
            for r in rs:
                assert torch.equal(r['normalization_std'], scale)
                assert r['advantage_statistics']['saturated_credit_normalization_fallback'] == int(fallback)
                assert r['advantage_statistics']['normalization_population_candidates'] == 8*g
            if fallback:
                assert rs[0]['adv_mask'].sum() == g
                assert all(r['adv_mask'].sum() == 0 for r in rs[1:])
            summaries = None
            for u in range(4):
                summaries = [S.validate_saturated_boundary(state(g, u), r, identity(g), credit=credit) for r in rs]
                S.validate_saturated_ranks(summaries, credit=credit)
            active = 0 if fallback else 4
            for change in ('delta', 'eligibility', 'dense'):
                bad = copy.deepcopy(rs[active])
                if change == 'delta': bad['saturated_credit']['delta'].requires_grad_(True)
                if change == 'eligibility': bad['saturated_credit']['eligible'] = False
                if change == 'dense': bad['dense_dac'] = {}
                with pytest.raises(ValueError): S.validate_saturated_boundary(state(g, 3), bad, identity(g), credit=credit)
            for change in ('normalization_std', 'statistics'):
                bad = copy.deepcopy(summaries)
                if change == 'normalization_std': bad[active]['normalization_std'] += .01
                else: bad[active]['advantage_statistics']['saturated_credit_activated_groups'] = 0
                with pytest.raises(ValueError): S.validate_saturated_ranks(bad, credit=credit)
    from starVLA.rl.qwen_world_safety_exploration.dense_credit import DenseRuntimeMixin
    class InvalidCombinedRuntime(S.SaturatedRuntimeMixin, DenseRuntimeMixin): pass
    with pytest.raises(ValueError): InvalidCombinedRuntime().make_credit(cfg(8))


def _native_worker(rank, initfile, root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+initfile, rank=rank, world_size=8,
                            timeout=timedelta(seconds=90))
    try:
        g = 8; credit = S.SaturatedCredit(g); ident = identity(g)
        x, recs, rewards, material, all_new = population(g, fallback=True)
        r = credit.finalize(rollout(rank, g, recs[rank]))
        actor = ToyModel(g); actor.device = torch.device('cpu')
        loss = S.SaturatedLoss(cfg(g))
        optimizer = torch.optim.AdamW(actor.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        generator = torch.Generator().manual_seed(42+rank)
        random.seed(42+rank); np.random.seed(42+rank); torch.manual_seed(42+rank)
        digest = tensor_digest(r)
        kwargs = dict(boundary_validator=lambda *a, **kw: S.validate_saturated_boundary(*a, credit=credit, **kw),
                      rank_validator=lambda ss: S.validate_saturated_ranks(ss, credit=credit))
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
            original, _ = ResearchLoss(cfg(g))._policy_loss(baseline, r)
            assert torch.equal(total, original)
            assert int(metrics['saturated_credit_activated_groups']) == int(rank == 0)
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
