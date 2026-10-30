"""Directed CPU math, finalized-loss and eight-rank recovery pilot contracts."""
import copy
import math

import pytest
import torch

from starVLA.rl.qwen_world_research_algorithms import safety_efficiency as se
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest


def groups():
    x = torch.ones(8, 8, 4, dtype=torch.float64)
    x[:, :, 3] = torch.linspace(.1, .8, 8, dtype=torch.float64)
    x[0, 4:, 2] = 0  # mixed feasible/failed, high EP unsafe
    x[1, 1:, 0] = 0  # exactly one feasible
    x[2, :, 0] = 0  # all-fail tied deficit despite EP/reward differences
    x[3, :, 0] = 0
    x[3, 4:, 1] = 0  # all-fail varying deficit
    x[5, :, 3] = .37  # all-feasible EP tie
    x[6, :, :3] = torch.tensor([0., .5, 1.], dtype=torch.float64)
    x[7, :, :3] = 1
    x[7, 1::2, 0] = .5  # partial NC is not feasible
    return x


def records(components):
    out = []
    for i, (nc, dac, ttc, ep) in enumerate(components.tolist()):
        comfort = float(i % 2)
        score = nc*dac*(5*ep+5*ttc+2*comfort)/12
        out.append(dict(status='OK', metrics=dict(
            no_at_fault_collisions=nc, drivable_area_compliance=dac,
            time_to_collision_within_bound=ttc, ego_progress=ep,
            comfort=comfort, score=score), readouts=dict(canonical_PDMS=score)))
    return out


def config():
    return dict(max_steps=10000, action_shape=[8, 3], model_family='QwenWorldPostCoT',
        prediction_type='x0', adaptation='ATTN_LORA_MEMORY', variant='R_CANONICAL',
        learning_rate=5e-5, memory_learning_rate=5e-6,
        grpo=dict(train=dict(kl_beta=50, ppo_clip_range=.005, adv_clip_max=5.,
            drop_zero_adv=True, adv_eps=1e-8, adv_clip_lower_quantile=0.,
            adv_clip_upper_quantile=1., adv_global_std=True, use_bc_loss=False)),
        research=dict(policy_objective='ppo', data_policy='uniform_full',
                      structured_advantage=se.STRUCTURED_SCHEMA))


def identity():
    return dict(world=8, sha256='new-safety-efficiency-arm',
        schema='world_postcot_cleanx_scope_RL_v1',
        projected_condition_identity='fixed-condition', recipe=config())


def state(update=0):
    return dict(updates=update, collections=1, next_inner=update,
        phase='ROLLOUT_COMMITTED' if update == 0 else 'INNER_COMMITTED',
        scene_cursor=8, raw_scene_exposures=8, new_candidates=64)


def rollout(rank, rec):
    return dict(rewards=se.canonical_components(rec)['canonical_rewards'], records=rec,
        chain=torch.zeros(8, 11, 8, 3), old_logp=torch.zeros(8, 3),
        timesteps=torch.arange(10), deltas=-torch.ones(10)/10, stoch_idx=torch.tensor([7, 8, 9]),
        old_probability={'source_clamped_logprob':torch.zeros(8, 3, 8, 3)},
        token=f'scene{rank}', log=f'log{rank}',
        behavior_version=dict(run='new-safety-efficiency-arm', update=0, adapter_sha256='fixed'),
        cond_bg=dict(schema='world_postcot_trainable_memory_observation_v1',
                     token=f'scene{rank}', identity='fixed-condition', repeats=8))


def mock_gather(monkeypatch):
    recs = [records(x) for x in groups()]
    rewards = torch.cat([se.canonical_components(r)['canonical_rewards'] for r in recs])
    material = torch.cat((rewards.double().unsqueeze(-1), groups()), -1)
    calls = []
    monkeypatch.setattr(se.dist, 'is_available', lambda:True)
    monkeypatch.setattr(se.dist, 'is_initialized', lambda:True)
    monkeypatch.setattr(se.dist, 'get_world_size', lambda:8)
    def gather(out, local):
        calls.append(local.clone())
        for rank in range(8):
            out[rank].copy_(material[rank:rank+1])
    monkeypatch.setattr(se.dist, 'all_gather', gather)
    return recs, rewards, material, calls


def independent_parts(x):
    """Scalar reference, independently follows the registered equations."""
    safe, efficiency = [], []
    for group in x.tolist():
        f = [float(nc == dac == ttc == 1) for nc, dac, ttc, ep in group]
        c = [(3-nc-dac-ttc)/3 for nc, dac, ttc, ep in group]
        n = sum(f)
        safe.append([v-n/8 for v in f] if n else [sum(c)/8-v for v in c])
        mean_ep = sum(v*row[3] for v, row in zip(f, group))/n if n else 0
        efficiency.append([v*(row[3]-mean_ep) if n >= 2 else 0 for v, row in zip(f, group)])
    return torch.tensor(safe, dtype=torch.float64), torch.tensor(efficiency, dtype=torch.float64)


def test_parts_follow_fixed_equations_and_no_unsafe_EP_credit():
    x = groups().requires_grad_()
    got = se.safety_efficiency_parts(x)
    expected_safe, expected_ep = independent_parts(x)
    torch.testing.assert_close(got['A_safe'], expected_safe, atol=1e-15, rtol=0)
    torch.testing.assert_close(got['A_EP'], expected_ep, atol=1e-15, rtol=0)
    torch.testing.assert_close(got['A_raw'].sum(1), torch.zeros(8, dtype=torch.float64), atol=1e-14, rtol=0)
    assert not any(v.requires_grad for v in got.values())
    assert bool((got['A_EP'][~got['feasible']] == 0).all())
    assert bool((got['A_raw'][0, 4:] < 0).all())  # unsafe high EP remains negative
    assert torch.equal(got['A_EP'][1], torch.zeros(8, dtype=torch.float64))
    assert torch.equal(got['A_raw'][2], torch.zeros(8, dtype=torch.float64))
    assert torch.equal(got['A_raw'][5], torch.zeros(8, dtype=torch.float64))
    assert got['A_raw'][3, 0] > 0 and got['A_raw'][3, 7] < 0
    assert not bool(got['feasible'][7, 1])


def test_one_full64_gather_raw_feedback_preserved_and_exact_zero_masks(monkeypatch):
    recs, rewards, material, calls = mock_gather(monkeypatch)
    safe, ep = independent_parts(groups())
    raw = safe+ep
    std = math.sqrt(sum(v*v for v in raw.reshape(-1).tolist())/64)
    snapshots = copy.deepcopy(recs)
    for rank in range(8):
        r = rollout(rank, recs[rank]); raw_id = id(r['rewards']); record_id = id(r['records'])
        se.finalize_safety_efficiency_advantages(r)
        assert len(calls) == rank+1 and torch.equal(calls[-1], material[rank:rank+1])
        assert id(r['rewards']) == raw_id and id(r['records']) == record_id
        assert torch.equal(r['rewards'], r['canonical_rewards'])
        assert r['canonical_rewards'].data_ptr() != r['rewards'].data_ptr()
        assert 'reward_transform' not in r
        assert abs(float(r['normalization_std'])-std) < 1e-15
        assert r['global_std'] == rewards.double().std(unbiased=False)
        torch.testing.assert_close(r['advantages'], (raw[rank]/(std+1e-8)).float(), atol=1e-7, rtol=1e-6)
        # Known exact ties must not be activated by scalar summation roundoff.
        assert r['adv_mask'].tolist() == [float(rank not in (2, 5, 6))]*8
        assert r['advantage_statistics']['normalization_population_candidates'] == 64
        assert r['structured_advantage']['protocol'] == se.STRUCTURED_PROTOCOL
        with pytest.raises(ValueError, match='exactly once'):
            se.finalize_safety_efficiency_advantages(r)
    assert recs == snapshots
    # Explicit inclusion of tied groups matters to the global scale.
    nonzero = raw[[0, 1, 3, 4, 7]]
    assert abs(float(nonzero.std(unbiased=False))-std) > .01


def test_all_zero_batch_is_finite_anchor_only_without_fabricated_direction():
    x = torch.ones(8, 8, 4, dtype=torch.float64)
    x[:, :, 3] = .3
    x[1::2, :, 0] = 0  # mix all-feasible EP ties and all-fail deficit ties
    rewards = torch.arange(8, dtype=torch.float64).expand(8, 8)/8
    material = torch.cat((rewards.unsqueeze(-1), x), -1)
    got = se.safety_efficiency_advantages(rewards, x, all_material=material)
    assert got['normalization_std'] == 0 and got['global_std'] > 0
    assert torch.equal(got['advantages'], torch.zeros(64))
    assert torch.equal(got['adv_mask'], torch.zeros(64))


def test_native_boundary_and_eight_rank_semantics_roundtrip_and_reject_corruption(monkeypatch, tmp_path):
    recs, rewards, material, calls = mock_gather(monkeypatch)
    summaries, rollouts = [], []
    for rank in range(8):
        r = se.finalize_safety_efficiency_advantages(rollout(rank, recs[rank]))
        before = tensor_digest(r)
        path = tmp_path/f'rank{rank}.pt'; torch.save(r, path)
        restored = torch.load(path, weights_only=False)
        assert tensor_digest(restored) == before
        # Every inner boundary retains the identical A/components/behavior.
        for update in range(4):
            se.validate_safety_efficiency_boundary(state(update), restored, identity(),
                optimizer=dict(state={0:dict(step=update)}, param_groups=[dict(params=[0])]),
                scheduler=dict(last_epoch=update))
        summaries.append(se.validate_safety_efficiency_boundary(state(), restored, identity()))
        assert tensor_digest(restored) == before
        rollouts.append(restored)
    assert len(calls) == 8
    se.validate_safety_efficiency_ranks(summaries)
    for key in ('A_safe', 'A_EP', 'A_raw', 'cost', 'components'):
        bad = copy.deepcopy(rollouts[0]); bad['structured_advantage'][key].reshape(-1)[0] += .01
        with pytest.raises(ValueError, match='material differs'):
            se.validate_safety_efficiency_boundary(state(), bad, identity())
    bad = copy.deepcopy(rollouts[0]); bad['canonical_rewards'][0, 0] += .01
    with pytest.raises(ValueError, match='canonical reward material'):
        se.validate_safety_efficiency_boundary(state(), bad, identity())
    for key in ('advantages', 'normalization_std', 'global_std'):
        bad = copy.deepcopy(summaries)
        if key == 'advantages':
            bad[0][key][0] += .1
        else:
            bad[0][key] += .1
        with pytest.raises(ValueError, match='globally finalized'):
            se.validate_safety_efficiency_ranks(bad)
    bad = copy.deepcopy(summaries); bad[2]['mask'] = [1.]*8
    with pytest.raises(ValueError, match='mask mismatch'):
        se.validate_safety_efficiency_ranks(bad)
    bad = copy.deepcopy(summaries); bad[1]['behavior']['adapter_sha256'] = 'changed'
    with pytest.raises(ValueError, match='collection-start'):
        se.validate_safety_efficiency_ranks(bad)
    bad = state(); bad['scene_cursor'] = 9
    with pytest.raises(ValueError, match='cursor'):
        se.validate_safety_efficiency_boundary(bad, rollouts[0], identity())
    with pytest.raises(ValueError, match='Adam step'):
        se.validate_safety_efficiency_boundary(state(1), rollouts[0], identity(),
            optimizer=dict(state={0:dict(step=0)}, param_groups=[dict(params=[0])]))
    with pytest.raises(ValueError, match='Scheduler'):
        se.validate_safety_efficiency_boundary(state(1), rollouts[0], identity(), scheduler=dict(last_epoch=0))
    # Nonpending native transactions retain structural checks and new identity.
    ready = dict(updates=0, collections=0, next_inner=0, phase='READY', scene_cursor=0,
                 raw_scene_exposures=0, new_candidates=0)
    se.validate_safety_efficiency_ranks([se.validate_safety_efficiency_boundary(ready, None, identity()) for _ in range(8)])


class ToyModel(torch.nn.Module):
    grpo_kl_beta = 50.
    grpo_denoising_discount = 1.
    grpo_use_bc_loss = False
    grpo_bc_coeff = .1
    grpo_kl_type = 'x'

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.linspace(-.001, .001, 8*3*8*3, dtype=torch.float64).reshape(8, 3, 8, 3))

    def action_chain_logprobs(self, *args, use_ref=False, return_terms=False, **kwargs):
        value = self.p*0 if use_ref else self.p
        logp = value.mean((-1, -2))
        terms = dict(mean=value, vel=value*2, std=torch.tensor([.05, .04, .03], dtype=torch.float64))
        return (logp, terms) if return_terms else logp

    def kl_weight_matrix(self, *, horizon, action_dim, device, dtype):
        return torch.tensor([.2, 1., 1.], device=device, dtype=dtype)[None, :].expand(horizon, action_dim)


def test_raw_ResearchLoss_consumes_same_A_four_inner_with_unchanged_anchor_and_cap(monkeypatch):
    recs, _, _, calls = mock_gather(monkeypatch)
    r = se.finalize_safety_efficiency_advantages(rollout(0, recs[0]))
    loss = ResearchLoss(config())
    assert loss.advantage_normalization is None and loss.adv_clip_max == 5
    model = ToyModel(); before = tensor_digest(r); a_id = id(r['advantages'])
    for _ in range(4):
        other = copy.deepcopy(model)
        got, diagnostics = loss._policy_loss(model, r)
        expected, original = SourceLoss(config())._policy_loss(other, r)
        got.backward(); expected.backward()
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
        torch.testing.assert_close(model.p.grad, other.p.grad, atol=0, rtol=0)
        torch.testing.assert_close(diagnostics['kl_loss'], original['kl_loss'], atol=0, rtol=0)
        assert bool(torch.isfinite(model.p.grad).all()) and bool(model.p.grad.abs().sum() > 0)
        assert tensor_digest(r) == before and id(r['advantages']) == a_id and len(calls) == 1
        model.p.grad = None
    tied = se.finalize_safety_efficiency_advantages(rollout(2, recs[2]))
    got, diagnostics = loss._policy_loss(model, tied)
    got.backward()
    assert diagnostics['policy_loss'] == 0 and diagnostics['kl_loss'] > 0
    assert bool(model.p.grad.abs().sum() > 0)  # anchor remains active


def test_protocol_and_feedback_fail_closed():
    se.validate_safety_efficiency_recipe(config())
    for key in ('advantage_normalization', 'reward_transform', 'anchor_ablation', 'memory_lr_ablation'):
        bad = config(); bad['research'][key] = 'another-experiment'
        with pytest.raises(ValueError, match='separate'):
            se.validate_safety_efficiency_recipe(bad)
    bad = config(); bad['grpo']['train']['kl_beta'] = 40
    with pytest.raises(ValueError, match='beta50'):
        se.validate_safety_efficiency_recipe(bad)
    rec = records(groups()[0]); rec[0]['status'] = 'FAIL'
    with pytest.raises(ValueError, match='Scoring failure'):
        se.canonical_components(rec)
    rec = records(groups()[0]); rec[0]['metrics']['ego_progress'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        se.canonical_components(rec)
    rec = records(groups()[0]); rec[0]['metrics']['score'] += .01
    with pytest.raises(ValueError, match='aggregation'):
        se.canonical_components(rec)
    rec = records(groups()[0]); rec[0]['readouts']['canonical_PDMS'] = float('nan')
    with pytest.raises(ValueError, match='readout'):
        se.canonical_components(rec)
    rec = records(groups()[0]); r = rollout(0, rec); r['rewards'] += .01
    with pytest.raises(ValueError, match='original canonical'):
        se.finalize_safety_efficiency_advantages(r)
