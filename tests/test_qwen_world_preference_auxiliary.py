"""Directed CPU evidence for independent canonical-PPO preference auxiliary."""
import copy
import json
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from starVLA.rl.qwen_world_research_algorithms import preference_auxiliary as p
from starVLA.rl.qwen_world_research_algorithms.preference_entry import PreferenceRuntimeMixin
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe.collection import finalize_advantages


def make_bank(root, *, bad_label=False, bad_endpoint=False):
    root.mkdir(parents=True, exist_ok=True)
    clean = np.linspace(-.8, .8, 2*8*8*3, dtype=np.float32).reshape(2, 8, 8, 3)
    requested = p.ActionCodec.denorm_odo(torch.from_numpy(clean)).numpy().astype(np.float64)
    metrics = np.ones((2, 8, 7), dtype=np.float64)
    metrics[:, :, 4] = np.linspace(.1, .8, 8)
    metrics[0, 1, 3] = 0  # failed TTC; candidate0/2 have feasible efficiency gap
    metrics[1, :, 1] = 0; metrics[1, 1, 2] = 0
    metrics[:, :, 0] = metrics[:, :, 1]*metrics[:, :, 2]*(5*metrics[:, :, 4]+5*metrics[:, :, 3]+2*metrics[:, :, 5])/12
    # candidate7 > candidate0 EP and raw meters, both pass
    progress = np.broadcast_to(np.arange(8, dtype=np.float64), (2, 8)).copy()
    if bad_endpoint:
        clean[0, 0, 0, 0] += .1
    np.savez_compressed(root/'candidates.npz', metrics=metrics, clean_normalized=clean,
        requested_trajectory=requested, raw_progress_meters=progress,
        tokens=np.asarray(['scene0', 'scene1']), logs=np.asarray(['log0', 'log1']), metric_keys=np.asarray(p.METRICS))
    groups = [dict(group=i, token=f'scene{i}', log=f'log{i}', cache_sha256='cache'+str(i),
        protocol_sha256='official-protocol', behavior_version=dict(update=0, run='old-run')) for i in range(2)]
    pairs = []
    for group, wi, li, reason in [(0, 0, 1, 'pass_over_fail'), (0, 7, 0, 'feasible_efficiency'), (1, 0, 1, 'allfail_deficit')]:
        w = dict(group=group, candidate=wi, trajectory_sha256=p.array_digest(requested[group, wi]))
        l = dict(group=group, candidate=li, trajectory_sha256=p.array_digest(requested[group, li]))
        if bad_label and reason == 'feasible_efficiency':
            reason = 'pass_over_fail'
        pairs.append(dict(pair_id=p.fingerprint(dict(token=f'scene{group}', winner=w['trajectory_sha256'],
            loser=l['trajectory_sha256'], reason=reason)), token=f'scene{group}', log=f'log{group}',
            winner=w, loser=l, reason=reason, origin='same_collection',
            winner_metrics=dict(zip(p.METRICS, metrics[group, wi].tolist())),
            loser_metrics=dict(zip(p.METRICS, metrics[group, li].tolist())),
            offline_only=True, on_policy_GRPO_eligible=False))
    for name, rows in [('groups.jsonl', groups), ('pairs.jsonl', pairs)]:
        (root/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest = dict(schema=p.BANK_SCHEMA, status='READY_OFFLINE_RESEARCH_ONLY', groups=2, candidates=16,
        pairs=3, unique_scenes=2, protocol_sha256='official-protocol',
        artifacts={name:dict(path=str(root/name), sha256=p.file_sha(root/name)) for name in
            ('candidates.npz', 'groups.jsonl', 'pairs.jsonl')})
    manifest['sha256'] = p.fingerprint(manifest)
    path = root/'manifest.json'; path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def bank(tmp_path):
    path = make_bank(tmp_path/'bank')
    return p.PreferenceBank(path, p.file_sha(path), ['scene0', 'scene1', 'sceneX'])


def cfg(bank):
    return dict(max_steps=50000, seed=42, action_shape=[8, 3], model_family='QwenWorldPostCoT',
        prediction_type='x0', adaptation='ATTN_LORA_MEMORY', variant='R_CANONICAL',
        learning_rate=5e-5, memory_learning_rate=5e-6,
        grpo=dict(train=dict(kl_beta=50, ppo_clip_range=.005, adv_clip_max=5., drop_zero_adv=True,
            adv_eps=1e-8, adv_clip_lower_quantile=0., adv_clip_upper_quantile=1.,
            adv_global_std=True, use_bc_loss=False)),
        research=dict(policy_objective='ppo', data_policy='preference_mixture',
            preference_distribution=p.DISTRIBUTION, preference_manifest='registered-manifest',
            preference_manifest_sha256=bank.sha256, preference_auxiliary=copy.deepcopy(p.AUX_CONFIG)))


def records(bank, token='scene0', saturated=False):
    group = 1 if token == 'scene1' else 0
    out = []
    for m in bank.arrays['metrics'][group]:
        m = dict(zip(p.METRICS, m.tolist()))
        if saturated:
            m.update(no_at_fault_collisions=1., drivable_area_compliance=1.,
                     time_to_collision_within_bound=1., ego_progress=1.)
            m['score'] = (10+2*m['comfort'])/12
        out.append(dict(status='OK', token=token, log='log0', metrics=m,
                        readouts=dict(canonical_PDMS=m['score'])))
    return out


def rollout(bank, token='scene0', saturated=False):
    rec = records(bank, token, saturated)
    return dict(records=rec, token=token, log='log0',
        rewards=torch.tensor([[r['metrics']['score'] for r in rec]], dtype=torch.float32),
        chain=torch.zeros(8, 11, 8, 3), old_logp=torch.zeros(8, 3), timesteps=torch.arange(10),
        deltas=-torch.ones(10)/10, stoch_idx=torch.tensor([7, 8, 9]),
        old_probability={'source_clamped_logprob':torch.zeros(8, 3, 8, 3)},
        protocol_sha256='official-protocol', behavior_version=dict(run='new-arm', update=0, adapter_sha256='fixed'),
        cond_bg=dict(schema='world_postcot_trainable_memory_observation_v1', token=token,
                     identity='fixed-condition', repeats=8))


class ToyActor(torch.nn.Module):
    grpo_kl_beta = 50.
    grpo_denoising_discount = 1.
    grpo_use_bc_loss = False
    grpo_bc_coeff = .1
    grpo_kl_type = 'x'

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.linspace(-.001, .001, 8*3*8*3, dtype=torch.float64).reshape(8, 3, 8, 3))
        self.device = torch.device('cpu'); self.calls = []; self._prepared_scope = None
        self._probability_audit = {}

    @contextmanager
    def _context_call(self, condition):
        previous = self._prepared_scope; self._prepared_scope = {}
        try:
            assert condition['repeats'] == 8
            yield condition
        finally:
            self._prepared_scope = previous

    def action_velocity(self, state, timestep, condition, use_ref=False):
        assert state.shape == (8, 8, 3) and timestep.shape == (8,)
        assert self._prepared_scope is not None
        self.calls.append(bool(use_ref))
        current = self.p.mean(1).float()
        return current*0 if use_ref else current

    def action_chain_logprobs(self, *args, use_ref=False, return_terms=False, **kwargs):
        value = self.p*0 if use_ref else self.p
        logp = value.mean((-1, -2))
        self._probability_audit['reference' if use_ref else 'current'] = dict(source_clamped_logprob=value.detach())
        terms = dict(mean=value, vel=value*2, std=torch.tensor([.05, .04, .03], dtype=torch.float64))
        return (logp, terms) if return_terms else logp

    def kl_weight_matrix(self, *, horizon, action_dim, device, dtype):
        return torch.tensor([.2, 1., 1.], device=device, dtype=dtype)[None, :].expand(horizon, action_dim)


def test_bank_requires_real_bound_endpoints_metrics_labels_hashes_and_TRAIN(tmp_path):
    path = make_bank(tmp_path/'good')
    b = p.PreferenceBank(path, p.file_sha(path), ['scene0', 'scene1'])
    assert len(b.pairs) == 3 and b.clean_pair(next(iter(b.pairs.values()))).shape == (2, 8, 3)
    with pytest.raises(ValueError, match='SHA'):
        p.PreferenceBank(path, '0'*64, ['scene0', 'scene1'])
    with pytest.raises(ValueError, match='TRAIN'):
        p.PreferenceBank(path, p.file_sha(path), ['scene0'])
    path = make_bank(tmp_path/'bad-label', bad_label=True)
    with pytest.raises(ValueError, match='metric witness'):
        p.PreferenceBank(path, p.file_sha(path), ['scene0', 'scene1'])
    path = make_bank(tmp_path/'bad-endpoint', bad_endpoint=True)
    with pytest.raises(ValueError, match='actually scored'):
        p.PreferenceBank(path, p.file_sha(path), ['scene0', 'scene1'])


def test_stateless_pair_and_four_common_draws_do_not_consume_SDE_or_global_RNG(bank):
    actor = ToyActor(); r = rollout(bank)
    wrong = rollout(bank); wrong['protocol_sha256'] = 'other-scorer'
    with pytest.raises(ValueError, match='protocol differs'):
        p.attach_preference_material(actor, wrong, bank, rank=2)
    original_rng = torch.random.get_rng_state().clone()
    sde = torch.Generator().manual_seed(42); original_sde = sde.get_state().clone()
    p.attach_preference_material(actor, r, bank, rank=2)
    material = r['preference_auxiliary']
    assert torch.equal(original_rng, torch.random.get_rng_state()) and torch.equal(original_sde, sde.get_state())
    other = rollout(bank); p.attach_preference_material(actor, other, bank, rank=2)
    assert material['material_sha256'] == other['preference_auxiliary']['material_sha256']
    assert material['sigma'].shape == (4,) and material['noise'].shape == (4, 8, 3)
    assert bool(((material['sigma'] >= .2) & (material['sigma'] <= .8)).all())
    state, sigma, target = p.forward_material(material['clean'], material['sigma'], material['noise'])
    assert state.shape == target.shape == (8, 8, 3) and torch.equal(sigma[:4], sigma[4:])
    expected_target = torch.cat((material['noise']-material['clean'][0], material['noise']-material['clean'][1]))
    torch.testing.assert_close(target, expected_target, atol=0, rtol=0)
    assert actor.calls == [True, True] and actor._prepared_scope is None
    with pytest.raises(ValueError, match='exactly once'):
        p.attach_preference_material(actor, r, bank, rank=2)


def test_reference_relative_CFM_gradient_sign_and_frozen_reference():
    velocity = torch.ones(8, 8, 3, requires_grad=True)
    reference = torch.ones(8, requires_grad=True)
    loss, _ = p.preference_objective(velocity, torch.zeros_like(velocity), reference)
    loss.backward()
    assert bool((velocity.grad[:4] > 0).all()) and bool((velocity.grad[4:] < 0).all())
    assert reference.grad is None and abs(float(loss)-np.log(2)) < 1e-7
    assert bool(torch.isfinite(velocity.grad).all())


def test_four_inner_reuse_cached_reference_preserve_PG_anchor_and_probability_audit(bank):
    actor = ToyActor(); r = rollout(bank); finalize_advantages(r)
    original = copy.deepcopy(r); p.attach_preference_material(actor, r, bank, rank=0)
    digest = p.payload_digest(r['preference_auxiliary']); a_id = id(r['advantages'])
    auxiliary = p.PreferenceAuxiliaryLoss(cfg(bank))
    for _ in range(4):
        control = copy.deepcopy(actor)
        expected, source_metrics = ResearchLoss(cfg(bank))._policy_loss(control, original)
        got, metrics = auxiliary._policy_loss(actor, r)
        # The explicit total delta is exactly the registered coefficient times
        # auxiliary. Source PG and anchor diagnostics are bit-identical.
        torch.testing.assert_close(got-expected, (.001*metrics['pref_aux_loss']).to(got), atol=1e-12, rtol=1e-6)
        for key in ('policy_loss', 'kl_loss', 'ratio_mean', 'clipfrac'):
            torch.testing.assert_close(metrics[key], source_metrics[key], atol=0, rtol=0)
        got.backward()
        assert bool(torch.isfinite(actor.p.grad).all()) and bool(actor.p.grad.abs().sum() > 0)
        assert p.payload_digest(r['preference_auxiliary']) == digest and id(r['advantages']) == a_id
        assert torch.equal(r['advantages'], original['advantages']) and torch.equal(r['rewards'], original['rewards'])
        assert torch.equal(actor._probability_audit['current']['source_clamped_logprob'], control._probability_audit['current']['source_clamped_logprob'])
        actor.p.grad = None
    assert actor.calls == [True, False, False, False, False] and actor._prepared_scope is None


@pytest.mark.parametrize('token,saturated,reason', [('scene0', True, 'fresh_safety_EP_saturated'), ('sceneX', False, 'no_bank_pair')])
def test_skip_is_exact_canonical_PPO_without_extra_forward(bank, token, saturated, reason):
    actor = ToyActor(); r = rollout(bank, token, saturated); finalize_advantages(r)
    p.attach_preference_material(actor, r, bank, rank=0)
    assert not r['preference_auxiliary']['enabled'] and r['preference_auxiliary']['skip_reason'] == reason
    assert 'noise' not in r['preference_auxiliary'] and actor.calls == []
    other = copy.deepcopy(actor)
    total, metrics = p.PreferenceAuxiliaryLoss(cfg(bank))._policy_loss(actor, r)
    expected, _ = ResearchLoss(cfg(bank))._policy_loss(other, r)
    total.backward(); expected.backward()
    torch.testing.assert_close(total, expected, atol=0, rtol=0)
    torch.testing.assert_close(actor.p.grad, other.p.grad, atol=0, rtol=0)
    assert metrics['pref_aux_loss'] == 0 and actor.calls == []


def test_native_CPU_roundtrip_eight_rank_source_A_validation_and_corruption_rejection(bank, monkeypatch, tmp_path):
    import starVLA.rl.simwam_recipe.source_advantages as source
    rs = [rollout(bank, token='sceneX' if rank == 6 else 'scene0', saturated=rank == 7) for rank in range(8)]
    all_rewards = torch.cat([r['rewards'].double() for r in rs])
    monkeypatch.setattr(source.dist, 'is_available', lambda:True)
    monkeypatch.setattr(source.dist, 'is_initialized', lambda:True)
    monkeypatch.setattr(source.dist, 'get_world_size', lambda:8)
    def gather(out, local):
        for rank in range(8): out[rank].copy_(all_rewards[rank:rank+1])
    monkeypatch.setattr(source.dist, 'all_gather', gather)
    identity = dict(world=8, sha256='new-arm', schema='world_postcot_cleanx_scope_RL_v1',
                    projected_condition_identity='fixed-condition', recipe=cfg(bank))
    state = dict(updates=0, collections=1, next_inner=0, phase='ROLLOUT_COMMITTED',
                 scene_cursor=8, raw_scene_exposures=8, new_candidates=64)
    summaries = []
    for rank, r in enumerate(rs):
        finalize_advantages(r); p.attach_preference_material(ToyActor(), r, bank, rank=rank)
        path = tmp_path/f'rank{rank}.pt'; torch.save(r, path); r = torch.load(path, weights_only=False); rs[rank] = r
        for inner in range(4):
            s = dict(state, updates=inner, next_inner=inner, phase='INNER_COMMITTED' if inner else 'ROLLOUT_COMMITTED')
            p.validate_preference_boundary(s, r, identity, bank=bank, rank=rank,
                optimizer=dict(state={0:dict(step=inner)}, param_groups=[dict(params=[0])]), scheduler=dict(last_epoch=inner))
        summaries.append(p.validate_preference_boundary(state, r, identity, bank=bank, rank=rank))
    p.validate_preference_ranks(summaries)
    for field in ('clean', 'sigma', 'noise', 'reference_velocity', 'reference_energy'):
        bad = copy.deepcopy(rs[0]); bad['preference_auxiliary'][field].reshape(-1)[0] += .01
        with pytest.raises(ValueError, match='digest'):
            p.validate_preference_boundary(state, bad, identity, bank=bank, rank=0)
    # Even with a recomputed payload digest, the stateless randomness and saved
    # reference velocity-energy relation remain independently reconstructable.
    bad = copy.deepcopy(rs[0]); bad['preference_auxiliary']['noise'][0, 0, 0] += .01
    bad['preference_auxiliary']['material_sha256'] = p.payload_digest(bad['preference_auxiliary'])
    with pytest.raises(ValueError, match='randomness'):
        p.validate_preference_boundary(state, bad, identity, bank=bank, rank=0)
    bad = copy.deepcopy(rs[0]); bad['preference_auxiliary']['reference_energy'][0] += .1
    bad['preference_auxiliary']['material_sha256'] = p.payload_digest(bad['preference_auxiliary'])
    with pytest.raises(ValueError, match='energy audit'):
        p.validate_preference_boundary(state, bad, identity, bank=bank, rank=0)
    bad = copy.deepcopy(summaries); bad[0]['advantages'][0] += .1
    with pytest.raises(ValueError, match='Advantage was not globally'):
        p.validate_preference_ranks(bad)
    bad = copy.deepcopy(rs[7]); bad['preference_auxiliary']['skip_reason'] = 'no_bank_pair'
    bad['preference_auxiliary']['material_sha256'] = p.payload_digest(bad['preference_auxiliary'])
    with pytest.raises(ValueError, match='skip identity'):
        p.validate_preference_boundary(state, bad, identity, bank=bank, rank=7)


def test_entry_finalizes_canonical_before_aux_and_rejects_recipe_mixing(bank):
    class Runtime(PreferenceRuntimeMixin):
        pass
    runtime = Runtime(); runtime.actor = ToyActor(); runtime.preference_bank = bank
    r = rollout(bank); runtime.finalize_advantages(r)
    assert 'advantages' in r and r['preference_auxiliary']['enabled']
    with pytest.raises(ValueError, match='once per collection'):
        runtime.finalize_advantages(r)
    for key in ('structured_advantage', 'reward_transform', 'advantage_normalization', 'memory_lr_ablation'):
        bad = cfg(bank); bad['research'][key] = 'another-arm'
        with pytest.raises(ValueError, match='independent'):
            p.validate_preference_recipe(bad)
    bad = cfg(bank); bad['research']['preference_auxiliary']['coefficient'] = .01
    with pytest.raises(ValueError, match='independent'):
        p.PreferenceAuxiliaryLoss(bad)
