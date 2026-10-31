"""Directed CPU source-collection/positive-CFM/SafetyEP native replay checks."""
import copy
import json
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from test_qwen_world_lastdet_contract import TinyActor as NativeTinyActor, Service, descriptor, recipe as lastdet_recipe
from starVLA.rl.qwen_world_research_algorithms import rescue_auxiliary as r
from starVLA.rl.qwen_world_research_algorithms.rescue_entry import RescueRuntimeMixin
from starVLA.rl.qwen_world_research_algorithms.lastdet_entry import require_lastdet
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.qwen_world_research_algorithms.safety_efficiency import (
    canonical_components, finalize_safety_efficiency_advantages,
)
from starVLA.rl.simwam_recipe.contracts import registered_stochastic_steps_for_recipe
from starVLA.rl.simwam_recipe.recovery_state import rollout_digest


class TinyActor(NativeTinyActor):
    @contextmanager
    def _context_call(self, condition):
        # ScopeRecipeActor rebuilds trainable Memory here per separate call.
        material = dict(self.materialize(condition), mock_current_memory=self.p.square())
        self.scope_calls = getattr(self, 'scope_calls', 0)+1
        yield material

    def action_velocity(self, state, timestep, condition, use_ref=False):
        return super().action_velocity(state, timestep, condition, use_ref=use_ref)+condition.get('mock_current_memory', 0)


def make_bank(root, *, bad_endpoint=False, bad_teacher=False):
    root.mkdir()
    clean = np.linspace(-.7, .7, 2*8*3, dtype=np.float32).reshape(2, 8, 3)
    requested = r.ActionCodec.denorm_odo(torch.from_numpy(clean)).numpy().astype(np.float64)
    metrics = np.ones((2, 7), dtype=np.float64); metrics[0, 4] = .8
    if bad_teacher: metrics[0, 3] = 0.
    metrics[:, 0] = metrics[:, 1]*metrics[:, 2]*(5*metrics[:, 4]+5*metrics[:, 3]+2*metrics[:, 5])/12
    rows = []
    for i, kind in enumerate(('low', 'high')):
        source = dict(arm='PPO_SAFETY_EP', updates=10000, sampler='ODE', checkpoint='sealed-source')
        sha = r.array_digest(requested[i]); token = f'scene{i}'
        rows.append(dict(teacher_id=r.fingerprint(dict(token=token, bank_kind=kind, source=source, requested_trajectory_sha256=sha)),
            token=token, log=f'log{i}', normalized_index=i, bank_kind=kind,
            metrics=dict(zip(r.METRICS, metrics[i].tolist())), protocol_sha256='official',
            cache_sha256=str(i)*64, source_scene='/sealed/scene.json', source_scene_sha256='a'*64,
            source=source, requested_trajectory_sha256=sha, normalized_sha256=r.array_digest(clean[i]),
            offline_only=True, on_policy_GRPO_eligible=False, metric_verified=True, human_optimal=False,
            baseline_SAFETY10k_score=0., codec_roundtrip=dict(bound_by_dimension=[1e-4]*3, max_abs_by_dimension=[0.]*3)))
    if bad_endpoint: clean[0, 0, 0] += .1
    np.savez(root/'teachers.npz', clean_normalized=clean, requested_trajectory=requested, metrics=metrics,
        metric_keys=np.asarray(r.METRICS), tokens=np.asarray(['scene0', 'scene1']),
        logs=np.asarray(['log0', 'log1']), bank_kind=np.asarray(['low', 'high']))
    (root/'teachers.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest = dict(schema=r.BANK_SCHEMA, status='READY_OFFLINE_VERIFIED_TEACHERS', count=2,
        low_tokens=['scene0'], high_tokens=['scene1'], protocol_sha256='official', metric_keys=list(r.METRICS),
        offline_only=True, on_policy_GRPO_eligible=False, NAVTEST_tokens_or_logs_included=False,
        artifacts={name:dict(path=str(root/name), sha256=r.file_sha(root/name)) for name in ('teachers.npz', 'teachers.jsonl')})
    manifest['sha256'] = r.fingerprint(manifest)
    path = root/'manifest.json'; path.write_text(json.dumps(manifest)); return path


@pytest.fixture
def bank(tmp_path):
    path = make_bank(tmp_path/'bank')
    return r.RescueTeacherBank(path, r.file_sha(path), ['scene0', 'scene1'])


def recipe(bank, aux=True):
    cfg = lastdet_recipe()
    cfg['research'].update(arm='PPO_RESCUE_CFM' if aux else 'PPO_RESCUE_REPLAY', data_policy='rescue_mixture',
        rescue_distribution=r.DISTRIBUTION, rescue_manifest='sealed-manifest', rescue_manifest_sha256=bank.sha256,
        structured_advantage=r.STRUCTURED_SCHEMA,
        warmstart=dict(schema='world_verified_RL_weights_only_warmstart_v1'))
    if aux: cfg['research']['rescue_auxiliary'] = copy.deepcopy(r.AUX_CONFIG)
    return cfg


class Feedback(Service):
    def score(self, value):
        rows = super().score(value)
        for i, row in enumerate(rows):
            m = row['metrics']
            row['protocol_sha256'] = 'official'
            row['cache'] = dict(path='/sealed/cache.pkl', sha256=str(self.rank)*64)
            m['driving_direction_compliance'] = 1.
            if self.rank == 2: m['ego_progress'] = 1.
            if self.rank == 4 and i == 0: m['time_to_collision_within_bound'] = 0.
            if self.rank == 5: m['no_at_fault_collisions'] = 0.
            m['score'] = m['no_at_fault_collisions']*m['drivable_area_compliance']*(5*m['ego_progress']+5*m['time_to_collision_within_bound']+2*m['comfort'])/12
            row['readouts']['canonical_PDMS'] = m['score']
        return rows


def collect(bank, rank=0, aux=True):
    cfg = recipe(bank, aux); actor = TinyActor(cfg)
    class Runtime(RescueRuntimeMixin): pass
    rt = Runtime(); rt.cfg = cfg
    out = rt.collect(actor, descriptor(rank), f'scene{rank}', f'log{rank}', Feedback(rank),
        torch.Generator().manual_seed(42+rank), variant='R_CANONICAL',
        version=dict(run='run', update=0, adapter_sha256='same'))
    return actor, out


def test_exact_two_roles_keep_old_entry_isolated_and_reject_mixed_recipes(bank):
    for aux in (True, False):
        cfg = recipe(bank, aux)
        assert r.validate_rescue_recipe(cfg) is cfg and registered_stochastic_steps_for_recipe(cfg) == [7, 8]
        with pytest.raises(ValueError, match='only'): require_lastdet(cfg)
    assert registered_stochastic_steps_for_recipe(lastdet_recipe()) == [7, 8]
    for key, value in [('arm', 'PPO_LASTDET'), ('data_policy', 'uniform_full'),
            ('structured_advantage', None), ('rescue_auxiliary', None), ('noise_level_ablation', 'eta003'),
            ('warmstart', {})]:
        bad = recipe(bank); bad['research'][key] = value
        with pytest.raises(ValueError): r.validate_rescue_recipe(bad)
    with pytest.raises(ValueError, match='exclusively'): r.RescueCFMLoss(recipe(bank, False))


def test_bank_requires_scored_endpoint_metric_witness_SHA_and_TRAIN(bank, tmp_path):
    assert len(bank.by_token) == 2
    path = make_bank(tmp_path/'valid')
    with pytest.raises(ValueError, match='SHA'): r.RescueTeacherBank(path, '0'*64, ['scene0', 'scene1'])
    with pytest.raises(ValueError, match='TRAIN'): r.RescueTeacherBank(path, r.file_sha(path), ['scene0'])
    path = make_bank(tmp_path/'bad_teacher', bad_teacher=True)
    with pytest.raises(ValueError, match='witness'): r.RescueTeacherBank(path, r.file_sha(path), ['scene0', 'scene1'])
    path = make_bank(tmp_path/'bad_endpoint', bad_endpoint=True)
    with pytest.raises(ValueError, match='identity changed'): r.RescueTeacherBank(path, r.file_sha(path), ['scene0', 'scene1'])


def test_positive_CFM_gradient_points_towards_detached_target():
    velocity = torch.ones(8, 8, 3, requires_grad=True)
    target = torch.zeros_like(velocity, requires_grad=True)
    loss = r.positive_cfm_objective(velocity, target); loss.backward()
    assert torch.all(velocity.grad > 0) and target.grad is None
    assert loss == 1 and torch.isfinite(velocity.grad).all()


def test_actual_W2_four_inner_positive_CFM_preserves_SafetyEP_PG_anchor_and_RNG(bank, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda:None)
    actor, rollout = collect(bank)
    finalize_safety_efficiency_advantages(rollout)
    original = copy.deepcopy(rollout)
    before = torch.random.get_rng_state().clone()
    generator = torch.Generator().manual_seed(42); sde = generator.get_state().clone()
    calls = len(actor.velocity_steps)
    r.attach_rescue_material(actor, rollout, bank, enabled=True, rank=0)
    assert len(actor.velocity_steps) == calls  # no additional model/ref forward in finalizer
    assert torch.equal(before, torch.random.get_rng_state()) and torch.equal(sde, generator.get_state())
    p = rollout['rescue_auxiliary']; saved_digest = p['material_sha256']
    assert p['sigma'].shape == (8,) and p['noise'].shape == (8, 8, 3) and p['coefficient'] == 1.
    assert bool(((p['sigma'] >= .2) & (p['sigma'] <= .8)).all())
    state, sigma, target = r.forward_material(p['clean'], p['sigma'], p['noise'])
    torch.testing.assert_close(target, p['noise']-p['clean'], atol=0, rtol=0)
    a_id = id(rollout['advantages']); actor._loss = r.RescueCFMLoss(recipe(bank))
    for _ in range(4):
        control = copy.deepcopy(actor)
        expected, source_metrics = ResearchLoss(recipe(bank))._policy_loss(control, original)
        total, metrics = actor._loss._policy_loss(actor, rollout)
        torch.testing.assert_close(total-expected, metrics['rescue_aux_weighted_loss'], atol=1e-7, rtol=1e-6)
        for k in ('policy_loss', 'kl_loss', 'ratio_mean', 'clipfrac'):
            torch.testing.assert_close(metrics[k], source_metrics[k], atol=0, rtol=0)
        total.backward(); assert torch.isfinite(actor.p.grad) and actor.p.grad.abs() > 0
        assert r.payload_digest(p) == saved_digest and id(rollout['advantages']) == a_id
        assert torch.equal(rollout['advantages'], original['advantages'])
        assert torch.equal(rollout['rewards'], original['rewards']) and rollout['old_logp'].shape == (8, 2)
        assert 'step9_ratio_mean' not in metrics and metrics['rescue_aux_enabled'] == 1
        actor.p.grad = None
    assert actor.scope_calls == 4  # no previous inner's Memory graph retained
    with pytest.raises(ValueError, match='exactly once'): r.attach_rescue_material(actor, rollout, bank, enabled=True, rank=0)


@pytest.mark.parametrize('rank,aux,reason', [(2, True, 'fresh_safety_comfort_EP_saturated'), (3, True, 'no_teacher'), (0, False, 'replay_control')])
def test_skip_and_replay_have_no_extra_forward_and_exact_source_gradient(bank, monkeypatch, rank, aux, reason):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda:None)
    actor, rollout = collect(bank, rank, aux); finalize_safety_efficiency_advantages(rollout)
    calls = len(actor.velocity_steps); r.attach_rescue_material(actor, rollout, bank, enabled=aux, rank=rank)
    assert len(actor.velocity_steps) == calls and rollout['rescue_auxiliary']['skip_reason'] == reason
    assert 'noise' not in rollout['rescue_auxiliary']
    other = copy.deepcopy(actor); control, _ = ResearchLoss(recipe(bank, aux))._policy_loss(other, rollout)
    loss = r.RescueCFMLoss(recipe(bank))._policy_loss(actor, rollout)[0] if aux else ResearchLoss(recipe(bank, False))._policy_loss(actor, rollout)[0]
    loss.backward(); control.backward()
    torch.testing.assert_close(loss, control, atol=0, rtol=0)
    torch.testing.assert_close(actor.p.grad, other.p.grad, atol=0, rtol=0)


def test_native_eight_ranks_W2_SafetyEP_four_inner_roundtrip_and_material_corruption(bank, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda:None)
    rs = [collect(bank, rank)[1] for rank in range(8)]
    material = torch.cat([torch.cat((out['rewards'].double().unsqueeze(-1), canonical_components(out['records'])['components']), -1) for out in rs])
    import starVLA.rl.qwen_world_research_algorithms.safety_efficiency as se
    monkeypatch.setattr(se.dist, 'is_available', lambda:True); monkeypatch.setattr(se.dist, 'is_initialized', lambda:True)
    monkeypatch.setattr(se.dist, 'get_world_size', lambda:8)
    def gather(out, local):
        for rank in range(8): out[rank].copy_(material[rank:rank+1])
    monkeypatch.setattr(se.dist, 'all_gather', gather)
    identity = dict(world=8, sha256='run', schema='world_postcot_cleanx_scope_RL_v1', recipe=recipe(bank), projected_condition_identity='condition')
    state = dict(updates=0, collections=1, next_inner=0, phase='ROLLOUT_COMMITTED', scene_cursor=8, raw_scene_exposures=8, new_candidates=64)
    summaries = []
    for rank, rollout in enumerate(rs):
        finalize_safety_efficiency_advantages(rollout)
        r.attach_rescue_material(TinyActor(recipe(bank)), rollout, bank, enabled=True, rank=rank)
        path = tmp_path/f'rank{rank}.pt'; torch.save(rollout, path); saved = torch.load(path, weights_only=False)
        assert rollout_digest(saved) == rollout_digest(rollout)
        for inner in range(4):
            s = dict(state, updates=inner, next_inner=inner, phase='INNER_COMMITTED' if inner else 'ROLLOUT_COMMITTED')
            r.validate_rescue_boundary(s, saved, identity, bank=bank, rank=rank,
                optimizer=dict(state={0:dict(step=inner)}, param_groups=[dict(params=[0])]), scheduler=dict(last_epoch=inner))
        summaries.append(r.validate_rescue_boundary(state, saved, identity, bank=bank, rank=rank))
    r.validate_rescue_ranks(summaries)
    assert rs[0]['rescue_auxiliary']['coefficient'] == 1 and rs[1]['rescue_auxiliary']['coefficient'] == .2
    assert summaries[2]['rescue_skip'] == 'fresh_safety_comfort_EP_saturated'
    assert not bool(rs[5]['adv_mask'].any())  # tied all-fail: no invented PG direction
    for key in ('clean', 'sigma', 'noise'):
        bad = copy.deepcopy(rs[0]); bad['rescue_auxiliary'][key].reshape(-1)[0] += .01
        with pytest.raises(ValueError, match='digest'): r.validate_rescue_boundary(state, bad, identity, bank=bank, rank=0)
        bad['rescue_auxiliary']['material_sha256'] = r.payload_digest(bad['rescue_auxiliary'])
        with pytest.raises(ValueError, match='randomness'): r.validate_rescue_boundary(state, bad, identity, bank=bank, rank=0)
    bad = copy.deepcopy(rs[0]); bad['rescue_auxiliary']['teacher']['source']['updates'] += 1
    bad['rescue_auxiliary']['material_sha256'] = r.payload_digest(bad['rescue_auxiliary'])
    with pytest.raises(ValueError, match='identity changed'): r.validate_rescue_boundary(state, bad, identity, bank=bank, rank=0)
    bad = copy.deepcopy(summaries); bad[0]['advantages'][0] += .1
    with pytest.raises(ValueError, match='globally finalized'): r.validate_rescue_ranks(bad)


def test_entry_once_finalizer_keeps_structured_credit_and_no_historical_A(bank, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda:None)
    class Runtime(RescueRuntimeMixin): pass
    runtime = Runtime(); runtime.cfg = recipe(bank); runtime.actor, out = collect(bank); runtime.rescue_bank = bank
    runtime.finalize_advantages(out)
    assert out['structured_advantage']['protocol'] == r.STRUCTURED_PROTOCOL and out['rescue_auxiliary']['enabled']
    assert 'advantages' not in out['rescue_auxiliary'] and out['rescue_auxiliary']['teacher']['on_policy_GRPO_eligible'] is False
    with pytest.raises(ValueError, match='once per collection'): runtime.finalize_advantages(out)


def test_fresh_verified_upgrade_never_uses_unsafe_or_worse_targets_and_replays_exactly(bank, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda:None)
    actor, out = collect(bank)
    # An unsafe EP=1 candidate must not outrank the best strict4 candidate.
    for idx, ep, ttc in ((5, .95, 1.), (7, 1., .99)):
        m = out['records'][idx]['metrics']; m.update(ego_progress=ep, time_to_collision_within_bound=ttc)
        m['score'] = (5*ep+5*ttc+2)/12
        out['records'][idx]['readouts']['canonical_PDMS'] = m['score']
    out['rewards'] = canonical_components(out['records'])['canonical_rewards']
    finalize_safety_efficiency_advantages(out)
    # Explicit eight identical rank math fixture: same local A and scale as
    # singleton CPU math, actual world64 count for boundary validation.
    out['advantage_statistics'].update(normalization_population_candidates=64, normalization_population_groups=8)
    before_A = out['advantages'].clone(); before_rewards = out['rewards'].clone()
    calls = len(actor.velocity_steps); r.attach_rescue_material(actor, out, bank, enabled=True, rank=0)
    p = out['rescue_auxiliary']; selected = p['target']
    assert len(actor.velocity_steps) == calls and selected['origin'] == 'online_upgrade'
    assert selected['candidate_index'] == 5 and selected['behavior_version'] == out['behavior_version']
    assert selected['cache']['sha256'] == bank.by_token['scene0']['cache_sha256']
    assert selected['metrics']['score'] > p['teacher']['metrics']['score']+1e-6
    assert torch.equal(p['clean'], out['chain'][5, -1]) and p['coefficient'] == 1.
    assert torch.equal(out['advantages'], before_A) and torch.equal(out['rewards'], before_rewards)
    identity = dict(world=8, sha256='run', schema='world_postcot_cleanx_scope_RL_v1', recipe=recipe(bank), projected_condition_identity='condition')
    state = dict(updates=0, collections=1, next_inner=0, phase='ROLLOUT_COMMITTED', scene_cursor=8, raw_scene_exposures=8, new_candidates=64)
    loss = r.RescueCFMLoss(recipe(bank)); digest = p['material_sha256']
    for inner in range(4):
        r.validate_rescue_boundary(dict(state, updates=inner, next_inner=inner, phase='INNER_COMMITTED' if inner else 'ROLLOUT_COMMITTED'), out, identity, bank=bank, rank=0)
        total, metrics = loss._policy_loss(actor, out); total.backward(); actor.p.grad = None
        assert metrics['rescue_aux_online_upgrade'] == 1 and metrics['rescue_aux_teacher_EP'] == pytest.approx(.95)
        assert r.payload_digest(p) == digest  # chosen before updates, not reselected each inner
    # Recomputed payload hashes cannot authorize changing candidate identity or
    # substituting an unscored endpoint. Restore rederives it from chain/records.
    bad = copy.deepcopy(out); bad['rescue_auxiliary']['target']['candidate_index'] = 7
    bad['rescue_auxiliary']['material_sha256'] = r.payload_digest(bad['rescue_auxiliary'])
    with pytest.raises(ValueError, match='identity changed'): r.validate_rescue_boundary(state, bad, identity, bank=bank, rank=0)
    bad = copy.deepcopy(out); bad['chain'][5, -1, 0, 0] += .1
    with pytest.raises(ValueError, match='actually scored'): r.validate_rescue_boundary(state, bad, identity, bank=bank, rank=0)
    # Existing worse fresh rollout retains the stronger static teacher; high
    # pool likewise never downgrades from its score=1 static teacher.
    for rank in (0, 1):
        _, other = collect(bank, rank)
        clean, target = r.select_verified_target(other, bank)
        assert target['origin'] == 'static_teacher'
        assert torch.equal(clean, bank.clean_teacher(f'scene{rank}'))
