"""Flow runner contracts/fault tests; synthetic fixtures are not real S0 evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from starVLA.model.modules.flow_posttraining import (
    FPOConfig, FlowRandomStreams, FrozenPolicySnapshot, group_centered_advantages,
    prepare_fpo_batch, sample_rollout_group, module_hash)
from starVLA.model.modules.st_world.consequence import METRICS, fingerprint
from tools.st_world_policy.flow_training_config import validate_config, target_contract
from tools.st_world_policy.training_config import PRECISION
from tools.st_world_policy.train_flow_policy import (
    FlowTrainingModule, MatchedOldMicrobatches, OfficialFeedbackLedger, encode_conditions,
    feedback_from_records, score_requests, numeric_execution, save_update_feedback)
from test_fpo import original_tiny_head, context


def config(cell="C1"):
    return {"schema": "st_flow_training_v1", "cell": cell,
        "execution": {"scope": "SMOKE", "global_batch": 2, "micro_batch": 1, "world_size": 1,
                      "max_updates": 4096, "stop_after": 1, "sample_limit": 2,
                      "precision_policy": PRECISION, "family": "consequence", "seed": 42},
        "fpo": FPOConfig().to_dict(), "loss": {"bc_weight": 1., "policy_weight": 0. if cell == "C0" else 1.},
        "optimizer": {"lr": 1e-5, "minimum_lr": 1e-6, "warmup": 100, "betas": [.9, .999],
                      "eps": 1e-8, "weight_decay": .01, "clip": 1.},
        "runtime": {"output_root": "/fixture/run", "loader_workers": 0, "save_every": 1,
                    "max_seconds": 1800., "deterministic": True, "gradient_checkpointing": False,
                    "mc_chunk": 5, "checkpoint_velocity": False},
        "data": {"root": "/fixture/data"},
        "feedback": {"navsim_root": "/fixture/navsim", "maps_root": "/fixture/maps", "cache_roots": ["/fixture/contexts"],
                     "campaign_id": "fixture", "score_protocol_hash": "c"*64, "workers": 1,
                     "max_calls": 0 if cell == "C0" else 16, "retry_attempts": 1}}


def setup_rollout(batch=2):
    cfg = config()
    model = FlowTrainingModule(original_tiny_head(), cfg).train()
    old = FrozenPolicySnapshot(model.policy, version=0, optimizer_step=0, base_binding_hash="a"*64)
    streams = FlowRandomStreams(77)
    cond = context(batch)
    entries = [{"scene_id": f"scene_{i}", "source_log": f"log_{i}", "observation_hash": "b"*64,
                "metric_cache": {"path": f"/fixture/contexts/log_{i}/unknown/scene_{i}/metric_cache.pkl", "sha256": "d"*64}}
               for i in range(batch)]
    rollout = sample_rollout_group(old, cond, scene_ids=[e["scene_id"] for e in entries],
                                   observation_hashes=[e["observation_hash"] for e in entries],
                                   noise_generator=streams["rollout_noise"])
    binding = {"binding_sha256": "a"*64, "checkpoint": {"sha256": "e"*64}, "complete_sha256": "f"*64,
               "inference_contract": {"precision": "FP32_master_eval_TF32_off"}}
    requests = score_requests(rollout, entries, binding, cfg["feedback"], fingerprint(numeric_execution(cfg, binding)))
    return cfg, model, old, streams, cond, rollout, requests


def scored(request, score=.7, nc=1., dac=1., ttc=1.):
    """Explicitly synthetic canonical-result fixture for failure/contract tests."""
    metrics = {k: 1. for k in METRICS}
    metrics.update(score=score, no_at_fault_collisions=nc, drivable_area_compliance=dac,
                   time_to_collision_within_bound=ttc)
    return {"identity": request.identity(), "valid": True, "status": "VALID_SCORED", "score_units": "0..1",
            "context_unchanged": True, "metrics": metrics, "metrics_valid": {k: True for k in METRICS},
            "requested_trajectory": request.trajectory, "context_hash": "1"*64,
            "reference_identity": {"full_cache_sha256": request.metric_cache_sha256,
                                   "score_protocol_hash": request.score_protocol_hash},
            "consequences": {"road": {"any": dac == 0}, "collision_summary": {"nc_min": nc},
                             "ttc_summary": {"any": ttc == 0}}}


def test_resolved_budget_precision_objective_reject_unknown_or_inert_options():
    for cell in ("C0", "C1", "C2", "C3", "C4"):
        cfg = config(cell)
        assert validate_config(cfg).mc_samples == 50
        assert target_contract(cfg)["BC"]["repeat"] == 8
    for path, value in ((["cell"], "C5"), (["fpo", "mc_samples"], 4),
                        (["execution", "stop_after"], 5), (["execution", "sample_limit"], 33),
                        (["execution", "precision_policy"], "float16"), (["loss", "bc_weight"], 0.),
                        (["feedback", "workers"], 8), (["runtime", "checkpoint_velocity"], "false")):
        cfg = config()
        node = cfg
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(ValueError):
            validate_config(cfg)
    cfg = config()
    cfg["old_reuse"] = 100
    with pytest.raises(ValueError, match="unknown"):
        validate_config(cfg)


def test_real_head_four_rollout_to_typed_request_no_GT_or_clipping():
    cfg, model, old, streams, cond, rollout, requests = setup_rollout()
    assert len(requests) == 8
    assert rollout.raw_actions.shape == (2, 4, 8, 4)
    for request in requests:
        assert request.generator_checkpoint_sha256 is None
        assert request.generator_kind == "policy_snapshot"
        assert request.generator_source_hash == fingerprint(request.generator_identity)
        assert request.generator_identity["state_sha256"] == old.identity.state_hash
        assert request.generator_identity["parent_checkpoint_sha256"] == "e"*64
        assert np.asarray(request.trajectory).shape == (8, 3)
    assert not np.all(np.abs(requests[0].trajectory[:, :2]) <= 1.)
    other = replace(requests[0], generator_identity={**requests[0].generator_identity, "policy_version": "wrong"})
    assert other.identity() != requests[0].identity()


def test_rewards_retain_half_collision_and_validate_all_official_fields():
    _, _, _, _, _, rollout, requests = setup_rollout(1)
    records = [scored(r, score=s, nc=n, dac=d, ttc=t) for r, s, n, d, t in zip(
        requests, [0., .1, .8, 1.], [0., .5, 1., 1.], [0., 1., 1., 1.], [1., 0., 1., 1.])]
    expected = {"C1": [0., .1, .8, 1.], "C2": [-1., 0., 0., 0.],
                "C3": [-1., -.5, 0., 0.], "C4": [0., -1., 0., 0.]}
    for cell, values in expected.items():
        feedback = feedback_from_records(rollout, requests, records, cell, "c"*64)
        torch.testing.assert_close(feedback.rewards[0], torch.tensor(values))
    for mutate in (lambda r: r.update(valid=False, status="TRANSIENT_SCORING_FAILURE"),
                   lambda r: r["metrics_valid"].update(score=False),
                   lambda r: r["consequences"]["collision_summary"].update(nc_min=1.),
                   lambda r: r["identity"].update(policy_version="wrong"),
                   lambda r: r.update(context_hash="9"*64)):
        wrong = deepcopy(records)
        mutate(wrong[0])
        with pytest.raises(ValueError):
            feedback_from_records(rollout, requests, wrong, "C1", "c"*64)


def test_MC50_matched_old_current_real_DiT_gradient_and_frozen_parameters():
    cfg, model, old, streams, cond, rollout, requests = setup_rollout()
    before = module_hash(old.policy)
    comparison = prepare_fpo_batch(MatchedOldMicrobatches(old, 1), rollout, cond,
        config=FPOConfig(), streams=streams, mc_chunk=5)
    feedback = feedback_from_records(rollout, requests,
        [scored(r, score=[0., .2, .8, 1.][i % 4]) for i, r in enumerate(requests)], "C1", "c"*64)
    advantages = group_centered_advantages(feedback.rewards, feedback.valid).flatten()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    original = module_hash(model)
    for i in range(2):
        sl = slice(4*i, 4*(i+1))
        output = model(cond[i:i+1], torch.zeros(1, 8, 4), streams=streams, global_scenes=2, world_size=1,
            actions=rollout.raw_actions.flatten(0, 1)[sl], samples=comparison.samples.slice(sl),
            old_cfm=comparison.old_cfm[sl], advantages=advantages[sl], valid=feedback.valid.flatten()[sl])
        torch.testing.assert_close(output["diagnostics"]["ratio"], torch.ones(4), rtol=0, atol=0)
        output["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.policy.action_head.qwen_proj.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.policy.action_head.model.parameters())
    optimizer.step()
    assert original != module_hash(model)
    assert before == module_hash(old.policy)
    assert all(p.grad is None for p in old.parameters())


def test_constant_reward_keeps_original_BC_and_dropout_RNG_exactly():
    cfg, model, old, streams, cond, rollout, requests = setup_rollout(1)
    reference = FlowTrainingModule(deepcopy(model.policy.action_head), config("C0")).train()
    comparison = prepare_fpo_batch(MatchedOldMicrobatches(old, 1), rollout, cond,
                                   config=FPOConfig(), streams=streams, mc_chunk=5)
    fb = feedback_from_records(rollout, requests, [scored(r, score=0.) for r in requests], "C1", "c"*64)
    rng = torch.get_rng_state().clone()
    stream_state = streams.state_dict()
    common = dict(condition=cond, targets=torch.zeros(1, 8, 4), streams=streams, global_scenes=1, world_size=1)
    output = model(**common, actions=rollout.raw_actions.flatten(0, 1), samples=comparison.samples,
        old_cfm=comparison.old_cfm, advantages=group_centered_advantages(fb.rewards, fb.valid).flatten(), valid=fb.valid.flatten())
    output["loss"].backward()
    after_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    streams.load_state_dict(stream_state)
    expected = reference(**common)
    expected["loss"].backward()
    assert torch.equal(output["loss"], expected["loss"])
    assert torch.equal(torch.get_rng_state(), after_rng)
    for p, q in zip(model.parameters(), reference.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


class FixtureScorer:
    def __init__(self, fail_once=()):
        self.fail_once, self.calls = set(fail_once), []

    def score(self, requests, *, expected_policy_version, require_valid):
        assert not require_valid
        results = []
        for request in requests:
            self.calls.append(request.candidate_id)
            if request.candidate_id in self.fail_once:
                self.fail_once.remove(request.candidate_id)
                results.append({"identity": request.identity(), "valid": False, "status": "TRANSIENT_SCORING_FAILURE",
                                "error": {"retryable": True, "message": "fixture timeout"}})
            else:
                results.append(scored(request))
        return results


def test_official_ledger_retry_only_failed_candidate_and_reuse_after_restart(tmp_path):
    *_, requests = setup_rollout(1)
    service = FixtureScorer([requests[1].candidate_id])
    ledger = OfficialFeedbackLedger(tmp_path, identity="id", max_calls=5, retry_attempts=1)
    results = ledger.score(requests, policy_version="0", service=service)
    assert all(r["valid"] for r in results)
    assert len(service.calls) == 5
    assert service.calls.count(requests[1].candidate_id) == 2
    restarted = OfficialFeedbackLedger(tmp_path, identity="id", max_calls=5, retry_attempts=1)
    again = restarted.score(requests, policy_version="0", service=service)
    assert len(service.calls) == 5 and restarted.admitted_calls == 5
    assert [r["identity"] for r in again] == [r["identity"] for r in results]
    assert len(list(tmp_path.glob("*/attempt_*/COMPLETE.json"))) == 2


def test_official_ledger_budget_partial_corruption_and_contract_failures(tmp_path):
    *_, requests = setup_rollout(1)
    ledger = OfficialFeedbackLedger(tmp_path / "budget", identity="id", max_calls=4, retry_attempts=1)
    with pytest.raises(ValueError, match="budget exhausted"):
        ledger.score(requests, policy_version="0", service=FixtureScorer([requests[0].candidate_id]))
    assert ledger.admitted_calls == 4
    good = OfficialFeedbackLedger(tmp_path / "corrupt", identity="id", max_calls=8, retry_attempts=1)
    scorer = FixtureScorer()
    good.score(requests, policy_version="0", service=scorer)
    next((tmp_path / "corrupt").glob("*/attempt_*/RESULTS.pt")).write_bytes(b"partial")
    with pytest.raises(ValueError, match="changed"):
        good.score(requests, policy_version="0", service=scorer)
    assert len(scorer.calls) == 4
    class ContractFailure(FixtureScorer):
        def score(self, requests, **kwargs):
            return [{"identity": r.identity(), "valid": False, "status": "CONTRACT_MISMATCH",
                     "error": {"retryable": False, "message": "wrong source"}} for r in requests]
    wrong = OfficialFeedbackLedger(tmp_path / "contract", identity="id", max_calls=8, retry_attempts=1)
    with pytest.raises(ValueError, match="Nonretryable"):
        wrong.score(requests, policy_version="0", service=ContractFailure())
    assert wrong.admitted_calls == 4


def test_worker_death_admission_is_counted_and_recovered_in_remaining_retry(tmp_path):
    *_, requests = setup_rollout(1)
    class WorkerDeath(FixtureScorer):
        def score(self, requests, **kwargs):
            raise RuntimeError("fixture process death after admission")
    ledger = OfficialFeedbackLedger(tmp_path, identity="id", max_calls=8, retry_attempts=1)
    with pytest.raises(RuntimeError, match="process death"):
        ledger.score(requests, policy_version="0", service=WorkerDeath())
    assert ledger.admitted_calls == 4
    restarted = OfficialFeedbackLedger(tmp_path, identity="id", max_calls=8, retry_attempts=1)
    scorer = FixtureScorer()
    assert all(r["valid"] for r in restarted.score(requests, policy_version="0", service=scorer))
    assert restarted.admitted_calls == 8
    assert len(list(tmp_path.glob("*/attempt_*/ADMITTED.json"))) == 2


def test_frozen_condition_extraction_does_not_receive_or_use_training_labels():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
            self.calls = 0
        def encode_current(self, observations):
            self.calls += 1
            return {"action_queries": torch.tensor([o["state"] for o in observations]).reshape(-1, 1, 1)*self.weight}
        def build_planner_condition(self, encoded):
            return encoded["action_queries"]
    encoder = Encoder().eval()
    observations = [{"state": x, "GT": -999} for x in range(3)]
    a = encode_conditions(encoder, observations, 2)
    assert encoder.calls == 2
    b = encode_conditions(encoder, [{"state": x, "GT": 999, "score": 0} for x in range(3)], 2)
    assert torch.equal(a, b) and not a.requires_grad
    encoder.train()
    with pytest.raises(ValueError, match="frozen and eval"):
        encode_conditions(encoder, observations, 2)


def test_pre_backward_rollout_MC_feedback_receipt_is_complete_and_idempotent(tmp_path):
    _, _, old, streams, cond, rollout, requests = setup_rollout(1)
    comparison = prepare_fpo_batch(MatchedOldMicrobatches(old, 1), rollout, cond,
                                   config=FPOConfig(), streams=streams, mc_chunk=5)
    feedback = feedback_from_records(rollout, requests, [scored(r) for r in requests], "C1", "c"*64)
    receipt = save_update_feedback(tmp_path, rollout, comparison, feedback, numeric_hash="3"*64, rank=0)
    assert save_update_feedback(tmp_path, rollout, comparison, feedback, numeric_hash="3"*64, rank=0) == receipt
    saved = torch.load(receipt["payload"]["path"], weights_only=False)
    assert saved["comparison"]["samples"]["noise"].shape == (4, 50, 8, 4)
    assert saved["rollout"]["initial_noise"].shape == (1, 4, 8, 4)
    assert torch.equal(saved["rollout"]["raw_actions"], rollout.raw_actions)
    with pytest.raises(ValueError, match="Regenerated"):
        save_update_feedback(tmp_path, rollout, comparison, feedback, numeric_hash="4"*64, rank=0)


def test_CPU_full_update_resume_restores_BC_dropout_policy_streams_and_next_ids():
    """Real original tiny DiT/Adam CPU state test, not a native CUDA ZeRO claim."""
    cfg = config()
    model = FlowTrainingModule(original_tiny_head(), cfg).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    streams = FlowRandomStreams(19)
    cond = context(1)
    def update(model, optimizer, streams, step):
        old = FrozenPolicySnapshot(model.policy, version=step, optimizer_step=step, base_binding_hash="a"*64)
        rollout = sample_rollout_group(old, cond, scene_ids=[f"fixed_scene_{step}"], observation_hashes=["b"*64],
                                       noise_generator=streams["rollout_noise"])
        paired = prepare_fpo_batch(MatchedOldMicrobatches(old, 1), rollout, cond, config=FPOConfig(), streams=streams, mc_chunk=5)
        optimizer.zero_grad()
        output = model(cond, torch.zeros(1, 8, 4), streams=streams, global_scenes=1, world_size=1,
            actions=rollout.raw_actions.flatten(0, 1), samples=paired.samples, old_cfm=paired.old_cfm,
            advantages=torch.tensor([-.5, -.3, .3, .5]), valid=torch.ones(4, dtype=torch.bool))
        output["loss"].backward()
        gradients = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        return rollout.scene_ids, rollout.raw_actions, output["loss"].detach(), gradients
    update(model, optimizer, streams, 0)
    saved = deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "streams": streams.state_dict(), "dropout_rng": torch.get_rng_state(), "completed": 1})
    expected = update(model, optimizer, streams, 1)
    expected_hash = module_hash(model)
    next_rng = torch.get_rng_state().clone()
    restored = FlowTrainingModule(original_tiny_head(), cfg).train()
    restored.load_state_dict(saved["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_streams = FlowRandomStreams(19)
    restored_streams.load_state_dict(saved["streams"])
    # No seed-setting operation after checkpoint RNG restoration.
    torch.set_rng_state(saved["dropout_rng"])
    actual = update(restored, restored_optimizer, restored_streams, saved["completed"])
    assert actual[0] == expected[0]
    assert torch.equal(actual[1], expected[1]) and torch.equal(actual[2], expected[2])
    for name in expected[3]:
        assert torch.equal(actual[3][name], expected[3][name])
    assert module_hash(restored) == expected_hash
    assert torch.equal(torch.get_rng_state(), next_rng)
    for key in streams.NAMES:
        assert torch.equal(streams[key].get_state(), restored_streams[key].get_state())
