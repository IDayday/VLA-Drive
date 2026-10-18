"""Behavior tests for the sole registered complete-rollout scale adaptation.

Synthetic CPU arithmetic and a two-process Gloo group only. No model/data/cache,
GPU or official simulator is opened. These tests are not real pipeline approval.
"""
from datetime import timedelta
import json
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from starVLA.model.modules.flow_posttraining import advantage_scale as scale
from starVLA.model.modules.flow_posttraining.fpo_objective import (
    FPOConfig, fpo_surrogate, group_centered_advantages,
)
from starVLA.model.modules.flow_posttraining.policy_snapshot import tensor_hash


def identities(scenes, rank=0):
    common = {"old_policy": {"version": 7, "optimizer_step": 7,
                            "state_hash": "1" * 64, "base_binding_hash": "2" * 64,
                            "role": "old"},
              "numeric_execution_hash": "3" * 64, "score_protocol_hash": "4" * 64,
              "reward_kind": "C1-PDMS", "global_scene_count": scenes}
    local = {"rollout_id": f"{rank + 5:064x}", "feedback_hash": f"{rank + 8:064x}"}
    return common, local


def sample_rewards(scenes=32):
    # Local generator does not consume the global training stream.
    value = torch.rand(scenes, 4, generator=torch.Generator().manual_seed(551))
    value[::4] = 0
    value[1::7] = .8
    return value


def normalize(rewards, valid=None, *, common=None, local=None, expected=None, **kwargs):
    default_common, default_local = identities(len(rewards))
    return scale.normalize_rollout_advantages(
        rewards, torch.ones_like(rewards, dtype=torch.bool) if valid is None else valid,
        collective_device="cpu", expected_global_count=rewards.numel() if expected is None else expected,
        common_identity=default_common if common is None else common,
        local_identity=default_local if local is None else local, **kwargs)


@pytest.mark.parametrize("scenes", [1, 5, 24, 32])
def test_complete_population_reference_and_receipt(scenes):
    rewards = sample_rewards(scenes)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    raw = group_centered_advantages(rewards, valid)
    output, receipt = normalize(rewards)
    std = raw.double().flatten().std(correction=1)
    expected = (raw.double() / (std + 1e-10)).float()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.dtype == torch.float32 and not output.requires_grad
    common = receipt["common"]
    assert common["count"] == 4 * scenes
    assert common["definition"]["correction"] == 1
    assert common["definition"]["epsilon_placement"] == "after_sqrt"
    assert common["std"] == pytest.approx(float(std), rel=2e-15, abs=0)
    assert receipt["common_sha256"] == scale.receipt_sha256(common)
    assert common["rank_records"][0]["raw_advantages_sha256"] == tensor_hash(raw)
    assert common["rank_records"][0]["normalized_advantages_sha256"] == tensor_hash(output)
    assert json.loads(json.dumps(receipt)) == receipt


def test_zero_groups_are_not_globally_recentered():
    value = torch.tensor([[1., 1., 1., 1.], [.5, .5, .5, .5]], dtype=torch.float32)
    value[0, 3] = torch.nextafter(torch.tensor(1.), torch.tensor(0.))
    raw = group_centered_advantages(value, torch.ones_like(value, dtype=torch.bool))
    assert raw.sum() < 0  # Original FP32 centering residual, not new re-centering.
    output, receipt = normalize(value)
    assert torch.equal(output[raw == 0], torch.zeros_like(output[raw == 0]))
    assert output[0, 3] < 0
    wrong = ((raw.double() - raw.double().mean()) / (raw.double().std() + 1e-10)).float()
    assert wrong[1].count_nonzero() == 4
    assert receipt["common"]["mean"] < 0


def test_all_zero_still_allows_original_bc_and_rng_unchanged():
    rewards = torch.tensor([[0.] * 4, [.5] * 4, [1.] * 4], dtype=torch.float32)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    output, receipt = normalize(rewards)
    assert not output.any() and receipt["common"]["all_zero"]
    assert receipt["common"]["std"] == 0
    assert receipt["common"]["gain"] is None
    assert receipt["common"]["denominator_reciprocal"] == 1e10
    parameter = torch.tensor(2., requires_grad=True)
    old = torch.zeros(rewards.numel(), 2)
    policy, _ = fpo_surrogate(old, old + parameter, output.flatten(),
                             torch.ones(rewards.numel(), dtype=torch.bool), FPOConfig(mc_samples=2),
                             global_count=rewards.numel(), world_size=1)
    ((parameter - 1).square() + policy).backward()
    assert parameter.grad == 2
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0] and np.array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert torch.equal(torch_state, torch.get_rng_state())


@pytest.mark.parametrize("case", ["double", "half", "trainable", "partial", "nan", "negative",
                                  "above_one", "wrong_group", "wrong_mask", "empty",
                                  "expected_mismatch", "identity_missing", "identity_nonjson",
                                  "identity_nan", "wrong_reward_kind", "wrong_local_hash",
                                  "old_reference", "bad_device"])
def test_bad_input_is_rejected_without_silent_fallback(case):
    rewards = sample_rewards(3)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    common, local = identities(3)
    kwargs = {}
    if case == "double": rewards = rewards.double()
    elif case == "half": rewards = rewards.half()
    elif case == "trainable": rewards.requires_grad_(True)
    elif case == "partial": valid[0, 0] = False
    elif case == "nan": rewards[0, 0] = float("nan")
    elif case == "negative": rewards[0, 0] = -.001
    elif case == "above_one": rewards[0, 0] = 1.01
    elif case == "wrong_group": rewards, valid = rewards[:, :3], valid[:, :3]
    elif case == "wrong_mask": valid = valid.float()
    elif case == "empty": rewards, valid, common = rewards[:0], valid[:0], identities(0)[0]
    elif case == "expected_mismatch": kwargs["expected"] = 8
    elif case == "identity_missing": common.pop("old_policy")
    elif case == "identity_nonjson": common["x"] = torch.tensor(1)
    elif case == "identity_nan": common["x"] = float("nan")
    elif case == "wrong_reward_kind": common["reward_kind"] = "C2-D"
    elif case == "wrong_local_hash": local["feedback_hash"] = "wrong"
    elif case == "old_reference": common["old_policy"]["role"] = "reference"
    elif case == "bad_device":
        with pytest.raises(scale.RolloutNormalizationError):
            scale.normalize_rollout_advantages(rewards, valid, collective_device="not-a-device",
                                               expected_global_count=12,
                                               common_identity=common, local_identity=local)
        return
    with pytest.raises(scale.RolloutNormalizationError):
        normalize(rewards, valid, common=common, local=local, **kwargs)


def test_nonzero_zero_variance_rejected(monkeypatch):
    monkeypatch.setattr(scale, "group_centered_advantages",
                        lambda rewards, valid, **kwargs: torch.ones_like(rewards))
    with pytest.raises(scale.RolloutNormalizationError, match="zero rollout variance"):
        normalize(sample_rewards(2))


def test_microbatch_and_positive_scale_gradient():
    rewards = sample_rewards(24)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    raw = group_centered_advantages(rewards, valid).flatten()
    normalized, receipt = normalize(rewards)
    adv = normalized.flatten().double()
    raw = raw.double()
    count = len(raw)
    config = FPOConfig(mc_samples=3)
    x = torch.linspace(-1, 1, count * 3, dtype=torch.float64).view(count, 3)
    old = .7 + x.square()
    p = torch.tensor([.2, -.3], dtype=torch.float64, requires_grad=True)
    current = old + p[0] * x + p[1] * x.square()
    unscaled, _ = fpo_surrogate(old, current, raw, valid.flatten(), config,
                              global_count=count, world_size=1)
    raw_grad = torch.autograd.grad(unscaled, p, retain_graph=True)[0]
    full, _ = fpo_surrogate(old, current, adv, valid.flatten(), config,
                          global_count=count, world_size=1)
    full_grad = torch.autograd.grad(full, p, retain_graph=True)[0]
    # Normalization output is FP32 by contract, so use its actual rounding bound
    # rather than claiming exact real-arithmetic scale proportionality.
    gain = receipt["common"]["gain"]
    torch.testing.assert_close(full_grad, raw_grad * gain, rtol=2e-6, atol=1e-12)
    parts = [slice(0, 4), slice(4, 28), slice(28, 60), slice(60, 96)]
    micro_loss = sum(fpo_surrogate(old[s], current[s], adv[s], valid.flatten()[s], config,
                                  global_count=count, world_size=1)[0] for s in parts)
    micro_grad = torch.autograd.grad(micro_loss, p)[0]
    torch.testing.assert_close(micro_loss, full, rtol=2e-15, atol=1e-15)
    torch.testing.assert_close(micro_grad, full_grad, rtol=2e-15, atol=1e-15)


def test_receipt_reconstruction_and_input_binding():
    rewards = sample_rewards(5)
    output, first = normalize(rewards)
    restored = json.loads(json.dumps(first))
    output2, second = normalize(rewards.clone())
    assert torch.equal(output, output2) and first == restored == second
    common, local = identities(5)
    common["old_policy"]["optimizer_step"] += 1
    _, changed_policy = normalize(rewards, common=common)
    assert changed_policy["common_sha256"] != first["common_sha256"]
    local["feedback_hash"] = "f" * 64
    _, changed_feedback = normalize(rewards, local=local)
    assert changed_feedback["common_sha256"] != first["common_sha256"]
    contract = scale.normalization_contract()
    contract["epsilon"] = 1
    assert scale.normalization_contract()["epsilon"] == 1e-10


def test_population_includes_zero_groups_and_autocast_does_not_change_fp32_path():
    rewards = torch.tensor([[.1, .2, .6, .9], [0., 0., 0., 0.]], dtype=torch.float32)
    expected, receipt = normalize(rewards)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, under_amp = normalize(rewards)
    assert torch.equal(actual, expected) and receipt == under_amp
    _, active_only = normalize(rewards[:1])
    assert receipt["common"]["count"] == 8
    assert receipt["common"]["gain"] != active_only["common"]["gain"]
    common, local = identities(2)
    with pytest.raises(scale.RolloutNormalizationError, match="global rollout count"):
        normalize(rewards[:1], common=common, local=local, expected=8)


def _gloo_worker(rank, rendezvous, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    records = []
    try:
        # Unequal partitions, the real 24-scene tail, and one empty local rank.
        for scenes, split in ((32, 13), (24, 8), (5, 0), (3, 3)):
            full = sample_rewards(scenes)
            start, end = (0, split) if rank == 0 else (split, scenes)
            rewards = full[start:end].clone()
            common, local = identities(scenes, rank)
            rng_before = torch.get_rng_state()
            output, receipt = normalize(rewards, common=common, local=local, expected=scenes * 4)
            assert torch.equal(rng_before, torch.get_rng_state())
            reference = group_centered_advantages(full, torch.ones_like(full, dtype=torch.bool))
            expected = (reference.double() / (reference.double().std() + 1e-10)).float()
            torch.testing.assert_close(output, expected[start:end], rtol=0, atol=0)
            hashes = [None, None]
            dist.all_gather_object(hashes, receipt["common_sha256"])
            assert hashes[0] == hashes[1]
            assert sum(item["local_count"] for item in receipt["common"]["rank_records"]) == scenes * 4
            # Actual global FPO loss/gradient with unequal rank populations and
            # rank-local microbatches. Exactly one rank averaging is applied.
            count = scenes * 4
            x = torch.linspace(-.8, .9, count * 2, dtype=torch.float64).view(count, 2)
            old = x.square() + .5
            p = torch.tensor(.13, dtype=torch.float64, requires_grad=True)
            current = old[start * 4:end * 4] + p * x[start * 4:end * 4]
            loss = p * 0
            for offset in range(0, output.numel(), 12):
                sl = slice(offset, offset + 12)
                loss = loss + fpo_surrogate(old[start * 4:end * 4][sl], current[sl],
                    output.flatten().double()[sl], torch.ones_like(output.flatten()[sl], dtype=torch.bool),
                    FPOConfig(mc_samples=2), global_count=count, world_size=2)[0]
            loss.backward()
            actual = torch.stack([loss.detach(), p.grad])
            dist.all_reduce(actual); actual /= 2
            ref_p = torch.tensor(.13, dtype=torch.float64, requires_grad=True)
            ref_loss = fpo_surrogate(old, old + ref_p * x, expected.flatten().double(),
                torch.ones(count, dtype=torch.bool), FPOConfig(mc_samples=2), global_count=count, world_size=1)[0]
            ref_loss.backward()
            torch.testing.assert_close(actual, torch.stack([ref_loss.detach(), ref_p.grad]), rtol=2e-14, atol=1e-15)
            records.append({"case": [scenes, split], "common_sha256": hashes[0], "status": "PASS"})
        # These faults originate on just one rank. All ranks must return the
        # same explicit error, without entering a mismatched stats collective.
        for fault in ("partial", "nan", "dtype", "requires_grad", "expected", "common",
                      "local", "device", "disabled", "count_mismatch"):
            common, local = identities(4, rank)
            rewards = sample_rewards(2)
            valid = torch.ones_like(rewards, dtype=torch.bool)
            expected_count = 16
            device = "cpu"
            distributed = True
            if rank == 1:
                if fault == "partial": valid[0, 0] = False
                elif fault == "nan": rewards[0, 0] = float("nan")
                elif fault == "dtype": rewards = rewards.double()
                elif fault == "requires_grad": rewards.requires_grad_(True)
                elif fault == "expected": expected_count, common["global_scene_count"] = 20, 5
                elif fault == "common": common["numeric_execution_hash"] = "a" * 64
                elif fault == "local": local["feedback_hash"] = "incorrect"
                elif fault == "device": device = "invalid-device"
                elif fault == "disabled": distributed = False
                elif fault == "count_mismatch": rewards, valid = rewards[:1], valid[:1]
            caught = None
            try:
                scale.normalize_rollout_advantages(rewards, valid, collective_device=device,
                    expected_global_count=expected_count, common_identity=common,
                    local_identity=local, distributed=distributed)
            except scale.RolloutNormalizationError as exc:
                caught = str(exc)
            errors = [None, None]
            dist.all_gather_object(errors, caught)
            assert errors[0] and errors[0] == errors[1], (fault, errors)
            records.append({"fault": fault, "status": "BOTH_RANKS_REJECTED", "message": errors[0]})
        with open(output_dir + f"/rank{rank}.json", "w") as handle:
            json.dump(records, handle, indent=2)
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_unequal_tail_empty_and_collective_failures(tmp_path):
    # One two-rank job, each child one thread, PG timeout 20s. The outer test
    # invocation additionally has a shell timeout; no indefinite worker launch.
    mp.spawn(_gloo_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        records = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert len(records) == 14
