"""New objective identity and real tiny-DiT state transitions; synthetic feedback."""
from copy import deepcopy

import pytest
import torch

from starVLA.model.modules.flow_posttraining import FlowRandomStreams, FrozenPolicySnapshot, module_hash, prepare_fpo_batch
from starVLA.model.modules.flow_posttraining.advantage_scale import normalize_rollout_advantages
from starVLA.model.modules.st_world.consequence import fingerprint
from tools.st_world_policy.flow_training_config import validate_config, target_contract
from tools.st_world_policy.train_flow_policy import (
    FlowTrainingModule, MatchedOldMicrobatches, feedback_from_records, numeric_execution,
    save_update_feedback, verify_restored_normalization, feedback_scale_identity)
from test_flow_training import config, setup_rollout, scored
from test_fpo import original_tiny_head, context


METHOD = "scene_center_rollout_std_v1"


def normalize(feedback, rollout, numeric_hash="3"*64):
    return normalize_rollout_advantages(feedback.rewards, feedback.valid,
        collective_device=torch.device("cpu"), expected_global_count=feedback.rewards.numel(),
        common_identity={"old_policy": rollout.policy.__dict__, "numeric_execution_hash": numeric_hash,
                         "score_protocol_hash": feedback.score_protocol_hash,
                         "reward_kind": feedback.reward_kind, "global_scene_count": len(rollout.scene_ids)},
        local_identity=feedback_scale_identity(rollout, feedback), distributed=False)


def test_new_method_is_explicit_C1_only_and_changes_numeric_target_identity():
    legacy = config()
    variant = deepcopy(legacy)
    variant["fpo"]["advantage"] = METHOD
    assert validate_config(variant).advantage == METHOD
    binding = {"binding_sha256": "a"*64, "inference_contract": {"precision": "fixture"}}
    old = numeric_execution(legacy, binding)
    new = numeric_execution(variant, binding)
    assert "advantage_normalization" not in old
    assert {k: v for k, v in new.items() if k != "advantage_normalization"} == old
    assert fingerprint(new) != fingerprint(old)
    assert target_contract(variant)["BC"] == target_contract(legacy)["BC"]
    for cell in ("C0", "C2", "C3", "C4"):
        wrong = config(cell)
        wrong["fpo"]["advantage"] = METHOD
        with pytest.raises(ValueError, match="only C1"):
            validate_config(wrong)
    variant["loss"]["policy_weight"] = 2.
    with pytest.raises(ValueError, match="weight stays one"):
        validate_config(variant)


def test_native_evaluation_reader_accepts_explicit_new_numeric_contract(tmp_path, monkeypatch):
    import test_flow_checkpoints as native_tests
    cfg = config()
    cfg["fpo"]["advantage"] = METHOD
    monkeypatch.setattr(native_tests, "config", lambda: deepcopy(cfg))
    fixture = native_tests.native_fixture(tmp_path, monkeypatch)
    actual_cfg, binding, record = native_tests.fc.flow_checkpoint_identity(fixture.reg, fixture.tag)
    assert actual_cfg["fpo"]["advantage"] == METHOD
    assert record["training_numeric_execution_hash"] == fixture.identity["numeric_execution_hash"]
    assert binding == fixture.binding


def test_normalized_feedback_native_boundary_checks_and_legacy_schema(tmp_path):
    _, _, old, streams, cond, rollout, requests = setup_rollout(1)
    paired = prepare_fpo_batch(MatchedOldMicrobatches(old, 1), rollout, cond,
        config=validate_config(config()), streams=streams, mc_chunk=5)
    fb = feedback_from_records(rollout, requests,
        [scored(r, score=s) for r,s in zip(requests, [0., .2, .8, 1.])], "C1", "c"*64)
    adv, norm = normalize(fb, rollout)
    folder = tmp_path / "rollouts" / "boundary_000000"
    receipt = save_update_feedback(folder, rollout, paired, fb, numeric_hash="3"*64, rank=0,
                                  normalization=norm, normalized_advantages=adv)
    assert save_update_feedback(folder, rollout, paired, fb, numeric_hash="3"*64, rank=0,
        normalization=norm, normalized_advantages=adv) == receipt
    payload = torch.load(receipt["payload"]["path"], weights_only=False)
    assert payload["schema"] == "st_flow_update_feedback_v2"
    assert torch.equal(payload["normalized_advantages"], adv)
    state = {"completed": 1, "last_completed_policy": rollout.policy.__dict__,
             "last_advantage_scale_common_sha256": norm["common_sha256"]}
    before = torch.get_rng_state().clone()
    verify_restored_normalization(tmp_path, state, rank=0, numeric_hash="3"*64)
    assert torch.equal(before, torch.get_rng_state())
    verify_restored_normalization(tmp_path, {"completed": 0, "last_advantage_scale_common_sha256": None},
                                 rank=0, numeric_hash="3"*64)
    for changed in ({"last_advantage_scale_common_sha256": "a"*64},
                    {"last_completed_policy": {**rollout.policy.__dict__, "version": 99}}):
        with pytest.raises(ValueError, match="last consumed"):
            verify_restored_normalization(tmp_path, {**state, **changed}, rank=0, numeric_hash="3"*64)
    with pytest.raises(ValueError, match="identities differ"):
        save_update_feedback(tmp_path / "wrong", rollout, paired, fb, numeric_hash="3"*64, rank=0,
                             normalization=norm, normalized_advantages=adv+1)
    # Same relative differences can have different official rewards/contexts.
    other = deepcopy(fb)
    object.__setattr__(other, "rewards", fb.rewards*.5)
    with pytest.raises(ValueError, match="identities differ"):
        save_update_feedback(tmp_path / "swapped", rollout, paired, other, numeric_hash="3"*64, rank=0,
                             normalization=norm, normalized_advantages=adv)
    old_receipt = save_update_feedback(tmp_path / "legacy", rollout, paired, fb, numeric_hash="3"*64, rank=0)
    old_payload = torch.load(old_receipt["payload"]["path"], weights_only=False)
    assert old_payload["schema"] == "st_flow_update_feedback_v1"
    assert "advantage_normalization" not in old_receipt["identity"]
    assert "normalized_advantages" not in old_payload
    with open(receipt["payload"]["path"], "ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="changed"):
        verify_restored_normalization(tmp_path, state, rank=0, numeric_hash="3"*64)


def test_scaled_original_DiT_update_and_exact_CPU_resume():
    """Real DiT/dropout/Adam arithmetic; not a native CUDA/ZeRO certification."""
    from starVLA.model.modules.flow_posttraining import sample_rollout_group
    cfg = config()
    cfg["fpo"]["advantage"] = METHOD
    model = FlowTrainingModule(original_tiny_head(), cfg).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    streams = FlowRandomStreams(19)
    cond = context(1)

    def update(model, optimizer, streams, step):
        old = FrozenPolicySnapshot(model.policy, version=step, optimizer_step=step, base_binding_hash="a"*64)
        rollout = sample_rollout_group(old, cond, scene_ids=[f"fixed_scene_{step}"], observation_hashes=["b"*64],
                                       noise_generator=streams["rollout_noise"])
        paired = prepare_fpo_batch(MatchedOldMicrobatches(old, 1), rollout, cond,
            config=validate_config(cfg), streams=streams, mc_chunk=5)
        from types import SimpleNamespace
        fb = SimpleNamespace(rewards=torch.tensor([[.8, .801, .799, .8]]),
            valid=torch.ones(1,4,dtype=torch.bool), score_protocol_hash="c"*64, reward_kind="C1-PDMS",
            rollout_id=rollout.rollout_id, context_hashes=("b"*64,), statuses=("VALID_SCORED",)*4)
        adv, norm = normalize(fb, rollout)
        optimizer.zero_grad()
        output = model(cond, torch.zeros(1,8,4), streams=streams, global_scenes=1, world_size=1,
            actions=rollout.raw_actions.flatten(0,1), samples=paired.samples, old_cfm=paired.old_cfm,
            advantages=adv.flatten(), valid=fb.valid.flatten())
        output["loss"].backward()
        gradients = {k: p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        return rollout.scene_ids, output["loss"].detach(), gradients, norm["common_sha256"]

    initial = module_hash(model)
    update(model, optimizer, streams, 0)
    assert initial != module_hash(model)
    saved = deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "streams": streams.state_dict(), "rng": torch.get_rng_state()})
    expected = update(model, optimizer, streams, 1)
    final_hash, final_rng = module_hash(model), torch.get_rng_state().clone()
    restored = FlowTrainingModule(original_tiny_head(), cfg).train()
    restored.load_state_dict(saved["model"], strict=True)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    opt2.load_state_dict(saved["optimizer"])
    streams2 = FlowRandomStreams(19)
    streams2.load_state_dict(saved["streams"])
    torch.set_rng_state(saved["rng"])
    actual = update(restored, opt2, streams2, 1)
    assert expected[0] == actual[0] and torch.equal(expected[1], actual[1]) and expected[3] == actual[3]
    assert expected[2].keys() == actual[2].keys()
    assert all(torch.equal(expected[2][k], actual[2][k]) for k in expected[2])
    assert final_hash == module_hash(restored) and torch.equal(final_rng, torch.get_rng_state())
    assert all(torch.equal(streams[k].get_state(), streams2[k].get_state()) for k in streams.NAMES)
