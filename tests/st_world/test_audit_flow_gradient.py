"""Auditor behavior with explicit CPU fixtures, not real driving evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from starVLA.model.modules.flow_posttraining import (
    DiTFlowPolicy, FPOConfig, module_hash, prepare_fpo_batch)
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
from tools.st_world_policy import audit_flow_gradient as audit
from tools.st_world_policy.train_flow_policy import (
    MatchedOldMicrobatches, OfficialFeedbackLedger, feedback_from_records, save_update_feedback)
from test_flow_training import FixtureScorer, scored, setup_rollout
from test_fpo import original_tiny_head


def fixture(tied=False, batch=2):
    cfg, model, old, streams, condition, rollout, requests = setup_rollout(batch)
    comparison = prepare_fpo_batch(MatchedOldMicrobatches(old, 1), rollout, condition,
                                  config=FPOConfig(), streams=streams, mc_chunk=5)
    records = [scored(r, score=1. if tied else [0., .2, .8, 1.][i % 4]) for i, r in enumerate(requests)]
    feedback = feedback_from_records(rollout, requests, records, "C1", "c" * 64)
    return cfg, old, condition, rollout, comparison, feedback, requests


@pytest.mark.parametrize("checkpoint_velocity", [False, True])
def test_actual_DiT_policy_only_gradient_is_not_BC_or_optimizer(checkpoint_velocity, monkeypatch):
    _, old, condition, rollout, comparison, feedback, _ = fixture()
    before = module_hash(old.policy)
    rng = torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("BC/optimizer must not be called in this audit")
    monkeypatch.setattr(DiTFlowPolicy, "bc_anchor", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    result = audit.audit_policy_gradients(old.policy, condition, rollout, comparison, feedback,
        micro_batch=1, mc_chunk=5, checkpoint_velocity=checkpoint_velocity)
    assert result["status"] == "POLICY_GRADIENT_REACHED_DIT_AND_PROJECTION"
    assert result["actual_nonzero_advantages"] == 8
    assert abs(result["reports"]["actual_feedback"]["loss"]) < 1e-6
    for name in ("DiT", "qwen_proj"):
        assert result["reports"]["actual_feedback"]["groups"][name]["l2"] > 0
    assert all(v["nonzero"] == 0 for v in result["reports"]["zero_advantage_control"]["groups"].values())
    assert result["BC_executed"] is False and result["optimizer_steps"] == 0
    assert module_hash(old.policy) == before and torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None and not p.requires_grad for p in old.parameters())
    assert all(value["bit_exact"] for value in result["reports"]["actual_feedback"]["current_CFM"])


def test_real_tied_feedback_remains_zero_signal():
    _, old, condition, rollout, comparison, feedback, _ = fixture(tied=True, batch=1)
    result = audit.audit_policy_gradients(old.policy, condition, rollout, comparison, feedback,
        micro_batch=1, mc_chunk=5, checkpoint_velocity=False)
    assert result["status"] == "NO_ADVANTAGE_SIGNAL"
    assert all(value["nonzero"] == 0 for value in result["reports"]["actual_feedback"]["groups"].values())


@pytest.mark.parametrize("change", ["condition", "old", "old_cfm", "invalid_feedback"])
def test_pure_gradient_rejects_wrong_replay_inputs(change):
    _, old, condition, rollout, comparison, feedback, _ = fixture(batch=1)
    if change == "condition":
        condition = condition + .01
    elif change == "old":
        with torch.no_grad():
            next(old.parameters()).add_(.01)
    elif change == "old_cfm":
        comparison.old_cfm[0, 0] += .01
    else:
        feedback.valid[0, 0] = False
    with pytest.raises(ValueError):
        audit.audit_policy_gradients(old.policy, condition, rollout, comparison, feedback,
            micro_batch=1, mc_chunk=5, checkpoint_velocity=False)


def test_actual_four_candidate_ten_euler_replay():
    _, old, condition, rollout, _, _, _ = fixture()
    rng = torch.get_rng_state().clone()
    result = audit.replay_endpoints(old.policy, condition, rollout.initial_noise)
    assert audit.exact_tensor(result, rollout.raw_actions, "rollout")["bit_exact"]
    assert torch.equal(torch.get_rng_state(), rng)
    changed = result.clone()
    changed.flatten()[0] += .01
    with pytest.raises(ValueError, match="no tolerance relaxation"):
        audit.exact_tensor(changed, rollout.raw_actions, "rollout")


def native_fixture():
    head = original_tiny_head()
    execution = deepcopy(head).bfloat16()
    receipt = {"state": {"completed": 1, "policy_version": 1, "pending_feedback": None}}
    native = {"global_steps": 1, **receipt["state"],
        "module": {audit.ACTION_PREFIX + k: v.clone() for k, v in execution.state_dict().items()},
        "param_shapes": [{audit.ACTION_PREFIX + k: v.shape for k, v in execution.named_parameters()}],
        "buffer_names": [audit.ACTION_PREFIX + k for k, _ in execution.named_buffers() if k in execution.state_dict()],
        "frozen_param_shapes": None, "frozen_param_fragments": None, "shared_params": {}}
    return head, native, receipt, module_hash(DiTFlowPolicy(execution))


def test_native_execution_loads_saved_BF16_values_not_FP32_masters():
    head, native, receipt, expected = native_fixture()
    saved = deepcopy(native["module"])
    policy = audit.restore_native_execution(head, native, receipt, expected)
    assert module_hash(policy) == expected
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in policy.parameters())
    assert all(not m.training for m in policy.modules())
    assert all(torch.equal(saved[k], native["module"][k]) for k in saved)


@pytest.mark.parametrize("change", ["step", "namespace", "dtype", "shape", "param_shapes", "buffer_names", "hash", "nan"])
def test_native_execution_rejects_invalid_source(change):
    head, native, receipt, expected = native_fixture()
    key = next(iter(native["module"]))
    if change == "step":
        native["global_steps"] = 2
    elif change == "namespace":
        native["module"]["foreign." + key] = native["module"].pop(key)
    elif change == "dtype":
        native["module"][key] = native["module"][key].float()
    elif change == "shape":
        native["module"][key] = native["module"][key].flatten()
    elif change == "param_shapes":
        native["param_shapes"][0][key] = (1,)
    elif change == "buffer_names":
        native["buffer_names"].append("policy.action_head.fake")
    elif change == "hash":
        expected = "0" * 64
    else:
        native["module"][key].flatten()[0] = float("nan")
    with pytest.raises(ValueError):
        audit.restore_native_execution(head, native, receipt, expected)


def saved_payload(tmp_path):
    cfg, _, _, rollout, comparison, feedback, _ = fixture(batch=1)
    save_update_feedback(tmp_path, rollout, comparison, feedback, numeric_hash="2" * 64, rank=0)
    kwargs = dict(numeric_hash="2" * 64, step=0, binding_hash="a" * 64,
                  entries=[{"scene_id": "scene_0", "observation_hash": "b" * 64}], fpo_config=cfg["fpo"])
    return kwargs


def test_payload_receipt_reconstructs_exact_saved_MC50_and_feedback(tmp_path):
    kwargs = saved_payload(tmp_path)
    rollout, comparison, feedback, receipt = audit.load_feedback_payload(tmp_path / "rank_0_COMPLETE.json", **kwargs)
    assert comparison.config.mc_samples == 50 and rollout.policy.version == 0
    assert feedback.rewards.shape == (1, 4) and receipt["sha256"] == file_sha256(tmp_path / "rank_0_COMPLETE.json")


@pytest.mark.parametrize("change", ["numeric", "step", "binding", "IDs", "MC", "partial"])
def test_payload_rejects_stale_or_corrupt_artifact(tmp_path, change):
    kwargs = saved_payload(tmp_path)
    if change == "numeric":
        kwargs["numeric_hash"] = "0" * 64
    elif change == "step":
        kwargs["step"] = 1
    elif change == "binding":
        kwargs["binding_hash"] = "0" * 64
    elif change == "IDs":
        kwargs["entries"][0]["scene_id"] = "foreign_scene"
    elif change == "MC":
        kwargs["fpo_config"]["mc_samples"] = 4
    else:
        (tmp_path / "rank_0.pt").write_bytes(b"interrupted")
    with pytest.raises(ValueError):
        audit.load_feedback_payload(tmp_path / "rank_0_COMPLETE.json", **kwargs)


def test_official_reader_only_reuses_completed_real_schema_and_retries(tmp_path):
    *_, requests = setup_rollout(1)
    service = FixtureScorer([requests[1].candidate_id])
    ledger = OfficialFeedbackLedger(tmp_path / "official_feedback", identity="fixture", max_calls=5, retry_attempts=1)
    original = ledger.score(requests, policy_version="0", service=service)
    calls = list(service.calls)
    records, receipts = audit.read_completed_official_feedback(tmp_path, requests, run_identity="fixture",
        cfg={"max_calls": 5, "retry_attempts": 1})
    assert len(receipts) == 2 and service.calls == calls
    assert [x["identity"] for x in records] == [x["identity"] for x in original]
    # Removing only a completion marker cannot trigger new scoring in a reader.
    Path(receipts[-1]["path"]).rename(Path(receipts[-1]["path"]).with_name("INCOMPLETE.json"))
    with pytest.raises(ValueError, match="incomplete"):
        audit.read_completed_official_feedback(tmp_path, requests, run_identity="fixture",
            cfg={"max_calls": 5, "retry_attempts": 1})
    assert service.calls == calls


def test_original_device_feedback_retains_physical_values_until_GPU_check(tmp_path):
    *_, rollout, requests = setup_rollout(1)
    ledger = OfficialFeedbackLedger(tmp_path / "official_feedback", identity="fixture", max_calls=4, retry_attempts=0)
    ledger.score(requests, policy_version="0", service=FixtureScorer())
    # Represents last-bit CPU-vs-GPU conversion divergence, without rewriting
    # the stored actual request/score. GPU audit must later check every hash.
    cpu_recomputed = [replace(r, trajectory_sha256="0" * 64) for r in requests]
    restored, records, _ = audit.read_original_device_feedback(tmp_path, cpu_recomputed,
        run_identity="fixture", cfg={"max_calls": 4, "retry_attempts": 0})
    assert [r.identity() for r in restored] == [r.identity() for r in requests]
    feedback_from_records(rollout, restored, records, "C1", "c" * 64)
    wrong = [replace(r, generator_source_hash="0" * 64) for r in cpu_recomputed]
    with pytest.raises(ValueError, match="unique"):
        audit.read_original_device_feedback(tmp_path, wrong, run_identity="fixture", cfg={"max_calls": 4, "retry_attempts": 0})


@pytest.mark.parametrize("change", ["identity", "admission", "result", "requests"])
def test_official_reader_rejects_modified_receipts(tmp_path, change):
    *_, requests = setup_rollout(1)
    ledger = OfficialFeedbackLedger(tmp_path / "official_feedback", identity="fixture", max_calls=4, retry_attempts=0)
    ledger.score(requests, policy_version="0", service=FixtureScorer())
    folder = tmp_path / "official_feedback" / identity_hash([r.identity() for r in requests])
    path = {"identity": tmp_path / "official_feedback" / "IDENTITY.json",
            "admission": folder / "attempt_00" / "ADMITTED.json", "result": folder / "attempt_00" / "RESULTS.pt",
            "requests": folder / "REQUESTS.json"}[change]
    path.write_bytes(b"{}")
    with pytest.raises(ValueError):
        audit.read_completed_official_feedback(tmp_path, requests, run_identity="fixture", cfg={"max_calls": 4, "retry_attempts": 0})


def test_native_next_batch_identity_and_epoch_transition():
    manifest = {"entries": [{"scene_id": str(i)} for i in range(5)]}
    execution = {"world_size": 1, "global_batch": 2, "seed": 42}
    first, cursor = audit.next_scene_entries(manifest, {"epoch": 0, "offset": 0}, execution)
    assert len(first) == 2 and cursor["manifest_indices"] == [2, 4]
    next_batch, cursor = audit.next_scene_entries(manifest, {"epoch": 0, "offset": 3}, execution)
    assert cursor["epoch"] == 1 and cursor["offset"] == 0
    assert next_batch != first
    with pytest.raises(ValueError, match="single-rank"):
        audit.next_scene_entries(manifest, {"epoch": 0, "offset": 0}, {**execution, "world_size": 2})
    with pytest.raises(ValueError, match="offset"):
        audit.next_scene_entries(manifest, {"epoch": 0, "offset": 4}, execution)


def test_current_input_does_not_read_any_GT_or_metric_cache(tmp_path, monkeypatch):
    (tmp_path / "current").mkdir()
    paths = []
    for index in range(3):
        path = tmp_path / f"view_{index}.png"
        Image.new("RGB", (32, 18), color=(index, 20, 30)).save(path)
        paths.append(str(path))
    indices = [{"token": "scene", "log": "log"}]
    identity = {"schema": "ddpolicy_current_cameras_v1", "identity": "fixture", "split": "train", "index_sha256": identity_hash(indices)}
    (tmp_path / "index.json").write_text(json.dumps(indices))
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    current = tmp_path / "current" / "scene.json"
    record = {"identity": "fixture", "token": "scene", "image_paths": paths,
              "global_pose_history": [[0, 0, 0]] * 4, "navigation": 1}
    current.write_text(json.dumps(record))
    data = {"cfg": {"data": {"root": str(tmp_path)}}, "manifest": {"data_identity": identity},
            "entries": [{"dataset_index": 0, "scene_id": "scene", "source_log": "log",
                         "current": {"path": str(current)}, "images": [{"path": p} for p in paths]}]}
    def forbidden(*args, **kwargs):
        raise AssertionError("Current-only input must not load torch label/metric payloads")
    monkeypatch.setattr(torch, "load", forbidden)
    before = audit.current_inputs(data)
    # Label metadata is not an allowed VLM input and does not alter context.
    record.update(GT="different", future_actors="unread", metric_cache="unread")
    current.write_text(json.dumps(record))
    after = audit.current_inputs(data)
    assert set(before[0]) == {"image", "state", "lang", "token"}
    assert before[0]["lang"] == after[0]["lang"]
    assert (before[0]["state"] == after[0]["state"]).all()
    assert all(a.tobytes() == b.tobytes() for a, b in zip(before[0]["image"], after[0]["image"]))
    data["entries"][0]["current"]["path"] = str(tmp_path / "foreign.json")
    with pytest.raises(ValueError, match="record"):
        audit.current_inputs(data)


@pytest.mark.parametrize("extra", [["--max-seconds", "901"], ["--max-scenes", "33"]])
def test_cli_refuses_unbounded_audit_before_reading_registration(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        audit.main(["--registration", "absent", "--tag", "step", "--output", str(tmp_path / "audit"), *extra])
    assert error.value.code == 2
