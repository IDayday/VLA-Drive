"""CPU native-schema/real-action-class tests, not real-image S0 qualification."""
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.dataloader.st_world_schema import ModelInputs
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
from tools.st_world_policy import flow_checkpoints as fc
from tools.st_world_policy.training_state import checkpoint_receipt
from tools.st_world_policy.train_flow_policy import numeric_execution
from test_flow_training import config
from test_fpo import original_tiny_head


class ParentFixture(nn.Module):
    """Only the encoder is a fixture; action class is the real original DiT."""
    def __init__(self):
        super().__init__()
        self.action_model = original_tiny_head()
        # Match the original parent loader's canonical saved-buffer behavior.
        for buffer in self.action_model.buffers():
            buffer.copy_(buffer.bfloat16().float())
        self.encoder = nn.Linear(4, 12)
        self.config = SimpleNamespace(framework=SimpleNamespace(action_model=self.action_model.config))
        self.foresight_config = SimpleNamespace(planner_condition_mode="action_only")
        self.inference_fp32, self.calls, self.stripped = True, 0, False

    def encode_current(self, observations):
        self.calls += 1
        x = torch.tensor([v["state"] for v in observations], dtype=torch.float32)
        self.last_sequence_lengths = torch.full((len(x),), 3)
        return {"action_queries": self.encoder(x).expand(-1, 3, -1)}

    def build_planner_condition(self, encoded):
        return encoded["action_queries"]

    def amp(self):
        return nullcontext()

    def strip_auxiliary_heads(self):
        self.stripped = True


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), "sha256": file_sha256(path)}


def native_fixture(tmp_path, monkeypatch):
    parent = ParentFixture().eval()
    cfg = config()
    cfg["runtime"]["output_root"] = str(tmp_path / "runs")
    binding = {"binding_sha256": "a"*64, "run": "/fixture/parent", "tag": "parent_100k",
        "parent_identity": {"fixture": True}, "source_contracts": {},
        "inference_contract": dict(fc.INFERENCE_CONTRACT), "complete_sha256": "b"*64,
        "checkpoint": {"sha256": "c"*64}}
    registration = {"schema": "st_world_run_v1", "run_id": "fixture_C1", "source_sha": "d"*40,
        "source_fingerprint": "e"*64, "execution": cfg["execution"],
        "weights_usable_for_formal_init": False,
        "artifacts": {"config": _json(tmp_path / "CONFIG.json", cfg),
                      "base_binding": _json(tmp_path / "BINDING.json", binding),
                      "environment": {"path": "/never/read/environment", "sha256": "f"*64}}}
    reg = tmp_path / "REGISTRATION.json"
    _json(reg, registration)
    numeric = {**numeric_execution(cfg, binding), "environment_sha256": "f"*64,
               "source_fingerprint": registration["source_fingerprint"]}
    body = {"schema": "st_flow_training_run_v1", "config": cfg, "base_binding": binding["binding_sha256"],
        "source_sha": registration["source_sha"], "registration_sha256": file_sha256(reg),
        "numeric_execution": numeric, "numeric_execution_hash": identity_hash(numeric),
        "weights_usable_for_formal_init": False, "new_optimizer_from_common_S0": True}
    run = Path(cfg["runtime"]["output_root"]) / registration["run_id"]
    identity = {**body, "identity": identity_hash(body)}
    _json(run / "identity.json", identity)
    tag = "step_000002_fixture"
    folder = run / "checkpoints" / tag
    folder.mkdir(parents=True)
    state = {"identity": identity["identity"], "tag": tag, "completed": 2,
             "termination_reason": "BUDGET_PAUSED", "policy_version": 2,
             "pending_old_policy": None, "pending_feedback": None}
    # Store deliberately rounded execution parameters and distinct FP32 masters.
    buffers = {fc.ACTION_PREFIX+k for k, _ in parent.action_model.named_buffers()}
    action = {k: v.clone() + (0. if fc.ACTION_PREFIX+k in buffers else 0.00000123)
              for k, v in parent.action_model.state_dict().items()}
    params = OrderedDict((fc.ACTION_PREFIX+k, v.shape) for k, v in parent.action_model.named_parameters())
    native = {**state, "global_steps": 2, "module": {fc.ACTION_PREFIX+k: v.bfloat16() for k, v in action.items()},
              "buffer_names": sorted(buffers), "param_shapes": [params], "shared_params": {}, "ds_version": "fixture"}
    flat = torch.cat([action[k[len(fc.ACTION_PREFIX):]].reshape(-1) for k in params])
    optimizer = {"optimizer_state_dict": {"zero_stage": 2, "partition_count": 1,
                   "single_partition_of_fp32_groups": [flat], "optimizer_state_dict": {"state": {}}}}
    torch.save(native, folder / "mp_rank_00_model_states.pt")
    torch.save(optimizer, folder / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt")
    torch.save({"fixture": True}, folder / "rng_rank0.pt")
    receipt = checkpoint_receipt(folder, state, world_size=1)
    calls = []
    def read_binding(path, *, verify_files=False):
        calls.append(verify_files)
        return deepcopy(binding)
    monkeypatch.setattr(fc, "read_binding", read_binding)
    monkeypatch.setattr(fc, "inference_source_contract", lambda *a: {"fixture_source": "g"*64})
    return SimpleNamespace(parent=parent, cfg=cfg, binding=binding, registration=registration,
        reg=reg, identity=identity, run=run, folder=folder, tag=tag, action=action,
        native=native, optimizer=optimizer, receipt=receipt, binding_checks=calls)


def test_native_flow_identity_metadata_and_exact_inventory(tmp_path, monkeypatch):
    f = native_fixture(tmp_path, monkeypatch)
    cfg, binding, record = fc.flow_checkpoint_identity(f.reg, f.tag)
    assert cfg == f.cfg and binding == f.binding
    assert record["native_checkpoint_sha256"] == identity_hash({k: v["sha256"] for k, v in f.receipt["files"].items()})
    assert record["native_receipt_sha256"] == file_sha256(f.folder / "COMPLETE.json")
    assert record["sha256"] == identity_hash({k: v for k, v in record.items() if k != "sha256"})
    assert record["scope"] == "SMOKE" and not record["weights_usable_for_formal_init"]
    (f.folder / "stale_model_states.pt").write_bytes(b"unreceipted file")
    with pytest.raises(ValueError, match="inventory"):
        fc.flow_checkpoint_identity(f.reg, f.tag)


@pytest.mark.parametrize("mutation", ["schema", "config", "parent", "numeric", "scope", "registration"])
def test_wrong_schema_lineage_or_precision_rejected(tmp_path, monkeypatch, mutation):
    f = native_fixture(tmp_path, monkeypatch)
    identity = deepcopy(f.identity)
    if mutation == "schema":
        identity["schema"] = "st_world_training_v1"
    elif mutation == "config":
        identity["config"]["execution"]["seed"] += 1
    elif mutation == "parent":
        identity["base_binding"] = "0"*64
    elif mutation == "numeric":
        identity["numeric_execution"]["encoder"] = "BF16_values_cast_to_float32"
        identity["numeric_execution_hash"] = identity_hash(identity["numeric_execution"])
    elif mutation == "scope":
        identity["weights_usable_for_formal_init"] = True
    else:
        identity["registration_sha256"] = "0"*64
    identity["identity"] = identity_hash({k: v for k, v in identity.items() if k != "identity"})
    _json(f.run / "identity.json", identity)
    # Re-sign a self-consistent native fixture as well: parent/precision cases
    # must fail the actual dependency rule, not just an old receipt identity.
    f.native["identity"] = identity["identity"]
    f.receipt["state"]["identity"] = identity["identity"]
    torch.save(f.native, f.folder / "mp_rank_00_model_states.pt")
    for name, info in f.receipt["files"].items():
        info.update(sha256=file_sha256(f.folder / name), bytes=(f.folder / name).stat().st_size)
    f.receipt["receipt_identity"] = identity_hash({k: v for k, v in f.receipt.items() if k != "receipt_identity"})
    _json(f.folder / "COMPLETE.json", f.receipt)
    with pytest.raises(ValueError):
        fc.flow_checkpoint_identity(f.reg, f.tag)


def test_installed_deepspeed_reads_real_master_groups_not_rounded_model_values(tmp_path, monkeypatch):
    f = native_fixture(tmp_path, monkeypatch)
    restored = fc._canonical_native_state(f.folder, f.receipt, f.parent.action_model)
    assert set(restored) == {fc.ACTION_PREFIX+k for k in f.action}
    for name, expected in f.action.items():
        torch.testing.assert_close(restored[fc.ACTION_PREFIX+name], expected, rtol=0, atol=0)
    assert any(not torch.equal(v, f.native["module"][k].float()) for k, v in restored.items())
    encoder = {k: v.clone() for k, v in f.parent.encoder.state_dict().items()}
    audit = fc.restore_action_masters(f.parent.action_model, restored)
    assert audit["changed_action_tensor_count"] > 0 and audit["action_update_l2"] > 0
    for name, value in f.parent.encoder.state_dict().items():
        torch.testing.assert_close(value, encoder[name], atol=0, rtol=0)


@pytest.mark.parametrize("mutation", ["extra_module", "prefix", "master_dtype", "native_step", "frozen", "shape", "buffer"])
def test_native_schema_corruption_rejected(tmp_path, monkeypatch, mutation):
    f = native_fixture(tmp_path, monkeypatch)
    if mutation == "extra_module":
        f.native["module"]["policy.encoder.weight"] = torch.zeros(1)
    elif mutation == "prefix":
        name = next(iter(f.native["module"]))
        f.native["module"][name.removeprefix(fc.ACTION_PREFIX)] = f.native["module"].pop(name)
    elif mutation == "master_dtype":
        f.optimizer["optimizer_state_dict"]["single_partition_of_fp32_groups"][0] = f.optimizer[
            "optimizer_state_dict"]["single_partition_of_fp32_groups"][0].bfloat16()
    elif mutation == "native_step":
        f.native["global_steps"] = 9
    elif mutation == "frozen":
        f.native["frozen_param_shapes"] = {"policy.encoder": [1]}
    elif mutation == "shape":
        f.native["param_shapes"][0]["policy.foreign"] = torch.Size([1])
    else:
        f.native["module"][f.native["buffer_names"][0]].view(-1)[0] += 1
    torch.save(f.native, f.folder / "mp_rank_00_model_states.pt")
    torch.save(f.optimizer, f.folder / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt")
    with pytest.raises(ValueError):
        fc._canonical_native_state(f.folder, f.receipt, f.parent.action_model)


@pytest.mark.parametrize("mutation", ["prefix", "missing", "dtype", "nonfinite", "shape"])
def test_strict_action_load_rejects_invalid_state_without_partial_update(mutation):
    head = original_tiny_head()
    before = {k: v.clone() for k, v in head.state_dict().items()}
    state = {fc.ACTION_PREFIX+k: v.clone() for k, v in before.items()}
    name = next(iter(state))
    if mutation == "prefix":
        state[name.replace(fc.ACTION_PREFIX, "base.action_model.")] = state.pop(name)
    elif mutation == "missing":
        state.pop(name)
    elif mutation == "dtype":
        state[name] = state[name].bfloat16()
    elif mutation == "nonfinite":
        state[name].view(-1)[0] = float("nan")
    else:
        state[name] = state[name].reshape(-1)[:1]
    with pytest.raises(ValueError):
        fc.restore_action_masters(head, state)
    for k, v in head.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def observations(n=2):
    return [ModelInputs({"image": [object(), object(), object()], "state": [[1., 0., 0., 1.]],
                         "lang": "keep lane", "token": "scene_"+str(i)}) for i in range(n)]


def test_loader_preserves_record_parent_freeze_and_real_ten_step_forward(tmp_path, monkeypatch):
    f = native_fixture(tmp_path, monkeypatch)
    _, _, identity_before = fc.flow_checkpoint_identity(f.reg, f.tag)
    load_calls = []
    def load_student(run, tag, identity, **kwargs):
        load_calls.append((run, tag, identity, kwargs))
        return deepcopy(f.parent)
    import tools.foresight.checkpoints
    monkeypatch.setattr(tools.foresight.checkpoints, "load_student", load_student)
    model, identity_after = fc.load_flow_checkpoint(f.reg, f.tag, device="cpu")
    assert identity_before == identity_after and True in f.binding_checks
    assert load_calls[0][3] == {"device": "cpu", "precision": "fp32", "strip": True}
    assert model.base.stripped and model.load_audit["all_parameters_frozen"]
    assert model.load_audit["optimizer_resume_claim"] is False
    model.train(True)
    assert all(not x.training for x in model.modules())
    calls = []
    hook = model.action_model.model.register_forward_hook(lambda *x: calls.append(1))
    noise = torch.randn(2, 8, 4)
    rng = torch.random.get_rng_state().clone()
    actual = model.predict_raw(observations(), initial_noise=noise)
    assert model.base.calls == 1 and len(calls) == 10
    assert torch.equal(torch.random.get_rng_state(), rng)
    hook.remove()
    reference = deepcopy(f.parent).requires_grad_(False).eval()
    reference.action_model.load_state_dict(f.action, strict=True)
    expected = reference.action_model.predict_action(reference.build_planner_condition(
        reference.encode_current([x.legacy_current() for x in observations()])), initial_noise=noise)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32 and not actual.requires_grad
    with pytest.raises(ValueError, match="permits only"):
        model.predict_raw([{**observations(1)[0].current, "GT": torch.zeros(8, 4)}], initial_noise=noise[:1])
    with pytest.raises(ValueError, match="FP32"):
        model.predict_raw(observations(), initial_noise=noise.bfloat16())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="AMP"):
            model.predict_raw(observations(), initial_noise=noise)


def test_native_changed_while_loading_parent_is_rechecked(tmp_path, monkeypatch):
    f = native_fixture(tmp_path, monkeypatch)
    import tools.foresight.checkpoints
    def load_student(*a, **kw):
        (f.folder / "stale_model_states.pt").write_bytes(b"changed during parent load")
        return deepcopy(f.parent)
    monkeypatch.setattr(tools.foresight.checkpoints, "load_student", load_student)
    with pytest.raises(ValueError, match="inventory"):
        fc.load_flow_checkpoint(f.reg, f.tag, device="cpu")


def test_flow_source_contract_covers_module_globals(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "INFERENCE_SOURCES", ("source.py",))
    (tmp_path / "source.py").write_text("SCALE = 1\ndef predict(x):\n    return SCALE*x\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "source fixture"], cwd=tmp_path, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    fc.inference_source_contract(tmp_path, sha)
    (tmp_path / "source.py").write_text("SCALE = 2\ndef predict(x):\n    return SCALE*x\n")
    with pytest.raises(ValueError, match="compatibility"):
        fc.inference_source_contract(tmp_path, sha)
