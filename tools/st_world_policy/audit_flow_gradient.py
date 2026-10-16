"""Read-only C1 replay and policy-only gradient evidence; no training/scoring.

Uses the exact saved BF16 execution policy, original current-only encoder path,
saved generation noise, coupled MC50 samples and completed official feedback.
It never starts DeepSpeed, constructs an optimizer, calls BC, changes a training
checkpoint, or obtains new official scores. Only a private action copy gets
gradients. GPU execution is a bounded, explicitly dispatched audit.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import random
import signal
import subprocess
import time
import traceback

import numpy as np
import torch

from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
from starVLA.dataloader.st_world_schema import ModelInputs
from starVLA.dataloader.st_world_targets import atomic_json
from starVLA.model.modules.flow_posttraining import (
    DiTFlowPolicy, RolloutBatch, CoupledFPOBatch, FeedbackBatch,
    group_centered_advantages, fpo_surrogate, module_hash, tensor_hash)
from starVLA.model.modules.st_world.consequence import ConsequenceRequest
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256, initialization_seed
from tools.ddpolicy_vehicle.training_state import epoch_batches
from tools.st_world_policy.flow_checkpoints import flow_checkpoint_identity, ACTION_PREFIX
from tools.st_world_policy.flow_training_config import NO_TEACHER, target_contract
from tools.st_world_policy.train_flow_policy import (
    encode_conditions, frozen_encoder_hash, score_requests, feedback_from_records, MatchedOldMicrobatches)
from tools.st_world_policy.training_config import environment_record


SCHEMA = "st_flow_policy_gradient_audit_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def verified_artifact(record):
    require(set(record) == {"path", "sha256"}, "Unknown/missing artifact fields")
    path = Path(record["path"])
    require(path.is_absolute() and file_sha256(path) == record["sha256"], "Artifact content/path mismatch: " + str(path))
    return path


def replay_source_contract(source_sha):
    """Verify all FPO computation, plus the precise reused training functions."""
    root = Path(__file__).resolve().parents[2]
    contract = {}
    for path in ("starVLA/model/modules/flow_posttraining/fpo_objective.py",
                 "starVLA/model/modules/flow_posttraining/policy_snapshot.py",
                 "starVLA/model/modules/flow_posttraining/rollout_contract.py",
                 "tools/ddpolicy_vehicle/training_state.py"):
        prior = subprocess.check_output(["git", "show", source_sha + ":" + path], cwd=root).decode()
        canonical = ast.dump(ast.parse(prior), include_attributes=False)
        require(canonical == ast.dump(ast.parse((root / path).read_text()), include_attributes=False),
                "FPO replay source changed: " + path)
        contract[path] = identity_hash(canonical)
    path = "tools/st_world_policy/train_flow_policy.py"
    prior = ast.parse(subprocess.check_output(["git", "show", source_sha + ":" + path], cwd=root).decode())
    current = ast.parse((root / path).read_text())
    for name in ("encode_conditions", "frozen_encoder_hash", "score_requests", "_validate_scored",
                 "feedback_from_records", "MatchedOldMicrobatches"):
        a = [node for node in prior.body if getattr(node, "name", None) == name]
        b = [node for node in current.body if getattr(node, "name", None) == name]
        require(len(a) == len(b) == 1 and ast.dump(a[0], include_attributes=False) == ast.dump(b[0], include_attributes=False),
                "Audited training function changed: " + name)
        contract[path + "::" + name] = identity_hash(ast.dump(a[0], include_attributes=False))
    return contract


def next_scene_entries(manifest, state, execution):
    """Resolve the next consumed batch from the saved optimizer-boundary cursor."""
    require(execution["world_size"] == 1, "This audit currently requires a single-rank source run")
    require(type(state["epoch"]) is int and type(state["offset"]) is int and state["epoch"] >= 0,
            "Invalid native sampler cursor")
    epoch, offset = state["epoch"], state["offset"]
    batches = epoch_batches(len(manifest["entries"]), execution["global_batch"], execution["seed"], epoch)
    require(0 <= offset <= len(batches), "Sampler offset exceeds original epoch")
    if offset == len(batches):
        epoch, offset = epoch + 1, 0
        batches = epoch_batches(len(manifest["entries"]), execution["global_batch"], execution["seed"], epoch)
    ids = batches[offset]
    return [manifest["entries"][index] for index in ids], {"epoch": epoch, "offset": offset, "manifest_indices": ids}


def load_feedback_payload(receipt_path, *, numeric_hash, step, binding_hash, entries, fpo_config):
    receipt_path = Path(receipt_path)
    receipt = read_json(receipt_path)
    require(set(receipt) == {"identity", "payload"}, "Feedback payload receipt schema mismatch")
    path = verified_artifact(receipt["payload"])
    require(path.resolve() == receipt_path.with_name("rank_0.pt").resolve(), "Feedback payload escaped its boundary")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(set(payload) == {"schema", "identity", "rollout", "comparison", "feedback"}
            and payload["schema"] == "st_flow_update_feedback_v1" and payload["identity"] == receipt["identity"],
            "Feedback payload/receipt mismatch")
    rollout = RolloutBatch.from_state_dict(payload["rollout"])
    comparison = CoupledFPOBatch.from_state_dict(payload["comparison"], rollout)
    feedback = FeedbackBatch.from_state_dict(payload["feedback"], rollout)
    feedback.validate(rollout)
    require(rollout.policy.version == rollout.policy.optimizer_step == step
            and rollout.policy.base_binding_hash == binding_hash, "Old policy is not the specified native boundary")
    require(rollout.scene_ids == tuple(e["scene_id"] for e in entries)
            and rollout.observation_hashes == tuple(e["observation_hash"] for e in entries),
            "Payload sample IDs/order/observations differ from native sampler cursor")
    require(comparison.config.to_dict() == fpo_config and comparison.config.mc_samples == 50,
            "Payload changed the registered MC/objective definition")
    expected = {"rollout_id": rollout.rollout_id, "comparison_id": comparison.comparison_id,
        "numeric_execution_hash": numeric_hash, "old_policy": vars(rollout.policy),
        "feedback": {"protocol": feedback.score_protocol_hash, "contexts": list(feedback.context_hashes),
                     "reward_kind": feedback.reward_kind, "rewards": feedback.rewards.tolist(),
                     "statuses": list(feedback.statuses)}}
    require(receipt["identity"] == expected, "Feedback identity fields do not describe actual payload")
    require(feedback.reward_kind == "C1-PDMS", "This narrow audit accepts actual C1 PDMS feedback only")
    return rollout, comparison, feedback, {"path": str(receipt_path), "sha256": file_sha256(receipt_path),
                                         "payload": receipt["payload"]}


def _read_completed_official_identities(run, identities, *, run_identity, cfg):
    """Read existing ledger only. No service object or admission can be created."""
    root = Path(run) / "official_feedback"
    require(read_json(root / "IDENTITY.json") == {"schema": "st_flow_feedback_ledger_v1", "identity": run_identity,
        "max_calls": cfg["max_calls"], "retry_attempts": cfg["retry_attempts"]}, "Official ledger lineage changed")
    require(bool(identities) and all(i["policy_version"] == identities[0]["policy_version"] for i in identities),
            "Mixed/empty official policy request")
    key = identity_hash(identities)
    folder = root / key
    version = identities[0]["policy_version"]
    require(read_json(folder / "REQUESTS.json") == {"policy_version": version, "requests": identities},
            "Official request content/order/version mismatch")
    results, receipts = [None] * len(identities), []
    for attempt in range(cfg["retry_attempts"] + 1):
        part = folder / f"attempt_{attempt:02d}"
        if not (part / "COMPLETE.json").is_file():
            continue
        complete = read_json(part / "COMPLETE.json")
        require(set(complete) == {"result", "admission_sha256"}
                and complete["admission_sha256"] == file_sha256(part / "ADMITTED.json"), "Official admission changed")
        admission = read_json(part / "ADMITTED.json")
        indices = admission["indices"]
        require(admission["request_hash"] == key and admission["policy_version"] == version
                and admission["calls"] == len(indices) and len(set(indices)) == len(indices)
                and all(type(i) is int and 0 <= i < len(identities) for i in indices), "Official admission indices/version changed")
        path = verified_artifact(complete["result"])
        require(path.resolve() == (part / "RESULTS.pt").resolve(), "Official result escaped its completed attempt")
        values = torch.load(path, map_location="cpu", weights_only=False)
        require(len(values) == len(indices), "Incomplete official result inventory")
        for i, value in zip(indices, values):
            require(value.get("identity") == identities[i], "Foreign candidate in official result")
            require(results[i] is None or results[i].get("valid") is not True, "Completed valid score was overwritten")
            results[i] = value
        receipts.append({"path": str(part / "COMPLETE.json"), "sha256": file_sha256(part / "COMPLETE.json")})
        if all(value is not None and value.get("valid") for value in results):
            break
    require(all(value is not None and value.get("valid") for value in results), "Official feedback incomplete; audit cannot score or retry")
    return results, receipts


def read_completed_official_feedback(run, requests, *, run_identity, cfg):
    return _read_completed_official_identities(run, [r.identity() for r in requests], run_identity=run_identity, cfg=cfg)


def read_original_device_feedback(run, expected_requests, *, run_identity, cfg):
    """Read the original GPU-converted physical values without CPU relabeling.

    CPU/GPU atan2 may differ in last bits. Only physical conversion verification
    is deferred to GPU execution; every other request field must match now.
    Completed result checksums bind the actual original physical trajectory.
    """
    expected = [r.identity() for r in expected_requests]
    without_physical_hash = lambda values: [{k: v for k, v in x.items() if k != "trajectory_sha256"} for x in values]
    matched = []
    for path in sorted((Path(run) / "official_feedback").glob("*/REQUESTS.json")):
        saved = read_json(path)
        if saved.get("policy_version") != expected[0]["policy_version"]:
            continue
        identities = saved.get("requests", [])
        if without_physical_hash(identities) == without_physical_hash(expected):
            require(path.parent.name == identity_hash(identities), "Original official request directory hash mismatch")
            matched.append(identities)
    require(len(matched) == 1, "No unique original-device official feedback for this exact rollout/policy/context")
    identities = matched[0]
    records, receipts = _read_completed_official_identities(run, identities, run_identity=run_identity, cfg=cfg)
    requests = [ConsequenceRequest(**identity, trajectory=record["requested_trajectory"], observe=True)
                for identity, record in zip(identities, records)]
    require([r.identity() for r in requests] == identities, "Original physical request reconstruction changed identity")
    return requests, records, receipts


def preflight(registration_path, tag, *, max_scenes):
    cfg, binding, native = flow_checkpoint_identity(registration_path, tag)
    require(cfg["cell"] == "C1", "Only C1 policy-gradient evidence is supported")
    require(type(max_scenes) is int and 1 <= max_scenes <= 32, "Audit scene cap must be 1..32")
    registration = read_json(registration_path)
    artifacts = registration["artifacts"]
    require(read_json(verified_artifact(artifacts["environment"])) == environment_record(), "Training/audit software environment differs")
    require(read_json(verified_artifact(artifacts["teacher_manifest"])) == NO_TEACHER, "Unexpected external teacher")
    require(read_json(verified_artifact(artifacts["target_contract"])) == target_contract(cfg), "Registered target contract mismatch")
    manifest = read_json(verified_artifact(artifacts["data_manifest"]))
    require(manifest.get("schema") == "st_flow_training_manifest_v1" and manifest.get("split") == "train"
            and manifest["scope"] == cfg["execution"]["scope"]
            and manifest["root"] == str(Path(cfg["data"]["root"]).resolve())
            and manifest["identity"] == identity_hash({k: v for k, v in manifest.items() if k != "identity"})
            and manifest["base_binding_hash"] == binding["binding_sha256"]
            and manifest["score_protocol_hash"] == cfg["feedback"]["score_protocol_hash"], "Invalid registered training manifest")
    run = Path(native["run"])
    identity = read_json(run / "identity.json")
    require(identity["data"] == manifest["identity"], "Training run uses another data manifest")
    receipt = read_json(run / "checkpoints" / tag / "COMPLETE.json")
    entries, cursor = next_scene_entries(manifest, receipt["state"], cfg["execution"])
    require(len(entries) <= max_scenes, "The complete saved batch exceeds the audit cap; do not truncate replay")
    require(manifest["data_identity"] == binding["parent_identity"]["data"]
            and manifest["ego_identity"] == binding["parent_identity"]["ego"], "Parent exposure/observation lineage changed")
    preprocessing = verified_artifact(manifest["current_preprocessing"])
    require(file_sha256(Path(__file__).resolve().parents[2] / "starVLA/dataloader/foresight_dataset.py") == file_sha256(preprocessing),
            "Current observation preprocessing changed")
    for entry in entries:
        require(Path(entry["current"]["path"]).resolve() ==
                (Path(cfg["data"]["root"]) / "current" / (entry["scene_id"] + ".json")).resolve(),
                "Manifest current input is not the original dataset record")
        for artifact in (entry["current"], *entry["images"]):
            verified_artifact(artifact)
        require(entry["observation_hash"] == identity_hash({"current": entry["current"], "images": entry["images"],
            "preprocess": manifest["current_preprocessing"]["sha256"]}), "Observation content hash changed")
    step = native["completed"]
    rollout, comparison, feedback, payload_receipt = load_feedback_payload(
        run / "rollouts" / f"boundary_{step:06d}" / "rank_0_COMPLETE.json",
        numeric_hash=identity["numeric_execution_hash"], step=step, binding_hash=binding["binding_sha256"],
        entries=entries, fpo_config=cfg["fpo"])
    expected_requests = score_requests(rollout, entries, binding, cfg["feedback"], identity["numeric_execution_hash"])
    requests, results, official_receipts = read_original_device_feedback(run, expected_requests,
        run_identity=native["run_identity"], cfg=cfg["feedback"])
    verified = feedback_from_records(rollout, requests, results, "C1", cfg["feedback"]["score_protocol_hash"])
    require(verified.state_dict().keys() == feedback.state_dict().keys()
            and all(torch.equal(verified.state_dict()[k], v) if torch.is_tensor(v) else verified.state_dict()[k] == v
                    for k, v in feedback.state_dict().items()), "Saved reward differs from actual completed official feedback")
    sources = replay_source_contract(registration["source_sha"])
    return {"cfg": cfg, "binding": binding, "native": native, "receipt": receipt, "manifest": manifest,
            "entries": entries, "cursor": cursor, "rollout": rollout, "comparison": comparison, "feedback": feedback,
            "payload_receipt": payload_receipt, "official_receipts": official_receipts,
            "original_request_identities": [r.identity() for r in requests],
            "replay_sources": sources, "frozen_encoder_hash": receipt["state"]["frozen_encoder_hash"]}


def current_inputs(data):
    """No training-dataset or GT/metric-cache reads are reachable here."""
    dataset = ForesightCurrentDataset(data["cfg"]["data"]["root"])
    require(dataset.identity == data["manifest"]["data_identity"] and dataset.identity["split"] == "train", "Current-only dataset identity differs")
    observations = []
    for row in data["entries"]:
        require(dataset.index[row["dataset_index"]] == {"token": row["scene_id"], "log": row["source_log"]}, "Current dataset ID/log changed")
        require(Path(row["current"]["path"]).resolve() ==
                (dataset.root / "current" / (row["scene_id"] + ".json")).resolve(), "Wrong current-only dataset record")
        record = read_json(row["current"]["path"])
        require([str(Path(p).resolve()) for p in record["image_paths"]] == [x["path"] for x in row["images"]], "Image view order changed")
        item = ModelInputs(dataset[row["dataset_index"]])
        require(item.current["token"] == row["scene_id"] and item.history is None, "Wrong current scene/history")
        observations.append(item.legacy_current())
    return observations


def restore_native_execution(action_head, native_model, receipt, expected_policy_hash):
    require(native_model.get("global_steps") == receipt["state"]["completed"]
            and all(native_model.get(k, "missing") == v for k, v in receipt["state"].items()), "Native model client state mismatch")
    before = action_head.state_dict()
    values = native_model["module"]
    require(set(values) == {ACTION_PREFIX + name for name in before}, "Wrong native action namespace/inventory")
    names = [name for group in native_model["param_shapes"] for name in group]
    shapes = {name: shape for group in native_model["param_shapes"] for name, shape in group.items()}
    buffers = {ACTION_PREFIX + name for name, _ in action_head.named_buffers() if name in before}
    require(len(set(names)) == len(names) and set(names) == {ACTION_PREFIX + name for name, _ in action_head.named_parameters()}
            and set(native_model["buffer_names"]) == buffers
            and not native_model.get("frozen_param_shapes") and not native_model.get("frozen_param_fragments")
            and not native_model.get("shared_params"), "Native execution parameter schema differs")
    require(all(tuple(shapes[ACTION_PREFIX + name]) == tuple(value.shape) for name, value in action_head.named_parameters()),
            "Native execution parameter metadata shape differs")
    for name, original in before.items():
        saved = values[ACTION_PREFIX + name]
        expected_dtype = torch.bfloat16 if original.is_floating_point() else original.dtype
        require(saved.shape == original.shape and saved.dtype == expected_dtype and torch.isfinite(saved).all(),
                "Native action execution shape/dtype/finite mismatch: " + name)
    action_head.bfloat16().load_state_dict({k[len(ACTION_PREFIX):]: v for k, v in values.items()}, strict=True)
    policy = DiTFlowPolicy(action_head).requires_grad_(False).eval()
    require(module_hash(policy) == expected_policy_hash, "Native BF16 execution is not the actual old policy")
    return policy


class ReadOnlyPolicy:
    """Expose verified old identity without snapshot allocation or DS hooks."""
    def __init__(self, policy, identity):
        self.policy, self.identity = policy, identity

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.policy(*args, **kwargs)


@torch.no_grad()
def replay_endpoints(policy, condition, initial_noise):
    require(not any(p.requires_grad for p in policy.parameters()) and not any(m.training for m in policy.modules()),
            "Replay policy must remain frozen eval")
    projected = policy.project_condition(condition)
    memory = projected[:, None].expand(-1, 4, -1, -1).reshape(-1, *projected.shape[1:])
    state = initial_noise.flatten(0, 1).clone()
    for index in range(10):
        buckets = torch.full((len(state),), int((index / 10.) * policy.action_head.num_timestep_buckets),
                             device=state.device, dtype=torch.long)
        state = state + .1 * policy.action_head.forward_onestep(memory, state, buckets)[0]
    return state.reshape_as(initial_noise)


def exact_tensor(actual, expected, label):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype, label + " shape/dtype differs")
    actual_hash, expected_hash = tensor_hash(actual), tensor_hash(expected)
    if actual_hash != expected_hash:
        delta = actual.detach().float().cpu() - expected.detach().float().cpu()
        raise ValueError(f"Strict replay failed: {label}; max_abs={delta.abs().max().item()}, "
                         f"actual_sha256={actual_hash}, expected_sha256={expected_hash}; no tolerance relaxation")
    return {"sha256": actual_hash, "elements": actual.numel(), "bit_exact": True}


def rng_identity(device):
    state = np.random.get_state()
    result = {"python": identity_hash(repr(random.getstate())),
              "numpy": identity_hash([state[0], state[1].tolist(), *state[2:]]),
              "torch_cpu": tensor_hash(torch.get_rng_state())}
    if torch.device(device).type == "cuda":
        result["torch_cuda"] = tensor_hash(torch.cuda.get_rng_state(device))
    return result


def gradient_statistics(policy):
    per_parameter, groups = {}, {name: {"elements": 0, "gradient_elements": 0, "nonzero": 0, "squared_l2": 0., "max_abs": 0.}
                                 for name in ("DiT", "qwen_proj", "other_action")}
    for name, parameter in policy.named_parameters():
        category = "qwen_proj" if name.startswith("action_head.qwen_proj.") else "DiT" if name.startswith("action_head.model.") else "other_action"
        group = groups[category]
        group["elements"] += parameter.numel()
        record = {"elements": parameter.numel(), "dtype": str(parameter.dtype), "gradient_present": parameter.grad is not None,
                  "gradient_dtype": None, "nonzero": 0, "squared_l2": 0., "max_abs": 0.}
        if parameter.grad is not None:
            record["gradient_dtype"] = str(parameter.grad.dtype)
            group["gradient_elements"] += parameter.grad.numel()
            for chunk in parameter.grad.detach().view(-1).split(1 << 20):
                values = chunk.double().cpu()
                require(bool(torch.isfinite(values).all()), "Nonfinite policy-only gradient: " + name)
                record["nonzero"] += int(torch.count_nonzero(values))
                record["squared_l2"] += float(values.square().sum())
                record["max_abs"] = max(record["max_abs"], float(values.abs().max()))
        record["l2"] = record["squared_l2"] ** .5
        per_parameter[name] = record
        group["nonzero"] += record["nonzero"]
        group["squared_l2"] += record["squared_l2"]
        group["max_abs"] = max(group["max_abs"], record["max_abs"])
    for value in groups.values():
        value["l2"] = value["squared_l2"] ** .5
    return {"groups": groups, "parameters": per_parameter, "all_finite": True,
            "all_parameters_reached": all(x["gradient_present"] for x in per_parameter.values()),
            "note": "Actual execution-parameter gradient dtype; no claim of FP32 backward"}


def audit_policy_gradients(old, condition, rollout, comparison, feedback, *, micro_batch, mc_chunk,
                           checkpoint_velocity, autocast_factory=nullcontext):
    """Pure FPO graph on a private action copy, with a genuine zero-A control."""
    require(not any(p.requires_grad for p in old.parameters()) and not any(m.training for m in old.modules()), "Old/reference policy not frozen")
    require(not condition.requires_grad and tuple(tensor_hash(x) for x in condition) == rollout.condition_hashes,
            "Frozen condition changed or has gradients")
    comparison.validate(rollout)
    feedback.validate(rollout)
    require(module_hash(old) == rollout.policy.state_hash, "Old policy hash differs")
    before_rng = rng_identity(condition.device)
    policy = deepcopy(old).requires_grad_(True).eval()
    require(all(not hasattr(p, "_hp_mapping") for p in policy.parameters()), "Audit copy contains DeepSpeed hooks/mappings")
    require(not ({p.untyped_storage().data_ptr() for p in old.parameters()} &
                 {p.untyped_storage().data_ptr() for p in policy.parameters()}), "Audit policy aliases frozen old")
    advantages = group_centered_advantages(feedback.rewards, feedback.valid).flatten()
    actions, valid = rollout.raw_actions.flatten(0, 1), feedback.valid.flatten()
    reports = {}
    for kind, advantage in (("actual_feedback", advantages), ("zero_advantage_control", torch.zeros_like(advantages))):
        policy.zero_grad(set_to_none=True)
        total, current_hashes = 0., []
        for at in range(0, len(condition), micro_batch):
            end = min(at + micro_batch, len(condition))
            sl = slice(4 * at, 4 * end)
            with autocast_factory():
                current = policy(condition[at:end].repeat_interleave(4, 0), actions[sl], comparison.samples.slice(sl),
                    mc_chunk=mc_chunk, checkpoint_velocity=checkpoint_velocity)
                current_hashes.append(exact_tensor(current, comparison.old_cfm[sl], "current/old CFM"))
                loss, diagnostics = fpo_surrogate(comparison.old_cfm[sl], current, advantage[sl], valid[sl], comparison.config,
                                                  global_count=len(condition) * 4, world_size=1)
            require(torch.isfinite(loss), "Nonfinite pure policy loss")
            require(torch.equal(diagnostics["ratio"], torch.ones_like(diagnostics["ratio"])), "Old=current ratio must be exactly one")
            loss.backward()
            total += float(loss.detach())
        reports[kind] = {"loss": total, "current_CFM": current_hashes, **gradient_statistics(policy)}
        require(reports[kind]["all_parameters_reached"], "Pure policy graph omitted an intended action parameter")
    require(all(g["nonzero"] == 0 for g in reports["zero_advantage_control"]["groups"].values()), "Zero advantage created policy gradients")
    signal_present = bool(torch.count_nonzero(advantages))
    if signal_present:
        require(all(reports["actual_feedback"]["groups"][k]["nonzero"] > 0 for k in ("DiT", "qwen_proj")),
                "Nonzero real advantage did not reach both DiT and qwen_proj")
    else:
        require(all(g["nonzero"] == 0 for g in reports["actual_feedback"]["groups"].values()), "Zero real signal created policy gradients")
    require(module_hash(old) == module_hash(policy) == rollout.policy.state_hash, "Audit changed policy weights/buffers")
    require(all(p.grad is None for p in old.parameters()), "Frozen reference accumulated gradients")
    require(rng_identity(condition.device) == before_rng, "Eval policy audit consumed training RNG")
    return {"status": "POLICY_GRADIENT_REACHED_DIT_AND_PROJECTION" if signal_present else "NO_ADVANTAGE_SIGNAL",
            "actual_nonzero_advantages": int(torch.count_nonzero(advantages)), "reports": reports,
            "BC_executed": False, "optimizer_steps": 0, "old_reference_unchanged": True,
            "policy_copy_weights_unchanged": True, "RNG_unchanged": True}


def execute(data, *, device):
    require(torch.device(device).type == "cuda" and torch.cuda.device_count() == 1, "Real audit requires exactly one explicitly allocated visible GPU")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg, binding = data["cfg"], data["binding"]
    torch.use_deterministic_algorithms(cfg["runtime"]["deterministic"])
    from tools.foresight.checkpoints import load_student
    with initialization_seed(cfg["execution"]["seed"]):
        encoder = load_student(binding["run"], binding["tag"], binding["parent_identity"], device="cpu", precision="fp32", strip=False)
    action = encoder.action_model
    del encoder.action_model
    native_path = Path(data["native"]["run"]) / "checkpoints" / data["native"]["tag"] / "mp_rank_00_model_states.pt"
    require(file_sha256(native_path) == data["receipt"]["files"][native_path.name]["sha256"], "Native model changed after preflight")
    native_model = torch.load(native_path, map_location="cpu", mmap=True, weights_only=False)
    old = restore_native_execution(action, native_model, data["receipt"], data["rollout"].policy.state_hash)
    del native_model
    old.action_head.model.gradient_checkpointing = cfg["runtime"]["gradient_checkpointing"]
    old.to(device)
    encoder.strip_auxiliary_heads()
    encoder.requires_grad_(False).eval()
    encoder.inference_fp32 = False
    encoder.qwen_vl_interface.model.model.language_model.gradient_checkpointing_disable()
    from tools.st_world_policy.training_config import official_runtime_policy
    official = official_runtime_policy(cfg)
    if official is not None:
        from tools.st_world_policy.official_runtime import apply_official_model_runtime
        apply_official_model_runtime(vlm=encoder.qwen_vl_interface.model,
            language=encoder.qwen_vl_interface.model.model.language_model,
            action_transformer=action.model, policy=official, training_copy=True)
        encoder.qwen_vl_interface.model.model.language_model.gradient_checkpointing_disable()
    encoder.bfloat16().to(device)
    require(frozen_encoder_hash(encoder) == data["frozen_encoder_hash"], "Original frozen training encoder identity changed")
    before_rng = rng_identity(device)
    observations = current_inputs(data)
    calls = {"visual": 0, "language": 0}
    def hook(name):
        def count(*_):
            calls[name] += 1
        return count
    handles = [encoder.qwen_vl_interface.model.model.visual.register_forward_hook(hook("visual")),
               encoder.qwen_vl_interface.model.model.language_model.register_forward_hook(hook("language"))]
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition = encode_conditions(encoder, observations, cfg["execution"]["micro_batch"])
    finally:
        for handle in handles:
            handle.remove()
    require(calls["visual"] == calls["language"] == (len(observations) + cfg["execution"]["micro_batch"] - 1) // cfg["execution"]["micro_batch"],
            "Encoder call count differs from original source microbatches")
    require(tuple(tensor_hash(x) for x in condition) == data["rollout"].condition_hashes, "Online current-only condition hash differs; no replay inference permitted")
    require(frozen_encoder_hash(encoder) == data["frozen_encoder_hash"] and all(p.grad is None for p in encoder.parameters()),
            "Frozen encoder changed during current-only reconstruction")
    del encoder
    torch.cuda.empty_cache()
    rollout = RolloutBatch.from_state_dict(data["rollout"].state_dict(), device=device)
    comparison = CoupledFPOBatch.from_state_dict(data["comparison"].state_dict(), rollout, device=device)
    feedback = FeedbackBatch.from_state_dict(data["feedback"].state_dict(), rollout, device=device)
    # The trainer converts detached raw endpoints outside its autocast region.
    with torch.no_grad():
        actual_requests = score_requests(rollout, data["entries"], binding, cfg["feedback"],
                                        data["native"]["training_numeric_execution_hash"])
        require([r.identity() for r in actual_requests] == data["original_request_identities"],
                "Original-device raw-to-physical conversion differs from saved official trajectories")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        endpoints = replay_endpoints(old, condition, rollout.initial_noise)
        endpoint_result = exact_tensor(endpoints, rollout.raw_actions, "ten-Euler actual endpoints")
        wrapper = MatchedOldMicrobatches(ReadOnlyPolicy(old, rollout.policy), cfg["execution"]["micro_batch"])
        old_cfm = wrapper(condition.repeat_interleave(4, 0), rollout.raw_actions.flatten(0, 1), comparison.samples,
                          mc_chunk=cfg["runtime"]["mc_chunk"])
        old_result = exact_tensor(old_cfm, comparison.old_cfm, "saved old CFM50")
    del endpoints, old_cfm, wrapper
    gradient = audit_policy_gradients(old, condition, rollout, comparison, feedback,
        micro_batch=cfg["execution"]["micro_batch"], mc_chunk=cfg["runtime"]["mc_chunk"],
        checkpoint_velocity=cfg["runtime"]["checkpoint_velocity"],
        autocast_factory=lambda: torch.autocast("cuda", dtype=torch.bfloat16))
    require(rng_identity(device) == before_rng, "Current replay/audit consumed RNG")
    return {"status": gradient["status"], "replay": {"condition_hashes": list(rollout.condition_hashes),
                "endpoints": endpoint_result, "old_CFM": old_result, "tolerance": "bit_exact_or_refuse",
                "original_GPU_physical_request_identities_exact": True},
            "policy_gradient": gradient, "encoder_calls": calls, "frozen_encoder_unchanged": True,
            "RNG_unchanged": True, "device": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "new_score_calls": 0, "training_steps": 0, "GT_read": False, "metric_cache_read": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--tag", required=True, help="Exact complete native checkpoint at the old-policy boundary")
    parser.add_argument("--output", required=True, help="New isolated audit directory")
    parser.add_argument("--max-scenes", type=int, default=4)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true", help="Verify real inputs/receipts without loading a model or CUDA")
    args = parser.parse_args(argv)
    if not 1 <= args.max_seconds <= 900 or not 1 <= args.max_scenes <= 32:
        parser.error("Audit limits are at most 900 seconds and 32 original scenes")
    output = Path(args.output).resolve()
    registration = read_json(args.registration)
    cfg = read_json(verified_artifact(registration["artifacts"]["config"]))
    run = (Path(cfg["runtime"]["output_root"]) / registration["run_id"]).resolve()
    if output == run or run in output.parents:
        parser.error("Audit output must be outside the existing training run")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    started = time.perf_counter()
    def timeout(*_):
        raise TimeoutError("Bounded audit time exhausted")
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    report = {"schema": SCHEMA, "command": vars(args), "training_steps": 0, "new_score_calls": 0,
              "formal_approval": False, "audit_source_sha256": file_sha256(Path(__file__)),
              "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]).decode().strip()}
    try:
        data = preflight(args.registration, args.tag, max_scenes=args.max_scenes)
        report.update(native_checkpoint=data["native"], payload_receipt=data["payload_receipt"],
                      official_receipts=data["official_receipts"], cursor=data["cursor"],
                      scene_ids=[e["scene_id"] for e in data["entries"]], replay_sources=data["replay_sources"],
                      audit_action_source="saved_native_BF16_execution_values_not_FP32_standalone",
                      audit_encoder_source="bound_original_S0_frozen_BF16_training_execution",
                      original_request_identity_hash=identity_hash(data["original_request_identities"]))
        if args.preflight_only:
            report.update(status="PREFLIGHT_VALIDATED", real_gradient_test="NOT_EXECUTED",
                          original_GPU_physical_conversion="PENDING_DEVICE_REPLAY")
        else:
            report.update(execute(data, device=args.device))
        code = 0
    except Exception as exc:
        report.update(status="AUDIT_REJECTED", error={"type": type(exc).__name__, "message": str(exc)}, traceback=traceback.format_exc())
        code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(output / "RESULT.json", report, immutable=True)
    print(json.dumps({"status": report["status"], "result": str(output / "RESULT.json"), "elapsed_seconds": report["elapsed_seconds"]}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
