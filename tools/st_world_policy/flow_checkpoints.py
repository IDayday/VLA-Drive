"""Strict action-only native Flow checkpoints -> original FP32 S0 deployment.

The frozen encoder comes from the bound S0's real FP32 masters. Only the exact
``policy.action_head.*`` namespace is reconstructed from the Flow ZeRO2 masters.
The BF16 training encoder, saved BF16 execution weights, feedback ledger and GT
are not deployment inputs. This reader is not an optimizer/resume converter.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

import torch
from torch import nn

from starVLA.dataloader.st_world_schema import ModelInputs
from starVLA.model.modules.flow_posttraining.policy_snapshot import tensor_hash
from starVLA.model.modules.vehicle_joint.initialization import (
    file_sha256, identity_hash, initialization_seed)
from tools.st_world_policy.base_binding import read_binding
from tools.st_world_policy.flow_training_config import validate_config
from tools.st_world_policy.fp32_master_init import SCHEMA as MASTER_INIT_SCHEMA
from tools.st_world_policy.training_state import verify_checkpoint


ACTION_PREFIX = "policy.action_head."
INFERENCE_CONTRACT = {
    "precision": "FP32_master_eval_TF32_off", "flow_steps": 10,
    "scene_seed_protocol": "foresight-action-v1", "raw": "8x4_xy_sin_cos",
    "physical": "8x3_xy_yaw", "dt_seconds": .5, "frame": "ego_t0_rear_axle"}

# Full modules, including imports/globals/helpers, not selected method bodies.
# No exemption changes the already-bound original action or input function.
INFERENCE_SOURCES = (
    "tools/foresight/checkpoints.py",
    "starVLA/dataloader/st_world_schema.py", "starVLA/dataloader/foresight_dataset.py",
    "starVLA/cache/navsim_feature_cache.py",
    "starVLA/model/framework/__init__.py", "starVLA/model/framework/base_framework.py",
    "starVLA/model/framework/share_tools.py", "starVLA/model/framework/QwenOFT.py",
    "starVLA/model/framework/DDPForesight.py", "starVLA/model/framework/ddp_full_foresight.py",
    "starVLA/model/framework/ddp_action_video_foresight.py",
    "starVLA/model/modules/foresight/config.py", "starVLA/model/modules/foresight/tokens.py",
    "starVLA/model/modules/foresight/queries.py",
    "starVLA/model/modules/action_model/GR00T_ActionHeader.py",
    "starVLA/model/modules/action_model/flow_matching_head/action_encoder.py",
    "starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py",
    "starVLA/model/modules/vlm/__init__.py", "starVLA/model/modules/vlm/QWen3.py",
    "starVLA/model/modules/vlm/qwen3_vl/modeling_qwen3_vl.py",
    "starVLA/model/modules/vlm/qwen3_vl/configuration_qwen3_vl.py",
    "starVLA/training/trainer_utils/trainer_tools.py")


def inference_source_contract(repo, source_sha):
    repo = Path(repo)
    result = {}
    for name in INFERENCE_SOURCES:
        original = subprocess.check_output(["git", "show", source_sha + ":" + name], cwd=repo).decode()
        before = ast.dump(ast.parse(original), include_attributes=False)
        after = ast.dump(ast.parse((repo / name).read_text()), include_attributes=False)
        if before != after:
            raise ValueError("Flow deployment source changed without compatibility review: " + name)
        result[name] = hashlib.sha256(after.encode()).hexdigest()
    return result


def _artifact(registration, name):
    artifact = registration["artifacts"][name]
    path = Path(artifact["path"])
    if not path.is_absolute() or file_sha256(path) != artifact["sha256"]:
        raise ValueError("Registered artifact changed: " + name)
    return path


def flow_checkpoint_identity(registration_path, tag):
    """Read identity without loading GT, metric contexts, feedback or a model."""
    registration_path = Path(registration_path)
    registration = json.loads(registration_path.read_text())
    if (registration.get("schema") != "st_world_run_v1"
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", registration.get("run_id", ""))
            or not re.fullmatch(r"[0-9a-f]{40}", registration.get("source_sha", ""))):
        raise ValueError("Foreign Flow run registration")
    cfg = json.loads(_artifact(registration, "config").read_text())
    validate_config(cfg)
    if cfg["execution"] != registration["execution"]:
        raise ValueError("Registered Flow execution differs from resolved config")
    run = Path(cfg["runtime"]["output_root"]) / registration["run_id"]
    identity = json.loads((run / "identity.json").read_text())
    body = {k: v for k, v in identity.items() if k != "identity"}
    if (identity.get("schema") != "st_flow_training_run_v1"
            or identity_hash(body) != identity.get("identity") or identity.get("config") != cfg
            or identity.get("registration_sha256") != file_sha256(registration_path)
            or identity.get("source_sha") != registration["source_sha"]
            or identity.get("new_optimizer_from_common_S0") is not True
            or identity.get("weights_usable_for_formal_init") != registration["weights_usable_for_formal_init"]
            or (cfg["execution"]["scope"] != "FORMAL" and identity["weights_usable_for_formal_init"] is not False)):
        raise ValueError("Foreign or modified native Flow training lineage")
    if not isinstance(tag, str) or not tag or Path(tag).name != tag or tag in (".", ".."):
        raise ValueError("Explicit safe native checkpoint tag required")
    folder = run / "checkpoints" / tag
    receipt = verify_checkpoint(folder, identity=identity["identity"], world_size=cfg["execution"]["world_size"])
    state = receipt["state"]
    if (type(state["completed"]) is not int or not 0 <= state["completed"] <= cfg["execution"]["max_updates"]
            or state.get("policy_version") != state["completed"]
            or state.get("pending_old_policy", "missing") is not None
            or state.get("pending_feedback", "missing") is not None):
        raise ValueError("Native Flow artifact is not a completed optimizer/policy boundary")
    binding = read_binding(_artifact(registration, "base_binding"))
    if (identity["base_binding"] != binding["binding_sha256"]
            or binding["inference_contract"] != INFERENCE_CONTRACT):
        raise ValueError("Flow parent binding or original FP32 inference precision differs")
    numeric = identity["numeric_execution"]
    numeric_schema = numeric.get("schema")
    if numeric_schema not in ("st_flow_numeric_execution_v1", "st_flow_numeric_execution_v2", "st_flow_numeric_execution_v3"):
        raise ValueError("Unsupported Flow numeric execution schema")
    expected = {
        "schema": numeric_schema, "precision": cfg["execution"]["precision_policy"],
        "encoder": "frozen_BF16_execution_eval_original_S0_AMP",
        "policy_rollout_old_current": "BF16_execution_cuda_autocast_eval",
        "BC": "BF16_execution_cuda_autocast_train_original_dropout",
        "Euler_residual": "float32", "CFM_reduction": "float32", "TF32": False,
        "deterministic": cfg["runtime"]["deterministic"],
        "gradient_checkpointing": cfg["runtime"]["gradient_checkpointing"],
        "base_binding": binding["binding_sha256"], "standalone_evaluation_contract": INFERENCE_CONTRACT,
        "environment_sha256": registration["artifacts"]["environment"]["sha256"],
        "source_fingerprint": registration["source_fingerprint"]}
    if numeric_schema in ("st_flow_numeric_execution_v2", "st_flow_numeric_execution_v3"):
        expected["master_initialization"] = MASTER_INIT_SCHEMA
    from tools.st_world_policy.training_config import official_runtime_policy
    official = official_runtime_policy(cfg)
    if (numeric_schema == "st_flow_numeric_execution_v3") != (official is not None):
        raise ValueError("Official runtime and numeric checkpoint version disagree")
    if official is not None:
        expected.update(encoder="frozen_BF16_execution_eval_official_FA2", official_execution=official)
    if numeric != expected or identity_hash(numeric) != identity["numeric_execution_hash"]:
        raise ValueError("Flow numeric execution identity changed or precision was relabeled")
    repo = Path(__file__).resolve().parents[2]
    for name, expected_hash in binding["source_contracts"].items():
        if file_sha256(repo / name) != expected_hash:
            raise ValueError("Protected original input/action source changed: " + name)
    record = {
        "schema": "st_flow_deployment_checkpoint_v1", "run": str(run), "tag": tag,
        "run_identity": identity["identity"], "completed": state["completed"],
        "receipt_identity": receipt["receipt_identity"],
        "native_checkpoint_sha256": identity_hash({k: v["sha256"] for k, v in receipt["files"].items()}),
        "native_receipt_sha256": file_sha256(folder / "COMPLETE.json"),
        "base_binding": binding["binding_sha256"], "training_source_sha": identity["source_sha"],
        "scope": cfg["execution"]["scope"],
        "weights_usable_for_formal_init": identity["weights_usable_for_formal_init"],
        "inference_contract": binding["inference_contract"],
        "deployment_source_contract": inference_source_contract(repo, identity["source_sha"]),
        "adapter_source_sha256": file_sha256(Path(__file__)),
        "parent_checkpoint_sha256": binding["checkpoint"]["sha256"],
        "parent_receipt_sha256": binding["complete_sha256"],
        "training_numeric_execution_hash": identity["numeric_execution_hash"],
        "master_initialization": numeric.get("master_initialization", "legacy_BF16_roundtrip_before_FP32_master"),
        "restoration": {"encoder": "original_bound_parent_FP32_masters",
                        "action": "native_Flow_ZeRO2_FP32_masters_strict_prefix_load",
                        "prefix": ACTION_PREFIX, "training_BF16_encoder_reused": False,
                        "optimizer_resume_conversion": False}}
    record["sha256"] = identity_hash(record)
    return cfg, binding, record


def _canonical_native_state(folder, receipt, action_head):
    """Check the installed native schema before DeepSpeed's glob reconstruction."""
    from deepspeed.checkpoint.constants import (
        OPTIMIZER_STATE_DICT, ZERO_STAGE, PARTITION_COUNT, SINGLE_PARTITION_OF_FP32_GROUPS)
    native = torch.load(folder / "mp_rank_00_model_states.pt", map_location="cpu", mmap=True, weights_only=False)
    if any(native.get(k, "missing") != v for k, v in receipt["state"].items()):
        raise ValueError("Native model client state differs from its completed receipt")
    if native.get("global_steps") != receipt["state"]["completed"]:
        raise ValueError("Native model optimizer step differs from completed receipt")
    expected = {ACTION_PREFIX + k for k in action_head.state_dict()}
    if set(native["module"]) != expected:
        raise ValueError("Native Flow model keys must be exactly policy.action_head.*")
    names = [name for group in native["param_shapes"] for name in group]
    params = {ACTION_PREFIX + k for k, _ in action_head.named_parameters()}
    buffers = {ACTION_PREFIX + k for k, _ in action_head.named_buffers() if ACTION_PREFIX + k in expected}
    if (len(set(names)) != len(names) or set(names) != params or set(native["buffer_names"]) != buffers
            or native.get("frozen_param_shapes") or native.get("frozen_param_fragments") or native.get("shared_params")):
        raise ValueError("Native Flow parameter/buffer inventory is not the registered action-only model")
    for name in buffers:
        value = native["module"][name]
        original = action_head.state_dict()[name[len(ACTION_PREFIX):]]
        # Buffers have no Adam master. The original S0 loader reconstructs
        # these from its model shard too. Require exact canonical parent values,
        # and retain that parent source instead of claiming learned FP32 buffers.
        if (value.shape != original.shape or not torch.equal(value.to(original.dtype), original)):
            raise ValueError("Native non-trained action buffer differs from the bound parent: " + name)
    for rank in range(receipt["world_size"]):
        opt = torch.load(folder / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
                         map_location="cpu", mmap=True, weights_only=False)[OPTIMIZER_STATE_DICT]
        counts = opt[PARTITION_COUNT]
        counts = counts if isinstance(counts, list) else [counts]
        groups = opt[SINGLE_PARTITION_OF_FP32_GROUPS]
        if (opt[ZERO_STAGE] != 2 or not counts or any(x != receipt["world_size"] for x in counts)
                or len(groups) != len(native["param_shapes"])
                or any(not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or x.ndim != 1 for x in groups)):
            raise ValueError("Native Flow requires real FP32 optimizer masters in the registered ZeRO2 layout")
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    state = get_fp32_state_dict_from_zero_checkpoint(str(folder.parent), tag=folder.name)
    for name in buffers:
        state[name] = action_head.state_dict()[name[len(ACTION_PREFIX):]].clone()
    return state


def restore_action_masters(action_head, state):
    """Strict namespace, dtype and shape checks; no load-from-BF16 fallback."""
    if not isinstance(state, dict) or not state or any(not k.startswith(ACTION_PREFIX) for k in state):
        raise ValueError("Only exact policy.action_head.* native state may be loaded")
    action = {k[len(ACTION_PREFIX):]: v for k, v in state.items()}
    before = action_head.state_dict()
    buffers = {k for k, _ in action_head.named_buffers()}
    if set(action) != set(before):
        raise ValueError("Missing or extra native action master keys")
    old_hashes, new_hashes = {}, {}
    changed, maximum, squared = [], 0., 0.
    for name, value in action.items():
        original = before[name]
        if (not isinstance(value, torch.Tensor) or value.shape != original.shape or value.dtype != original.dtype
                or (value.is_floating_point() and value.dtype != torch.float32)
                or not torch.isfinite(value).all()):
            raise ValueError("Action master shape/dtype/finite mismatch: " + name)
        if name in buffers and not torch.equal(value, original):
            raise ValueError("Non-trained action buffer changed: " + name)
        old_hashes[name], new_hashes[name] = tensor_hash(original), tensor_hash(value)
        if old_hashes[name] != new_hashes[name]:
            changed.append(name)
        for a, b in zip(original.reshape(-1).split(1 << 20), value.reshape(-1).split(1 << 20)):
            delta = a.double() - b.double()
            if delta.numel():
                maximum = max(maximum, delta.abs().max().item())
                squared += delta.square().sum().item()
    action_head.load_state_dict(action, strict=True)
    return {"parent_action_state_sha256": identity_hash(old_hashes),
            "flow_action_state_sha256": identity_hash(new_hashes),
            "changed_action_tensors": changed, "changed_action_tensor_count": len(changed),
            "action_update_max_abs": maximum, "action_update_l2": squared ** .5,
            "strict_action_load": True, "master_dtype": "float32",
            "action_buffers": "unchanged_bound_parent_values_verified_against_native"}


class FlowStandalone(nn.Module):
    """Current-only inference wrapper; compatible with world-export hooks."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        action = base.config.framework.action_model
        if (action.action_dim, action.action_horizon, action.repeated_diffusion_steps,
                action.num_inference_timesteps) != (4, 8, 8, 10) or base.foresight_config.planner_condition_mode != "action_only":
            raise ValueError("Standalone Flow must preserve the original S0 action/input architecture")
        if not getattr(base, "inference_fp32", False):
            raise ValueError("Standalone Flow must use the bound FP32 parent evaluation function")
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @property
    def action_model(self):
        return self.base.action_model

    @property
    def last_sequence_lengths(self):
        return self.base.last_sequence_lengths

    def strip_auxiliary_heads(self):
        self.base.strip_auxiliary_heads()
        return self

    @torch.no_grad()
    def predict_raw(self, observations, *, initial_noise):
        inputs = [x if isinstance(x, ModelInputs) else ModelInputs(x) for x in observations]
        if not inputs or any(x.history is not None for x in inputs):
            raise ValueError("Flow inference consumes only legal current observations, not history/GT/feedback")
        if (not isinstance(initial_noise, torch.Tensor) or initial_noise.dtype != torch.float32
                or initial_noise.shape != (len(inputs), 8, 4) or not torch.isfinite(initial_noise).all()):
            raise ValueError("Explicit finite FP32 8x4 scene-seeded initial noise required")
        if (any(p.dtype != torch.float32 for p in self.parameters())
                or torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda")):
            raise ValueError("Standalone FP32 execution cannot silently inherit AMP or rounded weights")
        if initial_noise.is_cuda and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ValueError("Bound standalone inference requires TF32 disabled")
        encoded = self.base.encode_current([x.legacy_current() for x in inputs])
        condition = self.base.build_planner_condition(encoded)
        if condition.dtype != torch.float32 or condition.device != initial_noise.device:
            raise ValueError("FP32 parent condition and explicit trajectory noise must share device/dtype")
        with self.base.amp():
            result = self.base.action_model.predict_action(condition, initial_noise=initial_noise)
        if result.shape != initial_noise.shape or result.dtype != torch.float32 or not torch.isfinite(result).all():
            raise ValueError("Invalid standalone Flow trajectory")
        return result

    def forward(self, observations, *, initial_noise):
        return self.predict_raw(observations, initial_noise=initial_noise)

    @torch.no_grad()
    def predict_action(self, observations, *, initial_noise):
        from starVLA.dataloader.foresight_dataset import decode_ego
        return decode_ego(self.predict_raw(observations, initial_noise=initial_noise))


def load_flow_checkpoint(registration_path, tag, *, device="cuda", strip=True):
    """Reconstruct both real master sources, then freeze an inference-only model."""
    cfg, binding, record = flow_checkpoint_identity(registration_path, tag)
    registration = json.loads(Path(registration_path).read_text())
    read_binding(_artifact(registration, "base_binding"), verify_files=True)
    from tools.foresight.checkpoints import load_student
    with initialization_seed(cfg["execution"]["seed"]):
        base = load_student(binding["run"], binding["tag"], binding["parent_identity"],
                            device="cpu", precision="fp32", strip=strip)
    folder = Path(record["run"]) / "checkpoints" / tag
    # Parent reconstruction can take minutes. Recheck immutable native bytes
    # immediately before consuming them, rather than trusting an earlier read.
    receipt = verify_checkpoint(folder, identity=record["run_identity"], world_size=cfg["execution"]["world_size"])
    if (receipt["receipt_identity"] != record["receipt_identity"]
            or file_sha256(folder / "COMPLETE.json") != record["native_receipt_sha256"]):
        raise ValueError("Native Flow checkpoint changed during parent reconstruction")
    state = _canonical_native_state(folder, receipt, base.action_model)
    audit = restore_action_masters(base.action_model, state)
    del state
    if any(p.dtype != torch.float32 for p in base.parameters()):
        raise ValueError("Bound parent/Flow masters did not reconstruct canonical FP32 parameters")
    base.inference_fp32 = True
    model = FlowStandalone(base).to(device).eval()
    if strip:
        model.strip_auxiliary_heads()
    model.load_audit = {
        "schema": "st_flow_fp32_restoration_audit_v1", "checkpoint_record_sha256": record["sha256"],
        "parent_checkpoint_sha256": binding["checkpoint"]["sha256"],
        "native_checkpoint_sha256": record["native_checkpoint_sha256"],
        "frozen_encoder_source": "unchanged_original_bound_S0_FP32_masters",
        "training_BF16_encoder_reused": False, "original_parent_modified": False,
        "optimizer_resume_claim": False, "training_feedback_required": False,
        "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()), **audit}
    return model, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--load", action="store_true", help="Strictly reconstruct real FP32 masters; no forward/training")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        if args.load:
            model, record = load_flow_checkpoint(args.registration, args.tag, device=args.device)
            result = {"record": record, "load_audit": model.load_audit}
        else:
            _, _, record = flow_checkpoint_identity(args.registration, args.tag)
            result = {"record": record, "strict_model_load": "NOT_RUN"}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}))
        return 2
    print(json.dumps({"status": "COMPLETE", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
