"""Read complete native ST checkpoints for single-trajectory deployment.

The parent experiment is never rewritten. Learned FP32 masters are reconstructed
with the existing DeepSpeed implementation and loaded strictly before removing
the label-only readout. Inference does not load teacher caches or future labels.
"""
from pathlib import Path
import ast
import hashlib
import json
import subprocess

from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash, initialization_seed
from tools.st_world_policy.base_binding import read_binding
from tools.st_world_policy.training_config import validate_config
from tools.st_world_policy.training_state import verify_checkpoint


def inference_source_contract(repo, source_sha, *, history):
    """Bind the complete inference module, exempting only its loss forward.

    Imports, globals, properties and helpers are executable dependencies too;
    selecting a handful of methods permits unreviewed changes to those inputs.
    """
    repo = Path(repo)
    methods = {"__init__", "train", "encode_inputs", "predict_raw", "predict_action", "strip_auxiliary_heads"}
    files = ["starVLA/model/framework/st_world_model.py", "tools/st_world_policy/world_expansion.py",
             "starVLA/model/framework/ddp_action_video_foresight.py",
             "starVLA/model/modules/st_world/config.py",
             "starVLA/dataloader/st_world_schema.py"]
    if history:
        files.append("starVLA/model/modules/st_world/history_adapter.py")
    result = {}
    for name in files:
        old = subprocess.check_output(["git", "show", source_sha + ":" + name], cwd=repo).decode()
        new = (repo / name).read_text()
        def representation(code):
            tree = ast.parse(code)
            if name.endswith("st_world_model.py"):
                cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SpaceTimeStudent")
                found = {node.name for node in cls.body
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in methods}
                if set(found) != methods:
                    raise ValueError("Incomplete deployment method contract")
                cls.body = [node for node in cls.body if not
                            (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward")]
            return ast.dump(tree, include_attributes=False)
        before, after = representation(old), representation(new)
        if before != after:
            raise ValueError("Deployment execution source changed without compatibility review: " + name)
        result[name] = hashlib.sha256(after.encode()).hexdigest()
    return result


def checkpoint_identity(registration_path, tag):
    registration_path = Path(registration_path)
    registration = json.loads(registration_path.read_text())
    cfg_path = Path(registration["artifacts"]["config"]["path"])
    if file_sha256(cfg_path) != registration["artifacts"]["config"]["sha256"]:
        raise ValueError("Registered training configuration changed")
    cfg = json.loads(cfg_path.read_text())
    validate_config(cfg)
    run = Path(cfg["runtime"]["output_root"]) / registration["run_id"]
    identity = json.loads((run / "identity.json").read_text())
    body = {k: v for k, v in identity.items() if k != "identity"}
    if (identity.get("schema") != "st_world_training_v1" or identity_hash(body) != identity["identity"]
            or identity["config"] != cfg
            or identity["registration_sha256"] != file_sha256(registration_path)):
        raise ValueError("Foreign or modified ST training lineage")
    if not tag or Path(tag).name != tag or tag in (".", ".."):
        raise ValueError("Explicit safe checkpoint tag required")
    receipt = verify_checkpoint(run / "checkpoints" / tag, identity=identity["identity"],
                                world_size=cfg["execution"]["world_size"])
    binding_path = Path(registration["artifacts"]["base_binding"]["path"])
    if file_sha256(binding_path) != registration["artifacts"]["base_binding"]["sha256"]:
        raise ValueError("Bound original S0 identity changed")
    binding = read_binding(binding_path)
    if identity["base_binding"] != binding["binding_sha256"]:
        raise ValueError("ST checkpoint belongs to another common S0")
    repo = Path(__file__).resolve().parents[2]
    for name, expected in binding["source_contracts"].items():
        if file_sha256(repo / name) != expected:
            raise ValueError("Protected original input/action source changed: " + name)
    record = {"schema": "st_world_deployment_checkpoint_v1", "run": str(run), "tag": tag,
        "run_identity": identity["identity"], "completed": receipt["state"]["completed"],
        "receipt_identity": receipt["receipt_identity"], "base_binding": binding["binding_sha256"],
        "native_checkpoint_sha256": identity_hash({k: v["sha256"] for k, v in receipt["files"].items()}),
        "native_receipt_sha256": file_sha256(run / "checkpoints" / tag / "COMPLETE.json"),
        "training_source_sha": identity["source_sha"], "scope": cfg["execution"]["scope"],
        "weights_usable_for_formal_init": identity["weights_usable_for_formal_init"],
        "inference_contract": binding["inference_contract"],
        "deployment_source_contract": inference_source_contract(repo, identity["source_sha"],
            history=validate_config(cfg).history)}
    record["sha256"] = identity_hash(record)
    return cfg, binding, record


def load_world_checkpoint(registration_path, tag, *, device="cuda", strip=True):
    """No BF16-to-FP32 relabeling: reconstruct actual learned FP32 masters."""
    cfg, binding, record = checkpoint_identity(registration_path, tag)
    from omegaconf import OmegaConf
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from starVLA.model.framework import build_framework
    from starVLA.model.framework.st_world_model import SpaceTimeStudent
    from starVLA.model.modules.st_world.config import SpaceTimeConfig
    parent_cfg = OmegaConf.create(binding["parent_identity"]["config"])
    parent_cfg.framework.qwenvl.device_map = "cpu"
    options = SpaceTimeConfig.from_dict(cfg["space_time"])
    with initialization_seed(cfg["execution"]["seed"]):
        parent = build_framework(parent_cfg)
        history = None
        if options.history:
            from starVLA.model.modules.st_world.history_adapter import build_history_encoder
            history = build_history_encoder(parent, initialization_seed=cfg["execution"]["seed"] + 3700)
        model = SpaceTimeStudent(parent, options, initialization_seed_value=cfg["execution"]["seed"],
                                  history_encoder=history)
    state = get_fp32_state_dict_from_zero_checkpoint(str(Path(record["run"]) / "checkpoints"), tag=tag)
    model.float().load_state_dict(state, strict=True)
    del state
    model.base.inference_fp32 = True
    model.eval().to(device)
    if strip:
        model.strip_auxiliary_heads()
    import torch
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Canonical ST deployment requires original FP32 master execution")
    return model, record
