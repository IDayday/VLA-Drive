"""Separate trajectory-IL experiment; original contracts are never overwritten."""
from copy import deepcopy
from pathlib import Path
from ..contracts import require
from ..io import atomic_json, digest, file_hash

CONTRACT_EXCLUDED = {"query_checkpoint", "query_checkpoint_hash", "binding_status", "workspace", "framework_contract_hash"}


def validate_contract(contract, repo):
    require(contract["framework_contract_hash"] == digest({k: v for k, v in contract.items() if k not in CONTRACT_EXCLUDED}),
            "framework contract content checksum mismatch")
    from ..query_base import bind_source
    bind_source(contract["source_root"], contract["source_commit"])
    for name, checksum in contract["source_hashes"].items():
        require(file_hash(Path(contract["source_root"]) / name) == checksum, "pinned original source changed: " + name)
    for name, checksum in (contract["query_model_code_hashes"] | contract["reused_M0_module_hashes"]).items():
        require(file_hash(Path(repo) / name) == checksum, "protected Query implementation changed: " + name)


def derive_contract(parent, output):
    """The original contract remains intact; this objective has a new identity."""
    result = deepcopy(parent)
    cfg = result["source_config"]["foresight"]
    cfg.update(arm="W_ACTION_ONLY", ablation="W_ACTION_ONLY", enable_current_dino=False,
               enable_future_dino=False, enable_interaction=False, lambda_cur=0., lambda_fut=0., lambda_int=0.)
    # Retain all current-input tokens, causal ordering, Query architecture, camera
    # transforms and normalizer. Only the three auxiliary task modules disappear.
    source_config_path = Path(output) / "SOURCE_CONFIG_IL_ONLY.json"
    atomic_json(source_config_path, result["source_config"], immutable=True)
    result["source_config_hash"] = file_hash(source_config_path)
    result["action"]["retained_auxiliary_losses"] = []
    result["action"]["auxiliary_only_during_query_base_training"] = False
    result["training_objective"] = {"name": "trajectory_il_only_v1", "losses": {"ego_il": 1.0},
        "parent_framework_contract_hash": parent["framework_contract_hash"],
        "input_tokens_unchanged": True, "repeat": 8,
        "authorization": "User requested Query Base trajectory loss only; disable all auxiliary tasks."}
    result.update(binding_status="FRAMEWORK_BOUND_QUERY_UNTRAINED", query_checkpoint=None, query_checkpoint_hash=None)
    result["framework_contract_hash"] = digest({k: v for k, v in result.items() if k not in CONTRACT_EXCLUDED})
    atomic_json(Path(output) / "IL_ONLY_CONTRACT.json", result, immutable=True)
    return result
