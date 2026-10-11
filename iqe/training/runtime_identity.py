"""Separate execution scheduling from the immutable optimization definition."""
import os
from copy import deepcopy
from pathlib import Path
import torch
from ..contracts import require
from ..io import file_hash, read_json, digest


def resume_dependencies(previous, current, *, allow_legacy_validation_metadata=False, allow_deterministic_backend_migration=False):
    if previous == current:
        return {"kind": "IDENTICAL", "previous_hash": digest(previous), "current_hash": digest(current)}
    normalized = deepcopy(previous)
    old_validation = normalized.get("config", {}).get("validation_every_steps")
    if allow_legacy_validation_metadata and isinstance(old_validation, int) and old_validation > 0:
        normalized["config"]["validation_every_steps"] = 0
    backend_migration = False
    if allow_deterministic_backend_migration:
        old_environment = normalized.get('environment', {})
        new_environment = current.get('environment', {})
        if old_environment.get('deterministic') is False and new_environment.get('deterministic') is True:
            old_environment['deterministic'] = True
            backend_migration = True
    require(normalized == current, "exact Base resume optimization/data/world/environment changed")
    return {"kind": "DETERMINISTIC_BACKEND_AND_METADATA" if backend_migration else "LEGACY_UNUSED_VALIDATION_FIELD", "previous_hash": digest(previous),
            "current_hash": digest(current), "old_value": old_validation, "new_value": 0,
            "reason": "1e876da Base trainer never consumed validation_every_steps; new trainer explicitly requires independent evaluation",
            "checkpoint_rewritten": False, "optimizer_schedule_changed": False,
            "bitwise_legacy_continuation": not backend_migration, "optimizer_state_preserved": True}



def configure_runtime(runtime):
    if runtime.get('deterministic_backward', False):
        os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True


def backend_flags():
    return {'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'flash_attention_deterministic':os.getenv('FLASH_ATTENTION_DETERMINISTIC','0')}


def validate_runtime(runtime, contract, world, *, mode, training_config=None):
    changed = runtime["prefetch_depth"] > 0
    report = {"options": runtime, "execution_hash": file_hash(Path(__file__).with_name("execution.py")),
              "precision": "unchanged_fp32_master_bf16_autocast_fp32_frozen_visual",
              "consumed_cursor_only": True, "world_size": world}
    report["backend_flags"] = backend_flags()
    if changed and mode == "full":
        require(runtime["qualification"], "BASE_RUNTIME_UNQUALIFIED: bounded real A/B qualification required")
        evidence = read_json(runtime["qualification"])
        require(evidence["status"] == "PASS" and evidence["no_precision_change"] and
                evidence["execution_hash"] == report["execution_hash"] and
                evidence["framework_contract_hash"] == contract["framework_contract_hash"] and
                evidence["world_size"] == world, "Base runtime qualification identity/fidelity mismatch")
        require(evidence["backend_flags"] == report["backend_flags"], "qualified numerical backend flags changed")
        variant = "vision_ahead" if runtime["frozen_vision_ahead"] else "cpu_prefetch"
        require(variant in evidence["timing"] and all(r["comparison"]["exact"] for r in evidence["all_ranks"]),
                "qualification does not cover this execution path")
        launch = read_json(Path(runtime["qualification"]).parent / "LAUNCH.json")
        tested = launch["receipt"]["dependencies"]["config"]
        require(training_config is not None and runtime["prefetch_depth"] == 1 and
                training_config["global_batch_size"] == tested["global_batch_size"] and
                training_config["micro_batch_size"] == tested["micro_batch_size"],
                "qualification batch/topology/prefetch depth differs from formal execution")
        report["qualification_hash"] = file_hash(runtime["qualification"])
    report["identity_hash"] = digest(report)
    return report
