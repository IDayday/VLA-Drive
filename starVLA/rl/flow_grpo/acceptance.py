"""Fail closed: bounded diagnostics cannot silently become formal experiments."""

from pathlib import Path
import json
import os
from .audit import source_fingerprints
from .config import config_hash
from .contracts import digest
from .loading import file_sha

GATES = (
    "parameter_contract",
    "own_visual_unchanged",
    "rl_sft_reference_gradients",
    "fp32_chunk_mathematics",
    "production_optimizer_update",
    "fixed_noise_ode",
    "production_repeat",
    "distributed_update_scaling",
    "checkpointing_accumulation",
    "exact_resume",
    "export_original_protocol",
    "inner_epochs_2",
    "distributed_faults",
    "global_metrics",
    "evaluation_protocol",
    "clean_checkout",
    "source_ode_sft_oracle",
)

# These are requirements for measured results, not declarations in a YAML profile.
# A single bundle may hold many tests, but each gate selects its own test_id.
GATE_REQUIREMENTS = {
    "parameter_contract": (
        "model",
        {"name_shape_alias_groups_equal", "visual_optimizer_excluded"},
    ),
    "own_visual_unchanged": (
        "production",
        {"own_visual_equal_initial", "reference_unchanged"},
    ),
    "rl_sft_reference_gradients": (
        "production",
        {
            "rl_full_parameter_coverage",
            "sft_coverage",
            "reference_loss_current_gradient",
            "visual_reference_no_grad",
        },
    ),
    "fp32_chunk_mathematics": (
        "fp32_model",
        {"gradients_equal", "optimizer_updates_equal", "fixed_noise_equal"},
    ),
    "production_optimizer_update": (
        "production",
        {"optimizer_update_comparison", "forward_weights_changed", "finite"},
    ),
    "fixed_noise_ode": ("production", {"fixed_chain_equal", "fixed_noise_equal"}),
    "production_repeat": ("production", {"repeat_equal"}),
    "distributed_update_scaling": ("production", {"single_multi_update_scaling_equal"}),
    "checkpointing_accumulation": (
        "production",
        {"checkpointing_equal", "accumulation_equal"},
    ),
    "exact_resume": (
        "production",
        {"weights_equal", "optimizer_equal", "rng_cursor_equal", "same_chain_equal"},
    ),
    "export_original_protocol": (
        "cuda_model",
        {"export_weights_equal", "original_inference_equal"},
    ),
    "inner_epochs_2": (
        "production",
        {
            "old_chain_unchanged",
            "advantages_unchanged",
            "policy_changed",
            "inner_boundary_resume_equal",
        },
    ),
    "distributed_faults": (
        "tool_distributed",
        {
            "output_conflict",
            "checkpoint_mkdir",
            "rank0_write",
            "reward_failure",
            "rank_exit",
        },
    ),
    "global_metrics": (
        "tool_distributed",
        {"global_min_max", "weighted_fraction", "counts", "global_quantiles"},
    ),
    "evaluation_protocol": (
        "tool",
        {
            "official_reward",
            "adjacent_aggregation",
            "token_noise",
            "evaluation_transaction",
        },
    ),
    "clean_checkout": ("tool", {"clean_prepare", "regression"}),
    "source_ode_sft_oracle": (
        "cuda_model",
        {"source_ode_equal", "source_sft_loss_equal"},
    ),
}


def evidence_json(artifact):
    if not isinstance(artifact, dict) or not artifact.get("path"):
        raise ValueError("evidence artifact path missing")
    path = Path(artifact["path"])
    if file_sha(path) != artifact.get("sha256"):
        raise ValueError(f"acceptance evidence changed: {path}")
    return json.loads(path.read_text())


def validate_observed_dtypes(artifact, world_size):
    payload = evidence_json(artifact)
    ranks = payload.get("ranks", [])
    if sorted(r.get("rank", -1) for r in ranks) != list(range(world_size)):
        raise ValueError("dtype inventory must cover every actual rank")
    for row in ranks:
        if row.get("device", {}).get("type") != "cuda":
            raise ValueError("dtype inventory is not an actual CUDA observation")
        inventory = row.get("dtype", {})
        if inventory.get("parameters", {}).get("torch.bfloat16", 0) <= 0:
            raise ValueError("missing actual BF16 parameter inventory")
        for field in ("accumulation_dtype", "communication_dtype"):
            if inventory.get(field) != "torch.float32":
                raise ValueError(f"wrong observed {field}")
        for field in ("master_weights", "communication_buffers"):
            if not inventory.get(field) or set(inventory[field]) != {"torch.float32"}:
                raise ValueError(f"unobserved/non-FP32 {field}")
        moments = inventory.get("optimizer_states", {})
        if moments.get("torch.float32", 0) <= 0 or any(
            k in moments for k in ("torch.bfloat16", "torch.float16")
        ):
            raise ValueError("missing/non-FP32 optimizer state observations")
        buffers = row.get("dtype_before_step", {}).get("partition_buffers", [])
        if not buffers or set(buffers) != {"torch.float32"}:
            raise ValueError("missing actual FP32 accumulation partitions")
        if not row.get("activation_dtypes"):
            raise ValueError("activation dtype observations missing")


def validate_gate_evidence(gate, pointer, context, cfg):
    if pointer.get("status") != "PASS" or pointer.get("test_id") != gate:
        raise ValueError(f"NOT_READY: explicit matching test_id required for {gate}")
    bundle = evidence_json(pointer)
    if bundle.get("schema_version") != 1 or not isinstance(bundle.get("tests"), list):
        raise ValueError("unparseable acceptance evidence schema")
    ids = [entry.get("test_id") for entry in bundle["tests"]]
    if len(ids) != len(set(ids)) or ids.count(pointer["test_id"]) != 1:
        raise ValueError("acceptance evidence test_id missing/duplicated")
    test = bundle["tests"][ids.index(pointer["test_id"])]
    if test.get("schema_version") != 1 or test.get("status") != "PASS":
        raise ValueError(f"NOT_READY: inner evidence status for {gate}")
    execution = test.get("execution", {})
    if (
        execution.get("exit_code") != 0
        or execution.get("completed") is not True
        or not execution.get("command")
    ):
        raise ValueError("acceptance test did not finish successfully")
    if execution.get("executable_sha256") != context.get(
        "executable_sha256"
    ) or not context.get("executable_sha256"):
        raise ValueError("evidence executed source differs from current context")
    category, checks = GATE_REQUIREMENTS[gate]
    scope, observed = test.get("scope", {}), test.get("observed_profile", {})
    for field in ("device_type", "precision", "backend", "world_size", "devices"):
        if not observed.get(field):
            raise ValueError(f"evidence missing actual {field}")
    if (
        not isinstance(observed["world_size"], int)
        or len(observed["devices"]) != observed["world_size"]
    ):
        raise ValueError("observed device/world-size mismatch")
    if scope.get("kind") != (
        "model_independent" if category.startswith("tool") else "full_model"
    ):
        raise ValueError("generic tool evidence cannot satisfy a model gate")
    results = test.get("results", {})
    if not checks <= set(scope.get("checks", [])) or any(
        results.get(check) is not True for check in checks
    ):
        raise ValueError(f"missing/pending/failed measured checks for {gate}")
    if category.startswith("tool"):
        if test.get("checkpoint") is not None:
            raise ValueError("model-independent evidence must declare checkpoint null")
        if category == "tool_distributed" and observed["world_size"] < 2:
            raise ValueError("distributed tool gate needs real multiple processes")
        return
    if test.get("checkpoint") != {
        "sha256": context["checkpoint_sha256"],
        "variant": cfg["checkpoint_contract"]["variant"],
    }:
        raise ValueError("evidence checkpoint initialization/variant mismatch")
    if category == "fp32_model":
        if observed["precision"] != "float32" or observed["backend"] != "pytorch":
            raise ValueError("mathematical model comparison needs actual FP32")
        if (
            results.get("chunks") != [1, 2]
            or results.get("groups") != [2, 8]
            or results.get("num_steps") != 10
        ):
            raise ValueError("mathematical comparison scope incomplete")
    if category in {"production", "cuda_model"}:
        if observed["device_type"] != "cuda" or observed["precision"] != "bfloat16":
            raise ValueError("CPU/FP32 evidence cannot satisfy a production BF16 gate")
    if category == "production":
        expected = context.get("resume_identity", {}).get("numerics", {})
        if test.get("declared_profile") != expected or not expected:
            raise ValueError("declared numerical profile mismatch")
        if (
            observed["backend"] != "deepspeed"
            or observed["world_size"] != context["world_size"]
            or observed.get("zero_stage") != 2
        ):
            raise ValueError("actual production backend/world-size mismatch")
        for field in (
            "candidate_chunk",
            "transition_chunk",
            "activation_checkpointing",
            "optimizer_offload",
            "attention_backend",
        ):
            if field not in observed or observed[field] != expected.get(field):
                raise ValueError(f"actual production profile mismatch: {field}")
        if test.get("config_sha256") != context["config_sha256"] or test.get(
            "resume_identity_sha256"
        ) != digest(context["resume_identity"]):
            raise ValueError("production evidence configuration/assets mismatch")
        validate_observed_dtypes(
            test.get("artifacts", {}).get("dtype_inventory"), context["world_size"]
        )
        if gate in {
            "production_optimizer_update",
            "inner_epochs_2",
            "exact_resume",
        } and results.get("optimizer_updates", 0) < (
            2 if gate != "production_optimizer_update" else 1
        ):
            raise ValueError("no actual optimizer updates in evidence")
        if (
            gate == "rl_sft_reference_gradients"
            and results.get("official_nonzero_advantage_groups", 0) < 1
        ):
            raise ValueError(
                "no nonzero official advantage group; cannot prove RL update"
            )
        if gate == "distributed_update_scaling" and not {
            1,
            context["world_size"],
        } <= set(results.get("compared_world_sizes", [])):
            raise ValueError("distributed scaling did not cover target world size")


def executable_identity():
    files = source_fingerprints()
    for root in ("tests/flow_grpo", "scripts/flow_grpo"):
        for path in sorted(Path(root).rglob("*")):
            if path.suffix in {".py", ".sh"}:
                files[str(path)] = file_sha(path)
    return digest(files)


def acceptance_context(cfg, resume_identity):
    return {
        "executable_sha256": executable_identity(),
        "config_sha256": config_hash(cfg),
        "checkpoint_sha256": cfg["checkpoint_contract"]["sha256"],
        "resume_identity": resume_identity,
        "world_size": int(os.getenv("WORLD_SIZE", "1")),
    }


def enforce_training_budget(cfg, context=None, *, record=None):
    from .diagnostic_loss import validate_scope

    runtime = cfg["runtime"]
    validate_scope(runtime)
    mode = runtime.get("run_mode", "diagnostic")
    maximum = runtime["max_updates"]
    if (cfg["sampling"].get("temporal_noise_correlation", 0.0)
            or cfg["sampling"].get("transition_mode", "flow_sde") != "flow_sde") and mode != "diagnostic":
        raise ValueError("correlated exploration is experimental and restricted to bounded diagnostics")
    if mode == "diagnostic":
        if maximum > 8:
            raise ValueError(
                "unreleased diagnostic mode is bounded to 8 optimizer updates"
            )
        if cfg["sampling"]["candidate_chunk_size"] > 1 and not runtime.get(
            "experimental_candidate_chunks", False
        ):
            raise ValueError(
                "chunk>=2 needs explicit experimental_candidate_chunks diagnostic opt-in"
            )
        return
    if mode not in {"paired_short", "formal"}:
        raise ValueError("unknown run_mode")
    if maximum > (100 if mode == "paired_short" else 2000):
        raise ValueError("pre-registered experiment budget exceeded")
    if cfg["sampling"]["candidate_chunk_size"] != 1:
        raise ValueError(
            "candidate_chunk>=2 is experimental; not accepted for paired training"
        )
    world = int(os.getenv("WORLD_SIZE", "1"))
    if runtime["accumulation_steps"] * world != 16:
        raise ValueError("paired global scene batch must be 16")
    required = {
        "sampling": {
            "group_size": 8,
            "num_steps": 10,
            "noise_level": 0.1,
            "transition_chunk_size": 1,
        },
        "algorithm": {
            "inner_epochs": 2,
            "ppo_clip_range": 0.02,
            "advantage_epsilon": 1e-6,
            "advantage_clip": 5.0,
            "reference_kl_coefficient": 0.01,
            "logprob_reduction": "flow_grpo_dimension_mean",
        },
        "retention": {"original_sft_coefficient": 0.1},
        "runtime": {
            "seed": 42,
            "deepspeed_stage": 2,
            "scene_microbatch": 1,
            "replay_microbatch": 1,
            "numerical_profile": "bf16_zero2_fp32_partition_v2",
        },
    }
    for section, fields in required.items():
        for key, value in fields.items():
            if cfg[section].get(key) != value:
                raise ValueError(f"paired recipe changed: {section}.{key}")
    path = runtime.get("acceptance_record")
    # The publisher may supply a not-yet-published record for identical semantic
    # validation before its atomic write. Trainer calls always load the file.
    if record is None:
        if not path or not Path(path).is_file():
            raise ValueError("NOT_READY: missing production profile acceptance record")
        record = json.loads(Path(path).read_text())
    if record.get("status") != "READY_FOR_THIS_PROFILE" or (
        context is not None and record.get("context") != context
    ):
        raise ValueError(
            "NOT_READY: acceptance record does not match current code/config/profile/assets"
        )
    for gate in GATES:
        evidence = record.get("gates", {}).get(gate, {})
        validate_gate_evidence(
            gate, evidence, record["context"] if context is None else context, cfg
        )
