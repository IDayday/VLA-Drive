"""Strict YAML keys and resolved execution contracts; placeholders never reach trainers."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import yaml
import json
from .contracts import require
from .io import digest, atomic_json, read_json
from .scoring.reference import COMPONENTS


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        require(key in result, f"unknown configuration field {key}")
        if isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_types(value, schema, path="", resolved=False):
    for key, default in schema.items():
        name = path + key
        current = value[key]
        if name in {"model.expert_architecture", "scorer.component_heads"} and resolved:
            continue
        if isinstance(default, dict):
            require(isinstance(current, dict), f"{name} must be an object")
            validate_types(current, default, name + ".", resolved)
        elif isinstance(default, bool):
            require(type(current) is bool, f"{name} must be a boolean")
        elif isinstance(default, int):
            require(type(current) is int, f"{name} must be an integer")
        elif isinstance(default, float):
            require(type(current) in {int, float}, f"{name} must be numeric")
        elif isinstance(default, str):
            require(isinstance(current, str), f"{name} must be a string")
        elif isinstance(default, list):
            require(isinstance(current, list) and current, f"{name} must be a nonempty list")
            if default and isinstance(default[0], str):
                require(all(isinstance(x, str) for x in current), f"{name} string list")
            elif default:
                require(all(type(x) in {int,float} for x in current), f"{name} numeric list")
        elif default is None:
            require(current is None or isinstance(current, str), f"{name} must be null or a real path string")


def load_config(path, *, resolved=False):
    path = Path(path).resolve()
    raw = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    require(isinstance(raw, dict), "config object required")
    saved_hash = raw.pop("config_hash", None)
    require(saved_hash is None or resolved, "resolved config hash in an unresolved input")
    if "extends" in raw:
        parent = load_config(path.parent / raw.pop("extends"))
        raw = merge(parent, raw)
    schema_path = Path(__file__).resolve().parents[1] / "configs/iqe/main.yaml"
    schema = yaml.safe_load(schema_path.read_text())
    # Applying to the reference schema rejects unknown nested keys too.
    raw = merge(schema, raw)
    validate_types(raw, schema, resolved=resolved)
    require(raw["method"] == "iqe_v1", "unsupported method")
    require(raw["s0"]["learned_driving_checkpoint"] is None, "user explicitly requested framework reuse, not learned driving initialization")
    require(not raw["model"]["cross_expert_attention"] and not raw["scorer"]["cross_candidate_attention"]
            and not raw["scorer"]["use_expert_identity"], "forbidden expert/candidate mixing or identity input")
    require(raw["metric"]["score_scale"] == "zero_one" and not raw["metric"]["vectorized_backend_enabled"], "locked reference and explicit score units required")
    require(raw["metric"]["protocol"] == "NAVSIM_v1.1_PDMS", "current source framework supports only its locked NAVSIM v1.1 PDMS; v2/EPDMS require another audited adapter")
    require(not raw["splits"]["final_test_read_for_training"], "final_test forbidden for fitting")
    require(not raw["expert_data"]["allow_unverified_synthetic"], "unverified synthetic data forbidden")
    require(raw["selection"]["calibration_split"] == "selector_cal", "calibration role")
    require(not raw["expert_train"]["wta_across_experts"] and not raw["expert_train"]["distill_all_experts_to_one"], "unsupported training scheme")
    for section in ("expert_train", "scorer", "query_base"):
        c = raw[section]
        require(0 <= c["warmup_steps"] < c["max_optimizer_steps"] and c["lr"] > 0 and c["micro_batch_size"] > 0, f"invalid {section} schedule/batch")
        require(c["weight_decay"] >= 0 and c["grad_clip_norm"] > 0 and c["validation_every_steps"] > 0 and 0 <= c["lr_min_ratio"] <= 1, f"invalid {section} optimizer")
    require(raw["model"]["variant"] in {"independent","query_only","adapter","residual"}, "unsupported expert variant; E1 uses train-policy-copy")
    require(raw["mining"]["variant"] in {"remaining_pool", "fixed_s0"}, "unsupported mining ablation")
    require(raw["metric"]["repetitions"] >= 1 and raw["execution"]["loader_workers"] >= 0 and raw["execution"]["num_threads"] > 0, "invalid resources/repetitions")
    require(raw["execution"]["cpu_score_workers"] > 0 and raw["execution"]["metric_cache_workers"] > 0, "positive CPU scoring/cache worker budgets")
    require(not raw["execution"]["shared_gpu"], "IQE has no authorization to contend with unrelated GPU jobs")
    require(raw["query_base"]["width"] % raw["query_base"]["heads"] == 0 and raw["query_base"]["width"] % raw["query_base"]["compressor_heads"] == 0, "Query attention width")
    locked = {"s0": {"preserve_original_entrypoints":True,"freeze_shared_after_s0":True,"retain_existing_auxiliary_objectives":True,"inherit_input_and_normalizer":True},
        "model":{"init_from":"s0","output_mode":"full_trajectory","precision":"fp32_master_bf16_autocast"},
        "metric":{"backend":"official_single_reference","require_candidate_set_invariance":True},
        "splits":{"grouping":"source_log_and_parent_group"},
        "scorer":{"type":"trajectory_conditioned_shared","direct_value_head":True,"sampling":"scene_uniform"},
        "selection":{"type":"scorer_base_fallback","sampling":"natural_scene_distribution","tie_break":"base_then_oldest_id"},
        "router_ablation":{"label_type":"true_reward_soft_targets","replay_and_refresh_targets_each_round":True,"run_only_selected_decoder":True}}
    for section, fields in locked.items():
        for key, expected in fields.items():
            require(raw[section][key] == expected, f"unsupported {section}.{key}; refusing silently ignored configuration")
    require(set(raw['selection']['protected_metrics']) <= set(COMPONENTS) and
            {'no_at_fault_collisions','drivable_area_compliance'} <= set(raw['selection']['protected_metrics']), 'protected metric schema')
    require(abs(sum(raw['splits']['fallback_fit_val_cal_log_ratios'])-1)<1e-8 and len(raw['splits']['fallback_fit_val_cal_log_ratios'])==3, 'fit/val/cal split ratios')
    require(raw['expert_data']['explicit_anchor_zero'] == (raw['expert_data']['mixture']['global_anchor']==0), 'anchor=0 must be explicitly declared')
    require(all(0<=v<=1 for v in raw['selection']['budgets'].values()), 'risk budget range')
    if resolved:
        require(raw["model"]["expert_architecture"] != "inherit_s0" and raw["scorer"]["component_heads"] != "from_metric_schema", "unresolved trainer config")
        require(raw["s0"]["query_checkpoint"] is not None, "Query S0 checkpoint not resolved")
        require(saved_hash is not None and digest(raw) == saved_hash, "resolved config checksum mismatch")
        raw["config_hash"] = saved_hash
    return raw


def resolve(config, contract, destination):
    result = deepcopy(config)
    result["model"]["expert_architecture"] = contract["query_architecture"]
    result["scorer"]["component_heads"] = list(COMPONENTS)
    result["s0"]["query_checkpoint"] = contract.get("query_checkpoint")
    result["config_hash"] = digest(result)
    atomic_json(destination, result)
    return result
