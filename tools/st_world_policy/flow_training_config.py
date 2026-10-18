"""Resolved C0--C4 runner contract; original S0 precision and BC are immutable.

The four rollouts are actions, not the fifty Monte Carlo points used to estimate
each action's FPO ratio. This runner uses one old-policy optimizer boundary per
update, without asynchronously stale feedback or cross-update reuse.
"""
from __future__ import annotations

import math
from pathlib import Path

from starVLA.model.modules.flow_posttraining.fpo_objective import FPOConfig
from tools.st_world_policy.check_formal_approval import validate_execution
from tools.st_world_policy.training_config import PRECISION, exact_fields, official_runtime_policy


REWARDS = {"C0": None, "C1": "C1-PDMS", "C2": "C2-D", "C3": "C3-C", "C4": "C4-T"}
NO_TEACHER = {"schema": "st_flow_no_teacher_v1", "external_teacher_loaded": False,
              "auxiliary_heads_executed": False, "supervision": "original_ego_GT_and_official_environment_only"}


def _absolute(value, name):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("An explicit absolute dependency is required: " + name)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("SHA256 identity required: " + name)


def target_contract(cfg):
    return {"schema": "st_flow_target_contract_v1", "cell": cfg["cell"],
            "raw": "8x4_original_normalized_xy_sin_cos", "physical": "8x3_ego_t0_rear_axle_xy_yaw",
            "horizon": 8, "dt": .5, "euler_steps": 10, "rollouts_per_scene": 4,
            "BC": {"source": "original_ego_labels_v1", "repeat": 8, "beta": [1.5, 1.],
                   "noise_s": .999, "time_transform": "(.999 - U**(1/1.5))/.999_no_added_clamp",
                   "mode": "train_original_dropout", "weight": 1.},
            "policy": {"mode": "eval_with_grad", "reward": REWARDS[cfg["cell"]],
                       "fpo": cfg["fpo"], "weight": cfg["loss"]["policy_weight"],
                       "old_refresh_updates": 1, "maximum_reuse": 1,
                       "incomplete_feedback": "fail_before_backward_retry_explicit_budget",
                       "zero_advantage": "retain_original_BC_for_every_scene"},
            "score_protocol_hash": cfg["feedback"]["score_protocol_hash"],
            "training_precision": PRECISION,
            "standalone_evaluation": "unchanged_bound_FP32_master_contract"}


def validate_config(cfg):
    exact_fields(cfg, {"schema", "execution", "cell", "fpo", "loss", "optimizer", "runtime", "data", "feedback"}, "Flow run")
    if cfg["schema"] != "st_flow_training_v1" or cfg["cell"] not in REWARDS:
        raise ValueError("Only registered C0/C1/C2/C3/C4 are implemented; C5 is not guessed")
    e = cfg["execution"]
    validate_execution(e)
    if e["family"] != "consequence" or e["precision_policy"] != PRECISION:
        raise ValueError("This consequence runner preserves original S0 training precision")
    if e["scope"] == "SMOKE" and (e["sample_limit"] > 32 or e["stop_after"] > 4):
        raise ValueError("Flow SMOKE is bounded to 32 scenes and four optimizer updates")
    if e["scope"] == "PROFILE" and e["stop_after"] > 256:
        raise ValueError("Flow PROFILE exceeds its 256-update bound")
    fpo = FPOConfig.from_dict(cfg["fpo"])
    if fpo.mc_samples != 50:
        raise ValueError("Registered original FPO uses MC50; four rollouts are a different dimension")
    exact_fields(cfg["loss"], {"bc_weight", "policy_weight"}, "Loss")
    weight = cfg["loss"]["policy_weight"]
    if (cfg["loss"]["bc_weight"] != 1. or isinstance(weight, bool)
            or not isinstance(weight, (int, float)) or not math.isfinite(weight)
            or (weight != 0 if cfg["cell"] == "C0" else weight <= 0)):
        raise ValueError("Original BC weight=1 and explicit positive C1--C4 policy coefficient required")
    if fpo.advantage == "scene_center_rollout_std_v1" and (cfg["cell"] != "C1" or weight != 1.):
        raise ValueError("Rollout-scale study changes only C1 advantage normalization; policy weight stays one")
    opt = cfg["optimizer"]
    exact_fields(opt, {"lr", "minimum_lr", "warmup", "betas", "eps", "weight_decay", "clip"}, "Optimizer")
    if any(isinstance(opt[k], bool) or not isinstance(opt[k], (int, float)) or not math.isfinite(opt[k])
           for k in ("lr", "minimum_lr", "eps", "weight_decay", "clip")):
        raise ValueError("Finite optimizer constants required")
    if not (0 < opt["minimum_lr"] <= opt["lr"] and opt["eps"] > 0 and opt["weight_decay"] >= 0
            and opt["clip"] == 1. and type(opt["warmup"]) is int and 0 <= opt["warmup"] < e["max_updates"]):
        raise ValueError("Invalid original matched fixed-horizon Adam/cosine recipe")
    if (not isinstance(opt["betas"], list) or len(opt["betas"]) != 2
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not 0 <= v < 1 for v in opt["betas"])):
        raise ValueError("Resolved Adam betas required")
    r = cfg["runtime"]
    exact_fields(r, {"output_root", "loader_workers", "save_every", "max_seconds", "deterministic",
                     "gradient_checkpointing", "mc_chunk", "checkpoint_velocity"}
                     | (set(r) & {"official_execution", "profile_mode"}), "Runtime")
    profile_mode = r.get("profile_mode", "none")
    if profile_mode not in ("none", "timing"):
        raise ValueError("Flow supports complete-window timing only; no implicit trace mode")
    if profile_mode != "none" and (e["scope"] != "PROFILE" or e["stop_after"] != 240):
        raise ValueError("Flow complete-window timing requires PROFILE and two120-update windows")
    _absolute(r["output_root"], "output_root")
    for k, low, high in (("loader_workers", 0, 8), ("mc_chunk", 1, 50), ("save_every", 1, 4096)):
        if type(r[k]) is not int or not low <= r[k] <= high:
            raise ValueError("Bounded integer runtime option required: " + k)
    if (isinstance(r["max_seconds"], bool) or not isinstance(r["max_seconds"], (int, float))
            or not math.isfinite(r["max_seconds"]) or r["max_seconds"] <= 0):
        raise ValueError("Finite invocation wall budget required")
    if any(type(r[k]) is not bool for k in ("deterministic", "gradient_checkpointing", "checkpoint_velocity")):
        raise ValueError("Backend flags must be resolved booleans")
    official_runtime_policy(cfg)
    exact_fields(cfg["data"], {"root"}, "Data")
    _absolute(cfg["data"]["root"], "data root")
    feedback = cfg["feedback"]
    exact_fields(feedback, {"navsim_root", "maps_root", "cache_roots", "campaign_id", "score_protocol_hash",
                            "workers", "max_calls", "retry_attempts"}, "Official feedback")
    for k in ("navsim_root", "maps_root"):
        _absolute(feedback[k], k)
    if not isinstance(feedback["cache_roots"], list) or not feedback["cache_roots"]:
        raise ValueError("Explicit official context roots required")
    for value in feedback["cache_roots"]:
        _absolute(value, "cache root")
    _sha(feedback["score_protocol_hash"], "score protocol")
    if not isinstance(feedback["campaign_id"], str) or not feedback["campaign_id"]:
        raise ValueError("Feedback campaign must be explicit")
    if type(feedback["workers"]) is not int or not 1 <= feedback["workers"] <= 4:
        raise ValueError("At most four global CPU scorers are allowed")
    if type(feedback["retry_attempts"]) is not int or not 0 <= feedback["retry_attempts"] <= 2:
        raise ValueError("At most two explicitly budgeted transient retries")
    if type(feedback["max_calls"]) is not int or feedback["max_calls"] < 0:
        raise ValueError("A finite total feedback call budget is required")
    if (cfg["cell"] == "C0" and feedback["max_calls"] != 0) or (cfg["cell"] != "C0" and feedback["max_calls"] < 4):
        raise ValueError("C0 never scores; C1--C4 must reserve real scoring calls")
    return fpo
