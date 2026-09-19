"""Explicit one-epoch F experiment; never releases historical failed profiles."""
import json
import math
import os
from pathlib import Path
from omegaconf import OmegaConf
from .config import split_tokens, config_hash
from .contracts import digest
from .loading import file_sha
from .research_budget import artifact


def epoch_budget(scenes, global_batch=16, inner_epochs=2, group_size=16):
    if any(type(v) is not int or v < 1 for v in (scenes, global_batch, inner_epochs, group_size)):
        raise ValueError("positive integer epoch dimensions required")
    batches = math.ceil(scenes / global_batch)
    exposures = batches * global_batch
    return dict(unique_scenes=scenes, global_scene_batch=global_batch,
                behavior_batches=batches, optimizer_updates=batches * inner_epochs,
                fresh_scene_exposures=exposures, padding_repeats=exposures-scenes,
                candidate_trajectories=exposures * group_size,
                sft_replay_exposures=exposures * inner_epochs,
                padding="deterministic cyclic SceneStream; final partial batch continues next seeded permutation")


def enforce_full_epoch(cfg, context=None, *, record=None):
    from .acceptance import validate_observed_dtypes
    runtime = cfg["runtime"]
    world = int(os.getenv("WORLD_SIZE", "1"))
    if world not in (8, 16) or runtime["accumulation_steps"] * world != 16:
        raise ValueError("full epoch requires world8/accum2 or world16/accum1, global16")
    train, dev = split_tokens(cfg)
    official = OmegaConf.load("navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml")
    if dev or len(train) != 103288 or set(train) != set(official.tokens):
        raise ValueError("full epoch must use all official navtrain, without dev holdout")
    budget = epoch_budget(len(train))
    if not 1 <= runtime["max_updates"] <= budget["optimizer_updates"]:
        raise ValueError("one-epoch experiment optimizer budget exceeded")
    required = {
        "sampling": {"group_size":16, "num_steps":10, "noise_level":.1,
                     "temporal_noise_correlation":.8, "candidate_chunk_size":1, "transition_chunk_size":1,
                     "train_step_fraction":1., "raw_noise_clipping":False},
        "algorithm": {"inner_epochs":2, "ppo_clip_range":.02, "reference_kl_coefficient":.04,
                      "logprob_reduction":"flow_grpo_dimension_mean", "advantage_normalization":"group",
                      "denoising_discount":1., "advantage_epsilon":1e-6, "advantage_clip":5.,
                      "adaptive_reference_kl":{"target":.02, "horizon_scenes":1024,
                                               "min_coefficient":.01, "max_coefficient":1.}},
        "retention": {"original_sft_enabled":True, "original_sft_coefficient":.1},
        "optimizer": {"initial_lr_multiplier":.1, "max_grad_norm":1.,
                      "schedule":{"type":"warmup_cosine", "total_updates":12912,
                                  "warmup_updates":388, "min_lr_ratio":.1}},
        "runtime": {"scene_microbatch":1, "replay_microbatch":1, "seed":42, "deepspeed_stage":2,
                    "numerical_profile":"bf16_zero2_fp32_partition_v2", "activation_checkpointing":True,
                    "noise_seed_schedule":"global_scene_v1", "overlap_reward_reference":True,
                    "reuse_inner_probe":True},
    }
    for section, fields in required.items():
        for key, value in fields.items():
            if cfg[section].get(key) != value:
                raise ValueError("full epoch registered recipe mismatch: " + section + "." + key)
    action_head = cfg.get("trainable_policy") == "action_head"
    if (cfg["checkpoint_contract"]["variant"] != "frozen_visual"
            or cfg["checkpoint_contract"]["sha256"] != "9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4"
            or cfg.get("lora") != "disabled" or cfg.get("trainable_policy") not in ("inherit_sft", "action_head")
            or cfg.get("rl_freeze_modules") != ["qwen_vl_interface.model.visual"]
            or cfg["sampling"].get("transition_mode", "flow_sde") != "flow_sde"):
        raise ValueError("full epoch source/model contract changed")
    path = runtime.get("epoch_evidence")
    if record is None:
        if not path or not Path(path).is_file():
            raise ValueError("full epoch needs completed native pilot and exact resume evidence")
        record = json.loads(Path(path).read_text())
    if (record.get("schema_version") != 1 or record.get("status") != "AUTHORIZED_FULL_DATA_EXPERIMENT"
            or record.get("budget") != budget
            or record.get("split_sha256") != file_sha(cfg["paths"]["split_manifest"])):
        raise ValueError("full epoch registration/data identity mismatch")
    if context is None:
        return
    if record.get("context") != context:
        raise ValueError("full epoch source/config/asset/world identity changed")
    if action_head:
        validate_action_head_evidence(record.get("action_head_cache"), cfg, context)
    if runtime.get("velocity_cuda_graph", False):
        validate_velocity_graph_evidence(record.get("velocity_graph"), cfg)
    pilot_context = artifact(record["pilot_context"])
    if pilot_context != context:
        raise ValueError("pilot differs from full epoch context")
    if artifact(record["resume_context"]) != context:
        raise ValueError("resumed pilot differs from full epoch context")
    if config_hash(artifact(record["pilot_config"])) != config_hash(cfg):
        raise ValueError("pilot recipe differs from full epoch")
    for key in ("pilot_control", "resume_control"):
        control = artifact(record[key])
        if control.get("status") != "PASS" or control.get("exit_codes") != [0] * (world // 8):
            raise ValueError("native target topology run did not finish: " + key)
    comparison = artifact(record["exact_resume"])
    files = comparison.get("files", {})
    required_files = {"scheduler.bin", "custom_checkpoint_0.pkl", "pytorch_model/mp_rank_00_model_states.pt"}
    required_files.update(f"rank_{rank}.pt" for rank in range(world))
    required_files.update(f"pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(world))
    if (comparison.get("status") != "PASS" or len(files) < 3 * world + 3
            or not required_files <= files.keys()
            or any(v.get("status") != "PASS" or not v.get("values")
                   or any(x.get("allclose") is not True for x in v["values"].values()) for v in files.values())):
        raise ValueError("exact optimizer/scheduler/controller resume comparison failed")
    binding = comparison.get("boundaries", {})
    state = binding.get("trainer_state", {})
    if (not binding.get("continuous") or not binding.get("resumed")
            or binding.get("config_hash") != config_hash(cfg) or state.get("update") != 2
            or state.get("world_size") != world
            or Path(binding.get("continuous", "")).parents[1] != Path(record["pilot_context"]["path"]).parent
            or Path(binding.get("resumed", "")).parents[1] != Path(record["resume_context"]["path"]).parent):
        raise ValueError("exact resume report is detached from pilot")
    rows = artifact(record["pilot_training"], lines=True)
    if [r.get("update") for r in rows] != [1,2] or [r.get("inner_epoch") for r in rows] != [0,1]:
        raise ValueError("pilot must execute both fixed behavior inner epochs")
    for row in rows:
        ranks = row.get("ranks", [])
        if sorted(r.get("rank", -1) for r in ranks) != list(range(world)):
            raise ValueError("pilot rank coverage incomplete")
        if row.get("scene_count") != 16 or row.get("candidate_count") != 256:
            raise ValueError("pilot scene/candidate scope differs")
        if row.get("scheduler_completed_updates") != row["update"]:
            raise ValueError("pilot scheduler did not count actual updates")
        if row.get("stability", {}).get("controller", {}).get("updates") != row["update"]:
            raise ValueError("pilot KL feedback did not count actual updates")
        for rank in ranks:
            if rank.get("device", {}).get("type") != "cuda" or rank.get("gradient_tensors") != (359 if action_head else 672):
                raise ValueError("pilot requires actual contracted GPU gradients")
    if rows[0].get("pre_update_ratio_min", 0) < .999 or rows[0].get("pre_update_ratio_max", 2) > 1.001:
        raise ValueError("pilot initial ratio mismatch")
    if rows[0].get("advantage_nonzero_group_fraction", 0) <= 0:
        raise ValueError("official reward has no nonzero group advantage in fixed pilot")
    for a, b in zip(rows[0]["ranks"], rows[1]["ranks"]):
        if any(a[k] != b[k] for k in ("behavior_sha256", "scene_tokens", "replay_tokens")):
            raise ValueError("pilot fixed behavior was not reused")
    validate_observed_dtypes(record["pilot_dtype"], world)
    if digest(artifact(record["pilot_dtype"])["ranks"]) != digest(rows[-1]["ranks"]):
        raise ValueError("pilot dtype evidence is detached")
    immutable = record.get("pilot_immutable", [])
    if len(immutable) != world:
        raise ValueError("missing immutable rank proofs")
    for rank, pointer in enumerate(immutable):
        proof = artifact(pointer)
        if (Path(pointer["path"]).name != f"immutable_update000002_rank{rank}.json"
                or proof.get("status") != "TESTED" or proof.get("end_update") != 2
                or any(proof["changed"].values()) or proof["before"] != proof["after"]):
            raise ValueError("own visual/reference immutability failed")


def validate_velocity_graph_evidence(pointer, cfg):
    if not pointer:
        raise ValueError("velocity graph requires real fixed-chain CUDA comparison")
    report = artifact(pointer)
    if (report.get("status") != "PASS" or report.get("device_type") != "cuda"
            or report.get("checkpoint_sha256") != cfg["checkpoint_contract"]["sha256"]
            or report.get("kernel_sha256") != file_sha("starVLA/rl/flow_grpo/velocity_graph.py")
            or report.get("script_sha256") != file_sha("scripts/analysis/cuda_velocity_probe.py")
            or report.get("live_weight_equal") is not True
            or report.get("capture_rng_unchanged") is not True):
        raise ValueError("velocity graph CUDA evidence/source/live-weight/RNG mismatch")
    scenes = report.get("scenes", [])
    if len(scenes) != 4 or len({tuple(s.get("tokens", [])) for s in scenes}) != 4:
        raise ValueError("velocity graph fixed four-scene coverage missing")
    for scene in scenes:
        if scene.get("chain_equal") is not True or scene.get("old_equal") is not True:
            raise ValueError("velocity graph changed real behavior chain")
        if len(scene.get("checks", [])) != 4:
            raise ValueError("velocity graph repeated comparison missing")
        for check in scene["checks"]:
            if set(check) != {"velocity", "mean", "std", "elementwise_logprob"} or any(
                    row.get("equal") is not True or row.get("max_abs") != 0 for row in check.values()):
                raise ValueError("velocity graph real transition mismatch")


def validate_action_head_evidence(pointer, cfg, context):
    if not pointer or not cfg.get("frozen_feature_cache"):
        raise ValueError("action-head epoch requires its own cached-prefix CUDA proof")
    proof = artifact(pointer)
    if (proof.get("status") != "PASS" or proof.get("device_type") != "cuda"
            or proof.get("checkpoint_sha256") != cfg["checkpoint_contract"]["sha256"]
            or proof.get("config_hash") != config_hash(cfg)
            or proof.get("executable_sha256") != context["executable_sha256"]
            or proof.get("script_sha256") != file_sha("scripts/analysis/action_head_cache_probe.py")
            or proof.get("trainable_numel") != 819503620
            or len(proof.get("trainable_names", [])) != 359
            or not all(n.startswith("action_model.") for n in proof["trainable_names"])
            or proof.get("prefix_matches_own_reference") is not True
            or proof.get("reference_no_grad") is not True):
        raise ValueError("action-head cache/gradient proof does not match actual profile")
    rows = proof.get("scenes", [])
    if len(rows) != 4 or len({r.get("token") for r in rows}) != 4:
        raise ValueError("fixed real-scene cache comparison incomplete")
    if rows[0].get("source_sft_max_abs", float('inf')) > 2e-6 or rows[0].get("source_ode_max_abs", float('inf')) > 2e-5:
        raise ValueError("cached action-head source oracle failed")
    if not any(r.get("official_reward_nonzero_advantage") for r in rows):
        raise ValueError("fixed cache probe has no official RL signal")
    for row in rows:
        if (any(row.get(k) is not True for k in ("condition_equal", "chain_equal", "old_equal", "frozen_no_grad"))
                or set(row.get("gradients", {})) != set(proof["trainable_names"])
                or any(g.get("equal") is not True for g in row["gradients"].values())
                or set(row.get("transitions_equal", {})) != {"velocity", "mean", "std", "elementwise_logprob"}
                or any(v is not True for v in row["transitions_equal"].values())
                or set(row.get("sft_equal", {})) != {"action_loss", "rgb_loss", "gs_loss", "reward_loss"}
                or any(v is not True for v in row["sft_equal"].values())):
            raise ValueError("cached action-head tensor/gradient equivalence failed")
