from pathlib import Path
import json
import hashlib
import os
from omegaconf import OmegaConf
from .contracts import digest


def resolve_config(
    path, sft_checkpoint=None, max_updates=None, output_dir=None, overrides=None
):
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides or [])),
        resolve=True,
    )
    if sft_checkpoint:
        cfg["sft_checkpoint"] = str(Path(sft_checkpoint).resolve())
    if max_updates is not None:
        cfg["runtime"]["max_updates"] = int(max_updates)
    if output_dir:
        cfg["runtime"]["output_dir"] = str(Path(output_dir).resolve())
    if cfg["paths"].get("asset_publication"):
        from .asset_publication import verify_publication

        published = Path(cfg["paths"]["asset_publication"])
        marker = verify_publication(published, verify_inputs=False)
        if (
            Path(cfg["paths"]["asset_manifest"]).resolve()
            != (published / "asset_manifest.json").resolve()
            or cfg["paths"]["asset_manifest_identity"] != marker["asset_identity"]
        ):
            raise ValueError(
                "training configuration references an unpublished/conflicting asset version"
            )
    ckpt = Path(cfg["sft_checkpoint"])
    config_path = ckpt / "config.yaml" if ckpt.is_dir() else ckpt.parent / "config.yaml"
    sft = OmegaConf.load(config_path)
    if sft.framework.name != "QwenOFT" or sft.framework.action_model.mlp_head != 0:
        raise ValueError(
            "this migration supports the audited QwenOFT GR00T flow checkpoint only"
        )
    if (
        sft.get("w_video_latent", 0)
        or sft.get("doing_s2", 0)
        or sft.datasets.reward_data.load_reward_data
    ):
        raise ValueError(
            "checkpoint has a different action dependency/reward contract; explicit audit required"
        )
    if (
        int(sft.ver_1225) != 1
        or int(sft.framework.action_model.action_dim) != 4
        or int(sft.framework.action_model.action_horizon) != 8
    ):
        raise ValueError(
            "unsupported action representation; expected audited absolute ego XY + sin/cos"
        )
    if cfg["checkpoint_contract"]["origin"] == "action_only_release":
        contract = cfg["checkpoint_contract"]
        source_path = Path(contract["source_config"])
        if (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            != contract["source_config_sha256"]
        ):
            raise ValueError("source SFT configuration SHA256 changed")
        expected = OmegaConf.load(source_path)
        if contract["variant"] == "frozen_visual":
            expected.trainer.freeze_modules = (
                "qwen_vl_interface.model.visual,qwen_vl_interface.model.lm_head"
            )
        elif contract["variant"] == "unfrozen_visual":
            expected.framework.retain_inactive_agent_dino_head = True
        else:
            raise ValueError("unknown audited action-only checkpoint variant")
        if digest(OmegaConf.to_container(expected, resolve=True)) != digest(
            OmegaConf.to_container(sft, resolve=True)
        ):
            raise ValueError(
                "checkpoint config no longer matches the exact audited SFT config"
            )
        if sft.framework.get("action_prompt_mode") != "minimal":
            raise ValueError(
                "action-only release requires the minimal history/action prompt"
            )
        if any(
            [
                sft.datasets.video_data.load_2d_data,
                sft.datasets.gs_data.load_3d_data,
                sft.datasets.reward_data.load_reward_data,
                sft.get("w_depth", 0),
                sft.get("w_video_latent", 0),
                sft.get("rgb_query_loss", 0),
            ]
        ):
            raise ValueError(
                "action-only checkpoint has no auxiliary training branches"
            )
        if cfg.get("rl_freeze_modules") != ["qwen_vl_interface.model.visual"]:
            raise ValueError(
                "the user-specified action-only GRPO visual freeze is mandatory"
            )
        if any(
            cfg["checkpoint_contract"][k] != "normalized"
            for k in ["sft_feature_output", "policy_feature_output"]
        ):
            raise ValueError("action-only checkpoints require normalized Qwen features")
    if cfg["trainable_policy"] != "inherit_sft" or cfg["lora"] != "disabled":
        raise ValueError("SFT parameter inheritance is mandatory")
    if (
        not cfg["retention"]["original_sft_enabled"]
        or cfg["retention"]["original_sft_coefficient"] <= 0
    ):
        raise ValueError("original SFT replay cannot be disabled")
    if cfg["algorithm"]["reference_kl_coefficient"] <= 0:
        raise ValueError("full reference required")
    if (
        cfg["sampling"]["train_step_fraction"] != 1
        or cfg["sampling"]["raw_noise_clipping"]
    ):
        raise ValueError("full untruncated chain required")
    if cfg["sampling"]["num_steps"] == "inherit_checkpoint":
        cfg["sampling"]["num_steps"] = int(
            sft.framework.action_model.num_inference_timesteps
        )
    if cfg["sampling"]["num_steps"] != int(
        sft.framework.action_model.num_inference_timesteps
    ):
        raise ValueError("do not change checkpoint inference step count")
    if cfg["sampling"]["noise_level"] <= 0:
        raise ValueError("training requires positive transition noise")
    if cfg["sampling"]["transition_chunk_size"] != 1:
        raise ValueError("transition_chunk_size must be 1; every step is processed")
    if cfg["algorithm"]["advantage_std_unbiased"]:
        raise ValueError("group population std required")
    if cfg["algorithm"]["inner_epochs"] < 1:
        raise ValueError("invalid inner_epochs")
    if cfg["runtime"]["deepspeed_stage"] not in (0, 1, 2):
        raise ValueError("validated backends are torch DDP and ZeRO 1/2")
    if cfg["algorithm"]["logprob_reduction"] not in (
        "flow_grpo_dimension_mean",
        "joint_sum",
    ):
        raise ValueError("invalid reduction")
    if (
        cfg["runtime"]["scene_microbatch"] != 1
        or cfg["runtime"]["replay_microbatch"] != 1
    ):
        raise ValueError(
            "exact original scene loss weighting currently uses microbatch=1; increase accumulation instead"
        )
    for key in [
        "base_vlm",
        "video_model",
        "data_root",
        "raw_root",
        "maps_root",
        "train_list",
        "test_list",
        "metric_cache",
    ]:
        if not Path(cfg["paths"][key]).exists():
            raise FileNotFoundError(f"{key}: {cfg['paths'][key]}")
    sft.framework.qwenvl.sft_feature_output = cfg["checkpoint_contract"][
        "sft_feature_output"
    ]
    sft.framework.qwenvl.policy_feature_output = cfg["checkpoint_contract"][
        "policy_feature_output"
    ]
    sft.framework.qwenvl.base_vlm = cfg["paths"]["base_vlm"]
    sft.framework.video_model.model_name = cfg["paths"]["video_model"]
    sft.datasets.vla_data.data_root = cfg["paths"]["data_root"]
    sft.datasets.vla_data.datalist_path = cfg["paths"]["train_list"]
    sft.datasets.vla_data.split = "train"
    sft.datasets.video_data.rgb_meta_dir = str(
        Path(cfg["paths"]["data_root"]) / "navsim_video"
    )
    if sft.datasets.vla_data.w_neg_traj is not None:
        raise ValueError("negative/teacher trajectory replacement is forbidden")
    os.environ["OPENSCENE_DATA_ROOT"] = cfg["paths"]["raw_root"]
    os.environ["NUPLAN_MAPS_ROOT"] = cfg["paths"]["maps_root"]
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    os.environ["NAVSIM_VIDEO_SOURCE"] = "images"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # RL does not consume externally cached trainable conditions.
    os.environ.pop("NAVSIM_FEATURE_CACHE_ROOT", None)
    return cfg, sft


def split_tokens(cfg):
    tokens = json.loads(Path(cfg["paths"]["train_list"]).read_text())
    test = set(json.loads(Path(cfg["paths"]["test_list"]).read_text()))
    if len(tokens) != len(set(tokens)) or set(tokens) & test:
        raise ValueError("duplicate train scenes or navtest leakage")
    manifest_path = cfg["paths"].get("split_manifest")
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        train, dev = manifest["train_tokens"], manifest["dev_tokens"]
        if set(train) & set(dev) or set(train + dev) != set(tokens):
            raise ValueError("split manifest differs from available train tokens")
        if set(train + dev) & test:
            raise ValueError("navtest leakage")
        return train, dev
    # Stable split selected by token hash, independent of rewards and seed search.
    ordered = sorted(tokens, key=lambda token: digest(token))
    n = cfg["runtime"]["validation_scenes"]
    if n < 1 or n >= len(tokens):
        raise ValueError("need disjoint training and validation scenes")
    return ordered[n:], ordered[:n]


def config_hash(cfg):
    copied = json.loads(json.dumps(cfg))
    # Run location and stop boundary are operational, not trajectory/optimizer semantics.
    copied["runtime"].pop("output_dir", None)
    copied["runtime"].pop("max_updates", None)
    copied["runtime"].pop("run_mode", None)
    copied["runtime"].pop("acceptance_record", None)
    # I/O/observation frequency does not change losses, draws, or optimizer state.
    # Keep the full values in the saved config; permit a bounded diagnostic to
    # save every update while a formal run saves every 100 updates.
    for field in ("save_every", "log_every", "diagnostic_optimizer_gradients"):
        copied["runtime"].pop(field, None)
    return digest(copied)
