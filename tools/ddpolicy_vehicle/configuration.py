"""Official DDP recipe plus the three explicit research definitions."""
from pathlib import Path
from omegaconf import OmegaConf


def make_config(asset_root, source_manifest, arm="A", seed=42, device="cuda"):
    root = Path(__file__).parents[2]
    cfg = OmegaConf.load(root/"starVLA/config/training/cfg_yaw_1225.yaml")
    assets = Path(asset_root)
    cfg.seed = seed
    cfg.framework.name = "DDPVehicle"
    cfg.framework.qwenvl.base_vlm = str(assets/"models/Qwen3-VL-2B-Instruct")
    cfg.framework.qwenvl.attn_implementation = "sdpa"
    cfg.framework.qwenvl.device_map = device
    cfg.framework.qwenvl.vl_hidden_dim = 2048
    cfg.framework.action_model.hidden_size = 1536
    cfg.framework.action_model.diffusion_model_cfg.cross_attention_dim = 1536
    cfg.framework.action_model.diffusion_model_cfg.output_dim = 1536
    cfg.framework.action_model.diffusion_model_cfg.num_layers = 24
    cfg.framework.action_model.repeated_diffusion_steps = 8
    cfg.framework.video_model.model_name = str(assets/"models/Wan2.1-Fun-V1.1-1.3B-InP")
    cfg.framework.video_model.config_path = str(root/"starVLA/model/modules/video_model/config/wan2.1/wan_civitai.yaml")
    cfg.datasets.video_data.load_2d_data = 1
    cfg.w_depth = cfg.gs_query_loss = cfg.rgb_query_loss = 1
    cfg.trainer.optimizer.weight_decay = .001
    cfg.trainer.max_train_steps = 100000
    cfg.trainer.num_warmup_steps = 5000
    cfg.from_scratch = {
        "arm": arm, "source_manifest": str(source_manifest),
        "vehicle_queries": 32, "gradient_checkpointing": True,
        "all_hidden_start": 90000, "auxiliary_cadence": 4, "auxiliary_weight": .1,
        "vehicle_xy_scale": 20., "graph": {"max_vehicles": 8, "max_context": 16},
        "graph_training": "predicted_only_from_start",
    }
    return cfg
