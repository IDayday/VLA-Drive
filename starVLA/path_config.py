"""Bind machine-local path environment variables into shared OmegaConf files."""

from __future__ import annotations

from collections.abc import Mapping
import os

from omegaconf import DictConfig, OmegaConf


ENVIRONMENT_PATH_BINDINGS: tuple[tuple[str, str], ...] = (
    ("NAVSIM_EXP_ROOT", "run_root_dir"),
    ("BASE_VLM", "framework.qwenvl.base_vlm"),
    ("VIDEO_MODEL", "framework.video_model.model_name"),
    ("GS_MODEL_PATH", "framework.gs_model.model_path"),
    ("DATA_ROOT", "datasets.vla_data.data_root"),
    ("NAVSIM_DATALIST_PATH", "datasets.vla_data.datalist_path"),
    ("NAVSIM_VIDEO_ROOT", "datasets.video_data.rgb_meta_dir"),
    ("NAVSIM_GS_ROOT", "datasets.gs_data.gs_meta_dir"),
    ("NAVSIM_REWARD_ROOT", "datasets.reward_data.reward_meta_dir"),
    ("NAVSIM_VGGT_CACHE_ROOT", "framework.vggt.cache.root"),
    (
        "NAVSIM_VGGT_DENSE_CACHE_ROOT",
        "framework.vggt_bottleneck.cache.root",
    ),
    (
        "NAVSIM_VGGT_DENSE_CACHE_ROOT",
        "framework.sq_3d_mix.cache.root",
    ),
)


def apply_environment_path_overrides(
    cfg: DictConfig,
    environment: Mapping[str, str] | None = None,
) -> DictConfig:
    """Apply non-empty local path variables to ``cfg`` in-place.

    Call this after loading shared YAML and before merging explicit CLI
    dotlist arguments. The resulting precedence is CLI > environment /
    ``env.local.sh`` > shared YAML.
    """

    values = os.environ if environment is None else environment
    for environment_name, config_key in ENVIRONMENT_PATH_BINDINGS:
        value = values.get(environment_name)
        if value is None or not str(value).strip():
            continue
        OmegaConf.update(cfg, config_key, str(value), force_add=True)
    return cfg
