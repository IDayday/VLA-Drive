from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
from omegaconf import OmegaConf

from starVLA.cache.navsim_feature_cache import CACHE_SCHEMA_VERSION
from starVLA.dataloader import navsim_dataset
from starVLA.path_config import apply_environment_path_overrides


REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(mode: str, cache_enabled: bool, cache_root: str = ""):
    return OmegaConf.create(
        {
            "enable_image_aug": 0,
            "w_depth": 0,
            "framework": {
                "name": "QwenOFT_SQ3DMix",
                "action_prompt_mode": "minimal",
                "vggt_bottleneck": {"cache": {"enabled": False}},
                "sq_3d_mix": {
                    "fusion_mode": mode,
                    "vggt": {
                        "feature_dim": 2048,
                        "view_order": ["cam_f0", "cam_l0", "cam_r0"],
                    },
                    "cache": {
                        "enabled": cache_enabled,
                        "root": cache_root,
                        "strict": True,
                        "component": "vggt_dense",
                    },
                },
            },
            "datasets": {
                "vla_data": {"w_neg_traj": None, "act_norm": 1},
                "video_data": {"load_2d_data": 0},
                "gs_data": {"load_3d_data": 0, "debug": 0},
                "reward_data": {"load_reward_data": 0},
            },
        }
    )


def _dataset(tmp_path: Path, cfg):
    datalist = tmp_path / "train.json"
    datalist.write_text("[]\n", encoding="utf-8")
    return navsim_dataset.NavSimDataset(
        datalist_path=datalist,
        split="train",
        video_data_cfg=cfg.datasets.video_data,
        gs_data_cfg=cfg.datasets.gs_data,
        reward_data_cfg=cfg.datasets.reward_data,
        ver_1225=True,
        dataset_cfg=cfg.datasets.vla_data,
        all_cfg=cfg,
        data_root=str(tmp_path / "processed"),
    )


def test_scene_only_does_not_open_dense_cache(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NAVSIM_FEATURE_CACHE_ROOT", raising=False)
    monkeypatch.delenv("NAVSIM_AGENT_DINO_CACHE_ROOT", raising=False)
    monkeypatch.delenv("NAVSIM_VGGT_CACHE_ROOT", raising=False)
    monkeypatch.setenv("NAVSIM_VGGT_DENSE_CACHE_ROOT", str(tmp_path / "missing"))

    def forbidden_reader(*args, **kwargs):
        raise AssertionError("scene_only opened a cache reader")

    monkeypatch.setattr(navsim_dataset, "NavsimFeatureCacheReader", forbidden_reader)
    dataset = _dataset(tmp_path, _config("scene_only", False))

    assert dataset.vggt_dense_cache is None


def test_dense_cache_capabilities_are_mutually_exclusive(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NAVSIM_FEATURE_CACHE_ROOT", raising=False)
    cfg = _config("gated", True, str(tmp_path / "cache"))
    cfg.framework.vggt_bottleneck.cache.enabled = True

    with pytest.raises(ValueError, match="cannot be enabled together"):
        _dataset(tmp_path, cfg)


def test_sq3dmix_reuses_exact_dense_manifest_contract(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NAVSIM_FEATURE_CACHE_ROOT", raising=False)
    datalist_content = "[]\n"
    cache_root = tmp_path / "dense cache"
    component = cache_root / "vggt_dense"
    component.mkdir(parents=True)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "complete": True,
        "world_size": 1,
        "component": "vggt_dense",
        "datalist_sha256": hashlib.sha256(datalist_content.encode()).hexdigest(),
        "sample_count": 0,
        "view_order": ["cam_f0", "cam_l0", "cam_r0"],
        "frame_index": 3,
        "teacher_layer_index": 23,
        "teacher_layer": "aggregator[-1]",
        "teacher_attention_branch": "full_aggregated_feature",
        "include_special_tokens": False,
        "patch_start_idx": 5,
        "spatial_pooling": None,
        "flatten_order": "view-major,row-major,col-major",
        "feature_dim": 2048,
        "preprocess": {
            "mode": "crop",
            "target_long_side": 518,
            "patch_size": 14,
            "preserve_aspect_ratio": True,
        },
    }
    (component / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    cfg = _config("gated", True, str(cache_root))

    dataset = _dataset(tmp_path, cfg)

    assert dataset.vggt_dense_cache is not None
    assert dataset.vggt_dense_cache.has_component("vggt_dense")


def test_dense_cache_environment_binding_precedes_yaml():
    cfg = _config("gated", True, "/yaml/cache")
    apply_environment_path_overrides(
        cfg,
        {"NAVSIM_VGGT_DENSE_CACHE_ROOT": "/environment/cache"},
    )

    assert cfg.framework.sq_3d_mix.cache.root == "/environment/cache"
    assert cfg.framework.vggt_bottleneck.cache.root == "/environment/cache"


@pytest.mark.parametrize("mode", ["scene_only", "projected_concat", "gated"])
def test_launcher_smoke_dry_run_for_all_modes(tmp_path: Path, mode: str):
    base_vlm = tmp_path / "base vlm"
    base_vlm.mkdir()
    data_root = tmp_path / "processed data"
    data_root.mkdir()
    datalist = tmp_path / "train list.json"
    datalist.write_text("[]\n", encoding="utf-8")
    cache_root = tmp_path / "dense cache"
    if mode != "scene_only":
        (cache_root / "vggt_dense").mkdir(parents=True)
        (cache_root / "vggt_dense" / "manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )

    environment = dict(os.environ)
    environment.update(
        {
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "BASE_VLM": str(base_vlm),
            "DATA_ROOT": str(data_root),
            "NAVSIM_DATALIST_PATH": str(datalist),
            "NAVSIM_EXP_ROOT": str(tmp_path / "experiments"),
            "NAVSIM_VGGT_DENSE_CACHE_ROOT": str(cache_root),
            "SQ3DMIX_FUSION_MODE": mode,
            "SQ3DMIX_SMOKE": "1",
            "SQ3DMIX_DRY_RUN": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "14-train_sq_3d_mix.sh")],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert f"--framework.sq_3d_mix.fusion_mode {mode}" in output
    expected_cache = "false" if mode == "scene_only" else "true"
    assert f"--framework.sq_3d_mix.cache.enabled {expected_cache}" in output
    if mode == "scene_only":
        assert "--framework.sq_3d_mix.cache.root" not in output
    else:
        assert "--framework.sq_3d_mix.cache.root" in output


def test_dlc_launcher_runs_dense_cache_before_gated_training():
    launcher = (REPO_ROOT / "run_sq3dmix_gated_dlc.sh").read_text(
        encoding="utf-8"
    )

    assert "export SQ3DMIX_FUSION_MODE=gated" in launcher
    assert "export SQ3DMIX_INTERVENTION=real" in launcher
    assert "export SQ3DMIX_SMOKE=0" in launcher
    cache_call = 'bash "$DRIVEDREAMER_ROOT/11-precompute_vggt_dense_cache.sh"'
    training_call = 'bash "$DRIVEDREAMER_ROOT/14-train_sq_3d_mix.sh"'
    assert cache_call in launcher
    assert training_call in launcher
    assert launcher.index(cache_call) < launcher.index(training_call)
