"""CPU-only contracts for portable, per-user path configuration."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PATH_VARIABLES = {
    "DRIVEDREAMER_ROOT",
    "DRIVEDREAMER_SHARED_ROOT",
    "SHARED_WEIGHT_ROOT",
    "HF_HOME",
    "NAVSIM_PUBLIC_ROOT",
    "NAVSIM_TRAINVAL_SENSOR_ROOT",
    "NAVSIM_TEST_LOG_ROOT",
    "NAVSIM_TEST_SENSOR_ROOT",
    "NUPLAN_MAPS_ROOT",
    "OPENSCENE_DATA_ROOT",
    "DATA_ROOT",
    "NAVSIM_EXP_ROOT",
    "BASE_VLM",
    "SOURCE_VLM",
    "VIDEO_MODEL",
    "PPD_MODEL",
    "DEPTH_ANYTHING_MODEL",
    "DA3_MODEL",
    "RELEASE_MODEL",
    "VGGT_REPO",
    "VGGT_CHECKPOINT",
    "VGGT_SOURCE_VLM",
    "VGGT_BASE_VLM",
    "NAVSIM_VGGT_CACHE_ROOT",
    "NAVSIM_VGGT_DENSE_CACHE_ROOT",
    "NAVSIM_VIDEO_ROOT",
    "TRAIN_CONFIG_YAML",
    "VIDEO_CONFIG",
    "VLA_DRIVE_ENV_LOADED",
    "VLA_DRIVE_SKIP_ENV_LOCAL",
}

ROOT_ENTRYPOINTS = (
    "0-process_data.sh",
    "1-gen_data_meta_list.sh",
    "2-gen_depth.sh",
    "3-gen_videos.sh",
    "3-stat_data.sh",
    "4-infer.sh",
    "5-eval_v1.sh",
    "6-eval_v2.sh",
    "7-add_token.sh",
    "7-add_vggt_tokens.sh",
    "8-train.sh",
    "8-train_action-only.sh",
    "8-train_action-only-qwen-visual.sh",
    "8-continue_action-only-qwen-visual-200k.sh",
    "8-train_agent_action.sh",
    "8-train_vggt_action.sh",
    "11-precompute_vggt_dense_cache.sh",
    "11-train_vggt_dense_bottleneck.sh",
    "14-train_sq_3d_mix.sh",
    "run_sq3dmix_gated_dlc.sh",
    "run_vggt_dense_bottleneck_dlc.sh",
    "run_vggt_pipeline.sh",
    "debug.sh",
    "pre_cache.sh",
    "training.sh",
)


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for variable in PATH_VARIABLES:
        environment.pop(variable, None)
    return environment


def _run_bash(script: str, *arguments: str, environment: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", "-c", script, "path-contract", *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _copy_environment_files(destination: Path) -> None:
    destination.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "env.sh", destination / "env.sh")
    shutil.copy2(REPO_ROOT / "load_env.sh", destination / "load_env.sh")


def test_load_env_uses_its_own_repository_and_loads_local_overrides(tmp_path: Path):
    checkout = tmp_path / "developer checkout with spaces"
    _copy_environment_files(checkout)
    (checkout / "env.local.sh").write_text(
        "\n".join(
            (
                'export SHARED_WEIGHT_ROOT="$DRIVEDREAMER_ROOT/user weights"',
                'export NAVSIM_PUBLIC_ROOT="$DRIVEDREAMER_ROOT/user navsim"',
                'export DATA_ROOT="${DATA_ROOT:-$DRIVEDREAMER_ROOT/user processed}"',
                'export NAVSIM_EXP_ROOT="$DRIVEDREAMER_ROOT/user experiments"',
                'export BASE_VLM="$DRIVEDREAMER_ROOT/user models/qwen"',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_bash(
        """
set -euo pipefail
export DRIVEDREAMER_ROOT=/stale/other/checkout
source "$1/load_env.sh"
printf 'ROOT=%s\n' "$DRIVEDREAMER_ROOT"
printf 'PWD=%s\n' "$PWD"
printf 'WEIGHTS=%s\n' "$SHARED_WEIGHT_ROOT"
printf 'PUBLIC=%s\n' "$NAVSIM_PUBLIC_ROOT"
printf 'DATA=%s\n' "$DATA_ROOT"
printf 'EXP=%s\n' "$NAVSIM_EXP_ROOT"
printf 'VLM=%s\n' "$BASE_VLM"
""",
        str(checkout),
        environment=_clean_environment(),
    )

    assert result.returncode == 0, result.stderr
    expected = {
        "ROOT": str(checkout),
        "PWD": str(checkout),
        "WEIGHTS": str(checkout / "user weights"),
        "PUBLIC": str(checkout / "user navsim"),
        "DATA": str(checkout / "user processed"),
        "EXP": str(checkout / "user experiments"),
        "VLM": str(checkout / "user models/qwen"),
    }
    actual = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert actual == expected


def test_shared_defaults_do_not_reference_a_developers_mount(tmp_path: Path):
    checkout = tmp_path / "portable-checkout"
    _copy_environment_files(checkout)
    result = _run_bash(
        """
set -euo pipefail
export VLA_DRIVE_SKIP_ENV_LOCAL=1
source "$1/load_env.sh"
printf 'SHARED=%s\n' "$DRIVEDREAMER_SHARED_ROOT"
printf 'WEIGHTS=%s\n' "$SHARED_WEIGHT_ROOT"
printf 'PUBLIC=%s\n' "$NAVSIM_PUBLIC_ROOT"
""",
        str(checkout),
        environment=_clean_environment(),
    )

    assert result.returncode == 0, result.stderr
    actual = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert actual == {
        "SHARED": str(checkout),
        "WEIGHTS": str(checkout / "weights"),
        "PUBLIC": str(checkout / "navsim_dataset_raw"),
    }


def test_all_root_entrypoints_use_the_same_environment_loader():
    expected = 'source "$project_root/load_env.sh"'
    offenders = []
    for relative_path in ROOT_ENTRYPOINTS:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        if expected not in text:
            offenders.append(relative_path)
    assert offenders == []


def test_personal_environment_files_are_git_ignored():
    ignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "env.local.sh" in ignore_text or "*.local.sh" in ignore_text
    assert (REPO_ROOT / "env.local.example.sh").is_file()


def test_one_shot_environment_override_has_highest_path_precedence(tmp_path: Path):
    checkout = tmp_path / "override-checkout"
    _copy_environment_files(checkout)
    shutil.copy2(REPO_ROOT / "env.local.example.sh", checkout / "env.local.example.sh")
    (checkout / "env.local.sh").write_text(
        'export DATA_ROOT="${DATA_ROOT:-$DRIVEDREAMER_ROOT/personal-data}"\n',
        encoding="utf-8",
    )
    one_shot_root = tmp_path / "one-shot-data"
    environment = _clean_environment()
    environment["DATA_ROOT"] = str(one_shot_root)
    result = _run_bash(
        'set -euo pipefail; source "$1/load_env.sh"; printf "%s\\n" "$DATA_ROOT"',
        str(checkout),
        environment=environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(one_shot_root)


def test_training_and_evaluation_datalists_are_configurable():
    env_text = (REPO_ROOT / "env.sh").read_text(encoding="utf-8")
    assert "export NAVSIM_DATALIST_PATH=" in env_text

    for relative_path in ("training.sh", "pre_cache.sh"):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert '"$project_root/train_meta.json"' not in text
        assert 'open("train_meta.json"' not in text
        assert "NAVSIM_DATALIST_PATH" in text

    for relative_path in ("5-eval_v1.sh", "6-eval_v2.sh"):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert 'DATALIST="${DATALIST:-' in text


def test_vggt_inference_does_not_require_training_cache_diagnostics():
    inference_text = (REPO_ROOT / "infer.py").read_text(encoding="utf-8")

    assert '"framework.vggt.cache.enabled"' in inference_text
    assert '"framework.vggt.alignment.log_scene_residual_metrics"' in inference_text
    assert "Inference consumes student queries" in inference_text


def test_vggt_checkpoint_evaluator_uses_versioned_default_steps():
    launcher = (REPO_ROOT / "9-eval_vggt_navtest_ckpts.sh").read_text(
        encoding="utf-8"
    )

    assert 'checkpoint_steps="${CHECKPOINT_STEPS:-10000 20000 30000}"' in launcher
    assert "CHECKPOINT_STEPS" in launcher


def test_environment_paths_override_yaml_but_explicit_cli_stays_highest_priority():
    from starVLA.path_config import apply_environment_path_overrides

    cfg = OmegaConf.create(
        {
            "run_root_dir": "yaml-output",
            "framework": {
                "qwenvl": {"base_vlm": "yaml-vlm"},
                "video_model": {"model_name": "yaml-video"},
                "vggt_bottleneck": {"cache": {"root": "yaml-dense-cache"}},
            },
            "datasets": {
                "vla_data": {"data_root": "yaml-data", "datalist_path": "yaml-list"},
                "video_data": {"rgb_meta_dir": "yaml-rgb"},
            },
        }
    )
    apply_environment_path_overrides(
        cfg,
        {
            "NAVSIM_EXP_ROOT": "/developer/output",
            "BASE_VLM": "/developer/vlm",
            "VIDEO_MODEL": "/developer/video",
            "DATA_ROOT": "/developer/data",
            "NAVSIM_DATALIST_PATH": "/developer/train.json",
            "NAVSIM_VIDEO_ROOT": "/developer/videos",
            "NAVSIM_VGGT_DENSE_CACHE_ROOT": "/developer/dense-cache",
        },
    )
    cli_cfg = OmegaConf.from_dotlist(
        [
            "framework.qwenvl.base_vlm=/one-shot/vlm",
            "framework.vggt_bottleneck.cache.root=/one-shot/dense-cache",
        ]
    )
    cfg = OmegaConf.merge(cfg, cli_cfg)

    assert cfg.run_root_dir == "/developer/output"
    assert cfg.framework.qwenvl.base_vlm == "/one-shot/vlm"
    assert cfg.framework.video_model.model_name == "/developer/video"
    assert cfg.datasets.vla_data.data_root == "/developer/data"
    assert cfg.datasets.vla_data.datalist_path == "/developer/train.json"
    assert cfg.datasets.video_data.rgb_meta_dir == "/developer/videos"
    assert cfg.framework.vggt_bottleneck.cache.root == "/one-shot/dense-cache"


def test_dense_cache_launcher_cli_paths_override_invalid_environment_and_support_multinode(
    tmp_path: Path,
):
    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    captured_arguments = tmp_path / "torchrun-arguments.txt"
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n',
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)

    teacher_repo = tmp_path / "teacher repo with spaces"
    teacher_repo.mkdir()
    teacher_checkpoint = tmp_path / "teacher weights" / "model.safetensors"
    teacher_checkpoint.parent.mkdir()
    teacher_checkpoint.touch()
    cache_root = tmp_path / "dense cache with spaces"
    datalist = tmp_path / "train list.json"
    datalist.write_text("[]\n", encoding="utf-8")
    environment = _clean_environment()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "CAPTURE_ARGS": str(captured_arguments),
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "VGGT_REPO": str(tmp_path / "invalid environment repo"),
            "VGGT_CHECKPOINT": str(tmp_path / "invalid environment checkpoint"),
            "NAVSIM_VGGT_DENSE_CACHE_ROOT": str(tmp_path / "invalid environment cache"),
            "VGGT_DENSE_CACHE_MAX_SAMPLES": "1",
            "VGGT_DENSE_CACHE_FULL": "1",
            "DATA_ROOT": str(tmp_path / "processed data"),
            "NAVSIM_DATALIST_PATH": str(datalist),
            "NAVSIM_TRAINVAL_SENSOR_ROOT": str(tmp_path / "sensors"),
            "NUM_MACHINES": "2",
            "MACHINE_RANK": "1",
            "LOCAL_NUM_PROCESSES": "4",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29671",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "11-precompute_vggt_dense_cache.sh"),
            "--vggt-repo",
            str(teacher_repo),
            "--vggt-checkpoint",
            str(teacher_checkpoint),
            "--cache-root",
            str(cache_root),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr

    arguments = captured_arguments.read_text(encoding="utf-8").splitlines()
    assert "--max-samples" not in arguments
    expected_torchrun = {
        "--nnodes": "2",
        "--node-rank": "1",
        "--nproc-per-node": "4",
        "--master-addr": "127.0.0.1",
        "--master-port": "29671",
    }
    for flag, expected in expected_torchrun.items():
        index = arguments.index(flag)
        assert arguments[index + 1] == expected
    expected_tool = {
        "--vggt-repo": str(teacher_repo),
        "--vggt-checkpoint": str(teacher_checkpoint),
        "--cache-root": str(cache_root),
    }
    for flag, expected in expected_tool.items():
        indices = [index for index, value in enumerate(arguments) if value == flag]
        assert arguments[indices[-1] + 1] == expected


def test_dense_training_launcher_cli_paths_override_invalid_environment(tmp_path: Path):
    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    captured_arguments = tmp_path / "accelerate-arguments.txt"
    fake_accelerate = fake_bin / "accelerate"
    fake_accelerate.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n',
        encoding="utf-8",
    )
    fake_accelerate.chmod(0o755)

    base_vlm = tmp_path / "base vlm with spaces"
    base_vlm.mkdir()
    cache_root = tmp_path / "dense cache with spaces"
    (cache_root / "vggt_dense").mkdir(parents=True)
    (cache_root / "vggt_dense" / "manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    datalist = tmp_path / "train list.json"
    datalist.write_text("[]\n", encoding="utf-8")
    action_checkpoint = tmp_path / "action checkpoint" / "pytorch_model.pt"
    action_checkpoint.parent.mkdir()
    action_checkpoint.touch()
    invalid_experiment_root = tmp_path / "invalid experiments"
    (invalid_experiment_root / "dense-cli-precedence").mkdir(parents=True)
    explicit_experiment_root = tmp_path / "explicit experiments"

    environment = _clean_environment()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "CAPTURE_ARGS": str(captured_arguments),
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "BASE_VLM": str(tmp_path / "invalid environment vlm"),
            "NAVSIM_VGGT_DENSE_CACHE_ROOT": str(tmp_path / "invalid environment cache"),
            "DATA_ROOT": str(tmp_path / "processed data"),
            "NAVSIM_DATALIST_PATH": str(datalist),
            "NAVSIM_EXP_ROOT": str(invalid_experiment_root),
            "ACTION_ONLY_CHECKPOINT": str(tmp_path / "missing action checkpoint"),
            "LOCAL_NUM_PROCESSES": "1",
            "NUM_PROCESSES": "1",
            "PER_DEVICE_BATCH_SIZE": "1",
            "TARGET_EFFECTIVE_BATCH_SIZE": "1",
            "RUN_ID": "dense-cli-precedence",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "11-train_vggt_dense_bottleneck.sh"),
            "--framework.qwenvl.base_vlm",
            str(base_vlm),
            "--framework.vggt_bottleneck.cache.root",
            str(cache_root),
            "--trainer.pretrained_checkpoint",
            str(action_checkpoint),
            "--run_root_dir",
            str(explicit_experiment_root),
            "--trainer.max_train_steps",
            "1",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr

    arguments = captured_arguments.read_text(encoding="utf-8").splitlines()
    expected = {
        "--framework.qwenvl.base_vlm": str(base_vlm),
        "--framework.qwenvl.attn_implementation": "sdpa",
        "--framework.vggt_bottleneck.cache.root": str(cache_root),
        "--trainer.pretrained_checkpoint": str(action_checkpoint),
        "--run_root_dir": str(explicit_experiment_root),
        "--trainer.max_train_steps": "1",
        "--trainer.num_warmup_steps": "5000",
        "--trainer.save_interval": "5000",
        "--trainer.logging_frequency": "50",
        "--framework.action_model.repeated_diffusion_steps": "8",
        "--framework.action_model.hidden_size": "1536",
        "--framework.action_model.diffusion_model_cfg.cross_attention_dim": "1536",
        "--framework.action_model.diffusion_model_cfg.output_dim": "1536",
        "--framework.action_model.diffusion_model_cfg.num_layers": "24",
    }
    for flag, value in expected.items():
        indices = [index for index, argument in enumerate(arguments) if argument == flag]
        assert arguments[indices[-1] + 1] == value


def test_dense_dlc_pipeline_dry_run_skips_cache_validation(tmp_path: Path):
    environment = _clean_environment()
    environment.update(
        {
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "VGGT_DENSE_DLC_DRY_RUN": "1",
            "LOCAL_NUM_PROCESSES": "16",
            "NUM_PROCESSES": "16",
            "PER_DEVICE_BATCH_SIZE": "2",
            "GRADIENT_ACCUMULATION_STEPS": "1",
            "TARGET_EFFECTIVE_BATCH_SIZE": "32",
            "PAI_JOB_ID": "dlc-contract-test",
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "run_vggt_dense_bottleneck_dlc.sh")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr

    output = result.stdout
    assert "local_ppus:16" in output
    assert "effective_batch=32" in output
    assert "attention=sdpa" in output
    assert "11-precompute_vggt_dense_cache.sh" in output
    assert "cache_validation=disabled" in output
    assert "--validate-only" not in output
    assert "MAX_TRAIN_STEPS=2" in output
    assert "11-train_vggt_dense_bottleneck.sh" in output
    assert "7-add_token.sh" not in output
    assert "7-add_vggt_tokens.sh" not in output


def test_dense_dlc_pipeline_auto_uses_batch_four_on_eight_ppus(tmp_path: Path):
    environment = _clean_environment()
    environment.update(
        {
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "VGGT_DENSE_DLC_DRY_RUN": "1",
            "LOCAL_NUM_PROCESSES": "8",
            "NUM_PROCESSES": "8",
            "PAI_JOB_ID": "dlc-eight-ppu-contract-test",
        }
    )
    for variable in (
        "PER_DEVICE_BATCH_SIZE",
        "GRADIENT_ACCUMULATION_STEPS",
        "TARGET_EFFECTIVE_BATCH_SIZE",
        "VGGT_DENSE_EXPECTED_PPU_COUNT",
    ):
        environment.pop(variable, None)

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "run_vggt_dense_bottleneck_dlc.sh")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "local_ppus:8" in result.stdout
    assert "effective_batch=32 (per_device=4 accumulation=1)" in result.stdout
    assert "--nproc-per-node=8" in result.stdout


def test_agent_training_launcher_forwards_paths_as_single_arguments(tmp_path: Path):
    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    captured_arguments = tmp_path / "accelerate-arguments.txt"
    fake_accelerate = fake_bin / "accelerate"
    fake_accelerate.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n',
        encoding="utf-8",
    )
    fake_accelerate.chmod(0o755)

    paths = {
        "BASE_VLM": str(tmp_path / "model with spaces"),
        "DATA_ROOT": str(tmp_path / "data with spaces"),
        "NAVSIM_DATALIST_PATH": str(tmp_path / "lists" / "train list.json"),
        "NAVSIM_EXP_ROOT": str(tmp_path / "outputs with spaces"),
        "TRAIN_CONFIG_YAML": str(tmp_path / "configs" / "train config.yaml"),
    }
    environment = _clean_environment()
    environment.update(paths)
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "CAPTURE_ARGS": str(captured_arguments),
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "RUN_ID": "path-contract-smoke",
            "NUM_PROCESSES": "1",
            "LOCAL_NUM_PROCESSES": "1",
            "PER_DEVICE_BATCH_SIZE": "1",
            "GRADIENT_ACCUMULATION_STEPS": "1",
            "MAX_TRAIN_STEPS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "8-train_agent_action.sh")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr

    arguments = captured_arguments.read_text(encoding="utf-8").splitlines()
    expected_flags = {
        "--config_yaml": paths["TRAIN_CONFIG_YAML"],
        "--framework.qwenvl.base_vlm": paths["BASE_VLM"],
        "--run_root_dir": paths["NAVSIM_EXP_ROOT"],
        "--datasets.vla_data.datalist_path": paths["NAVSIM_DATALIST_PATH"],
        "--datasets.vla_data.data_root": paths["DATA_ROOT"],
    }
    for flag, expected_value in expected_flags.items():
        flag_index = arguments.index(flag)
        assert arguments[flag_index + 1] == expected_value


def test_active_data_modules_do_not_embed_developer_mounts():
    active_modules = (
        "starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py",
        "navsim_data_process/data_engine/datasets/navsim/navsim_qa.py",
    )
    forbidden_prefixes = ("/mnt/petrelfs/", "/mnt/pfs/", "/shared_disk/")
    offenders = []
    for relative_path in active_modules:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for prefix in forbidden_prefixes:
            if prefix in text:
                offenders.append(f"{relative_path}: {prefix}")
    assert offenders == []


def test_evaluation_entrypoints_load_environment_from_any_working_directory(tmp_path: Path):
    environment = _clean_environment()
    environment.update(
        {
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "PRED_DIR": str(tmp_path / "missing predictions"),
            "DATALIST": str(tmp_path / "custom test list.json"),
        }
    )
    for launcher in ("5-eval_v1.sh", "6-eval_v2.sh"):
        result = subprocess.run(
            ["bash", str(REPO_ROOT / launcher)],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.returncode != 0
        assert "Prediction directory does not exist" in result.stderr
        assert "load_env.sh" not in result.stderr


def test_vggt_pipeline_dry_run_is_ppu_dlc_aware_and_preserves_paths(tmp_path: Path):
    paths = {
        "VGGT_REPO": str(tmp_path / "teacher repo with spaces"),
        "VGGT_CHECKPOINT": str(tmp_path / "teacher weights" / "model.safetensors"),
        "VGGT_SOURCE_VLM": str(tmp_path / "source model with spaces"),
        "VGGT_BASE_VLM": str(tmp_path / "derived model with spaces"),
        "NAVSIM_VGGT_CACHE_ROOT": str(tmp_path / "cache with spaces"),
        "DATA_ROOT": str(tmp_path / "data with spaces"),
        "NAVSIM_DATALIST_PATH": str(tmp_path / "lists" / "train list.json"),
        "NAVSIM_TRAINVAL_SENSOR_ROOT": str(tmp_path / "sensors with spaces"),
        "NAVSIM_EXP_ROOT": str(tmp_path / "experiments with spaces"),
    }
    environment = _clean_environment()
    environment.update(paths)
    environment.update(
        {
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "VGGT_PIPELINE_DRY_RUN": "1",
            "WORLD_SIZE": "1",
            "RANK": "0",
            "NPROC_PER_NODE": "16",
            "PAI_JOB_ID": "ppu-contract-job",
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "run_vggt_pipeline.sh")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "effective_batch=32" in result.stdout
    assert "topology=nodes:1 node_rank:0 local_ppus:16 global_processes:16" in result.stdout
    assert "7-add_vggt_tokens.sh" in result.stdout
    assert "tools/cache_vggt_queries.sh" in result.stdout
    assert "tools/check_ppu_runtime.py" in result.stdout
    assert "8-train_vggt_action.sh" in result.stdout
    for path in paths.values():
        assert path in result.stdout

    launcher_text = (REPO_ROOT / "run_vggt_pipeline.sh").read_text(encoding="utf-8")
    assert "ppu-smi" in launcher_text
    assert "nvidia-smi" not in launcher_text
    assert "NPROC_PER_NODE" in launcher_text
    assert "WORLD_SIZE" in launcher_text


def test_vggt_training_launcher_forwards_formal_batch_and_optimizer_contract(tmp_path: Path):
    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    captured_arguments = tmp_path / "vggt-accelerate-arguments.txt"
    fake_accelerate = fake_bin / "accelerate"
    fake_accelerate.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
        'printf "%s\\n" "${NAVSIM_AGENT_DINO_CACHE_ROOT-UNSET}" '
        '> "$CAPTURE_AGENT_DINO_ENV"\n',
        encoding="utf-8",
    )
    fake_accelerate.chmod(0o755)

    token_model = tmp_path / "token model with spaces"
    token_model.mkdir()
    cache_root = tmp_path / "cache with spaces"
    (cache_root / "vggt_query").mkdir(parents=True)
    (cache_root / "vggt_query" / "manifest.json").write_text("{}\n", encoding="utf-8")

    environment = _clean_environment()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "CAPTURE_ARGS": str(captured_arguments),
            "CAPTURE_AGENT_DINO_ENV": str(tmp_path / "agent-dino-env.txt"),
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            # A developer may configure this for another branch. VGGT action-only
            # training must remove the unrelated optional dependency.
            "NAVSIM_AGENT_DINO_CACHE_ROOT": str(tmp_path / "missing agent dino cache"),
            "VGGT_BASE_VLM": str(token_model),
            "NAVSIM_VGGT_CACHE_ROOT": str(cache_root),
            "NAVSIM_EXP_ROOT": str(tmp_path / "experiments with spaces"),
            "DATA_ROOT": str(tmp_path / "data with spaces"),
            "NAVSIM_DATALIST_PATH": str(tmp_path / "train list.json"),
            "NUM_MACHINES": "1",
            "MACHINE_RANK": "0",
            "LOCAL_NUM_PROCESSES": "16",
            "NUM_PROCESSES": "16",
            "PER_DEVICE_BATCH_SIZE": "2",
            "GRADIENT_ACCUMULATION_STEPS": "1",
            "TARGET_EFFECTIVE_BATCH_SIZE": "32",
            "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(16)),
            "TRITON_CACHE_DIR": str(tmp_path / "triton cache with spaces"),
            "RUN_ID": "vggt-formal-contract",
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "8-train_vggt_action.sh")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "VGGT effective batch: 32 (target=32)" in result.stdout
    assert (tmp_path / "agent-dino-env.txt").read_text(encoding="utf-8").strip() == "UNSET"

    arguments = captured_arguments.read_text(encoding="utf-8").splitlines()
    expected_flags = {
        "--num_processes": "16",
        "--num_machines": "1",
        "--mixed_precision": "bf16",
        "--framework.qwenvl.base_vlm": str(token_model),
        "--framework.vggt.cache.root": str(cache_root),
        "--datasets.vla_data.per_device_batch_size": "2",
        "--trainer.gradient_accumulation_steps": "1",
        "--trainer.optimizer.weight_decay": "1e-3",
        "--trainer.learning_rate.base": "1e-5",
        "--trainer.learning_rate.action_model": "1e-5",
        "--trainer.learning_rate.vggt_geometry_adapter": "3e-5",
        "--trainer.learning_rate.vggt_aligner": "3e-5",
        "--trainer.learning_rate.vggt_waypoint_reader": "3e-5",
        "--trainer.learning_rate.vggt_geometry_probe": "3e-5",
        "--trainer.learning_rate.vggt_aux_plan_head": "3e-5",
        "--framework.action_model.hidden_size": "1536",
        "--framework.action_model.diffusion_model_cfg.num_layers": "24",
    }
    for flag, expected_value in expected_flags.items():
        flag_index = arguments.index(flag)
        assert arguments[flag_index + 1] == expected_value


def test_vggt_resolution_probe_preserves_paths_and_cli_output_override(tmp_path: Path):
    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    captured_arguments = tmp_path / "probe-arguments.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    teacher_repo = tmp_path / "teacher repo with spaces"
    teacher_repo.mkdir()
    teacher_checkpoint = tmp_path / "teacher weights" / "model.safetensors"
    teacher_checkpoint.parent.mkdir()
    teacher_checkpoint.touch()
    paths = {
        "VGGT_REPO": str(teacher_repo),
        "VGGT_CHECKPOINT": str(teacher_checkpoint),
        "DATA_ROOT": str(tmp_path / "processed data with spaces"),
        "NAVSIM_DATALIST_PATH": str(tmp_path / "lists" / "train list.json"),
        "NAVSIM_TRAINVAL_SENSOR_ROOT": str(tmp_path / "sensors with spaces"),
        "NAVSIM_EXP_ROOT": str(tmp_path / "experiments with spaces"),
    }
    environment = _clean_environment()
    environment.update(paths)
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "CAPTURE_ARGS": str(captured_arguments),
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "VGGT_RESOLUTION_PROBE_OUTPUT": str(tmp_path / "environment report.json"),
            "VGGT_RESOLUTION_PROBE_SAMPLES": "17",
        }
    )
    cli_output = tmp_path / "one shot report with spaces.json"
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "tools/probe_vggt_spatial_resolution.sh"),
            "--output",
            str(cli_output),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    arguments = captured_arguments.read_text(encoding="utf-8").splitlines()
    assert arguments[0] == "tools/probe_vggt_spatial_resolution.py"
    for flag, expected_value in {
        "--datalist-path": paths["NAVSIM_DATALIST_PATH"],
        "--data-root": paths["DATA_ROOT"],
        "--sensor-root": paths["NAVSIM_TRAINVAL_SENSOR_ROOT"],
        "--vggt-repo": paths["VGGT_REPO"],
        "--vggt-checkpoint": paths["VGGT_CHECKPOINT"],
        "--max-samples": "17",
    }.items():
        flag_index = arguments.index(flag)
        assert arguments[flag_index + 1] == expected_value
    output_indices = [index for index, value in enumerate(arguments) if value == "--output"]
    assert arguments[output_indices[-1] + 1] == str(cli_output)


def test_vggt_geometry_probe_preserves_paths_and_cli_output_override(tmp_path: Path):
    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    captured_arguments = tmp_path / "geometry-probe-arguments.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    teacher_repo = tmp_path / "teacher repo with spaces"
    teacher_repo.mkdir()
    teacher_checkpoint = tmp_path / "teacher weights" / "model.safetensors"
    teacher_checkpoint.parent.mkdir()
    teacher_checkpoint.touch()
    paths = {
        "VGGT_REPO": str(teacher_repo),
        "VGGT_CHECKPOINT": str(teacher_checkpoint),
        "DATA_ROOT": str(tmp_path / "processed data with spaces"),
        "NAVSIM_DATALIST_PATH": str(tmp_path / "lists" / "train list.json"),
        "NAVSIM_TRAINVAL_SENSOR_ROOT": str(tmp_path / "sensors with spaces"),
        "NAVSIM_EXP_ROOT": str(tmp_path / "experiments with spaces"),
    }
    environment = _clean_environment()
    environment.update(paths)
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "CAPTURE_ARGS": str(captured_arguments),
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "VGGT_GEOMETRY_PROBE_OUTPUT": str(tmp_path / "environment report.json"),
            "VGGT_GEOMETRY_PROBE_TRAIN_SAMPLES": "19",
            "VGGT_GEOMETRY_PROBE_VAL_SAMPLES": "7",
        }
    )
    cli_output = tmp_path / "one shot geometry report with spaces.json"
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "tools/probe_vggt_geometry_signal.sh"),
            "--output",
            str(cli_output),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    arguments = captured_arguments.read_text(encoding="utf-8").splitlines()
    assert arguments[0] == "tools/probe_vggt_geometry_signal.py"
    for flag, expected_value in {
        "--datalist-path": paths["NAVSIM_DATALIST_PATH"],
        "--data-root": paths["DATA_ROOT"],
        "--sensor-root": paths["NAVSIM_TRAINVAL_SENSOR_ROOT"],
        "--vggt-repo": paths["VGGT_REPO"],
        "--vggt-checkpoint": paths["VGGT_CHECKPOINT"],
        "--train-samples": "19",
        "--val-samples": "7",
    }.items():
        flag_index = arguments.index(flag)
        assert arguments[flag_index + 1] == expected_value
    output_indices = [index for index, value in enumerate(arguments) if value == "--output"]
    assert arguments[output_indices[-1] + 1] == str(cli_output)


def test_navsim_dataset_loads_agent_dino_cache_only_for_agent_prompt():
    source = (REPO_ROOT / "starVLA/dataloader/navsim_dataset.py").read_text(
        encoding="utf-8"
    )
    assert 'action_prompt_mode == "minimal_agent"' in source
    assert "agent_dino_cache_root = (" in source
