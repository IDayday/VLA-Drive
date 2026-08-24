import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.training.config_loader import load_training_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "starVLA/config/training"
OFF_CONFIG = CONFIG_ROOT / "qwenpi_multitraj_score_suprim_off.yaml"
ON_CONFIG = CONFIG_ROOT / "qwenpi_multitraj_score_suprim_on.yaml"
OFF_LAUNCHER = ROOT / "run_qwenpi_multitraj_score_suprim_off_dlc.sh"
ON_LAUNCHER = ROOT / "run_qwenpi_multitraj_score_suprim_on_dlc.sh"
COMMON_LAUNCHER = ROOT / "train_qwenpi_drivor_suprim_dlc.sh"


def test_ablation_configs_differ_only_by_run_id_and_suprim_switch():
    off = load_training_config(OFF_CONFIG)
    on = load_training_config(ON_CONFIG)

    assert off.framework.hierarchical_scorer.joint.enabled is False
    assert on.framework.hierarchical_scorer.joint.enabled is True

    on.run_id = off.run_id
    on.framework.hierarchical_scorer.joint.enabled = False
    assert OmegaConf.to_container(on, resolve=False) == OmegaConf.to_container(
        off, resolve=False
    )


def test_both_arms_keep_multitrajectory_and_drivor_contract():
    for path in (OFF_CONFIG, ON_CONFIG):
        config = load_training_config(path)
        scorer = config.framework.hierarchical_scorer
        assert config.framework.scene_encoder.enabled is True
        assert config.framework.action_model.use_global_scene_tokens is True
        assert scorer.enabled is True
        assert scorer.dynamic.enabled is True
        assert scorer.dynamic.num_candidates == 64
        assert scorer.dynamic.candidate_chunk_size == 8
        assert scorer.dynamic.dynamic_topm == 32


def _validate_launcher(launcher: Path, cwd: Path) -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "VLA_PROJECT_ROOT": "/tmp/stale-vla-project-root",
            "DRIVEDREAMER_ROOT": "/tmp/stale-drivedreamer-root",
            "GIT_DIR": "/tmp/stale-linked-worktree-git-dir",
            "GIT_WORK_TREE": "/tmp/stale-linked-worktree",
            "QDS_LAUNCHER_VALIDATE_ONLY": "1",
            "QDS_LOCAL_PROCESSES": "8",
            "VLA_BATCH_SIZE": "4",
            "QDS_TARGET_EFFECTIVE_BATCH": "32",
        }
    )
    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout + completed.stderr


def test_launchers_resolve_own_worktree_and_validate_without_training(tmp_path):
    off_output = _validate_launcher(OFF_LAUNCHER, tmp_path)
    on_output = _validate_launcher(ON_LAUNCHER, tmp_path)

    for output in (off_output, on_output):
        assert f"project_root={ROOT}" in output
        assert "batch=micro:4 accumulation:1 effective:32" in output
        assert "formal_training=NOT_RUN" in output
        assert "git_metadata=linked-worktree-fallback" in output
        assert "/tmp/stale-vla-project-root" not in output
        assert "/tmp/stale-drivedreamer-root" not in output
    assert "drivesuprim_rerank=off" in off_output
    assert f"config={OFF_CONFIG}" in off_output
    assert "drivesuprim_assets=skipped" in off_output
    assert "drivesuprim_rerank=on" in on_output
    assert f"config={ON_CONFIG}" in on_output
    assert "drivesuprim_assets=required" in on_output


def test_bootstrap_failures_are_persisted_for_noninteractive_dlc(tmp_path):
    output_root = tmp_path / "output"
    environment = os.environ.copy()
    environment.update(
        {
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "VLA_OUTPUT_ROOT": str(output_root),
            "QDS_RUN_ID": "bootstrap-failure",
            "QDS_EXPECTED_BRANCH": "feature/not-the-active-branch",
            "QDS_LAUNCHER_VALIDATE_ONLY": "1",
            "QDS_LOCAL_PROCESSES": "8",
            "VLA_BATCH_SIZE": "4",
            "QDS_TARGET_EFFECTIVE_BATCH": "32",
        }
    )
    completed = subprocess.run(
        ["bash", str(ON_LAUNCHER)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    launcher_log = output_root / "launcher_logs/bootstrap-failure.log"
    assert launcher_log.is_file()
    content = launcher_log.read_text(encoding="utf-8")
    assert "wrong source worktree" in content
    assert "phase=source-contract" in content


def test_common_launcher_has_a_no_training_full_preflight_mode():
    source = COMMON_LAUNCHER.read_text(encoding="utf-8")
    marker = "full preflight passed; formal_training=NOT_RUN"

    assert "QDS_LAUNCHER_PREFLIGHT_ONLY" in source
    assert marker in source
    assert source.index(marker) < source.index("accelerate launch")


def test_launchers_are_portable_thin_wrappers():
    for launcher in (OFF_LAUNCHER, ON_LAUNCHER):
        source = launcher.read_text(encoding="utf-8")
        assert "${BASH_SOURCE[0]}" in source
        assert (
            'exec bash "$project_root/train_qwenpi_drivor_suprim_dlc.sh" "$@"'
            in source
        )
        assert "/mnt/" not in source
