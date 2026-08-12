from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN = REPO_ROOT / "starVLA/config/training/vggt_query_main.yaml"


def _merged(name: str):
    return OmegaConf.merge(
        OmegaConf.load(MAIN),
        OmegaConf.load(REPO_ROOT / f"starVLA/config/training/{name}"),
    )


def test_no_teacher_access_is_equal_capacity_without_cache_dependency():
    cfg = _merged("vggt_query_control_no_teacher_access.yaml")
    assert cfg.framework.vggt.supervision_enabled is False
    assert cfg.framework.vggt.access_enabled is True
    assert cfg.framework.vggt.cache.enabled is False
    assert cfg.framework.vggt.diagnostics.intervention_interval == 0
    assert cfg.framework.vggt.expected_memory_query_count == 195
    assert cfg.framework.vggt.layout.spatial_rows == 6
    assert cfg.framework.vggt.layout.spatial_cols == 10
    assert cfg.trainer.loss_weights.vggt_global_alignment == 0.0
    assert cfg.trainer.loss_weights.vggt_geometry == 0.0
    assert cfg.trainer.loss_weights.vggt_aux_plan == 0.05


def test_supervision_no_access_keeps_teacher_but_blocks_planner_path():
    cfg = _merged("vggt_query_control_supervision_no_access.yaml")
    assert cfg.framework.vggt.supervision_enabled is True
    assert cfg.framework.vggt.access_enabled is False
    assert cfg.framework.vggt.cache.enabled is True
    assert cfg.framework.vggt.expected_memory_query_count == 195
    assert cfg.framework.vggt.diagnostics.intervention_interval == 0
    assert cfg.trainer.loss_weights.vggt_global_alignment == 0.05
    assert cfg.trainer.loss_weights.vggt_spatial_alignment == 0.10
    assert cfg.trainer.loss_weights.vggt_geometry == 0.10
    assert cfg.trainer.loss_weights.vggt_aux_plan == 0.0


def test_control_launcher_only_waits_for_and_validates_shared_cache():
    launcher = (REPO_ROOT / "10-run_vggt_v2_control.sh").read_text(encoding="utf-8")
    assert "manifest_is_complete" in launcher
    assert "--validate-only" in launcher
    assert "cache_vggt_queries.sh" not in launcher
    assert "run_vggt_pipeline.sh" not in launcher


def test_main_debug_smoke_forces_intervention_without_changing_controls():
    trainer = (REPO_ROOT / "8-train_vggt_action.sh").read_text(encoding="utf-8")
    pipeline = (REPO_ROOT / "run_vggt_pipeline.sh").read_text(encoding="utf-8")
    assert "VGGT_INTERVENTION_INTERVAL" in trainer
    assert "--framework.vggt.diagnostics.intervention_interval" in trainer
    assert "VGGT_INTERVENTION_INTERVAL=1" in pipeline


def test_fixed_v2_restart_requires_confirmation_and_reuses_complete_cache():
    launcher = (REPO_ROOT / "run_vggt_v2_fixed.sh").read_text(encoding="utf-8")
    assert 'source "$project_root/load_env.sh"' in launcher
    assert "VGGT_FIXED_CONFIRM_OLD_JOB_STOPPED" in launcher
    assert 'VGGT_CACHE_FULL_VALIDATE="${VGGT_CACHE_FULL_VALIDATE:-0}"' in launcher
    assert 'VGGT_INTERVENTION_INTERVAL="${VGGT_INTERVENTION_INTERVAL:-500}"' in launcher
    assert "VGGT_RUN_SMOKE_BEFORE_FORMAL=1" in launcher
    assert "cache_vggt_queries.sh" not in launcher
    assert 'exec bash "$DRIVEDREAMER_ROOT/run_vggt_pipeline.sh"' in launcher
