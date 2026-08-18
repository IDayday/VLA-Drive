from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v3_overlay_preserves_capacity_and_separates_teacher_downstream_gate():
    main = OmegaConf.load(
        REPO_ROOT / "starVLA/config/training/vggt_query_main.yaml"
    )
    v3 = OmegaConf.load(REPO_ROOT / "starVLA/config/training/vggt_query_v3.yaml")
    config = OmegaConf.merge(main, v3)

    assert config.framework.vggt.version == 3
    assert config.framework.vggt.expected_memory_query_count == 195
    assert config.framework.vggt.teacher.layer_index == 11
    assert config.framework.vggt.teacher.attention_branch == "global"
    assert config.framework.vggt.teacher.codec_source_feature == "layer11_global"
    assert config.framework.vggt.teacher.frozen_tail_layers == [12, 23]
    assert config.framework.vggt.teacher.reused_native_heads == ["camera"]
    assert config.trainer.loss_weights.vggt_geometry == 0.0
    assert config.trainer.loss_weights.vggt_aux_plan == 0.0
    assert config.trainer.learning_rate.vggt_waypoint_reader is None
    assert config.trainer.learning_rate.vggt_geometry_probe is None
    assert config.trainer.learning_rate.vggt_aux_plan_head is None


def test_v3_dlc_pipeline_contains_all_three_separate_gates():
    pipeline = (REPO_ROOT / "12-run_vggt_v3_pipeline.sh").read_text(
        encoding="utf-8"
    )
    trainer = (REPO_ROOT / "8-train_vggt_v3_action.sh").read_text(
        encoding="utf-8"
    )

    assert 'source "$project_root/load_env.sh"' in pipeline
    assert "tools/train_vggt_native_codec.sh" in pipeline
    assert "validate_vggt_native_tail.py" in pipeline
    assert "tools/cache_vggt_v3_queries.sh" in pipeline
    assert "8-train_vggt_v3_action.sh" in pipeline
    assert "diagnose_vggt_training.py" in pipeline
    assert "evaluate_vggt_v3_native_downstream.py" in pipeline
    assert "teacher_codec_downstream" in pipeline
    assert "VGGT_PLANNER_VERSION=3" in trainer
    assert "NAVSIM_VGGT_V3_CACHE_ROOT" in trainer
