from pathlib import Path

from omegaconf import OmegaConf

from starVLA.model.modules.vggt_query.types import (
    VGGTQueryLayout,
    build_vggt_global_query_tokens,
    build_vggt_query_tokens,
)


def test_checked_in_token_file_matches_runtime_order():
    root = Path(__file__).resolve().parents[2]
    token_file = (
        root
        / "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/vggt_global_query_tokens_15.txt"
    )
    checked_in = tuple(
        line.strip() for line in token_file.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    layout = VGGTQueryLayout()
    assert checked_in == build_vggt_global_query_tokens(layout)
    assert len(checked_in) == layout.special_query_count == 15
    assert len(build_vggt_query_tokens(layout)) == layout.query_count == 195
    assert (layout.spatial_rows, layout.spatial_cols) == (6, 10)
    assert layout.teacher_dim == 1024


def test_v2_training_overlay_matches_layer11_global_query_contract():
    root = Path(__file__).resolve().parents[2]
    config = OmegaConf.load(root / "starVLA/config/training/vggt_query_main.yaml")
    assert config.framework.vggt.expected_global_query_count == 15
    assert config.framework.vggt.expected_memory_query_count == 195
    assert config.framework.vggt.teacher.layer_index == 11
    assert config.framework.vggt.teacher.attention_branch == "global"
    assert config.framework.vggt.layout.special_per_view == 5
    assert config.framework.vggt.layout.spatial_rows == 6
    assert config.framework.vggt.layout.spatial_cols == 10
    assert config.framework.vggt.layout.teacher_dim == 1024
    assert config.framework.vggt.alignment.mode == "raw"
    assert config.framework.vggt.alignment.scene_residual_enabled is False
    assert config.framework.vggt.planner_reader.output_query_count == 8
