import pytest

from starVLA.model.modules.action_model.multi_trajectory.config import (
    MultiTrajectoryConfig,
)


def test_scene_and_planning_dimensions_are_explicit():
    config = MultiTrajectoryConfig.from_mapping({"enabled": False})
    assert config.scene_compressor.scene_dim == 2048
    assert config.scene_compressor.num_heads == 32
    assert config.scene_compressor.scene_dim // config.scene_compressor.num_heads == 64
    assert config.planning.planning_dim == 256
    assert config.planning.num_heads == 8
    assert config.planning.planning_dim // config.planning.num_heads == 32


@pytest.mark.parametrize(
    "override,match",
    [
        ({"scene_compressor": {"num_queries": 8}}, "num_queries=16"),
        ({"scene_compressor": {"scene_dim": 256}}, "scene_dim=2048"),
        ({"scene_compressor": {"num_heads": 16}}, "num_heads=32"),
        ({"planning": {"planning_dim": 2048}}, "planning_dim must remain 256"),
        ({"planning": {"num_heads": 4}}, "num_heads=8"),
        ({"drivor": {"dynamic_topk": 65}}, "dynamic_topk=16"),
        ({"suprim": {"fine_memory_source": "patch_map"}}, "fine_memory_source"),
    ],
)
def test_fidelity_config_rejects_dimension_drift(override, match):
    with pytest.raises(ValueError, match=match):
        MultiTrajectoryConfig.from_mapping(override)


def test_joint_finetune_is_an_explicit_stage():
    config = MultiTrajectoryConfig.from_mapping(
        {"enabled": True, "training_stage": "joint_finetune"}
    )
    assert config.training_stage == "joint_finetune"
