import pytest
import torch
from torch import nn

from starVLA.model.modules.register_planner.checkpoint import (
    load_register_generator_checkpoint,
    load_stage_component_checkpoint,
    save_register_generator_checkpoint,
    save_stage_component_checkpoint,
    sha256_file,
    trainable_manifest_hash,
)
from starVLA.model.modules.register_planner.generator import RegisterTrajectoryGenerator
from starVLA.model.modules.scene_encoder import GlobalSceneQFormer
from starVLA.training.register_stage_utils import TrainingProgress


class QwenStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(8, 8)
        self.frozen = nn.Linear(8, 8)
        self.frozen.requires_grad_(False)


def _components():
    return (
        QwenStub(),
        nn.Linear(4, 8),
        GlobalSceneQFormer(
            input_dim=8,
            hidden_dim=8,
            output_dim=8,
            num_queries=4,
            num_layers=1,
            num_heads=1,
            ffn_dim=16,
            detach_qwen_input=False,
        ),
        RegisterTrajectoryGenerator(
            proposal_num=4,
            model_dim=8,
            ffn_dim=16,
            num_layers=1,
            num_heads=1,
            proj_drop=0.0,
            drop_path=0.0,
        ),
    )


def _metadata(qwen):
    return {
        "schema_version": 1,
        "stage": "register_generator",
        "qwen_base_model": "stub",
        "qwen_trainable_manifest_hash": trainable_manifest_hash(qwen),
        "proposal_num": 4,
        "num_poses": 8,
        "state_dim": 3,
        "scene_queries": 4,
        "scene_dim": 8,
        "decoder_layers": 1,
        "decoder_heads": 1,
        "proposal_head_style": "donor_mlp_v1",
        "commit": "abc",
        "config_hash": "def",
    }


def test_generator_checkpoint_roundtrip_and_manifest(tmp_path):
    qwen, action, scene, generator = _components()
    path = tmp_path / "generator.pt"
    save_register_generator_checkpoint(
        path,
        qwen_vl_interface=qwen,
        action_input_model=action,
        scene_encoder=scene,
        register_generator=generator,
        metadata=_metadata(qwen),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload["state_dict"]["qwen_trainable"]) == {
        "trainable.weight",
        "trainable.bias",
    }
    qwen2, action2, scene2, generator2 = _components()
    load_register_generator_checkpoint(
        path,
        qwen_vl_interface=qwen2,
        action_input_model=action2,
        scene_encoder=scene2,
        register_generator=generator2,
        expected_metadata={"proposal_num": 4, "scene_dim": 8},
    )
    assert len(sha256_file(path)) == 64


def test_generator_checkpoint_rejects_shape_contract(tmp_path):
    qwen, action, scene, generator = _components()
    path = tmp_path / "generator.pt"
    save_register_generator_checkpoint(
        path,
        qwen_vl_interface=qwen,
        action_input_model=action,
        scene_encoder=scene,
        register_generator=generator,
        metadata=_metadata(qwen),
    )
    with pytest.raises(RuntimeError, match="proposal_num"):
        load_register_generator_checkpoint(
            path,
            qwen_vl_interface=qwen,
            action_input_model=action,
            scene_encoder=scene,
            register_generator=generator,
            expected_metadata={"proposal_num": 64},
        )


def test_stage_checkpoint_rejects_bank_mismatch(tmp_path):
    module = nn.Linear(2, 2)
    path = tmp_path / "scorer.pt"
    save_stage_component_checkpoint(
        path,
        stage="drivor_scorer",
        module=module,
        metadata={"candidate_bank_manifest_hash": "one"},
    )
    with pytest.raises(RuntimeError, match="candidate_bank_manifest_hash"):
        load_stage_component_checkpoint(
            path,
            stage="drivor_scorer",
            module=nn.Linear(2, 2),
            expected_metadata={"candidate_bank_manifest_hash": "two"},
        )


def test_training_progress_resume_contract():
    source = TrainingProgress(
        epoch=7,
        completed_steps=123,
        early_best=0.42,
        early_bad_epochs=2,
    )
    restored = TrainingProgress()
    restored.load_state_dict(source.state_dict())
    assert restored == source
