import torch
from torch import nn

from starVLA.model.framework.QwenRegisterGenerator import QwenRegisterGenerator
from starVLA.model.modules.register_planner.generator import (
    RegisterTrajectoryGenerator,
)


def _generator(proposals):
    return RegisterTrajectoryGenerator(
        proposal_num=proposals,
        model_dim=32,
        ffn_dim=64,
        num_layers=2,
        num_heads=1,
        proj_drop=0.1,
        drop_path=0.2,
    )


def test_register64_output_shape():
    output = _generator(64)(torch.randn(2, 16, 32), torch.randn(2, 4))
    assert output.proposals.shape == (2, 64, 8, 3)
    assert len(output.proposal_list) == 3


def test_register1_output_shape():
    output = _generator(1)(torch.randn(2, 16, 32), torch.randn(2, 1, 4))
    assert output.proposals.shape == (2, 1, 8, 3)


def test_generator_is_deterministic_in_eval():
    generator = _generator(64).eval()
    scene, ego = torch.randn(1, 16, 32), torch.randn(1, 4)
    first = generator(scene, ego).proposals
    second = generator(scene, ego).proposals
    assert torch.equal(first, second)


def test_generator_has_no_flow_modules():
    names = {module.__class__.__name__.lower() for module in _generator(64).modules()}
    assert not any(
        marker in name
        for name in names
        for marker in ("flowmatchingactionhead", "dit", "beta", "euler")
    )


def test_one_register_one_trajectory():
    generator = _generator(64)
    assert generator.trajectory_registers.weight.shape == (64, 32)
    assert len(generator.proposal_heads) == 3
    assert all(head.network[-1].out_features == 24 for head in generator.proposal_heads)


def test_proposal_heads_match_donor_mlp():
    generator = _generator(64)
    for head in generator.proposal_heads:
        assert sum(isinstance(layer, nn.Linear) for layer in head.network) == 3
        assert sum(isinstance(layer, nn.LayerNorm) for layer in head.network) == 2


def test_production_generator_parameter_count():
    generator = RegisterTrajectoryGenerator()
    assert sum(parameter.numel() for parameter in generator.parameters()) == 11_207_032


def test_generator_loss_updates_baseline_trainable_qwen(tiny_factory):
    config = tiny_factory.config(4)
    model = QwenRegisterGenerator(
        config,
        qwen_vl_interface=tiny_factory.qwen(),
        qwen_hidden_extractor=tiny_factory.extractor,
    )
    model(tiny_factory.examples())["loss"].backward()
    language_grad = model.qwen_vl_interface.model.language.weight.grad
    assert language_grad is not None
    assert torch.count_nonzero(language_grad) > 0


def test_generator_loss_does_not_update_frozen_visual(tiny_factory):
    config = tiny_factory.config(4)
    model = QwenRegisterGenerator(
        config,
        qwen_vl_interface=tiny_factory.qwen(),
        qwen_hidden_extractor=tiny_factory.extractor,
    )
    model(tiny_factory.examples())["loss"].backward()
    assert not model.qwen_vl_interface.model.visual.weight.requires_grad
    assert model.qwen_vl_interface.model.visual.weight.grad is None


def test_scene_qformer_input_not_detached_in_generator_stage(tiny_factory):
    config = tiny_factory.config(4)
    model = QwenRegisterGenerator(
        config,
        qwen_vl_interface=tiny_factory.qwen(),
        qwen_hidden_extractor=tiny_factory.extractor,
    )
    assert model.scene_encoder.detach_qwen_input is False
