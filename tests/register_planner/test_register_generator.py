import torch
from torch import nn

from starVLA.model.framework.QwenRegisterGenerator import QwenRegisterGenerator
from starVLA.model.modules.register_planner.generator import (
    RegisterTrajectoryGenerator,
)
from starVLA.training.train_register_generator import (
    FirstBackwardGradientGate,
    assert_all_trainable_parameters_have_grad,
)


def _generator(proposals, stage_loss_mode="final_only"):
    return RegisterTrajectoryGenerator(
        proposal_num=proposals,
        model_dim=32,
        ffn_dim=64,
        num_layers=2,
        num_heads=1,
        proj_drop=0.1,
        drop_path=0.2,
        stage_loss_mode=stage_loss_mode,
    )


def test_register64_output_shape():
    output = _generator(64)(torch.randn(2, 16, 32), torch.randn(2, 4))
    assert output.proposals.shape == (2, 64, 8, 3)
    assert len(output.proposal_list) == 1
    assert len(output.token_list) == 2


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
    assert generator.proposal_heads is None
    assert generator.proposal_head_count == 1
    assert generator.final_proposal_head.network[-1].out_features == 24


def test_final_proposal_head_matches_donor_mlp():
    generator = _generator(64)
    for head in (generator.final_proposal_head,):
        assert sum(isinstance(layer, nn.Linear) for layer in head.network) == 3
        assert sum(isinstance(layer, nn.LayerNorm) for layer in head.network) == 2


def test_all_layers_ablation_keeps_five_stage_heads():
    generator = RegisterTrajectoryGenerator(
        proposal_num=4,
        model_dim=32,
        ffn_dim=64,
        num_layers=4,
        num_heads=1,
        proj_drop=0.0,
        drop_path=0.0,
        stage_loss_mode="all_layers",
    )
    output = generator(torch.randn(2, 4, 32), torch.randn(2, 4))
    assert generator.final_proposal_head is None
    assert generator.proposal_head_count == 5
    assert len(generator.proposal_heads) == 5
    assert len(output.proposal_list) == 5


def test_production_generator_parameter_count():
    generator = RegisterTrajectoryGenerator()
    assert sum(parameter.numel() for parameter in generator.parameters()) == 5_841_176


def test_final_only_loss_updates_every_trainable_parameter(tiny_factory):
    config = tiny_factory.config(4)
    model = QwenRegisterGenerator(
        config,
        qwen_vl_interface=tiny_factory.qwen(),
        qwen_hidden_extractor=tiny_factory.extractor,
    )
    gradient_gate = FirstBackwardGradientGate(model)
    model(tiny_factory.examples())["loss"].backward()
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == []
    assert_all_trainable_parameters_have_grad(model)
    assert gradient_gate.missing_local() == []
    gradient_gate.close()


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
