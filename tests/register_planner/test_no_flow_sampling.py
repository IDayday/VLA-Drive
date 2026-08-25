import inspect

import torch

from starVLA.model.framework.QwenRegisterGenerator import QwenRegisterGenerator
from starVLA.model.modules.register_planner.generator import RegisterTrajectoryGenerator
from starVLA.training import train_register_generator


def test_register_generator_has_no_sampling_interface():
    generator = RegisterTrajectoryGenerator(
        proposal_num=4,
        model_dim=32,
        ffn_dim=64,
        num_layers=2,
        num_heads=1,
    )
    assert not hasattr(generator, "predict_multi_action")
    assert not hasattr(generator, "num_inference_timesteps")
    assert not hasattr(generator, "candidate_chunk_size")


def test_register_forward_does_not_consume_random_noise(monkeypatch):
    generator = RegisterTrajectoryGenerator(
        proposal_num=4,
        model_dim=32,
        ffn_dim=64,
        num_layers=2,
        num_heads=1,
        proj_drop=0.0,
        drop_path=0.0,
    ).eval()

    def forbidden(*args, **kwargs):
        raise AssertionError("Register generation sampled random noise")

    monkeypatch.setattr(torch, "randn", forbidden)
    scene = torch.zeros(1, 4, 32)
    ego = torch.zeros(1, 4)
    assert generator(scene, ego).proposals.shape == (1, 4, 8, 3)


def test_stage_g_entry_does_not_reference_flow_or_scorers():
    source = inspect.getsource(train_register_generator)
    assert "FlowmatchingActionHead" not in source
    assert "DrivoRDynamicScorer" not in source
    assert "DriveSuprim" not in source
    assert "StaticVocabScoreStore" not in source


def test_framework_module_tree_has_no_flow(tiny_factory):
    model = QwenRegisterGenerator(
        tiny_factory.config(4),
        qwen_vl_interface=tiny_factory.qwen(),
        qwen_hidden_extractor=tiny_factory.extractor,
    )
    classes = [module.__class__.__name__ for module in model.modules()]
    assert "FlowmatchingActionHead" not in classes
    assert not any(name.startswith("DiT") for name in classes)
