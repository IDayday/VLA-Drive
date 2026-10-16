"""Control-flow/gradient tests; actual original S0 GPU audits remain separate."""
from contextlib import nullcontext
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from torch import nn

from starVLA.model.framework.st_world_model import SpaceTimeStudent
from starVLA.model.modules.foresight.queries import SpatialViewEncoding
from starVLA.model.modules.st_world.losses import optimizer_batch_counts
from starVLA.model.modules.st_world.requests import sample_requests, gather_current
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, ModelInputs


class Action(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Linear(8, 4)
        self.calls = []

    def forward(self, condition, target, *, noise, times):
        self.calls.append((len(condition), noise.detach().clone(), times.detach().clone()))
        return (self.weight(condition).float() - target).square().mean()

    def predict_action(self, condition, *, initial_noise):
        return self.weight(condition) + initial_noise


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_model = Action()
        self.language = nn.Linear(4, 8)
        self.foresight_queries = nn.Parameter(torch.randn(144, 8))
        self.query_geometry = SpatialViewEncoding(8, (6, 8))
        self.foresight_tokens = [f"<foresight_{i}>" for i in range(144)]
        self._special_token_ids = {"foresight": tuple(range(144))}
        self.qwen_vl_interface = nn.Module()
        self.qwen_vl_interface.model = nn.Module()
        self.qwen_vl_interface.model.config = NS(hidden_size=8)
        self.qwen_vl_interface.model.model = nn.Module()
        self.qwen_vl_interface.model.model.visual = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.5))
        self.qwen_vl_interface.model.model.visual.requires_grad_(False)
        self.dino_head = nn.Linear(8, 2)
        self.foresight_config = NS(planner_condition_mode="action_only")
        self.config = NS(framework=NS(action_model=NS(action_dim=4, action_horizon=8,
            repeated_diffusion_steps=8, num_inference_timesteps=10, noise_beta_beta=1.,
            noise_beta_alpha=1.5, noise_s=.999)))
        self.calls = 0

    def strip_auxiliary_heads(self):
        del self.dino_head

    def amp(self):
        return nullcontext()

    def encode_current(self, observations):
        self.calls += 1
        states = torch.as_tensor(np.stack([x["state"] for x in observations]))
        value = self.language(states)
        world = self.query_geometry(self.foresight_queries)[None] + value
        return {"W": world, "action_queries": value.expand(-1, 8, -1)}

    def build_planner_condition(self, encoded):
        return encoded["action_queries"]


def options():
    teacher = TeacherFeatureContract(image_hw=(288, 384), patch_hw=(16, 16),
        native_grid_hw=(18, 24), output_grid_hw=(18, 24), channels=2,
        storage_dtype="float32", compute_dtype="float32", view_names=("CAM_F0", "CAM_L0", "CAM_R0"),
        teacher_identity="a" * 64, transform_identity="b" * 64,
        pooling={"kind": "none", "kernel_hw": (1, 1), "stride_hw": (1, 1)})
    return {"cell": "S-P2", "teacher": teacher, "world_queries_per_view": 48,
            "sampled_positions": 96, "readout_dim": 16, "readout_heads": 4}


def inputs(n=2):
    return [ModelInputs({"image": [object(), object(), object()], "state": np.full((1, 4), i + 1., np.float32),
                         "lang": "current navigation", "token": str(i)}) for i in range(n)]


def test_wrapper_preserves_base_and_no_labels_or_readout_in_predict():
    parent = Parent().eval()
    observations = inputs()
    noise = torch.randn(2, 8, 4)
    with torch.no_grad():
        original = parent.action_model.predict_action(parent.build_planner_condition(
            parent.encode_current([x.current for x in observations])), initial_noise=noise)
    model = SpaceTimeStudent(parent, options(), initialization_seed_value=42).eval()
    assert torch.equal(model.predict_raw(observations, initial_noise=noise), original)
    model.strip_auxiliary_heads()
    assert torch.equal(model.predict_raw(observations, initial_noise=noise), original)


def test_one_encoder_repeat8_fm_once_feature_gradient_and_visual_eval():
    torch.manual_seed(7)
    parent = Parent()
    model = SpaceTimeStudent(parent, options(), initialization_seed_value=42).train()
    assert not parent.qwen_vl_interface.model.model.visual.training
    assert not hasattr(parent, "dino_head")
    req = sample_requests(2, 3, (18, 24), 96, patch_generator=torch.Generator())
    target = {"ego": torch.randn(2, 8, 4), "current_dense_features": torch.randn(2, 3, 18, 24, 2),
              "current_valid": torch.ones(2, 3, 18, 24, dtype=torch.bool)}
    _, mask = gather_current(target["current_dense_features"], target["current_valid"], req)
    counts = optimizer_batch_counts(2, current_valid=mask)
    counts["ego_scenes"] = 2
    output = model(inputs(), target, requests=req, global_counts=counts,
                   noise_generator=torch.Generator().manual_seed(2), time_generator=torch.Generator().manual_seed(3))
    output["loss"].backward()
    assert parent.calls == 1 and parent.action_model.calls[0][0] == 16
    assert set(output["losses"]) == {"ego_fm", "current"}
    assert int(output["metrics"]["current"].global_count) == 2 * 3 * 96 * 2
    for module in (parent.language, parent.action_model, model.sparse_readout):
        assert all(p.grad is not None and p.grad.isfinite().all() for p in module.parameters())
    assert parent.foresight_queries.grad.abs().sum() > 0
    assert all(p.grad is None for p in parent.qwen_vl_interface.model.model.visual.parameters())


def test_modelinputs_reject_label_injection_before_encoder():
    obs = inputs(1)[0]
    with pytest.raises(ValueError, match="permits only"):
        ModelInputs({**obs.current, "future_target": torch.randn(8, 3)})
