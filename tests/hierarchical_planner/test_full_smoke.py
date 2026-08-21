from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from starVLA.model.framework.QwenPI_DrivoRSuprim import QwenPIDrivoRSuprim
from starVLA.model.modules.trajectory_scorer.losses import SUPRIM_METRICS
from starVLA.training.hierarchical_schedule import HierarchicalTrainingSchedule
from starVLA.training.navsim_metric_supervisor import StubDynamicMetricSupervisor
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


class FakeFrozenQwen(nn.Module):
    def __init__(self, layers=2, width=12, length=7):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.layers = layers
        self.width = width
        self.length = length
        self.forward_calls = 0

    def build_qwenvl_inputs(self, images, instructions):
        return {
            "batch_size": torch.tensor(len(images)),
            "attention_mask": torch.ones(len(images), self.length, dtype=torch.long),
        }

    def forward(self, batch_size, attention_mask, **kwargs):
        self.forward_calls += 1
        batch = int(batch_size.item())
        base = torch.arange(
            batch * self.length * self.width, dtype=torch.float32
        ).reshape(batch, self.length, self.width)
        hidden = tuple((base + index) * self.scale for index in range(self.layers))
        return SimpleNamespace(hidden_states=hidden)


class FakeFlowDiT(nn.Module):
    action_dim = 4
    action_horizon = 8

    def __init__(self, qwen_dim=12, scene_dim=64, layers=2):
        super().__init__()
        holder = nn.Module()
        holder.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(layers)])
        self.model = holder
        self.qwen_proj = nn.Linear(qwen_dim, 16)
        self.scene_proj = nn.Linear(scene_dim, 16)
        self.dit_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, vl_embs_list, actions, state=None, global_scene_tokens=None):
        qwen_term = sum(self.qwen_proj(value).mean() for value in vl_embs_list)
        scene_term = self.scene_proj(global_scene_tokens).mean()
        state_term = state.mean() if state is not None else actions.new_zeros(())
        prediction = self.dit_weight * actions.mean() + qwen_term + scene_term + state_term
        return prediction.square()

    @torch.no_grad()
    def predict_multi_action(
        self,
        vl_embs_list,
        state=None,
        global_scene_tokens=None,
        num_candidates=4,
        candidate_chunk_size=2,
        initial_noise=None,
    ):
        batch = vl_embs_list[0].shape[0]
        if initial_noise is None:
            initial_noise = torch.randn(
                batch,
                num_candidates,
                self.action_horizon,
                self.action_dim,
                device=vl_embs_list[0].device,
                dtype=vl_embs_list[0].dtype,
            )
        return initial_noise + self.dit_weight.detach().tanh() * 0.01


class FakeStaticStore:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def get(self, tokens, *, device, dtype):
        base = torch.linspace(0.1, 0.9, self.vocab_size, device=device, dtype=dtype)
        return {
            name: base[None].expand(len(tokens), -1).clone()
            for name in SUPRIM_METRICS
        }


def _config():
    return OmegaConf.create(
        {
            "framework": {
                "name": "QwenPI-DrivoRSuprim",
                "qwenvl": {"vl_hidden_dim": 12, "base_vlm": "fake"},
                "scene_encoder": {
                    "input_dim": 12,
                    "hidden_dim": 64,
                    "output_dim": 64,
                    "num_queries": 4,
                    "num_layers": 2,
                    "num_heads": 4,
                    "ffn_dim": 128,
                    "dropout": 0.0,
                    "use_gradient_checkpointing": False,
                },
                "action_model": {
                    "hidden_size": 64,
                    "action_dim": 4,
                    "action_horizon": 8,
                    "state_dim": 4,
                    "future_action_window_size": 8,
                    "diffusion_model_cfg": {"num_layers": 2},
                },
                "hierarchical_scorer": {
                    "detach_scene_for_scorer": False,
                    "scorer_guides_dit": False,
                    "dynamic": {
                        "num_candidates": 4,
                        "candidate_chunk_size": 2,
                        "final_topm": 2,
                        "model_dim": 32,
                        "ffn_dim": 64,
                        "num_layers": 1,
                        "num_heads": 4,
                    },
                    "joint": {
                        "vocab_size": 16,
                        "vocab_num_poses": 40,
                        "model_dim": 32,
                        "ffn_dim": 64,
                        "num_heads": 4,
                        "coarse_layers": 1,
                        "coarse_topk": 4,
                        "sigma": 0.5,
                    },
                    "refinement": {
                        "num_layers": 2,
                        "use_mid_output": True,
                        "use_imitation": True,
                    },
                },
                "static_score_store": {
                    "cache_root": "unused-with-injected-store",
                    "split": "train",
                },
            },
            "datasets": {"vla_data": {}},
            "trainer": {
                "repeated_diffusion_steps": 1,
                "freeze_modules": "qwen_vl_interface",
                "learning_rate": {
                    "base": 1e-4,
                    "scene_encoder": 1e-4,
                    "action_model": 1e-4,
                    "hierarchical_scorer": 1e-4,
                },
            },
        }
    )


def _examples(batch=2):
    values = []
    for index in range(batch):
        heading = np.linspace(-0.3, 0.3, 8, dtype=np.float32)
        action = np.zeros((8, 4), dtype=np.float32)
        action[:, 0] = np.linspace(-0.5, 0.5, 8)
        action[:, 1] = index * 0.1
        action[:, 2] = np.sin(heading)
        action[:, 3] = np.cos(heading)
        values.append(
            {
                "image": [object()],
                "lang": "keep straight",
                "state": np.zeros((1, 4), dtype=np.float32),
                "action": action,
                "token": f"token-{index}",
            }
        )
    return values


def _model():
    return QwenPIDrivoRSuprim(
        _config(),
        qwen_vl_interface=FakeFrozenQwen(),
        action_model=FakeFlowDiT(),
        static_vocab=torch.randn(16, 40, 3),
        static_score_store=FakeStaticStore(16),
    )


def test_full_forward_backward_optimizer_step_and_qwen_freeze():
    torch.manual_seed(8)
    model = _model().train()
    assert not model.qwen_vl_interface.training
    assert all(not parameter.requires_grad for parameter in model.qwen_vl_interface.parameters())
    schedule = HierarchicalTrainingSchedule(
        progress=1.0,
        dynamic_enabled=True,
        num_dynamic_candidates=4,
        dynamic_topm=2,
        lambda_flow=1.0,
        lambda_drivor=1.0,
        lambda_suprim_coarse=1.0,
        lambda_suprim_fine=1.0,
    )
    supervisor = StubDynamicMetricSupervisor()
    output = model(
        _examples(), training_schedule=schedule, metric_supervisor=supervisor
    )
    assert set(output["losses"]) == {
        "flow",
        "drivor",
        "suprim_coarse",
        "suprim_fine",
    }
    assert all(torch.isfinite(value) for value in output["losses"].values())
    total = sum(output["losses"].values())
    groups = build_param_lr_groups(model, model.config)
    optimizer = torch.optim.AdamW(groups)
    optimizer.zero_grad()
    total.backward()
    assert model.qwen_vl_interface.scale.grad is None
    assert model.scene_encoder.input_proj.weight.grad is not None
    assert model.action_model.dit_weight.grad is not None
    assert model.hierarchical_scorer.dynamic_prescorer.metric_heads[
        "comfort"
    ][0].weight.grad is not None
    optimizer.step()
    assert model.qwen_vl_interface.forward_calls == 1
    assert supervisor.calls == 1


def test_individual_loss_gradient_boundaries():
    schedule = HierarchicalTrainingSchedule(
        progress=1.0,
        dynamic_enabled=True,
        num_dynamic_candidates=4,
        dynamic_topm=2,
        lambda_flow=1.0,
        lambda_drivor=1.0,
        lambda_suprim_coarse=1.0,
        lambda_suprim_fine=1.0,
    )
    for loss_name in ("flow", "drivor", "suprim_coarse", "suprim_fine"):
        model = _model().train()
        output = model(
            _examples(),
            training_schedule=schedule,
            metric_supervisor=StubDynamicMetricSupervisor(),
        )
        output["losses"][loss_name].backward()
        action_grad = model.action_model.dit_weight.grad
        scene_grad = model.scene_encoder.input_proj.weight.grad
        if loss_name == "flow":
            assert action_grad is not None and torch.count_nonzero(action_grad)
            assert scene_grad is not None and torch.count_nonzero(scene_grad)
            assert all(
                parameter.grad is None
                for parameter in model.hierarchical_scorer.parameters()
            )
        else:
            assert action_grad is None
            assert scene_grad is not None and torch.count_nonzero(scene_grad)
        assert model.qwen_vl_interface.scale.grad is None


def test_inference_calls_qwen_and_qformer_once_without_metric_supervisor():
    model = _model().eval()
    qformer_calls = []
    handle = model.scene_encoder.register_forward_hook(
        lambda *_args: qformer_calls.append(1)
    )
    try:
        result = model.predict_action(examples=_examples())
    finally:
        handle.remove()
    assert model.qwen_vl_interface.forward_calls == 1
    assert len(qformer_calls) == 1
    assert result["normalized_actions"].shape == (2, 8, 4)
    assert result["trajectory_navsim_8"].shape == (2, 8, 3)
    assert result["trajectory_navsim_40"].shape == (2, 40, 3)


def test_inference_does_not_construct_static_metric_store():
    config = _config()
    config.framework.static_score_store.cache_root = "/definitely/missing/pdm-cache"
    model = QwenPIDrivoRSuprim(
        config,
        qwen_vl_interface=FakeFrozenQwen(),
        action_model=FakeFlowDiT(),
        static_vocab=torch.randn(16, 40, 3),
        static_score_store=None,
    ).eval()

    result = model.predict_action(examples=_examples(batch=1))

    assert model.static_score_store is None
    assert result["normalized_actions"].shape == (1, 8, 4)
