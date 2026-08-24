from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from starVLA.model.framework.QwenPI_DrivoRSuprim import QwenPIDrivoRSuprim
from starVLA.model.framework.baseline_qwen import (
    apply_baseline_qwen_trainability,
    get_frozen_parameter_names,
    get_trainable_parameter_names,
)
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
)
from starVLA.model.modules.trajectory_scorer.losses import SUPRIM_METRICS
from starVLA.training.hierarchical_schedule import HierarchicalTrainingSchedule
from starVLA.training.navsim_metric_supervisor import StubDynamicMetricSupervisor
from starVLA.training.train_starvla import _combine_hierarchical_losses
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


class _FakeQwenCore(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width)
        self.visual = nn.Linear(width, width, bias=False)
        self.embed_tokens = nn.Linear(width, width, bias=False)
        self.language_model = nn.Linear(width, width, bias=False)
        self.lm_head = nn.Linear(width, width, bias=False)
        self.lm_head.weight = self.embed_tokens.weight


class FakeBaselineQwen(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.model = _FakeQwenCore(width)
        self.forward_calls = 0


def fake_baseline_features(framework, examples):
    qwen = framework.qwen_vl_interface
    qwen.forward_calls += 1
    core = qwen.model
    batch = len(examples)
    width = core.config.hidden_size
    base = torch.arange(
        batch * 9 * width,
        dtype=core.language_model.weight.dtype,
        device=core.language_model.weight.device,
    ).reshape(batch, 9, width)
    hidden = core.language_model(
        core.embed_tokens(base) + core.visual(base)
    )
    return hidden[:, :8], hidden, torch.ones(
        batch, 9, dtype=torch.long, device=hidden.device
    )


class FakeFlowDiT(nn.Module):
    action_dim = 4
    action_horizon = 8
    hidden_size = 64
    num_inference_timesteps = 2

    def __init__(self, qwen_dim=12, scene_dim=32, layers=2):
        super().__init__()
        holder = nn.Module()
        holder.transformer_blocks = nn.ModuleList(
            [nn.Identity() for _ in range(layers)]
        )
        self.model = holder
        self.qwen_proj = nn.Linear(qwen_dim, 16)
        self.scene_to_dit = nn.Sequential(
            nn.Linear(scene_dim, 16), nn.LayerNorm(16)
        )
        self.dit_weight = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        action_conditions,
        actions,
        video_token=None,
        state=None,
        global_scene_tokens=None,
    ):
        qwen_term = self.qwen_proj(action_conditions).mean()
        scene_term = (
            self.scene_to_dit(global_scene_tokens).mean()
            if global_scene_tokens is not None
            else actions.new_zeros(())
        )
        prediction = self.dit_weight * actions.mean() + qwen_term + scene_term
        return prediction.square()

    @torch.no_grad()
    def predict_multi_action(
        self,
        action_conditions,
        state=None,
        global_scene_tokens=None,
        num_candidates=4,
        candidate_chunk_size=2,
        initial_noise=None,
    ):
        batch = action_conditions.shape[0]
        if initial_noise is None:
            initial_noise = torch.randn(
                batch,
                num_candidates,
                self.action_horizon,
                self.action_dim,
                device=action_conditions.device,
                dtype=action_conditions.dtype,
            )
        return initial_noise + self.dit_weight.detach().tanh() * 0.01

    @torch.no_grad()
    def predict_action(
        self, action_conditions, state=None, global_scene_tokens=None
    ):
        return self.predict_multi_action(
            action_conditions,
            state=state,
            global_scene_tokens=global_scene_tokens,
            num_candidates=1,
            candidate_chunk_size=1,
        )[:, 0]


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
            "act_tok": 8,
            "framework": {
                "name": "QwenPI-DrivoRSuprim",
                "qwenvl": {"vl_hidden_dim": 12, "base_vlm": "fake"},
                "scene_encoder": {
                    "enabled": True,
                    "input_dim": 12,
                    "hidden_dim": 32,
                    "output_dim": 32,
                    "num_queries": 4,
                    "num_layers": 2,
                    "num_heads": 4,
                    "ffn_dim": 64,
                    "dropout": 0.0,
                    "detach_qwen_input": True,
                    "use_gradient_checkpointing": False,
                },
                "action_model": {
                    "hidden_size": 64,
                    "use_global_scene_tokens": True,
                    "scene_dim": 32,
                    "action_dim": 4,
                    "action_horizon": 8,
                    "state_dim": 4,
                    "flow_train_repeats": 8,
                    "num_inference_timesteps": 2,
                    "diffusion_model_cfg": {"num_layers": 2},
                },
                "hierarchical_scorer": {
                    "enabled": True,
                    "detach_scene_for_scorer": False,
                    "scorer_guides_dit": False,
                    "dynamic": {
                        "enabled": True,
                        "num_candidates": 4,
                        "candidate_chunk_size": 2,
                        "dynamic_topm": 2,
                        "model_dim": 32,
                        "ffn_dim": 64,
                        "num_layers": 1,
                        "num_heads": 4,
                    },
                    "joint": {
                        "enabled": True,
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
                        "num_stages": 1,
                        "num_layers": 2,
                        "memory_source": "dense_scene_memory",
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
                "freeze_modules": (
                    "qwen_vl_interface.model.visual,"
                    "qwen_vl_interface.model.lm_head"
                ),
                "learning_rate": {
                    "base": 1e-4,
                    "qwen_vl_interface": 1e-5,
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


def _model(config=None, *, static_store=True):
    config = config or _config()
    return QwenPIDrivoRSuprim(
        config,
        qwen_vl_interface=FakeBaselineQwen(),
        action_model=FakeFlowDiT(),
        static_vocab=torch.randn(16, 40, 3),
        static_score_store=FakeStaticStore(16) if static_store else None,
        qwen_feature_extractor=fake_baseline_features,
    )


def _schedule():
    return HierarchicalTrainingSchedule(
        progress=1.0,
        dynamic_enabled=True,
        num_dynamic_candidates=4,
        dynamic_topm=2,
        lambda_flow=1.0,
        lambda_drivor=1.0,
        lambda_suprim_coarse=1.0,
        lambda_suprim_fine=1.0,
    )


def test_qwen_trainable_parameter_manifest_matches_baseline():
    config = _config()
    baseline_qwen = FakeBaselineQwen()
    baseline = apply_baseline_qwen_trainability(baseline_qwen, config)
    model = _model(config)
    assert model.baseline_qwen_trainable_names == set(baseline.trainable_names)
    assert model.baseline_qwen_frozen_names == set(baseline.frozen_names)
    assert get_trainable_parameter_names(model.qwen_vl_interface) == set(
        baseline.trainable_names
    )
    assert get_frozen_parameter_names(model.qwen_vl_interface) == set(
        baseline.frozen_names
    )
    model.assert_qwen_trainability()


def test_baseline_qwen_modes_and_gradient_boundary_are_preserved():
    model = _model().train()
    core = model.qwen_vl_interface.model
    # Baseline freezes requires_grad only; parent train() still reaches these modules.
    assert core.visual.training and core.lm_head.training
    assert all(not p.requires_grad for p in core.visual.parameters())
    assert all(not p.requires_grad for p in core.lm_head.parameters())
    assert all(p.requires_grad for p in core.language_model.parameters())

    output = model(
        _examples(),
        training_schedule=_schedule(),
        metric_supervisor=StubDynamicMetricSupervisor(),
    )
    output["losses"]["flow"].backward()
    assert core.language_model.weight.grad is not None
    assert torch.count_nonzero(core.language_model.weight.grad)
    assert core.visual.weight.grad is None
    assert core.lm_head.weight.grad is None


def test_ego_state_encoder_matches_deepspeed_bf16_parameter_dtype():
    model = _model()
    model.action_input_model.to(dtype=torch.bfloat16)
    text_embeds = torch.zeros(2, 9, 12, dtype=torch.bfloat16)
    encoded = model._encode_ego_state_for_qwen(
        [example["state"] for example in _examples()], text_embeds
    )
    assert encoded.shape == (2, 12)
    assert encoded.dtype == torch.bfloat16


def test_full_forward_single_total_backward_and_optimizer_step():
    torch.manual_seed(8)
    model = _model().train()
    supervisor = StubDynamicMetricSupervisor()
    output = model(
        _examples(), training_schedule=_schedule(), metric_supervisor=supervisor
    )
    assert set(output["losses"]) == {
        "flow",
        "drivor",
        "suprim_coarse",
        "suprim_fine",
    }
    assert all(torch.isfinite(value) for value in output["losses"].values())
    total = _combine_hierarchical_losses(output["losses"], _schedule())
    groups = build_param_lr_groups(model, model.config)
    optimizer = torch.optim.AdamW(groups)
    optimizer.zero_grad()
    total.backward()
    assert model.qwen_vl_interface.model.language_model.weight.grad is not None
    assert model.scene_encoder.input_proj.weight.grad is not None
    assert model.action_model.dit_weight.grad is not None
    assert model.hierarchical_scorer.dynamic_prescorer.metric_heads[
        "comfort"
    ][0].weight.grad is not None
    optimizer.step()
    assert model.qwen_vl_interface.forward_calls == 1
    assert supervisor.calls == 1


def test_individual_loss_gradient_boundaries():
    for loss_name in ("flow", "drivor", "suprim_coarse", "suprim_fine"):
        model = _model().train()
        output = model(
            _examples(),
            training_schedule=_schedule(),
            metric_supervisor=StubDynamicMetricSupervisor(),
        )
        output["losses"][loss_name].backward()
        action_grad = model.action_model.dit_weight.grad
        scene_grad = model.scene_encoder.input_proj.weight.grad
        qwen_grad = model.qwen_vl_interface.model.language_model.weight.grad
        if loss_name == "flow":
            assert action_grad is not None and torch.count_nonzero(action_grad)
            assert scene_grad is not None and torch.count_nonzero(scene_grad)
            assert qwen_grad is not None and torch.count_nonzero(qwen_grad)
            assert all(
                parameter.grad is None
                for parameter in model.hierarchical_scorer.parameters()
            )
        else:
            assert action_grad is None
            assert scene_grad is not None and torch.count_nonzero(scene_grad)
            # detach_qwen_input=true blocks only the new scene/scorer path.
            assert qwen_grad is None


def _backward_one(loss_name):
    model = _model().train()
    output = model(
        _examples(),
        training_schedule=_schedule(),
        metric_supervisor=StubDynamicMetricSupervisor(),
    )
    output["losses"][loss_name].backward()
    return model


def test_flow_loss_updates_baseline_qwen_trainable_subset():
    model = _backward_one("flow")
    core = model.qwen_vl_interface.model
    assert core.language_model.weight.grad is not None
    assert core.visual.weight.grad is None
    assert core.lm_head.weight.grad is None


def test_flow_loss_updates_qformer_and_dit():
    model = _backward_one("flow")
    assert model.scene_encoder.input_proj.weight.grad is not None
    assert model.action_model.dit_weight.grad is not None


def test_scorer_losses_do_not_update_dit():
    for loss_name in ("drivor", "suprim_coarse", "suprim_fine"):
        assert _backward_one(loss_name).action_model.dit_weight.grad is None


def test_scorer_losses_update_qformer():
    for loss_name in ("drivor", "suprim_coarse", "suprim_fine"):
        assert _backward_one(loss_name).scene_encoder.input_proj.weight.grad is not None


def test_scene_branch_does_not_add_qwen_grad_when_detached():
    for loss_name in ("drivor", "suprim_coarse", "suprim_fine"):
        model = _backward_one(loss_name)
        assert model.qwen_vl_interface.model.language_model.weight.grad is None


def test_drivor_loss_does_not_update_flow_dit():
    assert _backward_one("drivor").action_model.dit_weight.grad is None


def test_drivor_loss_updates_qformer():
    assert _backward_one("drivor").scene_encoder.input_proj.weight.grad is not None


def test_single_total_backward():
    model = _model().train()
    output = model(
        _examples(),
        training_schedule=_schedule(),
        metric_supervisor=StubDynamicMetricSupervisor(),
    )
    total = _combine_hierarchical_losses(output["losses"], _schedule())
    total.backward()
    assert model.action_model.dit_weight.grad is not None
    assert model.hierarchical_scorer.dynamic_prescorer.metric_heads[
        "comfort"
    ][0].weight.grad is not None


def test_flow_train_repeat_expands_batch_only():
    model = _model()
    conditions = torch.randn(2, 8, 12)
    actions = torch.randn(2, 8, 4)
    scene = torch.randn(2, 4, 32)
    repeated_conditions, repeated_actions, repeated_scene = (
        model._repeat_flow_training_batch(conditions, actions, scene)
    )
    assert repeated_actions.shape == (16, 8, 4)
    assert repeated_conditions.shape == (16, 8, 12)
    assert repeated_scene.shape == (16, 4, 32)
    assert model.action_horizon == 8
    assert repeated_actions.ndim == 3  # no candidate axis


def test_flow_train_repeats_are_not_dynamic_candidates():
    model = _model()
    assert model.flow_train_repeats == 8
    assert model.num_dynamic_candidates == 4
    assert model._repeat_flow_training_batch.__name__ != "predict_multi_action"
    proposals = model.action_model.predict_multi_action(
        torch.randn(2, 8, 12),
        global_scene_tokens=torch.randn(2, 4, 32),
        num_candidates=4,
        candidate_chunk_size=2,
    )
    assert proposals.shape == (2, 4, 8, 4)


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
    model = _model(config, static_store=False).eval()
    result = model.predict_action(examples=_examples(batch=1))
    assert model.static_score_store is None
    assert result["normalized_actions"].shape == (1, 8, 4)


def test_single_trajectory_output_contract():
    config = _config()
    config.framework.scene_encoder.enabled = False
    config.framework.action_model.use_global_scene_tokens = False
    config.framework.hierarchical_scorer.enabled = False
    config.framework.hierarchical_scorer.dynamic.enabled = False
    config.framework.hierarchical_scorer.dynamic.num_candidates = 1
    config.framework.hierarchical_scorer.dynamic.dynamic_topm = 1
    config.framework.hierarchical_scorer.joint.enabled = False
    model = _model(config).eval()
    result = model.predict_action(examples=_examples(batch=2))
    assert result["normalized_actions"].shape == (2, 8, 4)
    assert result["trajectory_navsim_8"].shape == (2, 8, 3)
    assert result["dynamic_topm_indices"] is None


def test_small_model_gate_forward_backward_step_and_inference():
    config = _config()
    config.framework.qwenvl.vl_hidden_dim = 64
    config.framework.scene_encoder.input_dim = 64
    action = config.framework.action_model
    action.add_pos_embed = True
    action.max_seq_len = 32
    action.noise_beta_alpha = 1.5
    action.noise_beta_beta = 1.0
    action.noise_s = 0.999
    action.num_timestep_buckets = 1000
    action.DiTConfig = {
        "num_layers": 2,
        "input_embedding_dim": 64,
        "attention_head_dim": 64,
        "num_attention_heads": 1,
    }
    action.diffusion_model_cfg = {
        "num_layers": 2,
        "cross_attention_dim": 64,
        "dropout": 0.0,
        "final_dropout": False,
        "interleave_self_attention": False,
        "norm_type": "ada_norm",
        "output_dim": 64,
        "positional_embeddings": None,
    }
    real_flow = FlowmatchingActionHead(config)
    model = QwenPIDrivoRSuprim(
        config,
        qwen_vl_interface=FakeBaselineQwen(width=64),
        action_model=real_flow,
        static_vocab=torch.randn(16, 40, 3),
        static_score_store=FakeStaticStore(16),
        qwen_feature_extractor=fake_baseline_features,
    ).train()
    output = model(
        _examples(batch=1),
        training_schedule=_schedule(),
        metric_supervisor=StubDynamicMetricSupervisor(),
    )
    total = _combine_hierarchical_losses(output["losses"], _schedule())
    optimizer = torch.optim.AdamW(build_param_lr_groups(model, config))
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    assert torch.isfinite(total)
    model.eval()
    prediction = model.predict_action(examples=_examples(batch=1))
    assert prediction["normalized_actions"].shape == (1, 8, 4)
    assert prediction["trajectory_navsim_8"].shape == (1, 8, 3)


def test_dynamic_metric_scoring_overlaps_flow_forward(monkeypatch):
    model = _model().train()
    events = []
    original_flow_loss = model._flow_loss
    original_dynamic_forward = model.hierarchical_scorer.dynamic_prescorer.forward
    original_coarse_forward = model.hierarchical_scorer.joint_coarse_scorer.forward

    def tracked_flow_loss(*args, **kwargs):
        events.append("flow_forward")
        return original_flow_loss(*args, **kwargs)

    monkeypatch.setattr(model, "_flow_loss", tracked_flow_loss)

    def tracked_dynamic_forward(*args, **kwargs):
        events.append("dynamic_prescorer")
        return original_dynamic_forward(*args, **kwargs)

    def tracked_coarse_forward(*args, **kwargs):
        events.append("coarse_scorer")
        return original_coarse_forward(*args, **kwargs)

    monkeypatch.setattr(
        model.hierarchical_scorer.dynamic_prescorer,
        "forward",
        tracked_dynamic_forward,
    )
    monkeypatch.setattr(
        model.hierarchical_scorer.joint_coarse_scorer,
        "forward",
        tracked_coarse_forward,
    )

    class PendingTargets:
        def __init__(self, tokens, proposals):
            self.tokens = tokens
            self.proposals = proposals

        def result(self):
            events.append("metric_wait")
            assert "flow_forward" in events
            assert "dynamic_prescorer" in events
            assert "coarse_scorer" in events
            return StubDynamicMetricSupervisor().score(
                self.tokens, self.proposals
            )

    class AsyncSupervisor:
        def score(self, *_args, **_kwargs):
            raise AssertionError("the synchronous scoring path must not run")

        def score_async(self, tokens, proposals):
            events.append("metric_submit")
            return PendingTargets(tokens, proposals)

    output = model(
        _examples(),
        training_schedule=_schedule(),
        metric_supervisor=AsyncSupervisor(),
    )

    assert torch.isfinite(output["losses"]["flow"])
    assert events.index("metric_submit") < events.index("flow_forward")
    assert events.index("flow_forward") < events.index("dynamic_prescorer")
    assert events.index("dynamic_prescorer") < events.index("coarse_scorer")
    assert events.index("coarse_scorer") < events.index("metric_wait")
