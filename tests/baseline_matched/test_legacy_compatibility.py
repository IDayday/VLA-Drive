from types import SimpleNamespace

from omegaconf import OmegaConf
from torch import nn


class _Tokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return list(range(100, 100 + len(tokens)))
        return 99


class _FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.config = SimpleNamespace(hidden_size=12)
        self.model.visual = nn.Linear(12, 12)
        self.model.lm_head = nn.Linear(12, 12)
        self.model.language_model = nn.Linear(12, 12)
        self.processor = SimpleNamespace(tokenizer=_Tokenizer())


class _FakeAction(nn.Module):
    action_dim = 4
    action_horizon = 8


def _legacy_config():
    return OmegaConf.create(
        {
            "act_tok": 8,
            "vit_pre": 0,
            "w_depth": 0,
            "w_video_latent": 0,
            "framework": {
                "qwenvl": {"vl_hidden_dim": 12, "base_vlm": "fake"},
                "action_model": {
                    "action_dim": 4,
                    "action_horizon": 8,
                    "hidden_size": 64,
                    "mlp_head": 0,
                    "diffusion_model_cfg": {"num_layers": 2},
                },
                "reward_model": {"diffusion_model_cfg": {"num_layers": 1}},
            },
            "datasets": {
                "vla_data": {"load_act_data": 1},
                "video_data": {"load_2d_data": 0},
                "gs_data": {"load_3d_data": 0},
                "reward_data": {"load_reward_data": 0},
            },
            "trainer": {
                "freeze_modules": (
                    "qwen_vl_interface.model.visual,"
                    "qwen_vl_interface.model.lm_head"
                )
            },
        }
    )


def test_legacy_framework_still_builds(monkeypatch):
    import starVLA.model.framework.QwenOFT as qwen_oft_module

    monkeypatch.setattr(qwen_oft_module, "get_vlm_model", lambda config: _FakeQwen())
    monkeypatch.setattr(
        qwen_oft_module, "get_action_model", lambda config: _FakeAction()
    )
    model = qwen_oft_module.Qwenvl_OFT(_legacy_config(), infer_not_load_wan=1)
    assert isinstance(model.action_model, _FakeAction)
    assert model.action_input_model.layer1.in_features == 4
    assert model.action_input_model.layer2.out_features == 12
    assert all(not p.requires_grad for p in model.qwen_vl_interface.model.visual.parameters())
    assert all(not p.requires_grad for p in model.qwen_vl_interface.model.lm_head.parameters())
    assert all(p.requires_grad for p in model.qwen_vl_interface.model.language_model.parameters())
