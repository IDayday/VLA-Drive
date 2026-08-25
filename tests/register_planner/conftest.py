from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn


class TinyQwenBackbone(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_dim)
        self.visual = nn.Linear(hidden_dim, hidden_dim)
        self.language = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, hidden_dim, bias=False)


class TinyQwenInterface(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.model = TinyQwenBackbone(hidden_dim)


def tiny_hidden_extractor(framework, examples):
    parameter = framework.qwen_vl_interface.model.language.weight
    batch = len(examples)
    length = 12
    values = torch.stack(
        [torch.as_tensor(example["state"], dtype=parameter.dtype) for example in examples]
    ).to(parameter.device)
    seed = values.mean(dim=-1, keepdim=True).unsqueeze(-1).expand(
        batch, length, parameter.shape[1]
    )
    hidden = framework.qwen_vl_interface.model.language(seed)
    mask = torch.ones(batch, length, device=hidden.device, dtype=torch.bool)
    return hidden, mask


def make_config(proposal_num: int = 4, selector_type: str = "drivor"):
    return OmegaConf.create(
        {
            "act_tok": 8,
            "framework": {
                "name": "QwenRegisterGenerator",
                "qwenvl": {
                    "base_vlm": "tiny-qwen",
                    "vl_hidden_dim": 64,
                    "attn_implementation": "sdpa",
                },
                "scene_encoder": {
                    "input_dim": 64,
                    "hidden_dim": 32,
                    "output_dim": 32,
                    "num_queries": 4,
                    "num_layers": 2,
                    "num_heads": 4,
                    "ffn_dim": 64,
                    "dropout": 0.0,
                    "detach_qwen_input": False,
                },
                "register_generator": {
                    "proposal_num": proposal_num,
                    "num_poses": 8,
                    "state_dim": 3,
                    "model_dim": 32,
                    "ffn_dim": 64,
                    "num_layers": 2,
                    "num_heads": 1,
                    "one_token_per_trajectory": True,
                    "proj_drop": 0.0,
                    "drop_path": 0.0,
                    "layer_scale_init": 0.0,
                    "ego_state_dim": 4,
                },
                "generator_loss": {
                    "stage_loss_mode": "final_only",
                    "diversity_weight": 0.0,
                },
                "drivor_scorer": {
                    "scene_dim": 32,
                    "ego_state_dim": 4,
                    "model_dim": 32,
                    "ffn_dim": 64,
                    "num_layers": 2,
                    "num_heads": 1,
                    "decoder_style": "donor_register",
                    "proj_drop": 0.0,
                    "drop_path": 0.0,
                },
                "suprim": {
                    "static_vocab_path": "unused",
                    "coarse": {
                        "static_vocab_size": 16,
                        "coarse_topk": 4,
                        "coarse_layers": 2,
                        "scene_dim": 32,
                        "model_dim": 32,
                        "ffn_dim": 64,
                        "num_heads": 1,
                    },
                    "fine": {
                        "scene_dim": 32,
                        "model_dim": 32,
                        "ffn_dim": 64,
                        "num_heads": 1,
                        "refinement_layers": 2,
                        "use_mid_output": True,
                        "use_imitation": True,
                    },
                },
                "inference": {
                    "selector_type": selector_type,
                    "dynamic_topm": min(4, proposal_num),
                    "fine_memory_source": "global_scene_tokens",
                    "return_all_proposals": True,
                },
            },
            "trainer": {
                "freeze_modules": (
                    "qwen_vl_interface.model.visual,"
                    "qwen_vl_interface.model.lm_head"
                )
            },
        }
    )


def examples(batch: int = 2, with_action: bool = True):
    result = []
    for index in range(batch):
        item = {
            "token": f"token-{index}",
            "lang": "go straight",
            "state": np.asarray([1.0, 0.0, 0.0, float(index)], dtype=np.float32),
        }
        if with_action:
            action = np.zeros((8, 4), dtype=np.float32)
            action[:, 3] = 1.0
            item["action"] = action
        result.append(item)
    return result


@pytest.fixture
def tiny_factory():
    return SimpleNamespace(
        qwen=TinyQwenInterface,
        extractor=tiny_hidden_extractor,
        config=make_config,
        examples=examples,
    )
