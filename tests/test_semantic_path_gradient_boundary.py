from types import MethodType, SimpleNamespace

import torch
from torch import nn

from navsim.agents.EpisodeDrive.drivevla_backbone import DriveVLABackbone


def _boundary_backbone(*, no_grad: bool, backprop_to_vision: bool):
    backbone = DriveVLABackbone.__new__(DriveVLABackbone)
    nn.Module.__init__(backbone)
    backbone.semantic_frozen_llm_no_grad = no_grad
    backbone.semantic_backprop_to_vision = backprop_to_vision
    backbone.fake_llm = nn.Linear(4, 4, bias=False)

    def fake_forward(self, patch_features, *_args):
        return SimpleNamespace(hidden_states=(self.fake_llm(patch_features),))

    backbone._forward_internvl_from_patch_features = MethodType(
        fake_forward, backbone
    )
    return backbone


def _placeholder_inputs():
    integer = torch.ones(1, 1, dtype=torch.long)
    return integer, integer, integer, integer


def test_frozen_llm_boundary_blocks_vision_and_llm_gradients_but_not_qformer():
    backbone = _boundary_backbone(no_grad=True, backprop_to_vision=False)
    patch_features = torch.randn(2, 3, 4, requires_grad=True)
    language_output = backbone._forward_semantic_language_path(
        patch_features, *_placeholder_inputs()
    )
    llm_hidden = language_output.hidden_states[-1]
    assert not llm_hidden.requires_grad

    qformer = nn.Linear(4, 4)
    semantic_tokens = qformer(llm_hidden)
    semantic_tokens.square().mean().backward()
    assert patch_features.grad is None
    assert backbone.fake_llm.weight.grad is None
    assert qformer.weight.grad is not None and qformer.weight.grad.abs().sum() > 0


def test_nonformal_boundary_can_preserve_vision_gradient():
    backbone = _boundary_backbone(no_grad=False, backprop_to_vision=True)
    patch_features = torch.randn(2, 3, 4, requires_grad=True)
    output = backbone._forward_semantic_language_path(
        patch_features, *_placeholder_inputs()
    )
    output.hidden_states[-1].sum().backward()
    assert patch_features.grad is not None and patch_features.grad.abs().sum() > 0
    assert backbone.fake_llm.weight.grad is not None
