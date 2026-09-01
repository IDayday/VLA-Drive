from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.layers.planning_registers import (
    InternVLPlanningRegisters,
)
from navsim.agents.EpisodeDrive.drivevla_backbone import DriveVLABackbone


class _FakeEmbeddings(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(3, hidden_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = pixel_values.permute(0, 2, 3, 1).reshape(
            pixel_values.shape[0], -1, 3
        )
        patches = self.patch_projection(patches)
        return torch.cat((self.cls.expand(pixel_values.shape[0], -1, -1), patches), dim=1)


class _FakeEncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden_states + 0.01 * self.projection(hidden_states)


class _FakeEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeEncoderBlock(hidden_dim) for _ in range(layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return SimpleNamespace(last_hidden_state=hidden_states)


class _FakeVision(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_dim)
        self.embeddings = _FakeEmbeddings(hidden_dim)
        self.encoder = _FakeEncoder(hidden_dim)


class _RecordingMLP(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.last_shape = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.last_shape = tuple(inputs.shape)
        return self.projection(inputs)


class _FakeInternVL(nn.Module):
    def __init__(self, hidden_dim: int = 8, output_dim: int = 12) -> None:
        super().__init__()
        self.vision_model = _FakeVision(hidden_dim)
        self.select_layer = -1
        self.downsample_ratio = 1.0
        self.mlp1 = _RecordingMLP(hidden_dim, output_dim)
        self.pixel_shuffle_input_shape = None
        self.language_model = _FakeLanguageModel(output_dim)
        self.img_context_token_id = 9

    def pixel_shuffle(self, inputs: torch.Tensor, scale_factor: float):
        assert scale_factor == 1.0
        self.pixel_shuffle_input_shape = tuple(inputs.shape)
        return inputs

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.vision_model.embeddings(pixel_values)
        encoded = self.vision_model.encoder(
            inputs_embeds=embeddings,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state[:, 1:]
        grid = encoded.reshape(encoded.shape[0], 2, 2, encoded.shape[-1])
        shuffled = self.pixel_shuffle(grid, scale_factor=self.downsample_ratio)
        return self.mlp1(shuffled.reshape(shuffled.shape[0], -1, shuffled.shape[-1]))


class _FakeLanguageModel(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(32, hidden_dim)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, inputs_embeds: torch.Tensor, **kwargs):
        assert kwargs["output_hidden_states"]
        assert kwargs["return_dict"]
        return SimpleNamespace(hidden_states=(inputs_embeds, inputs_embeds + 1.0))


def test_register_and_patch_shapes_and_tile_mean() -> None:
    torch.manual_seed(4)
    model = _FakeInternVL()
    adapter = InternVLPlanningRegisters(
        vision_hidden_dim=8,
        num_registers=16,
        register_dim=256,
        tile_aggregation="mean",
    )
    pixels = torch.randn(3, 3, 2, 2)
    output = adapter(model, pixels, [1, 2])

    assert output.per_tile_registers.shape == (3, 16, 256)
    assert output.scene_registers.shape == (2, 16, 256)
    assert output.encoded_patches.shape == (3, 4, 8)
    assert output.patch_features.shape == (3, 4, 12)
    assert 0.25e-6 < adapter.planning_registers.detach().std().item() < 2.0e-6
    torch.testing.assert_close(
        output.scene_registers[1],
        output.per_tile_registers[1:].mean(dim=0),
    )
    assert all(layer.calls == 1 for layer in model.vision_model.encoder.layers)


def test_registers_never_enter_pixel_shuffle_or_mlp1() -> None:
    model = _FakeInternVL()
    adapter = InternVLPlanningRegisters(8, 16, 256)
    output = adapter(model, torch.randn(2, 3, 2, 2), [2])

    assert model.pixel_shuffle_input_shape == (2, 2, 2, 8)
    assert model.mlp1.last_shape == (2, 4, 8)
    assert output.encoded_patches.shape[1] == 4


def test_patch_only_semantic_path_keeps_legacy_dimensions() -> None:
    torch.manual_seed(8)
    model = _FakeInternVL()
    pixels = torch.randn(2, 3, 2, 2)
    legacy_features = model.extract_feature(pixels)
    adapter = InternVLPlanningRegisters(8, 16, 256)
    planning_features = adapter(model, pixels, [1, 1]).patch_features
    assert planning_features.shape == legacy_features.shape == (2, 4, 12)


def test_runtime_contract_fails_without_inputs_embeds() -> None:
    class WrongEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity()])

        def forward(self, pixel_values):
            return pixel_values

    model = _FakeInternVL()
    model.vision_model.encoder = WrongEncoder()
    adapter = InternVLPlanningRegisters(8, 16, 256)
    with pytest.raises(RuntimeError, match="inputs_embeds"):
        adapter(model, torch.randn(1, 3, 2, 2), [1])


def test_backbone_returns_hidden_state_and_planning_register_dict() -> None:
    model = _FakeInternVL()
    backbone = DriveVLABackbone.__new__(DriveVLABackbone)
    nn.Module.__init__(backbone)
    backbone.model = model
    backbone.skip_lm_head = False
    backbone.planning_registers_enabled = True
    backbone.planning_register_adapter = InternVLPlanningRegisters(8, 16, 256)

    input_ids = torch.tensor([[1, 9, 9, 9, 9, 2]])
    attention_mask = torch.ones_like(input_ids)
    result = backbone.forward_internvl_with_planning_registers(
        pixel_values=torch.randn(1, 3, 2, 2),
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0),
        image_flags=torch.ones(1, dtype=torch.long),
        num_patches_list=[1],
    )
    assert result["last_hidden_state"].shape == (1, 6, 12)
    assert result["planning_registers"].shape == (1, 16, 256)
    assert result["per_tile_registers"].shape == (1, 16, 256)


def test_tile_count_mismatch_is_not_silently_accepted() -> None:
    model = _FakeInternVL()
    adapter = InternVLPlanningRegisters(8, 16, 256)
    with pytest.raises(ValueError, match="aggregation mismatch"):
        adapter(model, torch.randn(2, 3, 2, 2), [1])
