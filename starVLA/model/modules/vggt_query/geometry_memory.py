"""Build the shared V2 geometry memory from existing Qwen hidden states."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _batched_image_grid(
    image_grid_thw: torch.Tensor,
    *,
    batch_size: int,
    view_count: int,
) -> torch.Tensor:
    assert image_grid_thw.ndim in (2, 3), "image_grid_thw must be [B*V,3] or [B,V,3]"
    if image_grid_thw.ndim == 2:
        assert image_grid_thw.shape == (batch_size * view_count, 3)
        image_grid_thw = image_grid_thw.reshape(batch_size, view_count, 3)
    else:
        assert image_grid_thw.shape == (batch_size, view_count, 3)
    return image_grid_thw


def extract_qwen_spatial_memory(
    last_hidden: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    spatial_merge_size: int,
    view_count: int = 3,
    output_size: tuple[int, int] = (6, 10),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample Qwen visual language states to ``[B,V*R*C,H]``.

    This function consumes the image-placeholder states already produced by
    the single Qwen forward. It never reruns the visual encoder and never adds
    the 180 spatial slots to the language sequence.
    """

    assert last_hidden.ndim == 3, "last_hidden must be [B,L,H]"
    assert input_ids.shape == last_hidden.shape[:2], "input_ids/hidden shape mismatch"
    assert spatial_merge_size > 0 and view_count > 0
    rows, cols = int(output_size[0]), int(output_size[1])
    assert rows > 0 and cols > 0
    batch, _, hidden_dim = last_hidden.shape
    grids = _batched_image_grid(
        image_grid_thw, batch_size=batch, view_count=view_count
    ).to(device=input_ids.device)
    sample_outputs = []
    for sample_index in range(batch):
        visual_states = last_hidden[sample_index][input_ids[sample_index].eq(image_token_id)]
        expected_counts = []
        for view_index in range(view_count):
            temporal, height, width = [
                int(value) for value in grids[sample_index, view_index].tolist()
            ]
            assert temporal == 1, "NAVSIM image contract expects one frame per Qwen image"
            assert height % spatial_merge_size == 0 and width % spatial_merge_size == 0
            expected_counts.append(
                temporal * height * width // (spatial_merge_size**2)
            )
        if int(visual_states.shape[0]) != sum(expected_counts):
            raise RuntimeError(
                "Qwen visual placeholder/grid mismatch: "
                f"sample={sample_index} placeholders={visual_states.shape[0]} "
                f"grid_tokens={sum(expected_counts)}"
            )
        offset = 0
        view_outputs = []
        for view_index, count in enumerate(expected_counts):
            _, height, width = [
                int(value) for value in grids[sample_index, view_index].tolist()
            ]
            map_height = height // spatial_merge_size
            map_width = width // spatial_merge_size
            view = visual_states[offset : offset + count].reshape(
                1, map_height, map_width, hidden_dim
            )
            offset += count
            pooled = F.adaptive_avg_pool2d(
                view.permute(0, 3, 1, 2).float(), (rows, cols)
            )
            view_outputs.append(
                pooled.permute(0, 2, 3, 1).reshape(rows * cols, hidden_dim)
            )
        sample_outputs.append(torch.cat(view_outputs, dim=0))
    spatial = torch.stack(sample_outputs, dim=0).to(dtype=last_hidden.dtype)
    valid = torch.ones(spatial.shape[:2], device=spatial.device, dtype=torch.bool)
    assert spatial.shape == (batch, view_count * rows * cols, hidden_dim)
    return spatial, valid


class SharedGeometryAdapter(nn.Module):
    """Map global and spatial Qwen states to one shared geometry space."""

    def __init__(self, input_dim: int, memory_dim: int, expansion: int = 2) -> None:
        super().__init__()
        assert input_dim > 0 and memory_dim > 0 and expansion > 0
        self.input_dim = int(input_dim)
        self.memory_dim = int(memory_dim)
        inner_dim = self.memory_dim * int(expansion)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, self.memory_dim),
            nn.LayerNorm(self.memory_dim),
        )

    def forward(
        self,
        global_queries: torch.Tensor,
        spatial_queries: torch.Tensor,
    ) -> torch.Tensor:
        """Return geometry memory ``[B,G+S,Dm]``."""

        assert global_queries.ndim == spatial_queries.ndim == 3
        assert global_queries.shape[0] == spatial_queries.shape[0]
        assert global_queries.shape[2] == spatial_queries.shape[2] == self.input_dim
        combined = torch.cat((global_queries, spatial_queries), dim=1)
        # DeepSpeed BF16 converts these parameters before diagnostic inference.
        # Casting activations unconditionally to FP32 then gives LayerNorm a
        # Float input and BFloat16 affine parameters outside autocast.  Follow
        # the actual module compute dtype; losses and diagnostics still promote
        # their reductions to FP32 where numerical precision matters.
        compute_dtype = self.adapter[0].weight.dtype
        memory = self.adapter(combined.to(dtype=compute_dtype))
        assert memory.shape == (*combined.shape[:2], self.memory_dim)
        return memory
