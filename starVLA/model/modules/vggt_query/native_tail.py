"""Resume a frozen native VGGT aggregator after the layer-11 bottleneck."""

from __future__ import annotations

from typing import Iterable

import torch


def _native_positions(
    aggregator,
    *,
    batch_size: int,
    view_count: int,
    source_rows: int,
    source_cols: int,
    special_per_view: int,
    device: torch.device,
) -> torch.Tensor | None:
    if aggregator.rope is None:
        return None
    positions = aggregator.position_getter(
        batch_size * view_count,
        source_rows,
        source_cols,
        device=device,
    )
    positions = positions + 1
    special = torch.zeros(
        batch_size * view_count,
        special_per_view,
        2,
        device=device,
        dtype=positions.dtype,
    )
    return torch.cat((special, positions), dim=1)


def resume_frozen_vggt_tail_from_global(
    aggregator,
    layer11_global: torch.Tensor,
    *,
    branch_dim: int = 1024,
    source_rows: int = 37,
    source_cols: int = 37,
    start_layer: int = 12,
    final_layer: int = 23,
    cached_layers: Iterable[int] = (17, 23),
) -> dict[int, torch.Tensor]:
    """Resume native layers 12--23 directly from layer-11 global.

    This is the canonical V3 path.  Its signature makes it impossible for
    layer-4 or the layer-11 frame branch to influence the resumed tail.
    """

    if layer11_global.ndim != 4:
        raise ValueError("native layer-11 global must be [B,S,P,C]")
    batch, views, token_count, width = layer11_global.shape
    if width != int(branch_dim):
        raise ValueError("native layer-11 global has the wrong branch width")
    special_count = token_count - int(source_rows) * int(source_cols)
    if special_count <= 0:
        raise ValueError("native layer-11 token grid has no special tokens")
    if start_layer < 0 or final_layer >= len(aggregator.frame_blocks):
        raise ValueError("requested VGGT tail layers are outside the aggregator")
    if final_layer >= len(aggregator.global_blocks) or start_layer > final_layer:
        raise ValueError("invalid VGGT tail layer range")
    requested = {int(index) for index in cached_layers}
    if not requested.issubset(set(range(start_layer, final_layer + 1))):
        raise ValueError("cached VGGT tail layers must lie inside the resumed range")

    tokens = layer11_global
    positions = _native_positions(
        aggregator,
        batch_size=batch,
        view_count=views,
        source_rows=source_rows,
        source_cols=source_cols,
        special_per_view=special_count,
        device=layer11_global.device,
    )
    outputs: dict[int, torch.Tensor] = {}
    for layer_index in range(start_layer, final_layer + 1):
        frame_input = tokens.reshape(batch * views, token_count, branch_dim)
        frame_output = aggregator.frame_blocks[layer_index](
            frame_input, pos=positions
        )
        global_input = frame_output.reshape(batch, views * token_count, branch_dim)
        global_positions = (
            positions.reshape(batch, views * token_count, 2)
            if positions is not None
            else None
        )
        global_output = aggregator.global_blocks[layer_index](
            global_input, pos=global_positions
        )
        tokens = global_output.reshape(batch, views, token_count, branch_dim)
        if layer_index in requested:
            outputs[layer_index] = torch.cat(
                (
                    frame_output.reshape(batch, views, token_count, branch_dim),
                    tokens,
                ),
                dim=-1,
            )
    if set(outputs) != requested:
        raise AssertionError("VGGT tail did not emit every requested layer")
    return outputs
