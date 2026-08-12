"""Static token and tensor contracts for VGGT query supervision."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VGGTQueryLayout:
    """Description of the compact VGGT teacher layout.

    The production layer-11 route maps three current-frame cameras to a 6x10
    pure-spatial grid per view: ``3 * 6 * 10 = 180`` student/teacher queries.
    VGGT camera/register tokens are excluded from this route.
    """

    view_count: int = 3
    special_per_view: int = 5
    spatial_rows: int = 6
    spatial_cols: int = 10
    teacher_dim: int = 1024

    @property
    def special_query_count(self) -> int:
        return self.view_count * self.special_per_view

    @property
    def spatial_query_count(self) -> int:
        return self.view_count * self.spatial_rows * self.spatial_cols

    @property
    def query_count(self) -> int:
        return self.special_query_count + self.spatial_query_count


def build_vggt_query_tokens(layout: VGGTQueryLayout = VGGTQueryLayout()) -> tuple[str, ...]:
    """Return the full 195-slot teacher/memory naming contract.

    Only :func:`build_vggt_global_query_tokens` is added to the language
    tokenizer. Spatial names describe memory slots, not text tokens.
    """

    special = tuple(
        f"<vggt_special_v{view}_t{token}>"
        for view in range(layout.view_count)
        for token in range(layout.special_per_view)
    )
    spatial = tuple(
        f"<vggt_spatial_v{view}_r{row}_c{col}>"
        for view in range(layout.view_count)
        for row in range(layout.spatial_rows)
        for col in range(layout.spatial_cols)
    )
    tokens = special + spatial
    assert len(tokens) == layout.query_count
    assert len(set(tokens)) == len(tokens)
    return tokens


def build_vggt_global_query_tokens(
    layout: VGGTQueryLayout = VGGTQueryLayout(),
) -> tuple[str, ...]:
    """Return the 15 global VGGT tokens inserted into the Qwen sequence."""

    tokens = tuple(
        f"<vggt_special_v{view}_t{token}>"
        for view in range(layout.view_count)
        for token in range(layout.special_per_view)
    )
    assert len(tokens) == layout.special_query_count
    assert len(set(tokens)) == len(tokens)
    return tokens
