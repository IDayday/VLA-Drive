"""Scoped conversion/capture of an existing Qwen visual forward."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from .types import VisualFeatureOutput


class VisualFeatureTap:
    """Expose ``[B,V,C,H,W]`` without rerunning the visual encoder.

    ``explicit`` converts an output already returned by the caller. ``hook``
    is a scoped fallback that captures exactly one forward and clears all
    references on consume/exit. It never stores tensors across batches.
    """

    def __init__(
        self,
        enabled: bool = True,
        mode: str = "explicit",
        num_views: int = 3,
        view_order: Optional[Sequence[int]] = None,
        view_names: Optional[Sequence[str]] = None,
    ) -> None:
        if mode not in {"explicit", "hook"}:
            raise ValueError("visual tap mode must be explicit or hook")
        self.enabled = bool(enabled)
        self.mode = mode
        self.num_views = int(num_views)
        self.view_order = tuple(view_order or range(self.num_views))
        if sorted(self.view_order) != list(range(self.num_views)):
            raise ValueError("view_order must be a permutation of range(num_views)")
        default_names = tuple(f"view_{i}" for i in range(self.num_views))
        self.view_names = tuple(view_names or default_names)
        self._captured_output = None
        self._captured_grid = None

    @property
    def has_pending_capture(self) -> bool:
        return self._captured_output is not None or self._captured_grid is not None

    def _clear(self) -> None:
        self._captured_output = None
        self._captured_grid = None

    @contextmanager
    def capture(self, visual_module: nn.Module):
        """Capture one visual forward; no hook is registered when disabled."""

        self._clear()
        if not self.enabled:
            yield self
            return
        if self.mode != "hook":
            yield self
            return

        def hook(_module, args, kwargs, output):
            if self._captured_output is not None:
                raise RuntimeError("VisualFeatureTap captured more than one visual forward")
            grid = kwargs.get("grid_thw")
            if grid is None and len(args) > 1:
                grid = args[1]
            if grid is None:
                raise RuntimeError("visual forward did not expose grid_thw")
            self._captured_output = output
            self._captured_grid = grid

        handle = visual_module.register_forward_hook(hook, with_kwargs=True)
        try:
            yield self
        finally:
            handle.remove()
            if self.has_pending_capture:
                self._clear()

    def from_visual_output(
        self,
        visual_module: nn.Module,
        visual_output,
        grid_thw: torch.Tensor,
    ) -> VisualFeatureOutput:
        """Convert existing merged tokens to ``[B,V,C,H,W]``."""

        if not self.enabled:
            raise RuntimeError("visual feature tap is disabled")
        if not hasattr(visual_module, "vit_tokens_to_featmap"):
            raise TypeError("visual module must provide vit_tokens_to_featmap")
        hidden_states = visual_output[0] if isinstance(visual_output, (tuple, list)) else visual_output
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
            raise ValueError("visual hidden states must have shape [N,C]")
        if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise ValueError("grid_thw must have shape [B*V,3]")
        _, per_view = visual_module.vit_tokens_to_featmap(
            hidden_states,
            grid_thw,
            num_views=self.num_views,
            view_order=self.view_order,
        )
        if per_view.ndim != 5 or per_view.shape[1] != self.num_views:
            raise ValueError("converted visual features must have shape [B,V,C,H,W]")
        names = tuple(self.view_names[index] for index in self.view_order)
        return VisualFeatureOutput(per_view, grid_thw, names)

    def consume(self, visual_module: nn.Module) -> VisualFeatureOutput:
        """Convert and clear the pending hook capture."""

        if not self.has_pending_capture:
            raise RuntimeError("no visual forward was captured")
        output, grid = self._captured_output, self._captured_grid
        try:
            return self.from_visual_output(visual_module, output, grid)
        finally:
            self._clear()
