"""Lazy external-teacher protocols; core training never imports teacher packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class CurrentPriorTeacherOutput:
    """Current/history teacher tensors before cache serialization.

    Shapes are ``features=[Th,V,C,H,W]`` and
    ``confidence=[Th,V,H,W]``. No future frame may be present.
    """

    features: np.ndarray
    confidence: np.ndarray

    def validate(self, history_length: int, views: int) -> "CurrentPriorTeacherOutput":
        if self.features.ndim != 5 or self.features.shape[:2] != (
            history_length,
            views,
        ):
            raise ValueError("prior teacher features must have shape [Th,V,C,H,W]")
        if self.confidence.shape != (
            history_length,
            views,
            self.features.shape[3],
            self.features.shape[4],
        ):
            raise ValueError("prior teacher confidence shape differs from features")
        if not np.isfinite(self.features).all() or not np.isfinite(self.confidence).all():
            raise ValueError("prior teacher output contains non-finite values")
        return self


class CurrentPriorTeacherAdapter(Protocol):
    """Interface implemented by a user-supplied local Driving-JEPA adapter."""

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def infer(
        self,
        video_uint8: np.ndarray,
        *,
        history_frame_indices: Sequence[int],
        output_hw: tuple[int, int],
        output_channels: int,
    ) -> CurrentPriorTeacherOutput: ...


@dataclass(frozen=True)
class PhysicalConsequenceOutput:
    """Physical values and availability ``[K,6]`` from a metric provider."""

    values: np.ndarray
    valid_mask: np.ndarray

    def validate(self, candidates: int) -> "PhysicalConsequenceOutput":
        if self.values.shape != (candidates, 6):
            raise ValueError("consequence values must have shape [K,6]")
        if self.valid_mask.shape != self.values.shape or self.valid_mask.dtype != np.bool_:
            raise ValueError("consequence valid_mask must be bool [K,6]")
        if not np.isfinite(self.values[self.valid_mask]).all():
            raise ValueError("valid consequence values contain non-finite values")
        return self


class PhysicalConsequenceProvider(Protocol):
    """NAVSIM/static-world metric adapter used only by the offline label tool."""

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def label(
        self,
        token: str,
        metadata: Mapping[str, Any],
        physical_trajectories: np.ndarray,
    ) -> PhysicalConsequenceOutput: ...
