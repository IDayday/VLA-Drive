"""Typed metadata that accompanies candidates but is never network input."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class CandidateMetadata:
    """Candidate provenance aligned with a ``[B,N,...]`` candidate tensor.

    ``source`` is zero for static vocabulary entries and one for generated
    dynamic entries. ``absolute_index`` is the unified-pool index.  These
    integer tensors are diagnostic/indexing metadata only.
    """

    source: Tensor
    source_index: Tensor
    absolute_index: Tensor

    def validate(self, batch_size: int, candidate_count: int) -> None:
        expected = (batch_size, candidate_count)
        for name, value in (
            ("source", self.source),
            ("source_index", self.source_index),
            ("absolute_index", self.absolute_index),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected}")
            if value.dtype != torch.long:
                raise TypeError(f"{name} must have dtype torch.long")
        if not torch.all((self.source == 0) | (self.source == 1)):
            raise ValueError("candidate source must be 0=static or 1=dynamic")

    def gather(self, indices: Tensor) -> "CandidateMetadata":
        """Gather all provenance fields with the same ``[B,K]`` indices."""

        if indices.ndim != 2 or indices.dtype != torch.long:
            raise TypeError("metadata gather indices must be a [B,K] long tensor")
        if indices.shape[0] != self.source.shape[0]:
            raise ValueError("metadata gather batch size differs")
        return CandidateMetadata(
            source=torch.gather(self.source, 1, indices),
            source_index=torch.gather(self.source_index, 1, indices),
            absolute_index=torch.gather(self.absolute_index, 1, indices),
        )
