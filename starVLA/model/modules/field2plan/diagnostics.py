"""Cheap scalar Field2Plan diagnostics; no large tensors are retained."""

from typing import Dict

import torch

from .types import GeometryFieldOutput, RefinerOutput, TubeReadoutOutput


def collect_mvp_metrics(
    geometry: GeometryFieldOutput,
    readout: TubeReadoutOutput,
    refinement: RefinerOutput,
) -> Dict[str, torch.Tensor]:
    """Collect detached scalar metrics for logging."""

    return {
        "field_valid_ratio": geometry.valid_ratio.mean().detach(),
        "tube_valid_ratio": readout.valid_mask.float().mean().detach(),
        "delta_norm": refinement.delta_norm.detach(),
        "source_gate_geo": readout.source_gates[..., 0].mean().detach(),
    }
