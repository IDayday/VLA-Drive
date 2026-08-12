"""Color mapping helpers for Retrieve Model semantic BEV outputs."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from .retrieve_features import AGENT_CLASS_NAMES, MAP_CLASS_NAMES


# Match navsim.visualization.config, which TransFuser uses for BEV rendering.
MAP_PALETTE = np.asarray(
    [
        [255, 255, 255],  # background
        [211, 211, 211],  # road: SemanticMapLayer.LANE fill
        [212, 209, 158],  # walkway: SemanticMapLayer.WALKWAYS fill
        [102, 102, 102],  # centerline: BASELINE_PATHS line
    ],
    dtype=np.uint8,
)
AGENT_PALETTE = np.asarray(
    [
        [255, 255, 255],  # background
        [105, 156, 219],  # vehicle
        [176, 122, 161],  # pedestrian
    ],
    dtype=np.uint8,
)


def semantic_labels_to_rgb(
    labels: np.ndarray | torch.Tensor,
    palette: np.ndarray,
) -> np.ndarray:
    """Convert a 2D integer label map into an RGB image."""
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label map, got {labels.shape}")
    if labels.size and (labels.min() < 0 or labels.max() >= len(palette)):
        raise ValueError(
            f"Labels must be in [0, {len(palette) - 1}], "
            f"got [{labels.min()}, {labels.max()}]"
        )
    return palette[labels.astype(np.int64)]


def semantic_logits_to_rgb(
    logits: torch.Tensor,
    palette: np.ndarray,
) -> np.ndarray:
    """Convert [C,H,W] raw logits to an RGB argmax prediction."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [C,H,W], got {tuple(logits.shape)}")
    return semantic_labels_to_rgb(logits.argmax(dim=0), palette)


def class_legend(
    class_names: Sequence[str],
    palette: np.ndarray,
) -> Mapping[str, list[int]]:
    """Return a JSON-serializable class-to-color mapping."""
    if len(class_names) != len(palette):
        raise ValueError("Class names and palette length differ")
    return {
        name: [int(channel) for channel in color]
        for name, color in zip(class_names, palette)
    }


MAP_LEGEND = class_legend(MAP_CLASS_NAMES, MAP_PALETTE)
AGENT_LEGEND = class_legend(AGENT_CLASS_NAMES, AGENT_PALETTE)
