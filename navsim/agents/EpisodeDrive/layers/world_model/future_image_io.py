"""Fixed-size, cache-safe UTF-8 path tensor encoding."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import torch


DEFAULT_PATH_BYTES = 1024
PathLike = Union[str, Path]


def encode_path_tensor(
    path: PathLike,
    max_bytes: int = DEFAULT_PATH_BYTES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    encoded = str(path).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(
            f"UTF-8 path needs {len(encoded)} bytes, exceeding fixed limit {max_bytes}: {path}"
        )
    tensor = torch.zeros(max_bytes, dtype=torch.uint8)
    if encoded:
        tensor[:len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)
    return tensor, torch.tensor(len(encoded), dtype=torch.long)


def decode_path_tensor(path_tensor: torch.Tensor, length: Union[int, torch.Tensor]) -> str:
    length_value = int(length.item()) if isinstance(length, torch.Tensor) else int(length)
    if path_tensor.ndim != 1:
        raise ValueError(
            f"A single path tensor must be rank 1, got {tuple(path_tensor.shape)}"
        )
    if length_value < 0 or length_value > path_tensor.numel():
        raise ValueError(
            f"Invalid encoded path length {length_value} for {path_tensor.numel()} bytes"
        )
    raw = bytes(path_tensor[:length_value].detach().cpu().to(torch.uint8).tolist())
    return raw.decode("utf-8")


def encode_path_tensor_batch(
    paths: Sequence[PathLike],
    max_bytes: int = DEFAULT_PATH_BYTES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_path_tensor(path, max_bytes=max_bytes) for path in paths]
    if not encoded:
        return (
            torch.zeros(0, max_bytes, dtype=torch.uint8),
            torch.zeros(0, dtype=torch.long),
        )
    return (
        torch.stack([item[0] for item in encoded]),
        torch.stack([item[1] for item in encoded]),
    )


def decode_path_tensor_batch(
    path_tensors: torch.Tensor,
    lengths: torch.Tensor,
) -> List[str]:
    if path_tensors.ndim != 2:
        raise ValueError(
            f"Path batch must have shape [N,L], got {tuple(path_tensors.shape)}"
        )
    if lengths.ndim != 1 or lengths.shape[0] != path_tensors.shape[0]:
        raise ValueError(
            "Path lengths must have shape [N] matching path tensors; "
            f"got {tuple(lengths.shape)} and {tuple(path_tensors.shape)}"
        )
    return [
        decode_path_tensor(path_tensor, length)
        for path_tensor, length in zip(path_tensors, lengths)
    ]
