"""Topology-independent initial noise for flow-matching inference."""

from __future__ import annotations

import hashlib

import torch


NOISE_MODES = ("legacy_rank_stream", "per_token")


def stable_token_seed(base_seed: int, scene_token: str, sample_index: int = 0) -> int:
    """Derive a stable torch seed without Python's process-randomized hash()."""

    if int(base_seed) < 0:
        raise ValueError("base_seed must be non-negative")
    if int(sample_index) < 0:
        raise ValueError("sample_index must be non-negative")
    if not isinstance(scene_token, str) or not scene_token:
        raise ValueError("scene_token must be a non-empty string")
    payload = f"{int(base_seed)}\0{scene_token}\0{int(sample_index)}".encode("utf-8")
    # torch.Generator.manual_seed accepts signed 64-bit seeds.  Masking avoids
    # backend-specific handling of values in the unsigned half of the range.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def diffusion_initial_noise(
    base_seed: int,
    scene_token: str,
    sample_index: int = 0,
    *,
    action_horizon: int = 8,
    action_dim: int = 4,
) -> torch.Tensor:
    """Generate canonical CPU FP32 noise for one scene.

    Generation intentionally happens on CPU.  The resulting values are then
    copied to the model device, so accelerator assignment cannot alter them.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_token_seed(base_seed, scene_token, sample_index))
    return torch.randn(
        (int(action_horizon), int(action_dim)),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
