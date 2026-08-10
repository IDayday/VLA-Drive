"""Offline dynamics-teacher protocol and explicit local V-JEPA adapter.

External repositories are imported only from :meth:`load_model`.  Dataset and
training code consume serialized cache tensors and never instantiate a teacher.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F


DEFAULT_DYNAMICS_VIEWS = ("cam_f0", "cam_l0", "cam_r0")


@dataclass(frozen=True)
class DynamicsTeacherSample:
    """One offline future-feature sample.

    Shapes:
        features: ``[H,V,Ct,Ht,Wt]``.
        confidence/valid_mask: ``[H,V,Ht,Wt]``.
        frame_indices/frame_times_s: ``[H]``.
        source_image_hw/feature_hw: ``[H,V,2]`` in ``[height,width]`` order.
    """

    token: str
    view_names: Tuple[str, ...]
    features: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    frame_indices: np.ndarray
    frame_times_s: np.ndarray
    source_image_hw: np.ndarray
    feature_hw: np.ndarray
    spatial_layout: str
    feature_normalization: str
    metadata: Dict[str, Any]

    def validate(self) -> "DynamicsTeacherSample":
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("dynamics teacher token must be a non-empty string")
        if self.features.ndim != 5:
            raise ValueError("features must have shape [H,V,Ct,Ht,Wt]")
        horizon, views, channels, height, width = self.features.shape
        if min(horizon, views, channels, height, width) <= 0:
            raise ValueError("dynamics feature dimensions must be positive")
        if len(self.view_names) != views:
            raise ValueError("view_names length must equal dynamics V")
        if self.features.dtype != np.float16:
            raise ValueError("dynamics features must use float16 cache storage")
        map_shape = (horizon, views, height, width)
        if self.confidence.shape != map_shape or self.confidence.dtype != np.float32:
            raise ValueError("confidence must be float32 [H,V,Ht,Wt]")
        if self.valid_mask.shape != map_shape or self.valid_mask.dtype != np.bool_:
            raise ValueError("valid_mask must be bool [H,V,Ht,Wt]")
        if self.frame_indices.shape != (horizon,) or self.frame_indices.dtype != np.int64:
            raise ValueError("frame_indices must be int64 [H]")
        if self.frame_times_s.shape != (horizon,) or self.frame_times_s.dtype != np.float32:
            raise ValueError("frame_times_s must be float32 [H]")
        if self.source_image_hw.shape != (horizon, views, 2):
            raise ValueError("source_image_hw must have shape [H,V,2]")
        if self.source_image_hw.dtype != np.int64 or np.any(self.source_image_hw <= 0):
            raise ValueError("source_image_hw must contain positive int64 sizes")
        if self.feature_hw.shape != (horizon, views, 2):
            raise ValueError("feature_hw must have shape [H,V,2]")
        if self.feature_hw.dtype != np.int64 or np.any(self.feature_hw <= 0):
            raise ValueError("feature_hw must contain positive int64 sizes")
        if not np.all(self.feature_hw == np.asarray([height, width], dtype=np.int64)):
            raise ValueError("feature_hw does not match the cached feature grid")
        if self.spatial_layout != "per_view_patch_grid":
            raise ValueError("unsupported dynamics spatial_layout")
        if self.feature_normalization not in {"none", "l2"}:
            raise ValueError("unsupported dynamics feature_normalization")
        if not np.isfinite(self.features).all() or not np.isfinite(self.confidence).all():
            raise ValueError("dynamics teacher tensors contain non-finite values")
        if np.any(self.confidence < 0.0) or np.any(self.confidence > 1.0):
            raise ValueError("dynamics confidence must be within [0,1]")
        if not np.all(np.diff(self.frame_indices) > 0):
            raise ValueError("future frame_indices must be strictly increasing")
        if not np.all(np.diff(self.frame_times_s) > 0):
            raise ValueError("future frame_times_s must be strictly increasing")
        return self


class DynamicsTeacherAdapter(Protocol):
    """Minimal protocol implemented by offline dynamics teachers."""

    name: str
    version: str

    def load_model(self) -> torch.nn.Module:
        """Load a user-provided local checkpoint without network access."""


class OfficialVJEPA2Adapter:
    """Pinned local adapter for the official V-JEPA 2.1 backbone API.

    ``pretrained=False`` is always passed to the official factory.  We then
    load ``ema_encoder`` from the explicitly provided checkpoint, preventing
    the upstream hub helper from attempting a network download.
    """

    name = "vjepa2_1"
    version = "official_local_v1"
    _VARIANTS = {
        "vjepa2_1_vit_base": "vjepa2_1_vit_base_384",
        "vjepa2_1_vit_base_384": "vjepa2_1_vit_base_384",
        "vjepa2_1_vit_large": "vjepa2_1_vit_large_384",
        "vjepa2_1_vit_large_384": "vjepa2_1_vit_large_384",
    }

    def __init__(
        self,
        *,
        local_repo: Path | str,
        checkpoint: Path | str,
        model_variant: str = "vjepa2_1_vit_large_384",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        num_frames: int = 12,
        image_size: int = 384,
    ) -> None:
        self.local_repo = Path(local_repo)
        self.checkpoint = Path(checkpoint)
        self.model_variant = str(model_variant)
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        if self.model_variant not in self._VARIANTS:
            raise ValueError(
                f"unsupported V-JEPA 2.1 variant {self.model_variant!r}; "
                f"choose one of {sorted(self._VARIANTS)}"
            )
        if self.num_frames < 2 or self.num_frames % 2:
            raise ValueError("V-JEPA num_frames must be an even integer >= 2")
        if self.image_size != 384:
            raise ValueError("the pinned V-JEPA 2.1 checkpoints require image_size=384")
        self._model: torch.nn.Module | None = None

    @staticmethod
    def _clean_encoder_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cleaned: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            name = str(key).replace("module.", "").replace("backbone.", "")
            cleaned[name] = value
        return cleaned

    @staticmethod
    def _install_rope_dtype_compatibility(module: object) -> None:
        """Keep official RoPE outputs in the Q/K dtype on strict SDPA runtimes.

        The official helper multiplies integer positions by ``1.0``, producing
        float32 positions.  PyTorch 2.4 PPU SDPA requires Q, K and V to have an
        identical dtype, so BF16 Q/K otherwise get promoted to FP32.  This
        process-local wrapper changes no checkpoint weights or vendored files.
        """

        original = getattr(module, "rotate_queries_or_keys", None)
        if not callable(original):
            raise RuntimeError("V-JEPA RoPE helper is unavailable")
        if getattr(original, "_field2plan_dtype_safe", False):
            return

        def dtype_safe_rotate(x, pos, n_registers, has_cls_first):
            output = original(
                x,
                pos.to(device=x.device, dtype=x.dtype),
                n_registers,
                has_cls_first,
            )
            return output.to(dtype=x.dtype)

        dtype_safe_rotate._field2plan_dtype_safe = True
        module.rotate_queries_or_keys = dtype_safe_rotate

    def load_model(self) -> torch.nn.Module:
        """Load the official encoder lazily from local files only."""

        if self._model is not None:
            return self._model
        if not self.local_repo.is_dir():
            raise FileNotFoundError(f"V-JEPA local repo not found: {self.local_repo}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"V-JEPA local checkpoint not found: {self.checkpoint}"
            )

        repo = str(self.local_repo.resolve())
        inserted = repo not in sys.path
        if inserted:
            sys.path.insert(0, repo)
        try:
            try:
                module = importlib.import_module("src.hub.backbones")
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "official V-JEPA 2.1 backbone module could not be imported "
                    "from the configured local repository"
                ) from error
            try:
                rope_module = importlib.import_module(
                    "app.vjepa_2_1.models.utils.modules"
                )
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "official V-JEPA 2.1 RoPE module could not be imported"
                ) from error
            self._install_rope_dtype_compatibility(rope_module)
            factory_name = self._VARIANTS[self.model_variant]
            factory = getattr(module, factory_name, None)
            if factory is None:
                raise RuntimeError(
                    f"official V-JEPA repository lacks factory {factory_name!r}"
                )
            built = factory(pretrained=False, num_frames=self.num_frames)
            encoder = built[0] if isinstance(built, tuple) else built
            if not isinstance(encoder, torch.nn.Module):
                raise RuntimeError("V-JEPA factory did not return a torch module")
            try:
                checkpoint = torch.load(
                    self.checkpoint, map_location="cpu", weights_only=True
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise RuntimeError(
                    f"failed to load local V-JEPA checkpoint: {self.checkpoint}"
                ) from error
            if not isinstance(checkpoint, dict) or "ema_encoder" not in checkpoint:
                raise RuntimeError("V-JEPA 2.1 checkpoint lacks ema_encoder state")
            encoder_state = checkpoint["ema_encoder"]
            if not isinstance(encoder_state, dict):
                raise RuntimeError("V-JEPA ema_encoder must be a state-dict mapping")
            cleaned = self._clean_encoder_state(encoder_state)
            try:
                encoder.load_state_dict(cleaned, strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    "V-JEPA checkpoint is incompatible with the selected local factory"
                ) from error
            encoder.requires_grad_(False).eval().to(device=self.device, dtype=self.dtype)
            self._model = encoder
            return encoder
        finally:
            if inserted and repo in sys.path:
                sys.path.remove(repo)

    def preprocess_video(self, frames: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert ``[V,T,H,W,3]`` RGB frames to ``[V,3,T,384,384]``.

        This is the manifest-pinned equivalent of the official deterministic
        evaluation recipe: center-square crop, resize to 384, then ImageNet
        normalization. Cropping before resize keeps camera-intrinsic updates
        exact and avoids resize-rounding ambiguity.
        """

        video = torch.as_tensor(frames)
        if video.ndim != 5 or video.shape[-1] != 3:
            raise ValueError("frames must have shape [V,T,H,W,3]")
        if video.shape[1] != self.num_frames:
            raise ValueError(f"expected {self.num_frames} temporal frames")
        if video.dtype == torch.uint8:
            video = video.float().div_(255.0)
        else:
            video = video.float()
            if video.min() < 0.0 or video.max() > 1.0:
                raise ValueError("floating RGB frames must be within [0,1]")
        video = video.permute(0, 1, 4, 2, 3).reshape(-1, 3, *video.shape[2:4])
        height, width = video.shape[-2:]
        crop_size = min(height, width)
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        video = video[..., top : top + crop_size, left : left + crop_size]
        video = F.interpolate(
            video,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = torch.tensor(
            (0.485, 0.456, 0.406), device=video.device, dtype=video.dtype
        ).reshape(1, 3, 1, 1)
        std = torch.tensor(
            (0.229, 0.224, 0.225), device=video.device, dtype=video.dtype
        ).reshape(1, 3, 1, 1)
        video = (video - mean) / std
        views = frames.shape[0]
        return video.reshape(views, self.num_frames, 3, self.image_size, self.image_size).permute(
            0, 2, 1, 3, 4
        ).contiguous()

    @torch.inference_mode()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``[V,3,T,384,384]`` into ``[V,T/2,24,24,C]``."""

        if video.ndim != 5 or video.shape[1:] != (
            3,
            self.num_frames,
            self.image_size,
            self.image_size,
        ):
            raise ValueError("video must have shape [V,3,T,384,384]")
        encoder = self.load_model()
        tokens = encoder(video.to(device=self.device, dtype=self.dtype))
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise RuntimeError("V-JEPA encoder must return [V,N,C] tokens")
        temporal = self.num_frames // 2
        spatial = self.image_size // 16
        if tokens.shape[1] != temporal * spatial * spatial:
            raise RuntimeError(
                "V-JEPA token count does not match tubelet/patch layout: "
                f"got {tokens.shape[1]}, expected {temporal * spatial * spatial}"
            )
        return tokens.reshape(tokens.shape[0], temporal, spatial, spatial, tokens.shape[-1])


def seeded_orthogonal_projection(
    input_dim: int,
    output_dim: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a deterministic ``[input_dim,output_dim]`` projection matrix."""

    if min(input_dim, output_dim) <= 0 or output_dim > input_dim:
        raise ValueError("projection dimensions require 0 < output_dim <= input_dim")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    orthogonal, _ = torch.linalg.qr(matrix, mode="reduced")
    return orthogonal.to(device=device)
