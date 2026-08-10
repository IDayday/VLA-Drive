"""Offline geometry-teacher adapters with explicit cache schemas.

The training path consumes serialized arrays only. External teacher packages
are imported lazily by offline tools and are never a dependency of the model
forward pass.
"""

from __future__ import annotations

import importlib
import pickle
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Protocol, Sequence, Tuple

import numpy as np


DEFAULT_GEOMETRY_VIEWS = ("cam_f0", "cam_l0", "cam_r0")


@dataclass(frozen=True)
class GeometryTeacherSample:
    """One metric-depth teacher sample.

    Shapes:
        depth_m/confidence/valid_mask: ``[V,Hd,Wd]``.
        source_image_hw/depth_hw/resize_scale_xy: ``[V,2]``. The first two
        are ordered ``[height,width]``; scales are ordered ``[sx,sy]``.
    """

    token: str
    view_names: Tuple[str, ...]
    depth_m: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    source_image_hw: np.ndarray
    depth_hw: np.ndarray
    resize_scale_xy: np.ndarray
    coordinate_frame: str
    metadata: Dict[str, Any]

    def validate(self) -> "GeometryTeacherSample":
        if self.depth_m.ndim != 3:
            raise ValueError("depth_m must have shape [V,Hd,Wd]")
        expected = self.depth_m.shape
        if self.confidence.shape != expected:
            raise ValueError("confidence must match depth_m shape [V,Hd,Wd]")
        if self.valid_mask.shape != expected:
            raise ValueError("valid_mask must match depth_m shape [V,Hd,Wd]")
        views = expected[0]
        if len(self.view_names) != views:
            raise ValueError("view_names length must match depth V")
        if self.source_image_hw.shape != (views, 2):
            raise ValueError("source_image_hw must have shape [V,2]")
        if self.depth_hw.shape != (views, 2):
            raise ValueError("depth_hw must have shape [V,2]")
        if self.resize_scale_xy.shape != (views, 2):
            raise ValueError("resize_scale_xy must have shape [V,2]")
        if self.depth_m.dtype != np.float32:
            raise ValueError("depth_m must be float32")
        if self.confidence.dtype != np.float32:
            raise ValueError("confidence must be float32")
        if self.valid_mask.dtype != np.bool_:
            raise ValueError("valid_mask must be bool")
        if not np.isfinite(self.depth_m).all():
            raise ValueError("depth_m must be finite after invalid-value masking")
        if not np.isfinite(self.confidence).all():
            raise ValueError("confidence must be finite")
        if np.any(self.confidence < 0.0) or np.any(self.confidence > 1.0):
            raise ValueError("confidence must be within [0,1]")
        declared_hw = np.repeat(
            np.asarray(expected[1:], dtype=np.int64)[None], views, axis=0
        )
        if not np.array_equal(self.depth_hw, declared_hw):
            raise ValueError("depth_hw does not match depth_m spatial shape")
        if self.coordinate_frame != "camera_optical_z_depth_m":
            raise ValueError("unsupported geometry coordinate_frame")
        return self


class GeometryTeacherAdapter(Protocol):
    """Protocol for offline geometry teachers; never invoked in training."""

    name: str
    version: str

    def infer(self, images: Sequence[Any]) -> GeometryTeacherSample:
        """Infer and return a validated, serializable geometry sample."""


class DA3LegacyDepthAdapter:
    """Read existing repository DA3 metric-depth pickle files.

    The legacy exporter saved metric depth only. Consequently this adapter
    exposes finite-positive validity as confidence and records that provenance
    explicitly instead of claiming access to DA3's discarded confidence head.
    """

    name = "depth_anything_3_metric_depth"
    version = "legacy_depth_vis_v1"

    def __init__(
        self,
        meta_root: Path | str,
        view_names: Sequence[str] = DEFAULT_GEOMETRY_VIEWS,
        frame_index: int = 3,
    ) -> None:
        self.meta_root = Path(meta_root)
        self.view_names = tuple(str(view) for view in view_names)
        self.frame_index = int(frame_index)
        if not self.view_names:
            raise ValueError("view_names cannot be empty")

    def cache_path(self, token: str) -> Path:
        return self.meta_root / f"{token}.pkl-depth.pkl"

    def load_cached(
        self,
        token: str,
        source_image_hw: np.ndarray,
    ) -> GeometryTeacherSample:
        """Load one legacy cache into the strict ``[V,Hd,Wd]`` schema."""

        path = self.cache_path(token)
        if not path.is_file():
            raise FileNotFoundError(f"DA3 depth cache not found: {path}")
        try:
            with path.open("rb") as stream:
                payload = pickle.load(stream)
        except (OSError, pickle.UnpicklingError, EOFError) as error:
            raise ValueError(f"corrupt DA3 depth cache: {path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"DA3 depth cache must contain a mapping: {path}")
        if set(payload) != set(self.view_names):
            raise ValueError(
                f"DA3 depth views must be exactly {self.view_names}, got {tuple(payload)}"
            )

        depths = []
        expected_shape = None
        for view in self.view_names:
            depth = np.asarray(payload[view])
            if depth.ndim != 2:
                raise ValueError(f"DA3 depth for {view} must have shape [Hd,Wd]")
            if expected_shape is None:
                expected_shape = depth.shape
            elif depth.shape != expected_shape:
                raise ValueError("all DA3 views must share the same depth shape")
            depths.append(depth.astype(np.float32, copy=False))

        depth_m = np.stack(depths, axis=0)
        valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
        depth_m = np.where(valid_mask, depth_m, 0.0).astype(np.float32, copy=False)
        confidence = valid_mask.astype(np.float32)
        source_hw = np.asarray(source_image_hw, dtype=np.int64)
        if source_hw.shape != (len(self.view_names), 2):
            raise ValueError("source_image_hw must have shape [V,2]")
        if np.any(source_hw <= 0):
            raise ValueError("source_image_hw must be positive")
        depth_hw = np.repeat(
            np.asarray(depth_m.shape[-2:], dtype=np.int64)[None],
            len(self.view_names),
            axis=0,
        )
        resize_scale_xy = np.stack(
            (
                depth_hw[:, 1] / source_hw[:, 1],
                depth_hw[:, 0] / source_hw[:, 0],
            ),
            axis=-1,
        ).astype(np.float32)
        return GeometryTeacherSample(
            token=str(token),
            view_names=self.view_names,
            depth_m=depth_m,
            confidence=confidence,
            valid_mask=valid_mask.astype(np.bool_, copy=False),
            source_image_hw=source_hw,
            depth_hw=depth_hw,
            resize_scale_xy=resize_scale_xy,
            coordinate_frame="camera_optical_z_depth_m",
            metadata={
                "teacher_name": self.name,
                "teacher_version": self.version,
                "frame_index": self.frame_index,
                "confidence_source": "finite_positive_validity",
                "legacy_source_path": str(path),
            },
        ).validate()


class VGGTAdapter:
    """Explicit, lazy offline VGGT integration point.

    No public API is guessed. Users must provide a local repository, local
    checkpoint, importable module name and factory name. The factory receives
    ``checkpoint_path=...`` and must return a callable accepting ``images``.
    """

    name = "vggt"
    version = "user_local_api_v1"

    def __init__(
        self,
        local_repo: Path | str,
        checkpoint: Path | str,
        module_name: str,
        factory_name: str,
    ) -> None:
        self.local_repo = Path(local_repo)
        self.checkpoint = Path(checkpoint)
        self.module_name = str(module_name)
        self.factory_name = str(factory_name)
        if not self.module_name or not self.factory_name:
            raise ValueError("VGGT module_name and factory_name are required")

    def infer(self, images: Sequence[Any]) -> GeometryTeacherSample:
        if not self.local_repo.is_dir():
            raise FileNotFoundError(f"VGGT local repo not found: {self.local_repo}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"VGGT local checkpoint not found: {self.checkpoint}")

        repo = str(self.local_repo.resolve())
        inserted = repo not in sys.path
        if inserted:
            sys.path.insert(0, repo)
        try:
            module = importlib.import_module(self.module_name)
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "VGGT local module could not be imported; provide the exact "
                "module_name implemented by the local repository"
            ) from error
        finally:
            if inserted and repo in sys.path:
                sys.path.remove(repo)
        if not hasattr(module, self.factory_name):
            raise RuntimeError(
                f"VGGT module {self.module_name!r} lacks explicit factory "
                f"{self.factory_name!r}"
            )
        factory = getattr(module, self.factory_name)
        teacher = factory(checkpoint_path=str(self.checkpoint))
        if not callable(teacher):
            raise RuntimeError("VGGT factory must return a callable offline teacher")
        sample = teacher(images)
        if not isinstance(sample, GeometryTeacherSample):
            raise RuntimeError(
                "VGGT callable must return GeometryTeacherSample; no output API is inferred"
            )
        return sample.validate()


def estimate_metric_scale_from_depth_reference(
    predicted_relative_depth: np.ndarray,
    metric_depth_reference_m: np.ndarray,
    *,
    min_metric_depth_m: float = 0.1,
    max_metric_depth_m: float = 200.0,
    min_valid_pixels: int = 4096,
) -> tuple[float, Dict[str, Any]]:
    """Estimate one robust scale from aligned relative/metric depth ``[V,H,W]``.

    This uses only current-frame geometry teachers.  The median is taken in
    log-ratio space and is robust to sparse teacher disagreement/outliers.
    """

    predicted = np.asarray(predicted_relative_depth, dtype=np.float64)
    reference = np.asarray(metric_depth_reference_m, dtype=np.float64)
    if predicted.ndim != 3 or reference.shape != predicted.shape:
        raise ValueError("predicted and metric reference depth must have the same shape [V,H,W]")
    if min_metric_depth_m < 0 or max_metric_depth_m <= min_metric_depth_m:
        raise ValueError("metric depth anchor range is invalid")
    if min_valid_pixels < 1:
        raise ValueError("min_valid_pixels must be positive")
    valid = (
        np.isfinite(predicted)
        & np.isfinite(reference)
        & (predicted > 0.0)
        & (reference >= min_metric_depth_m)
        & (reference <= max_metric_depth_m)
    )
    valid_count = int(valid.sum())
    if valid_count < min_valid_pixels:
        raise ValueError(
            f"metric depth anchor has {valid_count} valid pixels; "
            f"requires at least {min_valid_pixels}"
        )
    log_ratios = np.log(reference[valid]) - np.log(predicted[valid])
    log_median = float(np.median(log_ratios))
    absolute_deviation = np.abs(log_ratios - log_median)
    log_mad = float(np.median(absolute_deviation))
    # A non-zero floor prevents an exactly consistent majority from rejecting
    # all numerically perturbed inliers.  The second median remains robust.
    threshold = max(3.0 * log_mad, 0.05)
    inliers = absolute_deviation <= threshold
    inlier_count = int(inliers.sum())
    if inlier_count < min_valid_pixels:
        raise ValueError(
            f"metric depth anchor has only {inlier_count} robust inlier pixels"
        )
    robust_log_scale = float(np.median(log_ratios[inliers]))
    scale = float(np.exp(robust_log_scale))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("metric depth anchor scale must be finite and positive")
    return scale, {
        "method": "robust_log_median_metric_depth_ratio",
        "valid_pixel_count": valid_count,
        "inlier_pixel_count": inlier_count,
        "log_ratio_median": robust_log_scale,
        "log_ratio_mad": log_mad,
        "scale": scale,
    }


def estimate_metric_scale_from_camera_rig(
    predicted_world_to_camera: np.ndarray,
    known_camera_centers_m: np.ndarray,
    *,
    min_reference_baseline_m: float = 0.05,
    min_predicted_baseline: float = 1e-6,
) -> tuple[float, Dict[str, Any]]:
    """Metricize a scale-ambiguous reconstruction from a calibrated camera rig.

    Args:
        predicted_world_to_camera: OpenCV world-to-camera matrices ``[V,3,4]``.
        known_camera_centers_m: Corresponding physical camera centers ``[V,3]``.

    Pairwise camera distances are invariant to the arbitrary world-frame
    rotation and translation.  The returned scalar is the median ratio between
    the known metric baselines and VGGT's predicted baselines.  Future action
    and ground-truth trajectory are never inputs.
    """

    extrinsics = np.asarray(predicted_world_to_camera, dtype=np.float64)
    known = np.asarray(known_camera_centers_m, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError("predicted_world_to_camera must have shape [V,3,4]")
    if known.shape != (extrinsics.shape[0], 3):
        raise ValueError("known_camera_centers_m must have shape [V,3]")
    if extrinsics.shape[0] < 2:
        raise ValueError("camera-rig metricization requires at least two views")
    if not np.isfinite(extrinsics).all() or not np.isfinite(known).all():
        raise ValueError("camera-rig metricization inputs must be finite")
    if min_reference_baseline_m <= 0 or min_predicted_baseline <= 0:
        raise ValueError("camera-rig baseline thresholds must be positive")

    rotations = extrinsics[:, :, :3]
    translations = extrinsics[:, :, 3]
    predicted_centers = -np.einsum("vji,vj->vi", rotations, translations)
    pair_scales = []
    pair_diagnostics = []
    for first in range(extrinsics.shape[0]):
        for second in range(first + 1, extrinsics.shape[0]):
            known_distance = float(np.linalg.norm(known[first] - known[second]))
            predicted_distance = float(
                np.linalg.norm(predicted_centers[first] - predicted_centers[second])
            )
            if known_distance < min_reference_baseline_m:
                continue
            if predicted_distance < min_predicted_baseline:
                continue
            ratio = known_distance / predicted_distance
            if np.isfinite(ratio) and ratio > 0:
                pair_scales.append(ratio)
                pair_diagnostics.append(
                    {
                        "views": [first, second],
                        "known_baseline_m": known_distance,
                        "predicted_baseline": predicted_distance,
                        "scale": ratio,
                    }
                )
    if not pair_scales:
        raise ValueError(
            "camera rig has no non-degenerate known/predicted baseline pair"
        )
    scale = float(np.median(np.asarray(pair_scales, dtype=np.float64)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("estimated VGGT metric scale must be finite and positive")
    return scale, {
        "method": "median_known_to_predicted_camera_baseline_ratio",
        "valid_pair_count": len(pair_scales),
        "pair_scale_median": scale,
        "pair_scale_min": float(np.min(pair_scales)),
        "pair_scale_max": float(np.max(pair_scales)),
        "pairs": pair_diagnostics,
    }


@contextmanager
def _temporary_import_root(path: Path) -> Iterator[None]:
    """Temporarily expose a user-provided external repository for lazy import."""

    resolved = str(path.resolve())
    inserted = resolved not in sys.path
    if inserted:
        sys.path.insert(0, resolved)
    try:
        yield
    finally:
        if inserted and resolved in sys.path:
            sys.path.remove(resolved)


class OfficialVGGTMetricDepthAdapter:
    """Offline adapter for the pinned official ``facebookresearch/vggt`` API.

    VGGT predicts scale-ambiguous depth.  This adapter converts it to metric
    depth using only the known simultaneous multi-camera baselines.  Loading is
    lazy, all paths are explicit, and no network access is performed.
    """

    name = "vggt"
    version = "facebook_vggt_1b_r860abec_da3_anchor_v1"

    def __init__(
        self,
        local_repo: Path | str,
        checkpoint: Path | str,
        *,
        device: str = "cuda",
        output_hw: Sequence[int] = (144, 256),
        preprocess_mode: str = "crop",
        view_names: Sequence[str] = DEFAULT_GEOMETRY_VIEWS,
        frame_index: int = 3,
        use_bfloat16: bool = True,
        metricization: str = "da3_scale_anchor",
    ) -> None:
        self.local_repo = Path(local_repo)
        checkpoint_path = Path(checkpoint)
        self.checkpoint = (
            checkpoint_path / "model.safetensors"
            if checkpoint_path.is_dir()
            else checkpoint_path
        )
        self.device_name = str(device)
        self.output_hw = tuple(int(value) for value in output_hw)
        self.preprocess_mode = str(preprocess_mode)
        self.view_names = tuple(str(view) for view in view_names)
        self.frame_index = int(frame_index)
        self.use_bfloat16 = bool(use_bfloat16)
        self.metricization = str(metricization)
        if len(self.output_hw) != 2 or min(self.output_hw) <= 0:
            raise ValueError("VGGT output_hw must contain positive [height,width]")
        if self.preprocess_mode != "crop":
            raise ValueError(
                "official VGGT metric cache currently requires crop mode so "
                "wide NAVSIM images retain a padding-free pixel mapping"
            )
        if not self.view_names:
            raise ValueError("VGGT view_names cannot be empty")
        if self.metricization not in {"da3_scale_anchor", "camera_rig"}:
            raise ValueError(
                "VGGT metricization must be da3_scale_anchor or camera_rig"
            )
        self._model = None
        self._preprocess = None
        self._decode_pose = None
        self._torch = None

    def _validate_assets(self) -> None:
        if not self.local_repo.is_dir():
            raise FileNotFoundError(f"VGGT local repo not found: {self.local_repo}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"VGGT local checkpoint not found: {self.checkpoint}"
            )

    def load_model(self):
        """Lazily load the official model without point/track inference heads."""

        if self._model is not None:
            return self._model
        self._validate_assets()
        try:
            import torch
            from safetensors.torch import load_model as load_safetensors_model
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "official VGGT cache generation requires torch and safetensors"
            ) from error

        with _temporary_import_root(self.local_repo):
            try:
                model_module = importlib.import_module("vggt.models.vggt")
                load_module = importlib.import_module("vggt.utils.load_fn")
                pose_module = importlib.import_module("vggt.utils.pose_enc")
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "failed to import the pinned official VGGT repository"
                ) from error
        model = model_module.VGGT(
            enable_camera=True,
            enable_point=False,
            enable_depth=True,
            enable_track=False,
        )
        missing, unexpected = load_safetensors_model(
            model, self.checkpoint, strict=False, device="cpu"
        )
        if missing:
            raise RuntimeError(f"VGGT checkpoint has missing keys: {sorted(missing)}")
        allowed_prefixes = ("point_head.", "track_head.")
        disallowed = [
            key for key in unexpected if not key.startswith(allowed_prefixes)
        ]
        if disallowed:
            raise RuntimeError(
                f"VGGT checkpoint has unexpected non-disabled keys: {sorted(disallowed)}"
            )
        device = torch.device(self.device_name)
        model = model.to(device=device)
        model.eval()
        self._model = model
        self._preprocess = load_module.load_and_preprocess_images
        self._decode_pose = pose_module.pose_encoding_to_extri_intri
        self._torch = torch
        return self._model

    def infer(
        self,
        *,
        token: str,
        image_paths: Sequence[Path | str],
        source_image_hw: np.ndarray,
        known_camera_centers_m: np.ndarray | None = None,
        metric_depth_reference_m: np.ndarray | None = None,
    ) -> GeometryTeacherSample:
        """Infer one sample from ``V`` simultaneous images into ``[V,Hd,Wd]``."""

        if len(image_paths) != len(self.view_names):
            raise ValueError("VGGT image_paths length must match view_names")
        source_hw = np.asarray(source_image_hw, dtype=np.int64)
        if source_hw.shape != (len(self.view_names), 2) or np.any(source_hw <= 0):
            raise ValueError("source_image_hw must be positive [V,2]")
        known_centers = None
        if known_camera_centers_m is not None:
            known_centers = np.asarray(known_camera_centers_m, dtype=np.float32)
            if known_centers.shape != (len(self.view_names), 3):
                raise ValueError("known_camera_centers_m must have shape [V,3]")
        resolved_paths = [Path(path) for path in image_paths]
        missing_paths = [str(path) for path in resolved_paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"VGGT input image not found: {missing_paths[0]}"
            )

        model = self.load_model()
        torch = self._torch
        images = self._preprocess(
            [str(path) for path in resolved_paths], mode=self.preprocess_mode
        )
        if images.ndim != 4 or images.shape[:2] != (len(self.view_names), 3):
            raise RuntimeError("official VGGT preprocessor must return [V,3,H,W]")
        # NAVSIM views share one raw resolution and stay uncropped in official
        # crop mode.  Reject tall inputs because their center crop would need an
        # additional cache coordinate transform not represented by schema v1.
        if any(width < height for height, width in source_hw.tolist()):
            raise ValueError("VGGT crop mode does not support tall NAVSIM inputs")
        images = images.to(device=next(model.parameters()).device)
        device_type = images.device.type
        autocast_enabled = self.use_bfloat16 and device_type != "cpu"
        with torch.inference_mode():
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                prediction = model(images)
        required = {"pose_enc", "depth", "depth_conf"}
        if not required.issubset(prediction):
            raise RuntimeError(
                f"official VGGT output lacks keys: {sorted(required - prediction.keys())}"
            )
        depth = prediction["depth"]
        raw_confidence = prediction["depth_conf"]
        if depth.shape != (
            1,
            len(self.view_names),
            images.shape[-2],
            images.shape[-1],
            1,
        ):
            raise RuntimeError("official VGGT depth must have shape [1,V,H,W,1]")
        if raw_confidence.shape != depth.shape[:-1]:
            raise RuntimeError("official VGGT depth_conf must have shape [1,V,H,W]")
        relative_depth = depth[0, ..., 0].float()
        if self.metricization == "da3_scale_anchor":
            if metric_depth_reference_m is None:
                raise ValueError(
                    "da3_scale_anchor requires metric_depth_reference_m [V,Hr,Wr]"
                )
            reference = np.asarray(metric_depth_reference_m, dtype=np.float32)
            if reference.ndim != 3 or reference.shape[0] != len(self.view_names):
                raise ValueError(
                    "metric_depth_reference_m must have shape [V,Hr,Wr]"
                )
            aligned_relative = torch.nn.functional.interpolate(
                relative_depth[:, None],
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            metric_scale, scale_diagnostics = (
                estimate_metric_scale_from_depth_reference(
                    aligned_relative.cpu().numpy(), reference
                )
            )
        else:
            if known_centers is None:
                raise ValueError(
                    "camera_rig metricization requires known_camera_centers_m"
                )
            extrinsics, _ = self._decode_pose(
                prediction["pose_enc"].float(),
                image_size_hw=tuple(int(value) for value in images.shape[-2:]),
                build_intrinsics=False,
            )
            metric_scale, scale_diagnostics = estimate_metric_scale_from_camera_rig(
                extrinsics[0].detach().cpu().numpy(), known_centers
            )
        metric_depth = relative_depth * metric_scale
        # VGGT's expp1 confidence is positive but not calibrated to [0,1].
        # This bounded monotonic transform is recorded explicitly in metadata.
        confidence = raw_confidence[0].float().clamp_min(0.0)
        confidence = confidence / (1.0 + confidence)
        metric_depth = torch.nn.functional.interpolate(
            metric_depth[:, None],
            size=self.output_hw,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        confidence = torch.nn.functional.interpolate(
            confidence[:, None],
            size=self.output_hw,
            mode="bilinear",
            align_corners=False,
        )[:, 0].clamp(0.0, 1.0)
        valid = (
            torch.isfinite(metric_depth)
            & (metric_depth > 0.0)
            & torch.isfinite(confidence)
        )
        metric_depth = torch.where(valid, metric_depth, 0.0)
        confidence = torch.where(valid, confidence, 0.0)
        depth_m = metric_depth.cpu().numpy().astype(np.float32, copy=False)
        confidence_np = confidence.cpu().numpy().astype(np.float32, copy=False)
        valid_np = valid.cpu().numpy().astype(np.bool_, copy=False)
        depth_hw = np.repeat(
            np.asarray(self.output_hw, dtype=np.int64)[None],
            len(self.view_names),
            axis=0,
        )
        resize_scale_xy = np.stack(
            (
                depth_hw[:, 1] / source_hw[:, 1],
                depth_hw[:, 0] / source_hw[:, 0],
            ),
            axis=-1,
        ).astype(np.float32)
        return GeometryTeacherSample(
            token=str(token),
            view_names=self.view_names,
            depth_m=depth_m,
            confidence=confidence_np,
            valid_mask=valid_np,
            source_image_hw=source_hw,
            depth_hw=depth_hw,
            resize_scale_xy=resize_scale_xy,
            coordinate_frame="camera_optical_z_depth_m",
            metadata={
                "teacher_name": self.name,
                "teacher_version": self.version,
                "frame_index": self.frame_index,
                "preprocess_mode": self.preprocess_mode,
                "preprocess_hw": [int(value) for value in images.shape[-2:]],
                "confidence_source": "teacher_expp1_over_one_plus_expp1",
                "metric_scale": metric_scale,
                "metricization_mode": self.metricization,
                "metricization": scale_diagnostics,
            },
        ).validate()
