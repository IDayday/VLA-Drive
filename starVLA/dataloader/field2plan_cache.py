"""Strict, manifest-validated Field2Plan cache primitives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tokens(tokens: Iterable[str]) -> str:
    """Hash an ordered token list without ambiguous string concatenation."""

    digest = hashlib.sha256()
    for token in tokens:
        encoded = str(token).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_dataset_binding(
    *,
    split_metadata: Dict[str, Any],
    selected_tokens: Iterable[str],
    datalist_path: str,
    cache_name: str,
) -> None:
    """Bind a cache split to the exact datalist and selected token prefix.

    A cache may cover either the complete datalist or an explicit debug prefix.
    In both cases the manifest remains bound to the source datalist file hash,
    and the dataset selection must be an ordered prefix of that file.
    """

    path = Path(datalist_path)
    if not path.is_file():
        raise FileNotFoundError(f"{cache_name} datalist not found: {path}")
    if split_metadata["datalist_sha256"] != sha256_file(path):
        raise ValueError(f"{cache_name} cache datalist checksum mismatch")
    try:
        full_tokens = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {cache_name} datalist JSON: {path}") from error
    if not isinstance(full_tokens, list) or not all(
        isinstance(token, str) and token for token in full_tokens
    ):
        raise ValueError(f"{cache_name} datalist must contain non-empty token strings")
    selected = [str(token) for token in selected_tokens]
    if selected != full_tokens[: len(selected)]:
        raise ValueError(
            f"{cache_name} dataset tokens are not an ordered datalist prefix"
        )
    declared_count = split_metadata["entry_count"]
    declared_hash = split_metadata["tokens_sha256"]
    exact_selected = (
        declared_count == len(selected) and declared_hash == hash_tokens(selected)
    )
    exact_full = (
        declared_count == len(full_tokens) and declared_hash == hash_tokens(full_tokens)
    )
    if not (exact_selected or exact_full):
        raise ValueError(
            f"{cache_name} cache token count/hash does not cover the selected dataset"
        )


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON via fsync and same-directory atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write an NPZ payload atomically without pickle objects."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_npz_compressed(path: Path, **arrays: np.ndarray) -> None:
    """Atomically write a compressed, pickle-free NPZ payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_draft_entry(
    path: Path,
    expected_token: str,
    num_candidates: int,
) -> Dict[str, np.ndarray]:
    """Load and validate one ``[M,8,*]`` draft-cache entry."""

    try:
        with np.load(path, allow_pickle=False) as payload:
            if not {"draft_action", "physical_trajectory", "token"}.issubset(
                payload.files
            ):
                raise ValueError(f"draft cache entry lacks required keys: {path}")
            cached_token = str(payload["token"].item())
            draft = np.asarray(payload["draft_action"], dtype=np.float32)
            physical = np.asarray(payload["physical_trajectory"], dtype=np.float32)
    except (OSError, ValueError) as error:
        raise ValueError(f"corrupt draft cache entry: {path}") from error
    if cached_token != expected_token:
        raise ValueError(
            f"draft token mismatch: requested {expected_token!r}, "
            f"entry contains {cached_token!r}"
        )
    if draft.ndim == 2:
        draft = draft[None]
    if draft.shape != (num_candidates, 8, 4):
        raise ValueError(
            "draft_action must have shape "
            f"[{num_candidates},8,4], got {draft.shape}"
        )
    if physical.ndim == 2:
        physical = physical[None]
    if physical.shape != (num_candidates, 8, 3):
        raise ValueError(
            "physical_trajectory must have shape "
            f"[{num_candidates},8,3], got {physical.shape}"
        )
    if not np.isfinite(draft).all() or not np.isfinite(physical).all():
        raise ValueError(f"draft cache entry contains non-finite values: {path}")
    return {"draft_action": draft, "physical_trajectory": physical}


class DraftCacheReader:
    """Read frozen proposal actions from a strict cache.

    Layout is ``root/manifest.json`` plus ``root/{split}/{token}.npz``. The
    manifest must declare ``cache_type=baseline_draft`` and the verified
    ver-1225 trajectory schema.
    """

    def __init__(
        self,
        cache_root: str,
        split: str,
        expected_manifest_sha256: Optional[str] = None,
    ) -> None:
        self.root = Path(cache_root)
        self.split = str(split)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"draft cache manifest not found: {self.manifest_path}")
        actual_hash = sha256_file(self.manifest_path)
        if expected_manifest_sha256 and actual_hash != expected_manifest_sha256:
            raise ValueError(
                f"draft manifest checksum mismatch: expected {expected_manifest_sha256}, got {actual_hash}"
            )
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid draft cache manifest: {self.manifest_path}") from error
        self.manifest_sha256 = actual_hash
        self._validate_manifest()

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != 2:
            raise ValueError("unsupported draft cache schema_version")
        if manifest.get("cache_type") != "baseline_draft":
            raise ValueError("manifest cache_type must be baseline_draft")
        if manifest.get("status") != "complete":
            raise ValueError("draft manifest status must be complete")
        for section in ("checkpoint", "config"):
            checksum = manifest.get(section, {}).get("sha256")
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                raise ValueError(f"draft manifest requires {section}.sha256")
        git_commit = manifest.get("generator", {}).get("git_commit")
        if not isinstance(git_commit, str) or not git_commit:
            raise ValueError("draft manifest requires generator.git_commit")
        inference = manifest.get("inference", {})
        for name in ("seed", "steps", "num_candidates", "world_size", "batch_size_per_rank"):
            value = inference.get(name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"draft manifest inference.{name} must be positive")
        if inference.get("qwen_forward_mode") not in {"legacy", "optimized"}:
            raise ValueError("invalid draft qwen_forward_mode")
        self.num_candidates = int(inference["num_candidates"])
        splits = manifest.get("splits")
        if not isinstance(splits, dict) or self.split not in splits:
            raise ValueError(f"split {self.split!r} is absent from draft manifest")
        split_metadata = splits[self.split]
        if not isinstance(split_metadata, dict):
            raise ValueError("draft split metadata must be a mapping")
        for name in ("entry_count",):
            if not isinstance(split_metadata.get(name), int) or split_metadata[name] < 1:
                raise ValueError(f"draft split {name} must be positive")
        for name in ("tokens_sha256", "datalist_sha256"):
            checksum = split_metadata.get(name)
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError(f"draft split requires {name}")
        tensor = manifest.get("tensor_schema", {}).get("draft_action", {})
        if (
            tensor.get("last_dim") != 4
            or tensor.get("horizon") != 8
            or tensor.get("dtype") != "float32"
            or tensor.get("shape") != ["M", 8, 4]
        ):
            raise ValueError("draft manifest must declare action shape [...,8,4]")
        physical = manifest.get("tensor_schema", {}).get(
            "physical_trajectory", {}
        )
        if physical.get("dtype") != "float32" or physical.get("shape") != ["M", 8, 3]:
            raise ValueError("draft manifest must declare physical shape [M,8,3]")
        normalization = manifest.get("normalization", {})
        if normalization.get("version") != "ver_1225_act_norm_1":
            raise ValueError("draft normalization must be ver_1225_act_norm_1")

    def load(self, token: str) -> Dict[str, Any]:
        path = self.root / self.split / f"{token}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"draft cache entry not found: {path}")
        arrays = load_draft_entry(path, token, self.num_candidates)
        return {
            **arrays,
            "source": "cache",
            "manifest_sha256": self.manifest_sha256,
        }

    def validate_dataset_binding(
        self, tokens: Iterable[str], datalist_path: str
    ) -> None:
        """Require this manifest to cover the current ordered dataset tokens."""

        _validate_dataset_binding(
            split_metadata=self.manifest["splits"][self.split],
            selected_tokens=tokens,
            datalist_path=datalist_path,
            cache_name="draft",
        )


class GeometryCacheReader:
    """Read strict offline geometry teacher entries.

    Each entry contains metric depth and validity tensors with fixed shape
    ``[V,Hd,Wd]`` declared by the manifest. No teacher inference or fallback
    is allowed in this reader.
    """

    def __init__(
        self,
        cache_root: str,
        split: str,
        expected_manifest_sha256: Optional[str] = None,
    ) -> None:
        self.root = Path(cache_root)
        self.split = str(split)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"geometry cache manifest not found: {self.manifest_path}"
            )
        actual_hash = sha256_file(self.manifest_path)
        if expected_manifest_sha256 and actual_hash != expected_manifest_sha256:
            raise ValueError(
                "geometry manifest checksum mismatch: "
                f"expected {expected_manifest_sha256}, got {actual_hash}"
            )
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid geometry cache manifest: {self.manifest_path}"
            ) from error
        self.manifest_sha256 = actual_hash
        self._validate_manifest()

    @staticmethod
    def _require_sha256(value: Any, name: str) -> None:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"geometry manifest requires {name}")

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported geometry cache schema_version")
        if manifest.get("cache_type") != "geometry_teacher":
            raise ValueError("geometry manifest cache_type must be geometry_teacher")
        if manifest.get("status") != "complete":
            raise ValueError("geometry manifest status must be complete")
        teacher = manifest.get("teacher", {})
        if not isinstance(teacher.get("name"), str) or not teacher["name"]:
            raise ValueError("geometry manifest requires teacher.name")
        if not isinstance(teacher.get("version"), str) or not teacher["version"]:
            raise ValueError("geometry manifest requires teacher.version")
        self._require_sha256(
            teacher.get("source_index_sha256"), "teacher.source_index_sha256"
        )
        commit = manifest.get("generator", {}).get("git_commit")
        if not isinstance(commit, str) or not commit:
            raise ValueError("geometry manifest requires generator.git_commit")
        splits = manifest.get("splits")
        if not isinstance(splits, dict) or self.split not in splits:
            raise ValueError(f"split {self.split!r} is absent from geometry manifest")
        split_metadata = splits[self.split]
        if not isinstance(split_metadata.get("entry_count"), int) or split_metadata[
            "entry_count"
        ] < 1:
            raise ValueError("geometry split entry_count must be positive")
        self._require_sha256(
            split_metadata.get("tokens_sha256"), "splits.tokens_sha256"
        )
        self._require_sha256(
            split_metadata.get("datalist_sha256"), "splits.datalist_sha256"
        )
        schema = manifest.get("tensor_schema", {})
        view_names = schema.get("view_names")
        if not isinstance(view_names, list) or not view_names:
            raise ValueError("geometry tensor_schema.view_names is required")
        self.view_names = tuple(str(view) for view in view_names)
        self.array_schema = {
            "depth_m": ("float32", (len(view_names), None, None)),
            "confidence": ("float32", (len(view_names), None, None)),
            "valid_mask": ("bool", (len(view_names), None, None)),
            "source_image_hw": ("int64", (len(view_names), 2)),
            "depth_hw": ("int64", (len(view_names), 2)),
            "resize_scale_xy": ("float32", (len(view_names), 2)),
        }
        for name, (dtype, _) in self.array_schema.items():
            declaration = schema.get(name, {})
            shape = declaration.get("shape")
            if declaration.get("dtype") != dtype:
                raise ValueError(f"geometry schema {name}.dtype must be {dtype}")
            if not isinstance(shape, list) or any(
                not isinstance(dimension, int) or dimension <= 0 for dimension in shape
            ):
                raise ValueError(f"geometry schema {name}.shape must be positive")
            self.array_schema[name] = (dtype, tuple(shape))
        depth_shape = self.array_schema["depth_m"][1]
        for name in ("confidence", "valid_mask"):
            if self.array_schema[name][1] != depth_shape:
                raise ValueError(f"geometry schema {name} must match depth_m shape")
        coordinates = manifest.get("coordinates", {})
        if coordinates.get("frame") != "camera_optical_z_depth_m":
            raise ValueError("geometry cache coordinate frame is unsupported")
        if not isinstance(coordinates.get("frame_index"), int):
            raise ValueError("geometry coordinates.frame_index is required")
        if coordinates.get("confidence_source") not in {
            "finite_positive_validity",
            "teacher_confidence",
        }:
            raise ValueError("unsupported geometry confidence_source")
        self.coordinate_frame = coordinates["frame"]

    def load(self, token: str) -> Dict[str, Any]:
        """Load one entry and assert every declared tensor shape and dtype."""

        path = self.root / self.split / f"{token}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"geometry cache entry not found: {path}")
        required = {"token", *self.array_schema}
        try:
            with np.load(path, allow_pickle=False) as payload:
                if not required.issubset(payload.files):
                    raise ValueError("geometry cache entry lacks required arrays")
                cached_token = str(payload["token"].item())
                arrays = {name: np.asarray(payload[name]) for name in self.array_schema}
        except (OSError, ValueError) as error:
            raise ValueError(f"corrupt geometry cache entry: {path}") from error
        if cached_token != token:
            raise ValueError(
                f"geometry token mismatch: requested {token!r}, got {cached_token!r}"
            )
        dtype_map = {"float32": np.float32, "int64": np.int64, "bool": np.bool_}
        for name, (dtype, shape) in self.array_schema.items():
            array = arrays[name]
            if array.shape != shape:
                raise ValueError(
                    f"geometry {name} must have shape {shape}, got {array.shape}"
                )
            if array.dtype != dtype_map[dtype]:
                raise ValueError(
                    f"geometry {name} must have dtype {dtype}, got {array.dtype}"
                )
            if name != "valid_mask" and not np.isfinite(array).all():
                raise ValueError(f"geometry {name} contains non-finite values")
        if np.any(arrays["confidence"] < 0.0) or np.any(
            arrays["confidence"] > 1.0
        ):
            raise ValueError("geometry confidence must be within [0,1]")
        if np.any(arrays["depth_m"] < 0.0):
            raise ValueError("geometry depth_m cannot be negative")
        return {
            **arrays,
            "view_names": self.view_names,
            "coordinate_frame": self.coordinate_frame,
            "source": "cache",
            "manifest_sha256": self.manifest_sha256,
        }

    def validate_dataset_binding(
        self, tokens: Iterable[str], datalist_path: str
    ) -> None:
        """Require this manifest to cover the current ordered dataset tokens."""

        _validate_dataset_binding(
            split_metadata=self.manifest["splits"][self.split],
            selected_tokens=tokens,
            datalist_path=datalist_path,
            cache_name="geometry",
        )


class DynamicsCacheReader:
    """Read strictly pinned offline future-dynamics features.

    Each entry stores ``features=[H,V,Ct,Ht,Wt]`` plus temporal, confidence,
    and source-image metadata.  Missing or inconsistent entries fail fast;
    training never falls back to online V-JEPA inference.
    """

    _DTYPE_MAP = {
        "float16": np.float16,
        "float32": np.float32,
        "int64": np.int64,
        "bool": np.bool_,
    }

    def __init__(
        self,
        cache_root: str,
        split: str,
        expected_manifest_sha256: Optional[str] = None,
    ) -> None:
        self.root = Path(cache_root)
        self.split = str(split)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"dynamics cache manifest not found: {self.manifest_path}"
            )
        actual_hash = sha256_file(self.manifest_path)
        if expected_manifest_sha256 and actual_hash != expected_manifest_sha256:
            raise ValueError(
                "dynamics manifest checksum mismatch: "
                f"expected {expected_manifest_sha256}, got {actual_hash}"
            )
        try:
            self.manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid dynamics cache manifest: {self.manifest_path}"
            ) from error
        self.manifest_sha256 = actual_hash
        self._validate_manifest()

    @staticmethod
    def _require_sha(value: Any, name: str, length: int = 64) -> None:
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"dynamics manifest requires {name}")

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported dynamics cache schema_version")
        if manifest.get("cache_type") != "dynamics_teacher":
            raise ValueError("dynamics manifest cache_type must be dynamics_teacher")
        if manifest.get("status") != "complete":
            raise ValueError("dynamics manifest status must be complete")
        teacher = manifest.get("teacher", {})
        if teacher.get("name") not in {"vjepa2", "vjepa2_1", "drive_jepa"}:
            raise ValueError("dynamics manifest teacher.name is unsupported")
        if not isinstance(teacher.get("version"), str) or not teacher["version"]:
            raise ValueError("dynamics manifest requires teacher.version")
        self._require_sha(teacher.get("repo_commit"), "teacher.repo_commit", length=40)
        self._require_sha(
            teacher.get("checkpoint_sha256"), "teacher.checkpoint_sha256"
        )
        commit = manifest.get("generator", {}).get("git_commit")
        if not isinstance(commit, str) or not commit:
            raise ValueError("dynamics manifest requires generator.git_commit")
        splits = manifest.get("splits")
        if not isinstance(splits, dict) or self.split not in splits:
            raise ValueError(f"split {self.split!r} is absent from dynamics manifest")
        split_metadata = splits[self.split]
        if not isinstance(split_metadata.get("entry_count"), int) or split_metadata[
            "entry_count"
        ] < 1:
            raise ValueError("dynamics split entry_count must be positive")
        self._require_sha(split_metadata.get("tokens_sha256"), "splits.tokens_sha256")
        self._require_sha(
            split_metadata.get("datalist_sha256"), "splits.datalist_sha256"
        )

        schema = manifest.get("tensor_schema", {})
        view_names = schema.get("view_names")
        if not isinstance(view_names, list) or not view_names:
            raise ValueError("dynamics tensor_schema.view_names is required")
        self.view_names = tuple(str(view) for view in view_names)
        expected_dtypes = {
            "features": "float16",
            "confidence": "float32",
            "valid_mask": "bool",
            "frame_indices": "int64",
            "frame_times_s": "float32",
            "source_image_hw": "int64",
            "feature_hw": "int64",
        }
        self.array_schema: Dict[str, tuple[str, tuple[int, ...]]] = {}
        for name, expected_dtype in expected_dtypes.items():
            declaration = schema.get(name, {})
            shape = declaration.get("shape")
            if declaration.get("dtype") != expected_dtype:
                raise ValueError(
                    f"dynamics schema {name}.dtype must be {expected_dtype}"
                )
            if not isinstance(shape, list) or any(
                not isinstance(dimension, int) or dimension <= 0
                for dimension in shape
            ):
                raise ValueError(f"dynamics schema {name}.shape must be positive")
            self.array_schema[name] = (expected_dtype, tuple(shape))
        feature_shape = self.array_schema["features"][1]
        if len(feature_shape) != 5:
            raise ValueError("dynamics features must declare [H,V,Ct,Ht,Wt]")
        horizon, views, _, height, width = feature_shape
        if views != len(self.view_names):
            raise ValueError("dynamics feature V differs from view_names")
        expected_map = (horizon, views, height, width)
        for name in ("confidence", "valid_mask"):
            if self.array_schema[name][1] != expected_map:
                raise ValueError(f"dynamics {name} must have shape [H,V,Ht,Wt]")
        if self.array_schema["frame_indices"][1] != (horizon,):
            raise ValueError("dynamics frame_indices must have shape [H]")
        if self.array_schema["frame_times_s"][1] != (horizon,):
            raise ValueError("dynamics frame_times_s must have shape [H]")
        if self.array_schema["source_image_hw"][1] != (horizon, views, 2):
            raise ValueError("dynamics source_image_hw must have shape [H,V,2]")
        if self.array_schema["feature_hw"][1] != (horizon, views, 2):
            raise ValueError("dynamics feature_hw must have shape [H,V,2]")

        temporal = manifest.get("temporal", {})
        current = temporal.get("current_frame_index")
        history = temporal.get("history_frame_indices")
        future = temporal.get("future_frame_indices")
        interval = temporal.get("frame_interval_s")
        stride = temporal.get("teacher_temporal_stride")
        if not isinstance(current, int) or current < 0:
            raise ValueError("dynamics temporal.current_frame_index is required")
        if (
            not isinstance(history, list)
            or not history
            or any(not isinstance(index, int) or index > current for index in history)
        ):
            raise ValueError("dynamics history_frame_indices are invalid")
        if (
            not isinstance(future, list)
            or len(future) != horizon
            or any(not isinstance(index, int) or index <= current for index in future)
        ):
            raise ValueError("dynamics future_frame_indices are invalid")
        if any(right <= left for left, right in zip(future, future[1:])):
            raise ValueError("dynamics future_frame_indices must increase")
        if not isinstance(interval, (int, float)) or interval <= 0:
            raise ValueError("dynamics frame_interval_s must be positive")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("dynamics teacher_temporal_stride must be positive")
        self.current_frame_index = current
        self.history_frame_indices = tuple(history)
        self.future_frame_indices = tuple(future)
        self.frame_interval_s = float(interval)
        feature_metadata = manifest.get("features", {})
        if feature_metadata.get("spatial_layout") != "per_view_patch_grid":
            raise ValueError("unsupported dynamics spatial_layout")
        if feature_metadata.get("normalization") not in {"none", "l2"}:
            raise ValueError("unsupported dynamics feature normalization")
        projection = feature_metadata.get("projection", {})
        if not isinstance(projection.get("algorithm"), str) or not projection[
            "algorithm"
        ]:
            raise ValueError("dynamics feature projection algorithm is required")
        if not isinstance(projection.get("seed"), int):
            raise ValueError("dynamics feature projection seed is required")
        self.spatial_layout = feature_metadata["spatial_layout"]
        self.feature_normalization = feature_metadata["normalization"]
        preprocessing = manifest.get("preprocessing", {})
        input_image_hw = preprocessing.get("input_image_hw")
        if (
            not isinstance(input_image_hw, list)
            or len(input_image_hw) != 2
            or any(not isinstance(value, int) or value <= 0 for value in input_image_hw)
        ):
            raise ValueError("dynamics preprocessing.input_image_hw is required")
        if preprocessing.get("resize_policy") not in {
            "center_crop_square_then_bilinear",
        }:
            raise ValueError("unsupported dynamics preprocessing resize_policy")
        self._require_sha(preprocessing.get("sha256"), "preprocessing.sha256")
        self.input_image_hw = tuple(input_image_hw)
        self.preprocessing_hash = preprocessing["sha256"]

    def load(self, token: str) -> Dict[str, Any]:
        """Load one cache entry and validate every declared array."""

        path = self.root / self.split / f"{token}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"dynamics cache entry not found: {path}")
        required = {"token", *self.array_schema}
        try:
            with np.load(path, allow_pickle=False) as payload:
                if not required.issubset(payload.files):
                    raise ValueError("dynamics cache entry lacks required arrays")
                cached_token = str(payload["token"].item())
                arrays = {
                    name: np.asarray(payload[name]) for name in self.array_schema
                }
        except (OSError, ValueError) as error:
            raise ValueError(f"corrupt dynamics cache entry: {path}") from error
        if cached_token != token:
            raise ValueError(
                f"dynamics token mismatch: requested {token!r}, got {cached_token!r}"
            )
        for name, (dtype, shape) in self.array_schema.items():
            array = arrays[name]
            if array.shape != shape:
                raise ValueError(
                    f"dynamics {name} must have shape {shape}, got {array.shape}"
                )
            if array.dtype != self._DTYPE_MAP[dtype]:
                raise ValueError(
                    f"dynamics {name} must have dtype {dtype}, got {array.dtype}"
                )
            if name != "valid_mask" and not np.isfinite(array).all():
                raise ValueError(f"dynamics {name} contains non-finite values")
        confidence = arrays["confidence"]
        if np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise ValueError("dynamics confidence must be within [0,1]")
        if tuple(arrays["frame_indices"].tolist()) != self.future_frame_indices:
            raise ValueError("dynamics frame_indices differ from manifest")
        expected_times = (
            arrays["frame_indices"].astype(np.float32)
            - np.float32(self.current_frame_index)
        ) * np.float32(self.frame_interval_s)
        if not np.allclose(arrays["frame_times_s"], expected_times, atol=1e-6, rtol=0):
            raise ValueError("dynamics frame_times_s differ from manifest timing")
        if np.any(arrays["source_image_hw"] <= 0) or np.any(
            arrays["feature_hw"] <= 0
        ):
            raise ValueError("dynamics image/feature sizes must be positive")
        return {
            **arrays,
            "view_names": self.view_names,
            "future_frame_indices": self.future_frame_indices,
            "current_frame_index": self.current_frame_index,
            "spatial_layout": self.spatial_layout,
            "feature_normalization": self.feature_normalization,
            "source": "cache",
            "manifest_sha256": self.manifest_sha256,
        }

    def validate_dataset_binding(
        self, tokens: Iterable[str], datalist_path: str
    ) -> None:
        """Bind the dynamics cache to the exact ordered dataset split."""

        _validate_dataset_binding(
            split_metadata=self.manifest["splits"][self.split],
            selected_tokens=tokens,
            datalist_path=datalist_path,
            cache_name="dynamics",
        )
