"""Strict offline cache readers for GroundedWorld external and EMA targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from .field2plan_cache import _validate_dataset_binding, sha256_file


def _shape(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} shape must be a sequence")
    shape = tuple(int(item) for item in value)
    if not shape or min(shape) <= 0:
        raise ValueError(f"{name} shape entries must be positive")
    return shape


class _GroundedWorldCacheReader:
    cache_type: str

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        expected_manifest_sha256: Optional[str] = None,
    ) -> None:
        self.root = Path(cache_root)
        self.split = str(split)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"{self.cache_type} manifest not found: {self.manifest_path}"
            )
        actual_hash = sha256_file(self.manifest_path)
        if expected_manifest_sha256 and actual_hash != expected_manifest_sha256:
            raise ValueError(f"{self.cache_type} manifest checksum mismatch")
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {self.cache_type} manifest") from error
        self.manifest_sha256 = actual_hash
        self._validate_common()
        self._validate_specific()

    def _validate_common(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != 1:
            raise ValueError(f"unsupported {self.cache_type} schema_version")
        if manifest.get("cache_type") != self.cache_type:
            raise ValueError(f"manifest cache_type must be {self.cache_type}")
        if manifest.get("status") != "complete":
            raise ValueError(f"{self.cache_type} manifest status must be complete")
        splits = manifest.get("splits")
        if not isinstance(splits, dict) or self.split not in splits:
            raise ValueError(f"split {self.split!r} absent from {self.cache_type}")
        metadata = splits[self.split]
        if not isinstance(metadata, dict) or int(metadata.get("entry_count", 0)) <= 0:
            raise ValueError(f"invalid {self.cache_type} split metadata")
        for name in ("tokens_sha256", "datalist_sha256"):
            checksum = metadata.get(name)
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError(f"{self.cache_type} split requires {name}")

    def _validate_specific(self) -> None:
        raise NotImplementedError

    def validate_dataset_binding(
        self, tokens: Iterable[str], datalist_path: str | Path
    ) -> None:
        _validate_dataset_binding(
            split_metadata=self.manifest["splits"][self.split],
            selected_tokens=tokens,
            datalist_path=str(datalist_path),
            cache_name=self.cache_type,
        )

    def _load_arrays(self, token: str, required: set[str]) -> Dict[str, np.ndarray]:
        path = self.root / self.split / f"{token}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"{self.cache_type} entry not found: {path}")
        try:
            with np.load(path, allow_pickle=False) as payload:
                if not required.issubset(payload.files):
                    missing = sorted(required - set(payload.files))
                    raise ValueError(f"entry lacks keys {missing}")
                cached_token = str(payload["token"].item())
                arrays = {
                    name: np.asarray(payload[name]) for name in required - {"token"}
                }
        except (OSError, ValueError) as error:
            raise ValueError(f"corrupt {self.cache_type} entry: {path}") from error
        if cached_token != str(token):
            raise ValueError(f"{self.cache_type} token mismatch")
        return arrays


class PriorCacheReader(_GroundedWorldCacheReader):
    """Read current/history-only Driving-JEPA or control representations."""

    cache_type = "grounded_world_prior"

    def _validate_specific(self) -> None:
        teacher = self.manifest.get("teacher", {})
        for name in ("name", "domain", "checkpoint_sha256"):
            if not teacher.get(name):
                raise ValueError(f"prior teacher requires {name}")
        if len(str(teacher["checkpoint_sha256"])) != 64:
            raise ValueError("prior teacher checkpoint_sha256 must be SHA-256")
        temporal = self.manifest.get("temporal", {})
        self.current_frame_index = int(temporal.get("current_frame_index", -1))
        self.history_frame_indices = tuple(
            int(value) for value in temporal.get("history_frame_indices", [])
        )
        if (
            not self.history_frame_indices
            or self.history_frame_indices[-1] != self.current_frame_index
        ):
            raise ValueError("prior cache must end at current_frame_index")
        if any(index > self.current_frame_index for index in self.history_frame_indices):
            raise ValueError("prior cache cannot contain future frame indices")
        self.frame_interval_s = float(temporal.get("frame_interval_s", 0.0))
        if self.frame_interval_s <= 0:
            raise ValueError("prior frame_interval_s must be positive")
        schema = self.manifest.get("tensor_schema", {})
        self.feature_shape = _shape(
            schema.get("features", {}).get("shape"), "features"
        )
        self.confidence_shape = _shape(
            schema.get("confidence", {}).get("shape"), "confidence"
        )
        if self.feature_shape[0] != len(self.history_frame_indices):
            raise ValueError("prior feature Th differs from history indices")
        expected_confidence = (
            self.feature_shape[0],
            self.feature_shape[1],
            self.feature_shape[3],
            self.feature_shape[4],
        )
        if self.confidence_shape != expected_confidence:
            raise ValueError("prior confidence shape differs from features")

    def load(self, token: str) -> Dict[str, Any]:
        arrays = self._load_arrays(token, {"token", "features", "confidence"})
        features = arrays["features"].astype(np.float32, copy=False)
        confidence = arrays["confidence"].astype(np.float32, copy=False)
        if features.shape != self.feature_shape or confidence.shape != self.confidence_shape:
            raise ValueError("prior entry tensor shape differs from manifest")
        if not np.isfinite(features).all() or not np.isfinite(confidence).all():
            raise ValueError("prior entry contains non-finite values")
        return {
            "features": features,
            "confidence": confidence,
            "frame_indices": np.asarray(self.history_frame_indices, dtype=np.int64),
            "teacher": str(self.manifest["teacher"]["name"]),
            "manifest_sha256": self.manifest_sha256,
        }


class FutureTargetCacheReader(_GroundedWorldCacheReader):
    """Read shared student/EMA future targets, never an external future teacher."""

    cache_type = "grounded_world_future_target"

    def _validate_specific(self) -> None:
        producer = self.manifest.get("producer", {})
        if producer.get("source") != "student_ema":
            raise ValueError("future target source must be student_ema")
        if producer.get("shared_across_teacher_controls") is not True:
            raise ValueError("future target must be shared across teacher controls")
        if len(str(producer.get("checkpoint_sha256", ""))) != 64:
            raise ValueError("future target checkpoint_sha256 must be SHA-256")
        decay = float(producer.get("ema_decay", 0.0))
        if not 0.0 < decay < 1.0:
            raise ValueError("future target ema_decay must be in (0,1)")
        temporal = self.manifest.get("temporal", {})
        self.future_frame_indices = tuple(
            int(value) for value in temporal.get("future_frame_indices", [])
        )
        if not self.future_frame_indices:
            raise ValueError("future target requires future_frame_indices")
        self.frame_interval_s = float(temporal.get("frame_interval_s", 0.0))
        if self.frame_interval_s <= 0:
            raise ValueError("future target frame_interval_s must be positive")
        schema = self.manifest.get("tensor_schema", {})
        self.feature_shape = _shape(
            schema.get("features", {}).get("shape"), "features"
        )
        self.valid_shape = _shape(
            schema.get("valid_mask", {}).get("shape"), "valid_mask"
        )
        if self.feature_shape[0] != len(self.future_frame_indices):
            raise ValueError("future target H differs from frame indices")
        expected_valid = (
            self.feature_shape[0],
            self.feature_shape[2],
            self.feature_shape[3],
        )
        if self.valid_shape != expected_valid:
            raise ValueError("future valid_mask shape differs from features")

    def load(self, token: str) -> Dict[str, Any]:
        arrays = self._load_arrays(token, {"token", "features", "valid_mask"})
        features = arrays["features"].astype(np.float32, copy=False)
        valid = arrays["valid_mask"].astype(np.bool_, copy=False)
        if features.shape != self.feature_shape or valid.shape != self.valid_shape:
            raise ValueError("future target entry tensor shape differs from manifest")
        if not np.isfinite(features).all():
            raise ValueError("future target contains non-finite values")
        return {
            "features": features,
            "valid_mask": valid,
            "frame_indices": np.asarray(self.future_frame_indices, dtype=np.int64),
            "source": "student_ema",
            "manifest_sha256": self.manifest_sha256,
        }


class ConsequenceCacheReader(_GroundedWorldCacheReader):
    """Read perturbed trajectories and non-aggregate physical consequences."""

    cache_type = "grounded_world_consequence"

    def _validate_specific(self) -> None:
        producer = self.manifest.get("producer", {})
        if producer.get("source") != "navsim_physical_components":
            raise ValueError("consequence source must be navsim_physical_components")
        if producer.get("contains_aggregate_epdms") is not False:
            raise ValueError("consequence cache must not contain aggregate EPDMS")
        schema = self.manifest.get("tensor_schema", {})
        self.trajectory_shape = _shape(
            schema.get("physical_trajectories", {}).get("shape"),
            "physical_trajectories",
        )
        self.value_shape = _shape(schema.get("values", {}).get("shape"), "values")
        self.valid_shape = _shape(
            schema.get("valid_mask", {}).get("shape"), "valid_mask"
        )
        if len(self.trajectory_shape) != 3 or self.trajectory_shape[1:] != (8, 3):
            raise ValueError("consequence trajectories must have shape [K,8,3]")
        if self.value_shape != (self.trajectory_shape[0], 6):
            raise ValueError("consequence values must have shape [K,6]")
        if self.valid_shape != self.value_shape:
            raise ValueError("consequence valid_mask shape differs from values")
        components = tuple(schema.get("components", []))
        expected = (
            "clearance",
            "ttc",
            "collision",
            "lane_distance",
            "progress",
            "comfort",
        )
        if components != expected:
            raise ValueError("consequence component ordering is invalid")

    def load(self, token: str) -> Dict[str, Any]:
        arrays = self._load_arrays(
            token,
            {"token", "physical_trajectories", "values", "valid_mask"},
        )
        trajectories = arrays["physical_trajectories"].astype(np.float32, copy=False)
        values = arrays["values"].astype(np.float32, copy=False)
        valid = arrays["valid_mask"].astype(np.bool_, copy=False)
        if trajectories.shape != self.trajectory_shape:
            raise ValueError("consequence trajectory shape differs from manifest")
        if values.shape != self.value_shape or valid.shape != self.valid_shape:
            raise ValueError("consequence label shape differs from manifest")
        if not np.isfinite(trajectories).all() or not np.isfinite(values[valid]).all():
            raise ValueError("consequence entry contains non-finite valid values")
        return {
            "physical_trajectories": trajectories,
            "values": values,
            "valid_mask": valid,
            "manifest_sha256": self.manifest_sha256,
        }
