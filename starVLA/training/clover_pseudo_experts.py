# SPDX-License-Identifier: Apache-2.0
"""Strict loader for CLOVER evaluator-filtered pseudo-expert packages."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


def pseudo_expert_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_value(value: Any) -> float:
    if isinstance(value, Mapping):
        if "pdm_score" not in value:
            raise KeyError(
                "official CLOVER pseudo-expert score has no 'pdm_score' field"
            )
        return float(value["pdm_score"])
    return float(value)


def _scene_entries(payload: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if "scenes" in payload:
            payload = payload["scenes"]
        else:
            # Support a direct token -> record package without changing its
            # trajectory or score content.
            return [dict(record, token=str(token)) for token, record in payload.items()]
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise TypeError("pseudo-expert package must contain a scene sequence")
    return payload


class CloverPseudoExpertStore:
    """Token-indexed, donor-faithful high-score/FPS pseudo-expert selector.

    This consumes the official generated package.  It deliberately does not
    synthesize Gaussian perturbations: CLOVER pseudo experts use privileged
    route, drivable-area, future occupancy, and evaluator filtering, and the
    authors have not yet released that generator as of repository revision
    6aba8b7.  Treating simple perturbations as equivalent would violate the
    method's supervision contract.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        top_k: int = 8,
        score_threshold: float = 0.8,
        fps_min_distance: float = 0.05,
        gt_coverage_distance: float = 0.5,
        require_valid_flag: bool = True,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"CLOVER pseudo-expert package is missing: {self.path}")
        if top_k <= 0:
            raise ValueError("pseudo-expert top_k must be positive")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("pseudo-expert score threshold must lie in [0,1]")
        self.top_k = int(top_k)
        self.score_threshold = float(score_threshold)
        self.fps_min_distance = float(fps_min_distance)
        self.gt_coverage_distance = float(gt_coverage_distance)
        self.sha256 = pseudo_expert_sha256(self.path)
        with self.path.open("rb") as stream:
            payload = pickle.load(stream)
        self._index: dict[str, tuple[list[np.ndarray], list[float]]] = {}
        for scene in _scene_entries(payload):
            # Donor loader uses scene.get("valid", False): a missing flag is
            # not silently treated as a privileged/evaluator-validated scene.
            if require_valid_flag and not bool(scene.get("valid", False)):
                continue
            token = str(scene.get("token", ""))
            if not token:
                raise ValueError("pseudo-expert scene has no token")
            raw_trajectories = scene.get(
                "trajectories_relative", scene.get("trajectories")
            )
            raw_scores = scene.get("scores")
            if raw_trajectories is None or raw_scores is None:
                raise KeyError(f"pseudo-expert scene {token} lacks trajectories/scores")
            if len(raw_trajectories) != len(raw_scores):
                raise ValueError(f"pseudo-expert scene {token} trajectory/score count differs")
            trajectories: list[np.ndarray] = []
            scores: list[float] = []
            for trajectory, score in zip(raw_trajectories, raw_scores):
                array = np.asarray(trajectory, dtype=np.float32)
                if array.shape != (8, 3) or not np.isfinite(array).all():
                    raise ValueError(
                        f"pseudo-expert trajectory for {token} must be finite [8,3]"
                    )
                score_value = _score_value(score)
                if not np.isfinite(score_value):
                    raise ValueError(f"pseudo-expert score for {token} is not finite")
                trajectories.append(array)
                scores.append(score_value)
            if trajectories:
                self._index[token] = (trajectories, scores)
        if not self._index:
            raise ValueError("pseudo-expert package contains no usable scenes")

    def __len__(self) -> int:
        return len(self._index)

    def contains(self, token: str) -> bool:
        return str(token) in self._index

    def select(
        self, token: str, ground_truth: np.ndarray | Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply CLOVER score filtering, greedy FPS, and GT fallback."""

        gt = (
            ground_truth.detach().cpu().numpy()
            if torch.is_tensor(ground_truth)
            else np.asarray(ground_truth)
        ).astype(np.float32, copy=False)
        if gt.shape != (8, 3) or not np.isfinite(gt).all():
            raise ValueError("ground-truth trajectory must be finite [8,3]")
        entries = self._index.get(str(token))
        candidates: list[tuple[np.ndarray, float]] = []
        if entries is not None:
            candidates = [
                (trajectory, score)
                for trajectory, score in zip(*entries)
                if score >= self.score_threshold
            ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected: list[np.ndarray] = []
        used: set[int] = set()
        if candidates:
            selected.append(candidates[0][0])
            used.add(0)
            while len(selected) < self.top_k and len(used) < len(candidates):
                best_index = -1
                best_distance = -1.0
                for index, (candidate, _) in enumerate(candidates):
                    if index in used:
                        continue
                    minimum = min(
                        float(np.abs(candidate[:, :2] - prior[:, :2]).mean())
                        for prior in selected
                    )
                    if minimum > best_distance:
                        best_index = index
                        best_distance = minimum
                if best_index < 0 or best_distance < self.fps_min_distance:
                    break
                selected.append(candidates[best_index][0])
                used.add(best_index)
        gt_covered = any(
            float(np.abs(gt[:, :2] - candidate[:, :2]).mean())
            < self.gt_coverage_distance
            for candidate in selected
        )
        if not gt_covered and len(selected) < self.top_k:
            selected.append(gt)
        if not selected:
            selected.append(gt)

        output = np.zeros((self.top_k, 8, 3), dtype=np.float32)
        mask = np.zeros((self.top_k,), dtype=np.bool_)
        for index, trajectory in enumerate(selected[: self.top_k]):
            output[index] = trajectory
            mask[index] = True
        return output, mask

    def batch(
        self,
        tokens: Sequence[str],
        ground_truth: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        require_all_tokens: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if ground_truth.ndim != 3 or tuple(ground_truth.shape[-2:]) != (8, 3):
            raise ValueError("ground_truth must have shape [B,8,3]")
        if len(tokens) != ground_truth.shape[0]:
            raise ValueError("token count differs from ground-truth batch")
        missing = [str(token) for token in tokens if not self.contains(str(token))]
        if require_all_tokens and missing:
            preview = ", ".join(missing[:8])
            raise KeyError(
                f"pseudo-expert cache misses {len(missing)} batch tokens: {preview}"
            )
        selected = [
            self.select(str(token), ground_truth[index])
            for index, token in enumerate(tokens)
        ]
        trajectories = torch.as_tensor(
            np.stack([item[0] for item in selected]), device=device, dtype=dtype
        )
        mask = torch.as_tensor(
            np.stack([item[1] for item in selected]), device=device, dtype=torch.bool
        )
        return trajectories, mask

    def coverage(self, tokens: Sequence[str]) -> float:
        if not tokens:
            return 1.0
        return sum(self.contains(str(token)) for token in tokens) / len(tokens)
