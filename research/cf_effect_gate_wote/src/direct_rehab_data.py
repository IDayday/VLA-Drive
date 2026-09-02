"""Effect-free joins and complete-256 batching for Direct rehabilitation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from .direct_current_cache import (
    DIRECT_MODEL_INPUT_KEYS,
    SELECTOR_REFERENCE_KEYS,
    selector_index_is_maximal,
)
from .direct_rehab_contracts import (
    AccessAuditLog,
    AccessPolicy,
    assert_no_effect_input_stores,
)
from .feature_store import FeatureShardReader, stable_array_hash
from .independent_label_store import (
    SIX_FACTOR_LABEL_SCHEMA_VERSION,
    SixFactorIndependentCandidateLabelStore,
)
from .six_factor_metrics import pdms_from_six_factors


CANDIDATE_COUNT = 256


class DirectRehabDataError(RuntimeError):
    """A current-only feature/label join is unsafe or inconsistent."""


@dataclass(frozen=True)
class DirectScene:
    token: str
    trajectory: npt.NDArray[np.floating]
    ego_status: npt.NDArray[np.floating]
    current_bev_tokens: npt.NDArray[np.floating]
    candidate_current_feature: npt.NDArray[np.floating]
    factor_labels: npt.NDArray[np.float32]
    score_labels: npt.NDArray[np.float32]
    oracle_index: int
    wote_selected_index: int | None = None
    wote_final_rewards: npt.NDArray[np.floating] | None = None


@dataclass(frozen=True)
class DirectBatch:
    tokens: tuple[str, ...]
    trajectory: Tensor
    ego_status: Tensor
    current_bev_tokens: Tensor
    candidate_current_feature: Tensor
    factor_labels: Tensor
    score_labels: Tensor
    oracle_indices: Tensor

    def to(self, device: torch.device) -> "DirectBatch":
        return DirectBatch(
            tokens=self.tokens,
            trajectory=self.trajectory.to(device=device, dtype=torch.float32),
            ego_status=self.ego_status.to(device=device, dtype=torch.float32),
            current_bev_tokens=self.current_bev_tokens.to(
                device=device, dtype=torch.float32
            ),
            candidate_current_feature=self.candidate_current_feature.to(
                device=device, dtype=torch.float32
            ),
            factor_labels=self.factor_labels.to(device=device, dtype=torch.float32),
            score_labels=self.score_labels.to(device=device, dtype=torch.float32),
            oracle_indices=self.oracle_indices.to(device=device),
        )


@dataclass(frozen=True)
class DirectDataset:
    scenes: tuple[DirectScene, ...]

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(scene.token for scene in self.scenes)

    def __len__(self) -> int:
        return len(self.scenes)


def _feature_keys(require_selector_reference: bool) -> tuple[str, ...]:
    keys = {
        "trajectory",
        "ego_status_feature",
        "current_bev_tokens",
        "candidate_current_feature",
    }
    if require_selector_reference:
        keys.update({"selected_index", "final_rewards"})
    if not keys <= set(DIRECT_MODEL_INPUT_KEYS | SELECTOR_REFERENCE_KEYS):
        raise AssertionError("Direct loader field allow-list is internally inconsistent")
    return tuple(sorted(keys))


def load_direct_dataset(
    *,
    feature_root: Path,
    label_root: Path,
    expected_tokens: Sequence[str],
    access_policy: AccessPolicy,
    phase: str,
    access_log: Path | None = None,
    require_selector_reference: bool = False,
    configured_input_stores: Sequence[str] = (
        "frozen_current_features",
        "independent_six_factor_labels",
    ),
) -> DirectDataset:
    """Join only current features and independent labels by immutable identity."""

    assert_no_effect_input_stores(configured_input_stores)
    expected = tuple(str(token) for token in expected_tokens)
    if not expected or len(expected) != len(set(expected)):
        raise DirectRehabDataError("expected Direct token list must be unique/non-empty")
    expected_set = set(expected)
    for token in expected:
        access_policy.assert_token_access(token, phase)
    audit = AccessAuditLog(access_log, access_policy, phase) if access_log else None

    labels = SixFactorIndependentCandidateLabelStore(label_root)
    if labels.manifest.get("schema_version") != SIX_FACTOR_LABEL_SCHEMA_VERSION:
        raise DirectRehabDataError("Direct labels are not independent six-factor v2")
    label_index = labels.scene_index()
    reader = FeatureShardReader(feature_root, verify_shard_hashes=False)
    feature_identity = reader.manifest.get("identity", {})
    if feature_identity.get("label_source") != "none":
        raise DirectRehabDataError("Direct features must declare label_source=none")
    candidate_bank_hash = str(feature_identity.get("candidate_bank_hash"))

    scenes: list[DirectScene] = []
    keys = _feature_keys(require_selector_reference)
    for sidecar, arrays in reader.iter_shards(keys):
        array_names = set(arrays)
        forbidden = [
            name
            for name in array_names
            if "future" in name.lower() or "effect" in name.lower()
        ]
        if forbidden:
            raise DirectRehabDataError(f"Direct loader decoded forbidden fields: {forbidden}")
        for index, record in enumerate(sidecar["records"]):
            token = str(record["scene_token"])
            if token not in expected_set:
                continue
            access_policy.assert_token_access(token, phase)
            if audit is not None:
                audit.record(token, "direct_feature_label_read")
            if token not in label_index:
                raise DirectRehabDataError(f"independent labels missing {token}")
            label = label_index[token]
            if record.get("candidate_bank_hash") != candidate_bank_hash:
                raise DirectRehabDataError(f"feature candidate bank mismatch: {token}")
            if label.record.candidate_bank_hash != candidate_bank_hash:
                raise DirectRehabDataError(f"feature/label candidate bank mismatch: {token}")
            trajectory = np.asarray(arrays["trajectory"][index], dtype=np.float32)
            if trajectory.shape != (256, 8, 3):
                raise DirectRehabDataError(f"invalid candidate tensor for {token}")
            trajectory_hash = stable_array_hash(trajectory)
            if trajectory_hash != record.get("trajectory_hash"):
                raise DirectRehabDataError(f"feature trajectory hash mismatch: {token}")
            if trajectory_hash != label.record.trajectory_hash:
                raise DirectRehabDataError(f"feature/label trajectory mismatch: {token}")
            factors = np.asarray(label.factors, dtype=np.float32)
            scores = np.asarray(label.score, dtype=np.float32)
            if factors.shape != (256, 6) or scores.shape != (256,):
                raise DirectRehabDataError(f"invalid full-list labels for {token}")
            error = np.max(
                np.abs(pdms_from_six_factors(factors) - scores.astype(np.float64))
            )
            if float(error) > 1.0e-6:
                raise DirectRehabDataError(
                    f"six-factor score reconstruction failed for {token}: {error}"
                )
            selected = (
                int(np.asarray(arrays["selected_index"][index]).reshape(-1)[0])
                if require_selector_reference
                else None
            )
            final_rewards = (
                np.asarray(arrays["final_rewards"][index])
                if require_selector_reference
                else None
            )
            if selected is not None and not selector_index_is_maximal(
                final_rewards, selected
            ):
                raise DirectRehabDataError(f"invalid WoTE selector reference for {token}")
            scenes.append(
                DirectScene(
                    token=token,
                    trajectory=trajectory,
                    ego_status=np.asarray(
                        arrays["ego_status_feature"][index], dtype=np.float32
                    ),
                    current_bev_tokens=np.asarray(arrays["current_bev_tokens"][index]),
                    candidate_current_feature=np.asarray(
                        arrays["candidate_current_feature"][index]
                    ),
                    factor_labels=factors,
                    score_labels=scores,
                    oracle_index=int(label.oracle_index),
                    wote_selected_index=selected,
                    wote_final_rewards=final_rewards,
                )
            )
    actual = tuple(scene.token for scene in scenes)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise DirectRehabDataError(
            f"Direct scene order/identity mismatch: missing={missing}, extra={extra}"
        )
    return DirectDataset(tuple(scenes))


def concatenate_direct_datasets(
    datasets: Sequence[DirectDataset],
    expected_tokens: Sequence[str],
) -> DirectDataset:
    scenes = tuple(scene for dataset in datasets for scene in dataset.scenes)
    actual = tuple(scene.token for scene in scenes)
    expected = tuple(str(token) for token in expected_tokens)
    if actual != expected or len(actual) != len(set(actual)):
        raise DirectRehabDataError("nested-scale dataset composition changed token order")
    return DirectDataset(scenes)


def stack_direct_batch(scenes: Sequence[DirectScene]) -> DirectBatch:
    if not scenes:
        raise ValueError("cannot stack an empty Direct batch")
    return DirectBatch(
        tokens=tuple(scene.token for scene in scenes),
        trajectory=torch.from_numpy(
            np.stack([np.asarray(scene.trajectory) for scene in scenes])
        ),
        ego_status=torch.from_numpy(
            np.stack([np.asarray(scene.ego_status) for scene in scenes])
        ),
        current_bev_tokens=torch.from_numpy(
            np.stack([np.asarray(scene.current_bev_tokens) for scene in scenes])
        ),
        candidate_current_feature=torch.from_numpy(
            np.stack([np.asarray(scene.candidate_current_feature) for scene in scenes])
        ),
        factor_labels=torch.from_numpy(
            np.stack([scene.factor_labels for scene in scenes])
        ),
        score_labels=torch.from_numpy(
            np.stack([scene.score_labels for scene in scenes])
        ),
        oracle_indices=torch.as_tensor(
            [scene.oracle_index for scene in scenes], dtype=torch.long
        ),
    )


def iter_direct_batches(
    dataset: DirectDataset,
    *,
    batch_scenes: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> Iterator[DirectBatch]:
    if batch_scenes <= 0:
        raise ValueError("batch_scenes must be positive")
    order = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(int(seed) * 1_000_003 + int(epoch))
        rng.shuffle(order)
    for start in range(0, len(order), batch_scenes):
        yield stack_direct_batch(
            [dataset.scenes[int(index)] for index in order[start : start + batch_scenes]]
        )
