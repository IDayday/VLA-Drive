"""Leakage-safe data joins and deterministic schedules for Gate2O v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from .effect_tokenizer import (
    MODEL_VARIANTS,
    EffectTokenPacker,
    intervene_effects,
)
from .feature_store import (
    FeatureShardReader,
    atomic_write_json,
    sha256_file,
    stable_array_hash,
    stable_json_hash,
)
from .independent_label_store import (
    SIX_FACTOR_LABEL_SCHEMA_VERSION,
    SixFactorIndependentCandidateLabelStore,
)
from .six_factor_metrics import SIX_FACTOR_ORDER, pdms_from_six_factors


CANDIDATE_COUNT = 256
TRAIN_CANDIDATES = 64
PAIR_COUNT = 256
SPLIT_FILENAMES = {
    "train": "oracle_effect_v2_train_tokens.txt",
    "val": "oracle_effect_v2_val_tokens.txt",
    "test": "oracle_effect_v2_test_tokens.txt",
}
REQUIRED_FEATURE_KEYS = {
    "current_bev_tokens",
    "current_bev_pool",
    "ego_status_feature",
    "trajectory",
    "candidate_current_feature",
    "reward_feature",
    "future_ego_features_by_step",
    "future_bev_pool_by_step",
    "future_bev_tokens_by_step",
    "environment_only_future",
    "shared_environment_future",
    "selected_index",
    "final_rewards",
}
FORBIDDEN_FROZEN_LABEL_KEYS = {
    "factor_labels",
    "score_labels",
    "oracle_index",
    "published_labels",
    "formatted_pdm_score_256",
}
FORBIDDEN_EFFECT_KEYS = {
    "factor_labels",
    "score_labels",
    "oracle_index",
    "selected_index",
    "final_rewards",
    "nc",
    "dac",
    "ddc",
    "ep",
    "ttc",
    "comfort",
    "pdms",
    "epdms",
}

EXPECTED_ASSET_HASHES = {
    "checkpoint": "f5e73261cc55220d681bdfe2ce306a2f8e8cd555b10be51034e9b20e2967e53b",
    "candidate_bank": "44f64a763473c3a80482aaa3f78669445f56af40a1c00741a351c6c0650e758b",
    "evaluator_contract": "e1e376c9fc4c7e6020d0e18e5c2e061e2a7c53d91bb1c38da751139f4c69a98b",
}
EXPECTED_WOTE_COMMIT = "298957c128a91d41a1c6075bd0bb6e7e845e093f"

_LABEL_INDEX_CACHE: dict[
    str, tuple[Mapping[str, Any], Mapping[str, Any]]
] = {}


class OracleEffectDataError(RuntimeError):
    """A fixed split, cache, candidate, or six-factor label join is invalid."""


def asset_preflight(
    *,
    repo_root: Path,
    wote_root: Path,
    checkpoint: Path,
    candidate_bank: Path,
    evaluator_contract: Path,
    data_root: Path,
    map_root: Path,
) -> Mapping[str, Any]:
    actual = {
        "checkpoint": sha256_file(checkpoint),
        "candidate_bank": sha256_file(candidate_bank),
        "evaluator_contract": sha256_file(evaluator_contract),
    }
    wote_commit = subprocess.run(
        ["git", "-C", str(wote_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    paths = {
        "trainval_logs": data_root / "navsim_logs" / "trainval",
        "trainval_sensors": data_root / "sensor_blobs" / "trainval",
        "maps": map_root,
    }
    hashes_match = actual == EXPECTED_ASSET_HASHES
    paths_exist = all(path.is_dir() for path in paths.values())
    status = hashes_match and paths_exist and wote_commit == EXPECTED_WOTE_COMMIT
    return {
        "status": "PASS" if status else "FAIL",
        "repo_root": str(repo_root),
        "branch": branch,
        "base_or_current_commit": commit,
        "wote_root": str(wote_root),
        "wote_commit": wote_commit,
        "expected_wote_commit": EXPECTED_WOTE_COMMIT,
        "assets": {
            name: {
                "path": str(path),
                "sha256": actual[name],
                "expected_sha256": EXPECTED_ASSET_HASHES[name],
                "match": actual[name] == EXPECTED_ASSET_HASHES[name],
            }
            for name, path in (
                ("checkpoint", checkpoint),
                ("candidate_bank", candidate_bank),
                ("evaluator_contract", evaluator_contract),
            )
        },
        "paths": {
            name: {"path": str(path), "exists": path.is_dir()}
            for name, path in paths.items()
        },
        "published_formatted_pdm_score_used": False,
        "trajectory_offsets": False,
        "candidate_count": 256,
        "label_schema": "independent_wote_labels_4s_six_factor.v2",
    }


@dataclass(frozen=True)
class OracleEffectSplit:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self, g1_tokens: Sequence[str]) -> None:
        expected = {"train": 1024, "val": 256, "test": 512}
        values = {name: tuple(getattr(self, name)) for name in expected}
        for name, count in expected.items():
            if len(values[name]) != count or len(set(values[name])) != count:
                raise OracleEffectDataError(
                    f"{name} split expected {count} unique tokens, got {len(values[name])}"
                )
        sets = {name: set(value) for name, value in values.items()}
        if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
            raise OracleEffectDataError("oracle-effect train/val/test splits overlap")
        if sets["test"] & set(g1_tokens):
            raise OracleEffectDataError("oracle-effect test overlaps the fixed G1 set")


def read_tokens(path: Path) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not values or len(values) != len(set(values)):
        raise OracleEffectDataError(f"token file must be non-empty and unique: {path}")
    return values


def build_fixed_split(split_dir: Path) -> OracleEffectSplit:
    train = read_tokens(split_dir / "train_tokens.txt")
    val = read_tokens(split_dir / "val_tokens.txt")
    test = read_tokens(split_dir / "test_tokens.txt")
    g1 = read_tokens(split_dir / "relabel_headroom_tokens.txt")
    result = OracleEffectSplit(train[:1024], val[:256], test[200:712])
    result.validate(g1)
    if result.train != train[:1024] or result.val != val[:256] or result.test != test[200:712]:
        raise AssertionError("registered split slice changed")
    return result


def write_fixed_split(split_dir: Path, output_dir: Path) -> Mapping[str, Any]:
    split = build_fixed_split(split_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, filename in SPLIT_FILENAMES.items():
        path = output_dir / filename
        expected_text = "".join(f"{token}\n" for token in getattr(split, name))
        if path.exists() and path.read_text(encoding="utf-8") != expected_text:
            raise OracleEffectDataError(f"existing registered split differs: {path}")
        if not path.exists():
            path.write_text(expected_text, encoding="utf-8")
        paths[name] = path
    determinism_tokens = split.train[:8] + split.val[:4] + split.test[:4]
    determinism_path = output_dir / "oracle_effect_v2_determinism16_tokens.txt"
    expected_determinism = "".join(f"{token}\n" for token in determinism_tokens)
    if determinism_path.exists() and determinism_path.read_text(encoding="utf-8") != expected_determinism:
        raise OracleEffectDataError(
            f"existing registered determinism split differs: {determinism_path}"
        )
    if not determinism_path.exists():
        determinism_path.write_text(expected_determinism, encoding="utf-8")
    payload = {
        "schema_version": "oracle_effect_split.v2",
        "selection": {
            "train": "original_train[:1024]",
            "val": "original_val[:256]",
            "test": "original_test[200:712]",
        },
        "g1_overlap": 0,
        "determinism_selection": "train[:8] + val[:4] + test[:4]",
        "determinism": {
            "path": str(determinism_path.relative_to(output_dir.parent.parent.parent.parent)),
            "count": len(determinism_tokens),
            "sha256": sha256_file(determinism_path),
        },
        "splits": {
            name: {
                "path": str(path.relative_to(output_dir.parent.parent.parent.parent)),
                "count": len(getattr(split, name)),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    payload["logical_content_sha256"] = stable_json_hash(payload)
    return payload


def deterministic_candidate_schedule(
    scene_token: str,
    seed: int,
    epoch: int,
    selected_index: int,
    count: int = TRAIN_CANDIDATES,
) -> npt.NDArray[np.int64]:
    """Label-free 64/256 schedule shared by every model type."""

    if not 0 <= selected_index < CANDIDATE_COUNT:
        raise OracleEffectDataError(f"invalid frozen WoTE selected index: {selected_index}")
    if not 1 <= count <= CANDIDATE_COUNT:
        raise ValueError("candidate schedule count must be in [1,256]")
    if count == CANDIDATE_COUNT:
        return np.arange(CANDIDATE_COUNT, dtype=np.int64)
    digest = hashlib.sha256(
        f"candidate:{scene_token}:{seed}:{epoch}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    pool = np.delete(np.arange(CANDIDATE_COUNT, dtype=np.int64), selected_index)
    result = np.concatenate(
        [np.asarray([selected_index], dtype=np.int64), rng.choice(pool, count - 1, replace=False)]
    )
    result.sort()
    if selected_index not in result or len(np.unique(result)) != count:
        raise AssertionError("candidate schedule failed its registered contract")
    return result


def deterministic_pair_schedule(
    scene_token: str,
    seed: int,
    epoch: int,
    candidate_count: int,
    pair_count: int = PAIR_COUNT,
) -> npt.NDArray[np.int64]:
    """Score-independent pair indices into the deterministic candidate subset."""

    if candidate_count < 2 or pair_count <= 0:
        raise ValueError("pair schedule requires at least two candidates and one pair")
    digest = hashlib.sha256(
        f"pair:{scene_token}:{seed}:{epoch}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    left = rng.integers(0, candidate_count, size=pair_count, dtype=np.int64)
    offset = rng.integers(1, candidate_count, size=pair_count, dtype=np.int64)
    right = (left + offset) % candidate_count
    result = np.stack([left, right], axis=-1)
    if np.any(result[:, 0] == result[:, 1]):
        raise AssertionError("pair schedule produced a self-pair")
    return result


@dataclass(frozen=True)
class OracleEffectScene:
    token: str
    frozen: Mapping[str, npt.NDArray[Any]]
    effects: Mapping[str, npt.NDArray[Any]]
    factor_labels: npt.NDArray[np.float32]
    score_labels: npt.NDArray[np.float32]
    raw_progress: npt.NDArray[np.float32]
    oracle_index: int


@dataclass(frozen=True)
class RawProbeBatch:
    tokens: tuple[str, ...]
    trajectory: Tensor
    ego_status: Tensor
    current_bev_tokens: Tensor
    auxiliary_tokens: Tensor
    factor_labels: Tensor
    score_labels: Tensor
    candidate_indices: Tensor
    pair_indices: Tensor
    wote_selected_indices: npt.NDArray[np.int64]


def _feature_keys_for_model(model_type: str) -> tuple[str, ...]:
    keys = {
        "current_bev_tokens",
        "ego_status_feature",
        "trajectory",
        "selected_index",
        "final_rewards",
    }
    if model_type == "wote_full_future":
        keys.update(
            {
                "reward_feature",
                "future_ego_features_by_step",
                "future_bev_pool_by_step",
            }
        )
    elif model_type == "wote_environment_only":
        keys.add("environment_only_future")
    return tuple(sorted(keys))


def _effect_keys_for_model(model_type: str) -> tuple[str, ...]:
    """Decode no score-adjacent or unused effect arrays during probe epochs."""

    if model_type in {
        "trajectory_only",
        "direct_current",
        "wote_full_future",
        "wote_environment_only",
    }:
        return ()
    if model_type == "shared_logged_future":
        return ("shared_actor_mask", "shared_logged_future")
    keys = {
        "primitive_actor_effect",
        "primitive_actor_mask",
        "primitive_ego_effect",
        "primitive_interaction_mask",
        "primitive_map_effect",
    }
    if model_type == "full_engineered_action_effect":
        keys.update({"actor_engineered_effect", "map_engineered_effect"})
    return tuple(sorted(keys))


def validate_label_free_feature_cache(root: Path) -> Mapping[str, Any]:
    reader = FeatureShardReader(root)
    identity = reader.manifest.get("identity", {})
    if identity.get("label_source") != "none":
        raise OracleEffectDataError("frozen feature cache must declare label_source=none")
    first = reader.manifest["shards"][0]
    sidecar = json.loads((root / first["sidecar"]).read_text(encoding="utf-8"))
    keys = set(sidecar["arrays"])
    missing = sorted(REQUIRED_FEATURE_KEYS - keys)
    leakage = sorted(FORBIDDEN_FROZEN_LABEL_KEYS & keys)
    if missing or leakage:
        raise OracleEffectDataError(
            f"frozen feature cache missing={missing}, forbidden={leakage}"
        )
    return {
        "label_source": "none",
        "required_keys_present": True,
        "forbidden_label_keys": leakage,
        "scene_count": int(reader.manifest["scene_count"]),
        "logical_content_sha256": str(reader.manifest["logical_content_sha256"]),
    }


def validate_effect_cache(root: Path) -> Mapping[str, Any]:
    reader = FeatureShardReader(root)
    first = reader.manifest["shards"][0]
    sidecar = json.loads((root / first["sidecar"]).read_text(encoding="utf-8"))
    keys = set(sidecar["arrays"])
    leakage = sorted({key.lower() for key in keys} & FORBIDDEN_EFFECT_KEYS)
    required = {
        "primitive_ego_effect",
        "primitive_map_effect",
        "primitive_actor_effect",
        "primitive_actor_mask",
        "primitive_interaction_mask",
        "map_engineered_effect",
        "actor_engineered_effect",
        "shared_logged_future",
        "shared_actor_mask",
    }
    missing = sorted(required - keys)
    actor_sidecar = root / "actor_selection_audit.json"
    if leakage or missing or not actor_sidecar.is_file():
        raise OracleEffectDataError(
            f"effect cache missing={missing}, forbidden={leakage}, "
            f"actor_sidecar={actor_sidecar.is_file()}"
        )
    actor_audit = json.loads(actor_sidecar.read_text(encoding="utf-8"))
    if actor_audit.get("candidate_independent") is not True or actor_audit.get("score_independent") is not True:
        raise OracleEffectDataError("actor selection sidecar is not label/candidate independent")
    return {
        "required_keys_present": True,
        "forbidden_keys": leakage,
        "scene_count": int(reader.manifest["scene_count"]),
        "logical_content_sha256": str(reader.manifest["logical_content_sha256"]),
        "actor_selection_sha256": str(actor_audit["logical_content_sha256"]),
    }


def _records_match(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return all(
        first.get(key) == second.get(key)
        for key in (
            "scene_token",
            "candidate_indices",
            "trajectory_hash",
            "candidate_bank_hash",
        )
    )


def _validate_frozen_scene(token: str, frozen: Mapping[str, npt.NDArray[Any]]) -> None:
    trajectory = np.asarray(frozen["trajectory"])
    current = np.asarray(frozen["current_bev_tokens"])
    status = np.asarray(frozen["ego_status_feature"])
    if trajectory.shape != (256, 8, 3) or current.shape != (64, 256) or status.shape != (8,):
        raise OracleEffectDataError(
            f"{token}: invalid trajectory/BEV/status shapes "
            f"{trajectory.shape}/{current.shape}/{status.shape}"
        )
    selected = np.asarray(frozen["selected_index"])
    if selected.size != 1 or not 0 <= int(selected.reshape(-1)[0]) < 256:
        raise OracleEffectDataError(f"{token}: invalid selected_index")


def iter_oracle_scenes(
    frozen_root: Path,
    effect_root: Path,
    label_root: Path,
    model_type: str,
) -> Iterator[OracleEffectScene]:
    """Strictly join label-free features, replay effects, and v2 labels by identity."""

    if model_type not in MODEL_VARIANTS:
        raise OracleEffectDataError(f"unknown model type: {model_type}")
    # Full hashes are verified once during preflight.  Re-hashing multi-GB
    # immutable shards on every training epoch would change no scientific
    # contract and would dominate runtime.
    frozen_reader = FeatureShardReader(frozen_root, verify_shard_hashes=False)
    effect_reader = FeatureShardReader(effect_root, verify_shard_hashes=False)
    cache_key = str(label_root.resolve())
    if cache_key not in _LABEL_INDEX_CACHE:
        labels = SixFactorIndependentCandidateLabelStore(label_root)
        _LABEL_INDEX_CACHE[cache_key] = (labels.manifest, labels.scene_index())
    label_manifest, label_index = _LABEL_INDEX_CACHE[cache_key]
    if label_manifest.get("schema_version") != SIX_FACTOR_LABEL_SCHEMA_VERSION:
        raise OracleEffectDataError("label store is not independent six-factor v2")
    if int(frozen_reader.manifest["scene_count"]) != len(label_index):
        raise OracleEffectDataError("feature/label scene counts differ")
    effect_iter = effect_reader.iter_shards(_effect_keys_for_model(model_type))
    count = 0
    for frozen_sidecar, frozen_arrays in frozen_reader.iter_shards(
        _feature_keys_for_model(model_type)
    ):
        try:
            effect_sidecar, effect_arrays = next(effect_iter)
        except StopIteration as error:
            raise OracleEffectDataError("effect cache has fewer shards") from error
        if len(frozen_sidecar["records"]) != len(effect_sidecar["records"]):
            raise OracleEffectDataError("feature/effect shard lengths differ")
        for index, frozen_record in enumerate(frozen_sidecar["records"]):
            effect_record = effect_sidecar["records"][index]
            if not _records_match(frozen_record, effect_record):
                raise OracleEffectDataError("feature/effect record identities differ")
            token = str(frozen_record["scene_token"])
            if token not in label_index:
                raise OracleEffectDataError(f"labels missing {token}")
            label = label_index[token]
            if label.record.candidate_bank_hash != frozen_record.get("candidate_bank_hash"):
                raise OracleEffectDataError(f"{token}: candidate-bank join mismatch")
            if label.record.trajectory_hash != frozen_record.get("trajectory_hash"):
                raise OracleEffectDataError(f"{token}: trajectory join mismatch")
            frozen = {key: value[index] for key, value in frozen_arrays.items()}
            effects = {key: value[index] for key, value in effect_arrays.items()}
            _validate_frozen_scene(token, frozen)
            if stable_array_hash(
                np.asarray(frozen["trajectory"], dtype=np.float32)
            ) != frozen_record.get("trajectory_hash"):
                raise OracleEffectDataError(
                    f"{token}: stored trajectory no longer equals the fixed base anchors"
                )
            factors = np.asarray(label.factors, dtype=np.float32)
            scores = np.asarray(label.score, dtype=np.float32)
            if factors.shape != (256, 6) or scores.shape != (256,):
                raise OracleEffectDataError(f"{token}: invalid six-factor label shapes")
            reconstructed = pdms_from_six_factors(factors)
            error = float(np.max(np.abs(reconstructed - scores.astype(np.float64))))
            if error > 1.0e-6:
                raise OracleEffectDataError(
                    f"{token}: stored six-factor score reconstruction error {error}"
                )
            count += 1
            yield OracleEffectScene(
                token=token,
                frozen=frozen,
                effects=effects,
                factor_labels=factors,
                score_labels=scores,
                raw_progress=np.asarray(label.raw_progress, dtype=np.float32),
                oracle_index=int(label.oracle_index),
            )
    try:
        next(effect_iter)
    except StopIteration:
        pass
    else:
        raise OracleEffectDataError("effect cache has more shards")
    expected = int(frozen_reader.manifest["scene_count"])
    if count != expected:
        raise OracleEffectDataError(f"read {count} scenes, expected {expected}")


def _scene_batch_values(
    scene: OracleEffectScene,
    model_type: str,
    candidate_indices: npt.NDArray[np.int64],
    intervention: str,
) -> tuple[npt.NDArray[Any], ...]:
    effects = (
        scene.effects
        if intervention == "none"
        else intervene_effects(scene.effects, intervention, scene.token)
    )
    packed = EffectTokenPacker().pack(model_type, effects, scene.frozen)
    current = np.asarray(scene.frozen["current_bev_tokens"], dtype=np.float32)
    if not packed.use_current_bev:
        current = np.zeros_like(current)
    return (
        np.asarray(scene.frozen["trajectory"], dtype=np.float32)[candidate_indices],
        np.asarray(scene.frozen["ego_status_feature"], dtype=np.float32),
        current,
        packed.auxiliary_tokens[candidate_indices],
        scene.factor_labels[candidate_indices],
        scene.score_labels[candidate_indices],
        candidate_indices,
        int(np.asarray(scene.frozen["selected_index"]).reshape(-1)[0]),
    )


def _stack_batch(
    rows: Sequence[tuple[str, tuple[npt.NDArray[Any], ...]]],
    seed: int,
    epoch: int,
) -> RawProbeBatch:
    counts = {len(row[1][0]) for row in rows}
    if len(counts) != 1:
        raise OracleEffectDataError("candidate counts differ within a batch")
    count = next(iter(counts))
    pair_indices = np.stack(
        [deterministic_pair_schedule(token, seed, epoch, count) for token, _ in rows]
    )
    return RawProbeBatch(
        tokens=tuple(token for token, _ in rows),
        trajectory=torch.from_numpy(np.stack([value[0] for _, value in rows])),
        ego_status=torch.from_numpy(np.stack([value[1] for _, value in rows])),
        current_bev_tokens=torch.from_numpy(np.stack([value[2] for _, value in rows])),
        auxiliary_tokens=torch.from_numpy(np.stack([value[3] for _, value in rows])),
        factor_labels=torch.from_numpy(np.stack([value[4] for _, value in rows])),
        score_labels=torch.from_numpy(np.stack([value[5] for _, value in rows])),
        candidate_indices=torch.from_numpy(np.stack([value[6] for _, value in rows])),
        pair_indices=torch.from_numpy(pair_indices),
        wote_selected_indices=np.asarray([value[7] for _, value in rows], dtype=np.int64),
    )


def iter_raw_batches(
    *,
    frozen_root: Path,
    effect_root: Path,
    label_root: Path,
    model_type: str,
    batch_scenes: int,
    seed: int,
    epoch: int,
    full_candidates: bool,
    intervention: str = "none",
    scene_limit: int | None = None,
) -> Iterator[RawProbeBatch]:
    if batch_scenes <= 0:
        raise ValueError("batch_scenes must be positive")
    pending: list[tuple[str, tuple[npt.NDArray[Any], ...]]] = []
    for scene_number, scene in enumerate(
        iter_oracle_scenes(frozen_root, effect_root, label_root, model_type)
    ):
        if scene_limit is not None and scene_number >= scene_limit:
            break
        selected = int(np.asarray(scene.frozen["selected_index"]).reshape(-1)[0])
        indices = (
            np.arange(256, dtype=np.int64)
            if full_candidates
            else deterministic_candidate_schedule(scene.token, seed, epoch, selected)
        )
        pending.append(
            (
                scene.token,
                _scene_batch_values(scene, model_type, indices, intervention),
            )
        )
        if len(pending) == batch_scenes:
            yield _stack_batch(pending, seed, epoch)
            pending.clear()
    if pending:
        yield _stack_batch(pending, seed, epoch)


def compare_deterministic_stores(first: Path, second: Path, kind: str) -> Mapping[str, Any]:
    if kind == "labels":
        first_store = SixFactorIndependentCandidateLabelStore(first)
        second_store = SixFactorIndependentCandidateLabelStore(second)
        first_index = first_store.scene_index()
        second_index = second_store.scene_index()
        if tuple(first_store.scene_tokens) != tuple(second_store.scene_tokens):
            raise OracleEffectDataError("determinism label token order changed")
        mismatches: list[str] = []
        for token in first_store.scene_tokens:
            a, b = first_index[token], second_index[token]
            if not (
                np.array_equal(a.factors, b.factors)
                and np.array_equal(a.score, b.score)
                and np.array_equal(a.raw_progress, b.raw_progress)
                and a.oracle_index == b.oracle_index
            ):
                mismatches.append(token)
        logical_equal = first_store.logical_content_sha256 == second_store.logical_content_sha256
    elif kind == "effects":
        first_reader = FeatureShardReader(first)
        second_reader = FeatureShardReader(second)
        logical_equal = (
            first_reader.manifest["logical_content_sha256"]
            == second_reader.manifest["logical_content_sha256"]
        )
        mismatches = []
        first_shards = list(first_reader.iter_shards())
        second_shards = list(second_reader.iter_shards())
        if len(first_shards) != len(second_shards):
            mismatches.append("shard_count")
        for shard_index, ((first_sidecar, first_arrays), (second_sidecar, second_arrays)) in enumerate(
            zip(first_shards, second_shards)
        ):
            if first_sidecar["records"] != second_sidecar["records"]:
                mismatches.append(f"shard-{shard_index}:records")
            if set(first_arrays) != set(second_arrays):
                mismatches.append(f"shard-{shard_index}:array_keys")
                continue
            for name in first_arrays:
                if not np.array_equal(first_arrays[name], second_arrays[name]):
                    mismatches.append(f"shard-{shard_index}:{name}")
        first_actor = json.loads((first / "actor_selection_audit.json").read_text(encoding="utf-8"))
        second_actor = json.loads((second / "actor_selection_audit.json").read_text(encoding="utf-8"))
        if first_actor != second_actor:
            mismatches.append("actor_selection_audit")
    else:
        raise ValueError("kind must be labels or effects")
    return {
        "kind": kind,
        "exact_tensor_match": not mismatches,
        "logical_sha256_match": logical_equal,
        "mismatches": mismatches,
        "status": "PASS" if not mismatches and logical_equal else "FAIL",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--repo-root", type=Path, required=True)
    preflight.add_argument("--wote-root", type=Path, required=True)
    preflight.add_argument("--checkpoint", type=Path, required=True)
    preflight.add_argument("--candidate-bank", type=Path, required=True)
    preflight.add_argument("--evaluator-contract", type=Path, required=True)
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument("--map-root", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    split = commands.add_parser("build-splits")
    split.add_argument("--split-dir", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--manifest", type=Path, required=True)
    audit = commands.add_parser("compare-determinism")
    audit.add_argument("--first", type=Path, required=True)
    audit.add_argument("--second", type=Path, required=True)
    audit.add_argument("--kind", choices=("labels", "effects"), required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        payload = asset_preflight(
            repo_root=args.repo_root,
            wote_root=args.wote_root,
            checkpoint=args.checkpoint,
            candidate_bank=args.candidate_bank,
            evaluator_contract=args.evaluator_contract,
            data_root=args.data_root,
            map_root=args.map_root,
        )
        atomic_write_json(args.output, payload)
        if payload["status"] != "PASS":
            return 4
    elif args.command == "build-splits":
        payload = write_fixed_split(args.split_dir, args.output_dir)
        atomic_write_json(args.manifest, payload)
    elif args.command == "compare-determinism":
        payload = compare_deterministic_stores(args.first, args.second, args.kind)
        atomic_write_json(args.output, payload)
        if payload["status"] != "PASS":
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
