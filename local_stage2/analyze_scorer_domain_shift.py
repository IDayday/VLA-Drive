"""Audit why Navtrain scorer gains fail to transfer to complete Navtest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from local_stage2.train_independent_scorer import physical_log_name


TARGET_FACTOR_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)
SOURCE_FACTOR_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)
SOURCE_TO_TARGET = {
    "no_at_fault_collisions": 0,
    "drivable_area_compliance": 1,
    "driving_direction_compliance": 5,
    "time_to_collision_within_bound": 3,
    "ego_progress": 2,
    "comfort": 4,
}
TOP_K_VALUES = (2, 4, 8, 16)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    output = np.empty_like(value, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def _safe_mean(value: np.ndarray) -> float:
    return float(np.mean(value, dtype=np.float64)) if value.size else float("nan")


def _quantiles(value: np.ndarray) -> Dict[str, float]:
    points = np.quantile(value.astype(np.float64), [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "p05": float(points[0]),
        "p25": float(points[1]),
        "p50": float(points[2]),
        "p75": float(points[3]),
        "p95": float(points[4]),
    }


def _feature_vector(
    scene_features: np.ndarray,
    ego_features: np.ndarray,
    base_scores: np.ndarray,
    factor_logits: np.ndarray,
) -> np.ndarray:
    selected = base_scores.argmax(axis=1)
    row = np.arange(len(selected))
    ordered = np.sort(base_scores.astype(np.float32), axis=1)
    score_statistics = np.stack(
        (
            base_scores.mean(axis=1),
            base_scores.std(axis=1),
            ordered[:, -1],
            ordered[:, -1] - ordered[:, -2],
            ordered[:, 0],
        ),
        axis=1,
    )
    return np.concatenate(
        (
            scene_features.astype(np.float32).mean(axis=1),
            ego_features.astype(np.float32).reshape(len(selected), -1),
            score_statistics.astype(np.float32),
            factor_logits[row, selected].astype(np.float32),
        ),
        axis=1,
    )


@dataclass
class SplitAccumulator:
    name: str
    tokens: List[str] = field(default_factory=list)
    log_names: List[str] = field(default_factory=list)
    values: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    selected_factors: List[np.ndarray] = field(default_factory=list)
    oracle_factors: List[np.ndarray] = field(default_factory=list)
    all_factor_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(len(TARGET_FACTOR_NAMES), dtype=np.float64)
    )
    all_factor_count: int = 0
    factor_brier_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(len(SOURCE_FACTOR_NAMES), dtype=np.float64)
    )
    factor_logloss_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(len(SOURCE_FACTOR_NAMES), dtype=np.float64)
    )
    factor_accuracy_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(len(SOURCE_FACTOR_NAMES), dtype=np.float64)
    )
    factor_target_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(len(SOURCE_FACTOR_NAMES), dtype=np.float64)
    )
    factor_prediction_count: int = 0
    pair_correct: int = 0
    pair_total: int = 0
    representations: List[np.ndarray] = field(default_factory=list)

    def _append(self, key: str, value: np.ndarray) -> None:
        self.values.setdefault(key, []).append(np.asarray(value))

    def update(
        self,
        *,
        tokens: Sequence[str],
        log_names: Sequence[str],
        base_scores: np.ndarray,
        factor_logits: np.ndarray,
        target_factors: np.ndarray,
        scene_features: np.ndarray,
        ego_features: np.ndarray,
    ) -> None:
        if not len(tokens):
            return
        if target_factors.shape[1:] != (64, 7):
            raise ValueError(f"Unexpected target shape: {target_factors.shape}")
        scores = target_factors[..., -1]
        order = np.argsort(-base_scores, axis=1, kind="stable")
        row = np.arange(len(tokens))
        base_index = order[:, 0]
        oracle_index = scores.argmax(axis=1)
        base_values = scores[row, base_index]
        oracle_values = scores[row, oracle_index]
        self.tokens.extend(str(value) for value in tokens)
        self.log_names.extend(str(value) for value in log_names)
        self._append("base_selected_pdms", base_values)
        self._append("oracle_pdms", oracle_values)
        self._append("regret", oracle_values - base_values)
        self._append("candidate_pdms_mean", scores.mean(axis=1))
        self._append("candidate_pdms_std", scores.std(axis=1))
        sorted_base = np.take_along_axis(base_scores, order, axis=1)
        self._append("base_score_margin", sorted_base[:, 0] - sorted_base[:, 1])
        self._append("base_oracle_hit", (base_index == oracle_index).astype(np.float32))
        for top_k in TOP_K_VALUES:
            ordered_scores = np.take_along_axis(scores, order[:, :top_k], axis=1)
            self._append(f"best_of_base_top{top_k}", ordered_scores.max(axis=1))

        self.selected_factors.append(target_factors[row, base_index])
        self.oracle_factors.append(target_factors[row, oracle_index])
        self.all_factor_sums += target_factors.sum(axis=(0, 1), dtype=np.float64)
        self.all_factor_count += int(np.prod(target_factors.shape[:2]))

        top_prediction = np.take_along_axis(base_scores, order[:, :16], axis=1)
        top_target = np.take_along_axis(scores, order[:, :16], axis=1)
        left, right = np.triu_indices(16, k=1)
        target_delta = top_target[:, left] - top_target[:, right]
        valid = np.abs(target_delta) >= 0.02
        prediction_delta = top_prediction[:, left] - top_prediction[:, right]
        self.pair_correct += int(
            ((np.sign(prediction_delta) == np.sign(target_delta)) & valid).sum()
        )
        self.pair_total += int(valid.sum())

        probabilities = _sigmoid(factor_logits.astype(np.float64))
        for source_index, name in enumerate(SOURCE_FACTOR_NAMES):
            target_index = SOURCE_TO_TARGET[name]
            target = target_factors[..., target_index].astype(np.float64)
            prediction = probabilities[..., source_index]
            if name != "ego_progress":
                target = (target >= 1.0 - 1e-6).astype(np.float64)
            clipped = np.clip(prediction, 1e-7, 1.0 - 1e-7)
            self.factor_brier_sums[source_index] += np.square(prediction - target).sum()
            self.factor_logloss_sums[source_index] += (
                -target * np.log(clipped) - (1.0 - target) * np.log(1.0 - clipped)
            ).sum()
            self.factor_accuracy_sums[source_index] += (
                (prediction >= 0.5) == (target >= 0.5)
            ).sum()
            self.factor_target_sums[source_index] += target.sum()
        self.factor_prediction_count += int(np.prod(target_factors.shape[:2]))
        self.representations.append(
            _feature_vector(scene_features, ego_features, base_scores, factor_logits)
        )

    def finalize(self) -> Tuple[Dict[str, object], np.ndarray]:
        values = {key: np.concatenate(parts) for key, parts in self.values.items()}
        selected_factors = np.concatenate(self.selected_factors)
        oracle_factors = np.concatenate(self.oracle_factors)
        log_names = np.asarray(self.log_names)
        unique_logs = np.unique(log_names)
        per_log_regret = np.asarray(
            [_safe_mean(values["regret"][log_names == name]) for name in unique_logs]
        )
        factor_prediction = {}
        for index, name in enumerate(SOURCE_FACTOR_NAMES):
            count = max(self.factor_prediction_count, 1)
            factor_prediction[name] = {
                "target_mean": float(self.factor_target_sums[index] / count),
                "brier": float(self.factor_brier_sums[index] / count),
                "log_loss": float(self.factor_logloss_sums[index] / count),
                "threshold_accuracy": float(self.factor_accuracy_sums[index] / count),
            }
        report = {
            "scene_count": len(self.tokens),
            "log_count": len(unique_logs),
            "base_selected_pdms": _safe_mean(values["base_selected_pdms"]),
            "best_of_64_pdms": _safe_mean(values["oracle_pdms"]),
            "base_scorer_regret": _safe_mean(values["regret"]),
            "candidate_pdms_mean": _safe_mean(values["candidate_pdms_mean"]),
            "candidate_pdms_std_mean": _safe_mean(values["candidate_pdms_std"]),
            "base_score_margin_mean": _safe_mean(values["base_score_margin"]),
            "base_oracle_hit_rate": _safe_mean(values["base_oracle_hit"]),
            "top16_pairwise_accuracy_delta_ge_0_02": float(
                self.pair_correct / max(self.pair_total, 1)
            ),
            "top16_pair_count_delta_ge_0_02": self.pair_total,
            "per_log_regret": {
                "mean": _safe_mean(per_log_regret),
                **_quantiles(per_log_regret),
            },
            "selected_factor_means": {
                name: _safe_mean(selected_factors[:, index])
                for index, name in enumerate(TARGET_FACTOR_NAMES)
            },
            "oracle_factor_means": {
                name: _safe_mean(oracle_factors[:, index])
                for index, name in enumerate(TARGET_FACTOR_NAMES)
            },
            "all_candidate_factor_means": {
                name: float(self.all_factor_sums[index] / max(self.all_factor_count, 1))
                for index, name in enumerate(TARGET_FACTOR_NAMES)
            },
            "factor_prediction": factor_prediction,
        }
        for top_k in TOP_K_VALUES:
            report[f"best_of_base_top{top_k}_pdms"] = _safe_mean(
                values[f"best_of_base_top{top_k}"]
            )
        return report, np.concatenate(self.representations)


def _load_log_split(path: Path) -> Tuple[set[str], set[str]]:
    config = OmegaConf.load(path)
    # Resolved Hydra training configs use ``train_logs``/``val_logs`` while
    # the locked scorer split manifest deliberately names the physical-log
    # boundary explicitly.  Accept both schemas, but never guess a split from
    # scene ordering.
    train_values = config.get("train_physical_logs", config.get("train_logs"))
    val_values = config.get(
        "validation_physical_logs", config.get("val_logs")
    )
    if train_values is None or val_values is None:
        raise ValueError(
            "log split must define train_physical_logs/validation_physical_logs "
            "or train_logs/val_logs"
        )
    train_logs = {str(value) for value in train_values}
    val_logs = {str(value) for value in val_values}
    if not train_logs or not val_logs:
        raise ValueError("log split train and validation sets must both be non-empty")
    if train_logs & val_logs:
        raise RuntimeError("Navtrain train/validation log sets overlap")
    return train_logs, val_logs


def _update_navtrain(
    source_root: Path,
    label_root: Path,
    train_logs: set[str],
    val_logs: set[str],
) -> Dict[str, SplitAccumulator]:
    accumulators = {
        "navtrain_train": SplitAccumulator("navtrain_train"),
        "navtrain_validation": SplitAccumulator("navtrain_validation"),
    }
    source_paths = sorted(source_root.glob("*_shard_*-of-*/chunk_*.pt"))
    if not source_paths:
        raise RuntimeError(f"No source chunks found under {source_root}")
    seen_scene_count = 0
    assigned_scene_count = 0
    available_physical_logs: set[str] = set()
    for chunk_index, source_path in enumerate(source_paths, start=1):
        label_path = label_root / source_path.relative_to(source_root)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        source = torch.load(source_path, map_location="cpu")
        labels = torch.load(label_path, map_location="cpu")
        if list(source["tokens"]) != list(labels["tokens"]):
            raise RuntimeError(f"Source/label token mismatch: {source_path}")
        if tuple(source["factor_keys"]) != SOURCE_FACTOR_NAMES:
            raise RuntimeError(f"Unexpected source factors: {source_path}")
        if tuple(labels["target_factor_keys"]) != TARGET_FACTOR_NAMES:
            raise RuntimeError(f"Unexpected label factors: {label_path}")
        # Feature caches retain NAVSIM segment-directory names.  Split
        # manifests are intentionally defined at the physical-log boundary;
        # use the same audited normalization as scorer training.
        log_names = np.asarray(
            [physical_log_name(value) for value in source["log_names"]]
        ).astype(str)
        seen_scene_count += len(log_names)
        available_physical_logs.update(log_names.tolist())
        for split_name, allowed_logs in (
            ("navtrain_train", train_logs),
            ("navtrain_validation", val_logs),
        ):
            mask = np.asarray([name in allowed_logs for name in log_names])
            if not mask.any():
                continue
            indices = np.flatnonzero(mask)
            assigned_scene_count += len(indices)
            accumulators[split_name].update(
                tokens=[source["tokens"][index] for index in indices],
                log_names=log_names[indices],
                base_scores=source["base_scores"][mask].float().numpy(),
                factor_logits=source["factor_logits"][mask].float().numpy(),
                target_factors=labels["target_factors"][mask].float().numpy(),
                scene_features=source["scene_features"][mask].float().numpy(),
                ego_features=source["ego_features"][mask].float().numpy(),
            )
        if chunk_index % 50 == 0 or chunk_index == len(source_paths):
            print(
                f"NAVTRAIN_DOMAIN_AUDIT chunks={chunk_index}/{len(source_paths)}",
                flush=True,
            )
    if assigned_scene_count != seen_scene_count:
        missing_logs = sorted(
            available_physical_logs.difference(train_logs | val_logs)
        )
        raise RuntimeError(
            "Navtrain split does not cover every cached scene: "
            f"assigned={assigned_scene_count}, total={seen_scene_count}, "
            f"missing_physical_logs={missing_logs[:10]}"
        )
    for split_name, accumulator in accumulators.items():
        if not accumulator.tokens:
            raise RuntimeError(f"Navtrain split is empty after log matching: {split_name}")
    return accumulators


def _update_navtest(
    feature_cache_path: Path,
    candidate_matrix_path: Path,
) -> SplitAccumulator:
    with feature_cache_path.open("rb") as file:
        cache = pickle.load(file)
    with np.load(candidate_matrix_path, allow_pickle=False) as archive:
        matrix = {key: archive[key] for key in archive.files}
    tokens = matrix["tokens"].astype(str)
    if set(tokens) != set(cache):
        raise RuntimeError("Navtest feature/matrix token sets differ")
    factor_names = tuple(matrix["candidate_factor_names"].astype(str))
    if factor_names != TARGET_FACTOR_NAMES:
        raise RuntimeError(f"Unexpected Navtest factors: {factor_names}")
    accumulator = SplitAccumulator("navtest")
    batch_size = 256
    for start in range(0, len(tokens), batch_size):
        batch_tokens = tokens[start : start + batch_size]
        base_scores = np.stack([cache[token]["predicted_scores"] for token in batch_tokens])
        cached_scores = matrix["predicted_scores"][start : start + len(batch_tokens)]
        if float(np.max(np.abs(base_scores.astype(np.float64) - cached_scores))) > 1e-8:
            raise RuntimeError("Navtest feature/matrix Base scores differ")
        accumulator.update(
            tokens=batch_tokens,
            log_names=matrix["log_names"][start : start + len(batch_tokens)].astype(str),
            base_scores=base_scores.astype(np.float32),
            factor_logits=np.stack(
                [cache[token]["base_factor_logits"] for token in batch_tokens]
            ).astype(np.float32),
            target_factors=matrix["candidate_factors"][
                start : start + len(batch_tokens)
            ].astype(np.float32),
            scene_features=np.stack(
                [cache[token]["scene_features"] for token in batch_tokens]
            ).astype(np.float32),
            ego_features=np.stack(
                [cache[token]["ego_features"] for token in batch_tokens]
            ).astype(np.float32),
        )
    return accumulator


def _stable_sample_indices(tokens: Sequence[str], limit: int) -> np.ndarray:
    if len(tokens) <= limit:
        return np.arange(len(tokens))
    hashes = np.fromiter(
        (
            int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
            for token in tokens
        ),
        dtype=np.uint64,
        count=len(tokens),
    )
    return np.argpartition(hashes, limit)[:limit]


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive = labels == 1
    positive_count = int(positive.sum())
    negative_count = len(labels) - positive_count
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2)
        / max(positive_count * negative_count, 1)
    )


def _domain_probe(
    name_a: str,
    accumulator_a: SplitAccumulator,
    representation_a: np.ndarray,
    name_b: str,
    accumulator_b: SplitAccumulator,
    representation_b: np.ndarray,
    *,
    seed: int,
    max_samples_per_domain: int,
) -> Dict[str, object]:
    index_a = _stable_sample_indices(accumulator_a.tokens, max_samples_per_domain)
    index_b = _stable_sample_indices(accumulator_b.tokens, max_samples_per_domain)
    features = np.concatenate((representation_a[index_a], representation_b[index_b]))
    labels = np.concatenate(
        (np.zeros(len(index_a), dtype=np.float32), np.ones(len(index_b), dtype=np.float32))
    )
    groups = np.concatenate(
        (
            np.asarray(accumulator_a.log_names)[index_a],
            np.asarray(accumulator_b.log_names)[index_b],
        )
    )
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(labels), dtype=bool)
    for domain_value in (0.0, 1.0):
        domain_indices = np.flatnonzero(labels == domain_value)
        domain_logs = np.unique(groups[domain_indices])
        rng.shuffle(domain_logs)
        train_logs = set(domain_logs[: max(1, int(0.8 * len(domain_logs)))])
        train_mask[domain_indices] = np.asarray(
            [groups[index] in train_logs for index in domain_indices]
        )
    test_mask = ~train_mask
    mean = features[train_mask].mean(axis=0, dtype=np.float64)
    std = features[train_mask].std(axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-5)
    normalized = ((features - mean) / std).astype(np.float32)
    train_x = torch.from_numpy(normalized[train_mask])
    train_y = torch.from_numpy(labels[train_mask, None])
    test_x = torch.from_numpy(normalized[test_mask])
    test_y = labels[test_mask]
    torch.manual_seed(seed)
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    model = torch.nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(30):
        permutation = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(train_x), 1024):
            batch = permutation[start : start + 1024]
            logits = model(train_x[batch])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, train_y[batch]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        test_scores = model(test_x).squeeze(1).numpy()
    return {
        "domain_a": name_a,
        "domain_b": name_b,
        "feature_dimension": int(features.shape[1]),
        "train_scene_count": int(train_mask.sum()),
        "test_scene_count": int(test_mask.sum()),
        "train_log_count": int(len(np.unique(groups[train_mask]))),
        "test_log_count": int(len(np.unique(groups[test_mask]))),
        "test_auroc": _auc(test_y, test_scores),
        "test_accuracy": float(np.mean((test_scores >= 0.0) == (test_y >= 0.5))),
        "positive_rate": float(test_y.mean()),
    }


def _markdown(payload: Mapping[str, object]) -> str:
    splits = payload["splits"]
    lines = [
        "# Scorer Navtrain-to-Navtest Domain-Shift Audit",
        "",
        "This report is diagnostic only. Navtest factors are not used to tune or train a scorer.",
        "",
        "## Candidate-bank and Base-scorer difficulty",
        "",
        "| Split | Scenes | Logs | Base PDMS | Best-64 | Regret | Top-16 pairwise | Base oracle hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("navtrain_train", "navtrain_validation", "navtest"):
        value = splits[name]
        lines.append(
            f"| {name} | {value['scene_count']} | {value['log_count']} | "
            f"{value['base_selected_pdms']:.6f} | {value['best_of_64_pdms']:.6f} | "
            f"{value['base_scorer_regret']:.6f} | "
            f"{value['top16_pairwise_accuracy_delta_ge_0_02']:.4f} | "
            f"{value['base_oracle_hit_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Current-feature domain probes",
            "",
            "A linear classifier is trained on current-observation scene/ego/scorer features and evaluated on held-out logs. AUROC near 0.5 means the domains are hard to distinguish; high AUROC indicates representation shift.",
            "",
            "| Domains | Held-out-log AUROC | Accuracy | Train scenes | Test scenes |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for probe in payload["domain_probes"]:
        lines.append(
            f"| {probe['domain_a']} vs {probe['domain_b']} | {probe['test_auroc']:.4f} | "
            f"{probe['test_accuracy']:.4f} | {probe['train_scene_count']} | "
            f"{probe['test_scene_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--log-split", type=Path, required=True)
    parser.add_argument("--navtest-feature-cache", type=Path, required=True)
    parser.add_argument("--navtest-candidate-matrix", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-domain-samples", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_logs, val_logs = _load_log_split(args.log_split)
    accumulators = _update_navtrain(
        args.source_root, args.label_root, train_logs, val_logs
    )
    accumulators["navtest"] = _update_navtest(
        args.navtest_feature_cache, args.navtest_candidate_matrix
    )
    split_reports = {}
    representations = {}
    for name, accumulator in accumulators.items():
        split_reports[name], representations[name] = accumulator.finalize()
    probes = []
    for name_a, name_b in (
        ("navtrain_train", "navtrain_validation"),
        ("navtrain_train", "navtest"),
        ("navtrain_validation", "navtest"),
    ):
        probes.append(
            _domain_probe(
                name_a,
                accumulators[name_a],
                representations[name_a],
                name_b,
                accumulators[name_b],
                representations[name_b],
                seed=args.seed,
                max_samples_per_domain=args.max_domain_samples,
            )
        )
    validation = split_reports["navtrain_validation"]
    navtest = split_reports["navtest"]
    regret_ratio = float(
        navtest["base_scorer_regret"] / max(validation["base_scorer_regret"], 1e-12)
    )
    validation_test_probe = next(
        probe
        for probe in probes
        if probe["domain_a"] == "navtrain_validation"
        and probe["domain_b"] == "navtest"
    )
    interpretation = (
        "Navtest Base scorer regret is "
        f"{regret_ratio:.2f}x the Navtrain-validation regret, while the best-of-64 "
        "ceiling remains high. "
        f"The held-out-log current-feature domain AUROC is {validation_test_probe['test_auroc']:.3f}. "
        "This separates candidate-generation headroom from scorer-domain generalization: "
        "new rankers must be selected by multi-domain/worst-fold Navtrain criteria rather "
        "than a single official validation split."
    )
    payload = {
        "schema_version": 1,
        "future_inputs_used_by_domain_probe": False,
        "navtest_used_for_training_or_hyperparameter_selection": False,
        "splits": split_reports,
        "domain_probes": probes,
        "navtrain_validation_to_navtest_regret_ratio": regret_ratio,
        "interpretation": interpretation,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_name(f".{args.output_json.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_json)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    temporary_md = args.output_md.with_name(f".{args.output_md.name}.tmp")
    temporary_md.write_text(_markdown(payload))
    temporary_md.replace(args.output_md)
    print(json.dumps({
        "splits": {
            name: {
                key: report[key]
                for key in (
                    "scene_count",
                    "log_count",
                    "base_selected_pdms",
                    "best_of_64_pdms",
                    "base_scorer_regret",
                    "top16_pairwise_accuracy_delta_ge_0_02",
                )
            }
            for name, report in split_reports.items()
        },
        "domain_probes": probes,
        "regret_ratio": regret_ratio,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
