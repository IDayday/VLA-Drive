#!/usr/bin/env python3
"""Task-level audit of an immutable, officially scored 64-candidate bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


PREDICTED_COMPONENTS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "comfort",
)
OFFICIAL_COMPONENTS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "pdm_score",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = target.astype(bool).reshape(-1)
    score = score.reshape(-1)
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(score, method="average")
    auc = (ranks[target].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def _average_precision(target: np.ndarray, score: np.ndarray) -> float | None:
    target = target.astype(bool).reshape(-1)
    score = score.reshape(-1)
    positives = int(target.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="stable")
    sorted_target = target[order]
    precision = np.cumsum(sorted_target) / np.arange(1, len(target) + 1)
    return float(precision[sorted_target].sum() / positives)


def _ece(target: np.ndarray, score: np.ndarray, bins: int = 15) -> Tuple[float, list]:
    target = target.reshape(-1).astype(np.float64)
    score = np.clip(score.reshape(-1).astype(np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(score, edges[1:-1]), bins - 1)
    ece = 0.0
    rows = []
    for index in range(bins):
        mask = indices == index
        if not mask.any():
            continue
        predicted = float(score[mask].mean())
        observed = float(target[mask].mean())
        fraction = float(mask.mean())
        ece += fraction * abs(predicted - observed)
        rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(mask.sum()),
                "predicted_mean": predicted,
                "observed_mean": observed,
            }
        )
    return float(ece), rows


def _row_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_rank = rankdata(left, axis=1, method="average")
    right_rank = rankdata(right, axis=1, method="average")
    left_rank -= left_rank.mean(axis=1, keepdims=True)
    right_rank -= right_rank.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left_rank, axis=1) * np.linalg.norm(right_rank, axis=1)
    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = (left_rank[valid] * right_rank[valid]).sum(axis=1) / denominator[valid]
    return result


def _binary_metrics(target: np.ndarray, score: np.ndarray) -> dict:
    target = target.astype(np.float64)
    label = target >= 0.5
    predicted = score >= 0.5
    positive = label.sum()
    negative = (~label).sum()
    tp = np.logical_and(predicted, label).sum()
    tn = np.logical_and(~predicted, ~label).sum()
    fp = np.logical_and(predicted, ~label).sum()
    fn = np.logical_and(~predicted, label).sum()
    ece, calibration = _ece(target, score)
    return {
        "count": int(target.size),
        "positive_rate": float(label.mean()),
        "predicted_positive_rate": float(predicted.mean()),
        "auroc": _binary_auc(label, score),
        "average_precision": _average_precision(label, score),
        "accuracy_at_0p5": float((predicted == label).mean()),
        "balanced_accuracy_at_0p5": float(
            0.5 * (tp / max(positive, 1) + tn / max(negative, 1))
        ),
        "precision_at_0p5": float(tp / max(tp + fp, 1)),
        "recall_at_0p5": float(tp / max(tp + fn, 1)),
        "brier": float(np.mean(np.square(score - target))),
        "ece_15_bin": ece,
        "calibration": calibration,
    }


def _continuous_metrics(target: np.ndarray, score: np.ndarray) -> dict:
    target_flat = target.reshape(-1).astype(np.float64)
    score_flat = score.reshape(-1).astype(np.float64)
    correlation = spearmanr(target_flat, score_flat).statistic
    pearson = np.corrcoef(target_flat, score_flat)[0, 1]
    ece, calibration = _ece(target_flat, score_flat)
    return {
        "count": int(target_flat.size),
        "target_mean": float(target_flat.mean()),
        "prediction_mean": float(score_flat.mean()),
        "mae": float(np.mean(np.abs(target_flat - score_flat))),
        "rmse": float(np.sqrt(np.mean(np.square(target_flat - score_flat)))),
        "spearman": _safe_float(correlation),
        "pearson": _safe_float(pearson),
        "ece_15_bin": ece,
        "calibration": calibration,
    }


def _cluster_bootstrap(
    values: Dict[str, np.ndarray], groups: np.ndarray, *, samples: int, seed: int
) -> dict:
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = {name: np.bincount(inverse, weights=array) for name, array in values.items()}
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(samples, dtype=np.float64) for name in values}
    for draw in range(samples):
        chosen = rng.integers(0, len(unique), size=len(unique))
        denominator = counts[chosen].sum()
        for name, group_sums in sums.items():
            draws[name][draw] = group_sums[chosen].sum() / denominator
    return {
        name: {
            "mean": float(array.mean()),
            "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        }
        for name, array in draws.items()
    }


def _metric_cache_logs(tokens: Iterable[str], metric_cache: Path) -> np.ndarray:
    from navsim.common.dataloader import MetricCacheLoader

    loader = MetricCacheLoader(metric_cache)
    logs = []
    for token in tokens:
        path = loader.metric_cache_paths.get(str(token))
        if path is None:
            raise KeyError(f"Metric cache does not contain token {token}")
        relative = Path(path).relative_to(metric_cache)
        logs.append(relative.parts[0])
    return np.asarray(logs)


def _validate(payload) -> None:
    required = {
        "tokens",
        "proposals",
        "predicted_pdms",
        "selected_indices",
        "component_probabilities",
        "component_names",
        "candidate_scores",
        "official_component_names",
    }
    missing = required - set(payload.files)
    if missing:
        raise KeyError(f"Scored candidate bank is missing: {sorted(missing)}")
    scenes = len(payload["tokens"])
    if scenes != 12146:
        raise RuntimeError(f"Formal Navtest requires 12,146 scenes, found {scenes}")
    if payload["proposals"].shape != (scenes, 64, 8, 3):
        raise RuntimeError(f"Expected proposals [12146,64,8,3], got {payload['proposals'].shape}")
    if tuple(payload["component_names"].tolist()) != PREDICTED_COMPONENTS:
        raise RuntimeError("Predicted component ordering differs from the audited contract")
    if tuple(payload["official_component_names"].tolist()) != OFFICIAL_COMPONENTS:
        raise RuntimeError("Official component ordering differs from the audited contract")


def _plot(report_arrays: dict, output_dir: Path) -> None:
    official = report_arrays["official"]
    predicted = report_arrays["predicted"]
    selected = report_arrays["selected"]
    oracle = report_arrays["oracle"]
    selected_rank = report_arrays["selected_rank"]
    selected_indices = report_arrays["selected_indices"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    subsample = np.random.default_rng(0).choice(official.size, size=min(120000, official.size), replace=False)
    axes[0, 0].hexbin(predicted.reshape(-1)[subsample], official.reshape(-1)[subsample], gridsize=55, bins="log", mincnt=1)
    axes[0, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0, 0].set(xlabel="scorer predicted PDMS", ylabel="official PDMS", title="Scorer calibration (sampled candidates)")

    axes[0, 1].hist(selected_rank, bins=np.arange(0.5, 65.6, 1), color="#4472c4")
    axes[0, 1].set(xlabel="official rank of selected trajectory (1=best)", ylabel="scenes", title="Selection rank")

    axes[0, 2].hist(oracle - selected, bins=50, color="#c44e52")
    axes[0, 2].set(xlabel="offline Oracle@64 - selected PDMS", ylabel="scenes", title="Scorer regret")

    axes[1, 0].hist(official.reshape(-1), bins=50, alpha=0.45, label="all candidates")
    axes[1, 0].hist(selected, bins=50, alpha=0.55, label="selected")
    axes[1, 0].hist(oracle, bins=50, histtype="step", linewidth=2, label="offline Oracle@64")
    axes[1, 0].set(xlabel="official PDMS", ylabel="count", title="Proposal and selection quality")
    axes[1, 0].legend()

    counts = np.bincount(selected_indices, minlength=64)
    axes[1, 1].bar(np.arange(64), counts)
    axes[1, 1].set(xlabel="trajectory-query index", ylabel="times selected", title="Query usage")

    scene_mean = official.mean(axis=1)
    axes[1, 2].scatter(scene_mean, oracle - selected, s=2, alpha=0.25)
    axes[1, 2].set(xlabel="scene candidate-mean PDMS", ylabel="scorer regret", title="Regret vs scene difficulty")
    for axis in axes.reshape(-1):
        axis.grid(alpha=0.2)
    fig.savefig(output_dir / "scorer_task_audit.png", dpi=180)
    plt.close(fig)


def audit(path: Path, metric_cache: Path, output_dir: Path, bootstrap_samples: int) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        _validate(payload)
        tokens = np.asarray(payload["tokens"])
        proposals = np.asarray(payload["proposals"], dtype=np.float32)
        predicted_pdms = np.asarray(payload["predicted_pdms"], dtype=np.float64)
        selected_indices = np.asarray(payload["selected_indices"], dtype=np.int64)
        predicted_components = np.asarray(payload["component_probabilities"], dtype=np.float64)
        official_components = np.asarray(payload["candidate_scores"], dtype=np.float64)
        registers = np.asarray(payload["planning_registers"], dtype=np.float64) if "planning_registers" in payload else None
        tile_gate = np.asarray(payload["tile_gate"], dtype=np.float64) if "tile_gate" in payload else None
        semantic_gate = np.asarray(payload["semantic_gate"], dtype=np.float64) if "semantic_gate" in payload else None

    if not all(np.isfinite(array).all() for array in (proposals, predicted_pdms, predicted_components, official_components)):
        raise RuntimeError("Candidate bank contains non-finite values")
    scenes, candidates = predicted_pdms.shape
    official_pdms = official_components[..., -1]
    rows = np.arange(scenes)
    selected = official_pdms[rows, selected_indices]
    oracle_indices = official_pdms.argmax(axis=1)
    oracle = official_pdms[rows, oracle_indices]
    regret = oracle - selected
    selected_rank = rankdata(-official_pdms, axis=1, method="min")[rows, selected_indices]
    row_corr = _row_spearman(predicted_pdms, official_pdms)

    selection = {
        "selected_pdms": float(selected.mean()),
        "offline_oracle_at_64_upper_bound": float(oracle.mean()),
        "scorer_regret": float(regret.mean()),
        "selected_score_predicted_mean": float(predicted_pdms[rows, selected_indices].mean()),
        "selected_score_calibration_bias": float((predicted_pdms[rows, selected_indices] - selected).mean()),
        "selected_true_rank_mean": float(selected_rank.mean()),
        "selected_true_rank_median": float(np.median(selected_rank)),
        "selected_is_official_oracle_fraction": float(np.isclose(selected, oracle, atol=1e-7).mean()),
        "selected_in_official_top_3_fraction": float((selected_rank <= 3).mean()),
        "selected_in_official_top_5_fraction": float((selected_rank <= 5).mean()),
        "selected_in_official_top_10_fraction": float((selected_rank <= 10).mean()),
        "catastrophic_misselection_oracle_gt_0p9_selected_lt_0p5_fraction": float(np.logical_and(oracle > 0.9, selected < 0.5).mean()),
        "scene_spearman_mean": float(np.nanmean(row_corr)),
        "scene_spearman_median": float(np.nanmedian(row_corr)),
        "scene_spearman_valid_fraction": float(np.isfinite(row_corr).mean()),
        "global_spearman": _safe_float(spearmanr(predicted_pdms.reshape(-1), official_pdms.reshape(-1)).statistic),
    }
    retrieval = {}
    predicted_order = np.argsort(-predicted_pdms, axis=1)
    for k in (1, 2, 4, 8, 16, 32, 64):
        chosen = predicted_order[:, :k]
        best = np.take_along_axis(official_pdms, chosen, axis=1).max(axis=1)
        retrieval[f"best_official_among_scorer_top_{k}"] = float(best.mean())
        # PDMS often has ties.  Keep exact argmax-index recovery separate from
        # the scientifically relevant question of whether top-k contains any
        # proposal attaining the offline-oracle score.
        retrieval[f"canonical_argmax_index_recall_at_{k}"] = float(
            np.any(chosen == oracle_indices[:, None], axis=1).mean()
        )
        retrieval[f"offline_oracle_score_covered_at_{k}"] = float(
            np.isclose(best, oracle, atol=1e-7).mean()
        )

    official_index = {name: index for index, name in enumerate(OFFICIAL_COMPONENTS)}
    head_metrics = {}
    for pred_index, name in enumerate(PREDICTED_COMPONENTS):
        target = official_components[..., official_index[name]]
        prediction = predicted_components[..., pred_index]
        if name in {"no_at_fault_collisions", "driving_direction_compliance"}:
            target = np.where(target == 0.5, 0.0, target)
        if name == "ego_progress":
            metrics = _continuous_metrics(target, prediction)
            metrics["task_type"] = "soft_binary_cross_entropy_regression"
        else:
            metrics = _binary_metrics(target, prediction)
            metrics["task_type"] = "binary_classification"
        metrics["scene_rank_spearman_mean"] = float(
            np.nanmean(_row_spearman(prediction, target))
        )
        head_metrics[name] = metrics

    aggregate_metrics = _continuous_metrics(official_pdms, predicted_pdms)
    aggregate_metrics["scene_rank_spearman_mean"] = float(np.nanmean(row_corr))
    aggregate_metrics["scene_rank_spearman_median"] = float(np.nanmedian(row_corr))

    selected_components = official_components[rows, selected_indices, :-1]
    oracle_components = official_components[rows, oracle_indices, :-1]
    component_attribution = {
        name: {
            "selected_mean": float(selected_components[:, index].mean()),
            "oracle_mean": float(oracle_components[:, index].mean()),
            "selected_minus_oracle": float((selected_components[:, index] - oracle_components[:, index]).mean()),
        }
        for index, name in enumerate(OFFICIAL_COMPONENTS[:-1])
    }

    counts = np.bincount(selected_indices, minlength=candidates)
    probabilities = counts / counts.sum()
    nonzero = probabilities > 0
    query_entropy = -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))
    query_usage = {
        "used_query_count": int(nonzero.sum()),
        "effective_selected_query_count": float(np.exp(query_entropy)),
        "normalized_selection_entropy": float(query_entropy / np.log(candidates)),
        "max_query_selection_fraction": float(probabilities.max()),
        "selection_counts": counts.tolist(),
    }

    flattened = proposals.reshape(scenes, candidates, -1).astype(np.float64)
    endpoint = proposals[:, :, -1, :2].astype(np.float64)
    upper = np.triu_indices(candidates, 1)
    pair_sum = 0.0
    endpoint_pair_sum = 0.0
    clusters = []
    for scene, scene_endpoint in zip(flattened, endpoint):
        distance = np.linalg.norm(scene[:, None] - scene[None, :], axis=-1)
        pair_sum += distance[upper].sum()
        endpoint_pair_sum += np.linalg.norm(scene_endpoint[:, None] - scene_endpoint[None, :], axis=-1)[upper].sum()
        adjacency = distance / math.sqrt(scene.shape[-1]) <= 0.5
        remaining = set(range(candidates))
        count = 0
        while remaining:
            frontier = [remaining.pop()]
            while frontier:
                neighbors = set(np.flatnonzero(adjacency[frontier.pop()])) & remaining
                remaining.difference_update(neighbors)
                frontier.extend(neighbors)
            count += 1
        clusters.append(count)
    pairs = scenes * len(upper[0])
    proposal_quality = {
        "candidate_mean_pdms": float(official_pdms.mean()),
        "candidate_median_pdms": float(np.median(official_pdms)),
        "candidate_p10_pdms": float(np.percentile(official_pdms, 10)),
        "candidate_p25_pdms": float(np.percentile(official_pdms, 25)),
        "candidate_above_0p8_fraction": float((official_pdms > 0.8).mean()),
        "candidate_above_0p9_fraction": float((official_pdms > 0.9).mean()),
        "mean_pairwise_full_trajectory_l2": float(pair_sum / pairs),
        "mean_pairwise_endpoint_l2": float(endpoint_pair_sum / pairs),
        "effective_clusters_rms_0p5_mean": float(np.mean(clusters)),
        "effective_clusters_rms_0p5_median": float(np.median(clusters)),
        "effective_clusters_rms_0p5_p10": float(np.percentile(clusters, 10)),
        "exact_duplicate_fraction": float(1.0 - np.mean([len(np.unique(np.round(scene, 4), axis=0)) for scene in flattened]) / candidates),
    }

    representation = {}
    if registers is not None:
        normalized = registers / np.maximum(np.linalg.norm(registers, axis=-1, keepdims=True), 1e-12)
        cosine = normalized @ np.swapaxes(normalized, -1, -2)
        offdiag = cosine[:, ~np.eye(cosine.shape[1], dtype=bool)]
        centered = registers - registers.mean(axis=1, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        energy = np.square(singular)
        probability = energy / np.maximum(energy.sum(axis=1, keepdims=True), 1e-12)
        effective_rank = np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1))
        representation.update(
            {
                "register_effective_rank_mean": float(effective_rank.mean()),
                "register_effective_rank_p10": float(np.percentile(effective_rank, 10)),
                "register_effective_rank_p90": float(np.percentile(effective_rank, 90)),
                "register_pairwise_cosine_mean": float(offdiag.mean()),
                "register_pairwise_cosine_p90": float(np.percentile(offdiag, 90)),
                "register_feature_std": float(registers.std()),
                "mean_singular_value_energy_fraction": probability.mean(axis=0).tolist(),
            }
        )
    if tile_gate is not None:
        representation["tile_gate_tanh_mean"] = float(tile_gate.mean())
        representation["tile_gate_tanh_std"] = float(tile_gate.std())
        representation["tile_gate_abs_gt_0p1_fraction"] = float((np.abs(tile_gate) > 0.1).mean())
    if semantic_gate is not None:
        representation["semantic_gate_probability_mean"] = float(semantic_gate.mean())
        representation["semantic_gate_probability_std"] = float(semantic_gate.std())

    logs = _metric_cache_logs(tokens, metric_cache)
    bootstrap = _cluster_bootstrap(
        {"selected_pdms": selected, "offline_oracle_at_64": oracle, "scorer_regret": regret, "candidate_scene_mean": official_pdms.mean(axis=1)},
        logs,
        samples=bootstrap_samples,
        seed=0,
    )
    report = {
        "schema_version": 1,
        "candidate_bank": str(path.resolve()),
        "candidate_bank_sha256": _sha256(path),
        "metric_cache": str(metric_cache.resolve()),
        "scene_count": scenes,
        "log_count": int(len(np.unique(logs))),
        "candidate_count": candidates,
        "inference_uses_future_inputs": False,
        "offline_oracle_definition": "Offline best of the same 64 frozen proposals; not an online deployable policy.",
        "selection": selection,
        "scorer_top_k_retrieval": retrieval,
        "scorer_aggregate_task": aggregate_metrics,
        "scorer_component_tasks": head_metrics,
        "official_component_selection_attribution": component_attribution,
        "query_usage": query_usage,
        "proposal_bank_task": proposal_quality,
        "planning_representation": representation,
        "log_cluster_bootstrap": bootstrap,
        "scorer_reads_predicted_future_registers": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_task_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(
        {
            "official": official_pdms,
            "predicted": predicted_pdms,
            "selected": selected,
            "oracle": oracle,
            "selected_rank": selected_rank,
            "selected_indices": selected_indices,
        },
        output_dir,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    report = audit(args.candidate_bank, args.metric_cache, args.output_dir, args.bootstrap_samples)
    print(json.dumps({"selection": report["selection"], "bootstrap": report["log_cluster_bootstrap"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
