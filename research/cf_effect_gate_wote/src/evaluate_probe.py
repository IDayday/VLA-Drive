"""Evaluate matched G2/G3 scorers with scene-level paired uncertainty."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import yaml

from .feature_store import atomic_write_json
from .metrics import (
    FACTOR_NAMES,
    candidate_ranks,
    mean_kendall_tau,
    paired_scene_bootstrap,
    pairwise_ranking_accuracy,
    pdms_from_factors,
)
from .models.probe_heads import MatchedCapacityFactorProbe, MatchedInputComposer
from .train_probe import (
    G2_MODEL_TYPES,
    ProbeScene,
    _stack_raw_batch,
    iter_probe_scenes,
    raw_scene_inputs,
)


@dataclass(frozen=True)
class EvaluationResult:
    model_type: str
    seed: int
    tokens: tuple[str, ...]
    predicted_factors: npt.NDArray[np.float32]
    predicted_scores: npt.NDArray[np.float32]
    true_factors: npt.NDArray[np.float32]
    official_selected_indices: npt.NDArray[np.int64]
    train_metadata: Mapping[str, Any]


def _effect_permutation(token: str, seed: int, candidates: int = 256) -> npt.NDArray[np.int64]:
    digest = hashlib.sha256(f"effect-swap:{token}:{seed}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    permutation = rng.permutation(candidates).astype(np.int64)
    if np.array_equal(permutation, np.arange(candidates)):
        permutation = np.roll(permutation, 1)
    return permutation


def _iter_evaluation_batches(
    frozen_cache: Path,
    effect_cache: Path | None,
    model_type: str,
    batch_scenes: int,
    swap_effects: bool,
    seed: int,
) -> Iterator[tuple[Any, tuple[str, ...]]]:
    pending: list[tuple[str, tuple[npt.NDArray[Any], ...]]] = []
    for scene in iter_probe_scenes(frozen_cache, effect_cache):
        permutation = _effect_permutation(scene.token, seed) if swap_effects else None
        raw = raw_scene_inputs(
            scene,
            model_type,
            effect_permutation=permutation,
        )
        pending.append((scene.token, raw))
        if len(pending) == batch_scenes:
            batch = _stack_raw_batch(pending)
            yield batch, batch.tokens
            pending.clear()
    if pending:
        batch = _stack_raw_batch(pending)
        yield batch, batch.tokens


def evaluate_checkpoint(
    checkpoint_path: Path,
    frozen_cache: Path,
    effect_cache: Path | None,
    batch_scenes: int,
    device: torch.device,
    swap_effects: bool = False,
) -> EvaluationResult:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "matched_factor_probe.v1":
        raise ValueError(f"unsupported probe checkpoint: {checkpoint_path}")
    model_type = str(payload["model_type"])
    seed = int(payload["seed"])
    composer = MatchedInputComposer().to(device)
    probe = MatchedCapacityFactorProbe(hidden_dim=int(payload["hidden_dim"])).to(device)
    composer.load_state_dict(payload["composer_state_dict"], strict=True)
    probe.load_state_dict(payload["probe_state_dict"], strict=True)
    composer.eval()
    probe.eval()
    predicted_factors: list[npt.NDArray[np.float32]] = []
    predicted_scores: list[npt.NDArray[np.float32]] = []
    true_factors: list[npt.NDArray[np.float32]] = []
    selected_indices: list[npt.NDArray[np.int64]] = []
    tokens: list[str] = []
    with torch.inference_mode():
        for batch, batch_tokens in _iter_evaluation_batches(
            frozen_cache,
            effect_cache,
            model_type,
            batch_scenes,
            swap_effects,
            seed,
        ):
            output = probe(
                composer(
                    batch.base.to(device),
                    batch.current.to(device),
                    batch.auxiliary.to(device),
                )
            )
            predicted_factors.append(output["factors"].cpu().numpy().astype(np.float32))
            predicted_scores.append(output["score"].cpu().numpy().astype(np.float32))
            true_factors.append(batch.factor_labels.numpy().astype(np.float32))
            selected_indices.append(batch.selected_indices)
            tokens.extend(batch_tokens)
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("evaluation produced empty or duplicate scene tokens")
    return EvaluationResult(
        model_type=(
            "effect_swap"
            if swap_effects and model_type == "oracle_replay_effect"
            else f"{model_type}_swap"
            if swap_effects
            else model_type
        ),
        seed=seed,
        tokens=tuple(tokens),
        predicted_factors=np.concatenate(predicted_factors, axis=0),
        predicted_scores=np.concatenate(predicted_scores, axis=0),
        true_factors=np.concatenate(true_factors, axis=0),
        official_selected_indices=np.concatenate(selected_indices, axis=0),
        train_metadata={
            key: payload[key]
            for key in (
                "learning_rate",
                "pairwise_weight",
                "best_epoch",
                "training_steps",
                "parameter_audit",
                "peak_gpu_memory_bytes",
            )
        },
    )


def _binary_auc(target: npt.ArrayLike, prediction: npt.ArrayLike) -> float:
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    score = np.asarray(prediction, dtype=np.float64).reshape(-1)
    positive = truth >= 0.5
    count_positive = int(positive.sum())
    count_negative = int((~positive).sum())
    if count_positive == 0 or count_negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[positive].sum()
    return float(
        (rank_sum - count_positive * (count_positive + 1) / 2.0)
        / (count_positive * count_negative)
    )


def scene_outcomes(result: EvaluationResult) -> pd.DataFrame:
    target_scores = pdms_from_factors(result.true_factors)
    selected = result.predicted_scores.argmax(axis=1)
    rows = np.arange(len(selected))
    selected_true = target_scores[rows, selected]
    oracle = target_scores.max(axis=1)
    official_selected = target_scores[rows, result.official_selected_indices]
    selected_factors = result.true_factors[rows, selected]
    false_safe = (
        (selected_factors[:, 0] == 0)
        | (selected_factors[:, 1] == 0)
        | (selected_factors[:, 3] == 0)
    )
    ranks = candidate_ranks(target_scores)[rows, selected]
    return pd.DataFrame(
        {
            "scene_token": result.tokens,
            "gate": "G2",
            "model": result.model_type,
            "seed": result.seed,
            "selected_index": selected,
            "official_selected_index": result.official_selected_indices,
            "selected_pdms": selected_true,
            "official_selected_pdms": official_selected,
            "oracle_pdms": oracle,
            "regret": oracle - selected_true,
            "candidate_rank": ranks,
            "false_safe": false_safe,
            "official_failure_recovered": (official_selected == 0) & (selected_true > 0),
        }
    )


def aggregate_metrics(result: EvaluationResult, outcomes: pd.DataFrame) -> dict[str, Any]:
    target_scores = pdms_from_factors(result.true_factors)
    official_failures = outcomes["official_selected_pdms"].to_numpy() == 0
    recovery_rate = (
        float(outcomes.loc[official_failures, "official_failure_recovered"].mean())
        if official_failures.any()
        else float("nan")
    )
    row: dict[str, Any] = {
        "gate": "G2",
        "model": result.model_type,
        "seed": result.seed,
        "future_input": {
            "trajectory_only": "none",
            "direct_current": "none",
            "shared_logged_future": "shared logged future",
            "oracle_replay_effect": "oracle replay effect",
            "effect_swap": "swapped replay effect",
            "predicted_replay_effect": "predicted replay effect",
            "wote_full_future": "WoTE reward feature",
            "wote_environment_only": "masked environment-only rollout",
            "predicted_replay_effect_swap": "swapped predicted replay effect",
        }[result.model_type],
        "candidate_specific": result.model_type
        in {
            "oracle_replay_effect",
            "effect_swap",
            "predicted_replay_effect",
            "wote_full_future",
            "wote_environment_only",
        },
        "selected_pdms": float(outcomes["selected_pdms"].mean()),
        "top1_regret": float(outcomes["regret"].mean()),
        "mean_candidate_rank": float(outcomes["candidate_rank"].mean()),
        "pairwise_accuracy": pairwise_ranking_accuracy(result.predicted_scores, target_scores),
        "kendall_tau": mean_kendall_tau(result.predicted_scores, target_scores),
        "false_safe_rate": float(outcomes["false_safe"].mean()),
        "failure_recovery_rate": recovery_rate,
    }
    for index, name in enumerate(FACTOR_NAMES):
        row[f"factor_mae_{name}"] = float(
            np.abs(result.predicted_factors[..., index] - result.true_factors[..., index]).mean()
        )
        row[f"factor_auc_{name}"] = (
            _binary_auc(
                result.true_factors[..., index], result.predicted_factors[..., index]
            )
            if name in {"NC", "DAC", "TTC"}
            else float("nan")
        )
    row.update(
        {
            "trainable_parameters": int(
                result.train_metadata["parameter_audit"]["trainable_parameters"]
            ),
            "flops_per_candidate_estimate": int(
                result.train_metadata["parameter_audit"]["approximate_flops_per_candidate"]
            ),
            "peak_gpu_memory_bytes": int(result.train_metadata["peak_gpu_memory_bytes"]),
            "training_steps": int(result.train_metadata["training_steps"]),
            "best_validation_epoch": int(result.train_metadata["best_epoch"]),
            "learning_rate": float(result.train_metadata["learning_rate"]),
            "pairwise_weight": float(result.train_metadata["pairwise_weight"]),
        }
    )
    return row


def _paired_model_arrays(
    outcomes: pd.DataFrame, left: str, right: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    left_frame = (
        outcomes[outcomes["model"] == left]
        .groupby("scene_token", sort=True)["selected_pdms"]
        .mean()
    )
    right_frame = (
        outcomes[outcomes["model"] == right]
        .groupby("scene_token", sort=True)["selected_pdms"]
        .mean()
    )
    if not left_frame.index.equals(right_frame.index):
        raise ValueError(f"paired model scene tokens differ: {left}/{right}")
    return left_frame.to_numpy(), right_frame.to_numpy()


def summarize_g2(
    metrics: pd.DataFrame,
    outcomes: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    required = set(G2_MODEL_TYPES) | {"effect_swap"}
    missing = sorted(required - set(metrics["model"]))
    if missing:
        raise ValueError(f"G2 metrics missing models: {missing}")
    seed_sets = {
        model: set(metrics.loc[metrics["model"] == model, "seed"].astype(int))
        for model in required
    }
    if len({tuple(sorted(value)) for value in seed_sets.values()}) != 1:
        raise ValueError(f"G2 seed sets differ: {seed_sets}")
    counts = metrics.groupby("model")["trainable_parameters"].unique()
    all_counts = [int(value) for values in counts for value in values]
    parameter_spread = (max(all_counts) - min(all_counts)) / min(all_counts)
    if parameter_spread > 0.05:
        raise ValueError(f"probe parameter spread exceeds 5%: {parameter_spread}")

    per_seed: list[dict[str, Any]] = []
    for seed in sorted(seed_sets["direct_current"]):
        rows = metrics[metrics["seed"] == seed].set_index("model")
        direct_regret = float(rows.loc["direct_current", "top1_regret"])
        effect_regret = float(rows.loc["oracle_replay_effect", "top1_regret"])
        shared_regret = float(rows.loc["shared_logged_future", "top1_regret"])
        direct_reduction = (
            (direct_regret - effect_regret) / direct_regret
            if direct_regret > 0
            else -float("inf")
        )
        shared_reduction = (
            (shared_regret - effect_regret) / shared_regret
            if shared_regret > 0
            else -float("inf")
        )
        per_seed.append(
            {
                "seed": int(seed),
                "direct_regret_reduction_fraction": direct_reduction,
                "direct_selected_pdms_gain": float(
                    rows.loc["oracle_replay_effect", "selected_pdms"]
                    - rows.loc["direct_current", "selected_pdms"]
                ),
                "shared_regret_reduction_fraction": shared_reduction,
                "swap_selected_pdms_drop": float(
                    rows.loc["oracle_replay_effect", "selected_pdms"]
                    - rows.loc["effect_swap", "selected_pdms"]
                ),
            }
        )
    averaged = metrics.groupby("model", sort=False).mean(numeric_only=True)
    direct_regret = float(averaged.loc["direct_current", "top1_regret"])
    effect_regret = float(averaged.loc["oracle_replay_effect", "top1_regret"])
    shared_regret = float(averaged.loc["shared_logged_future", "top1_regret"])
    direct_reduction = (
        (direct_regret - effect_regret) / direct_regret if direct_regret > 0 else -float("inf")
    )
    shared_reduction = (
        (shared_regret - effect_regret) / shared_regret if shared_regret > 0 else -float("inf")
    )
    pdms_gain = float(
        averaged.loc["oracle_replay_effect", "selected_pdms"]
        - averaged.loc["direct_current", "selected_pdms"]
    )
    effect_scene, direct_scene = _paired_model_arrays(
        outcomes, "oracle_replay_effect", "direct_current"
    )
    effect_shared_scene, shared_scene = _paired_model_arrays(
        outcomes, "oracle_replay_effect", "shared_logged_future"
    )
    effect_swap_scene, swap_scene = _paired_model_arrays(
        outcomes, "oracle_replay_effect", "effect_swap"
    )
    ci_direct = paired_scene_bootstrap(
        effect_scene,
        direct_scene,
        samples=bootstrap_samples,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    ci_shared = paired_scene_bootstrap(
        effect_shared_scene,
        shared_scene,
        samples=bootstrap_samples,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed + 1,
    )
    ci_swap = paired_scene_bootstrap(
        effect_swap_scene,
        swap_scene,
        samples=bootstrap_samples,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed + 2,
    )
    direction_consistent = all(
        row["direct_regret_reduction_fraction"] > 0
        and row["direct_selected_pdms_gain"] > 0
        for row in per_seed
    )
    swap_significant = ci_swap.lower > 0 and all(
        row["swap_selected_pdms_drop"] > 0 for row in per_seed
    )
    conditions = {
        "effect_vs_direct_regret_reduction_at_least_20pct": direct_reduction >= 0.20,
        "effect_vs_direct_pdms_gain_at_least_0_005": pdms_gain >= 0.005,
        "effect_vs_shared_regret_reduction_at_least_10pct": shared_reduction >= 0.10,
        "three_seed_direction_consistent": direction_consistent,
        "effect_swap_significantly_worse": swap_significant,
    }
    return {
        "schema_version": "gate_g2.v1",
        "gate_g2_pass": all(conditions.values()),
        "conditions": conditions,
        "mean_direct_regret_reduction_fraction": direct_reduction,
        "mean_direct_selected_pdms_gain_raw": pdms_gain,
        "mean_direct_selected_pdms_gain_points": pdms_gain * 100.0,
        "mean_shared_regret_reduction_fraction": shared_reduction,
        "parameter_spread_fraction": parameter_spread,
        "per_seed": per_seed,
        "paired_scene_bootstrap": {
            "effect_minus_direct_selected_pdms": asdict(ci_direct),
            "effect_minus_shared_selected_pdms": asdict(ci_shared),
            "effect_minus_swap_selected_pdms": asdict(ci_swap),
        },
        "failure_reason": (
            None
            if all(conditions.values())
            else "replay-grounded candidate effect adds no planning information"
        ),
    }


def evaluate_suite(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing evaluation output: {args.output_dir}")
    manifest = json.loads((args.training_root / "training_manifest.json").read_text())
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metric_rows: list[dict[str, Any]] = []
    outcome_frames: list[pd.DataFrame] = []
    for trial in manifest["selected_trials"]:
        checkpoint = Path(trial["checkpoint"])
        result = evaluate_checkpoint(
            checkpoint,
            args.test_cache,
            args.test_effects,
            int(config["probe"]["batch_scenes"]),
            device,
        )
        outcomes = scene_outcomes(result)
        metric_rows.append(aggregate_metrics(result, outcomes))
        outcome_frames.append(outcomes)
        if result.model_type == "oracle_replay_effect":
            swapped = evaluate_checkpoint(
                checkpoint,
                args.test_cache,
                args.test_effects,
                int(config["probe"]["batch_scenes"]),
                device,
                swap_effects=True,
            )
            swapped_outcomes = scene_outcomes(swapped)
            metric_rows.append(aggregate_metrics(swapped, swapped_outcomes))
            outcome_frames.append(swapped_outcomes)
    metrics = pd.DataFrame(metric_rows)
    outcomes = pd.concat(outcome_frames, ignore_index=True)
    summary = summarize_g2(
        metrics,
        outcomes,
        int(config["bootstrap"]["samples"]),
        float(config["bootstrap"]["confidence"]),
        int(config["bootstrap"]["seed"]),
    )
    metrics.to_csv(args.output_dir / "probe_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    outcomes.to_parquet(args.output_dir / "scene_level_g2.parquet", index=False)
    atomic_write_json(args.output_dir / "g2_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--test-effects", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    evaluate_suite(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
