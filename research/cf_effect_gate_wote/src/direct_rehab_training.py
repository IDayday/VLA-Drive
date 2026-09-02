"""Train and evaluate the preregistered current-only Direct Scorer V3."""

from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .direct_rehab_contracts import AccessPolicy
from .direct_rehab_data import (
    DirectDataset,
    iter_direct_batches,
    load_direct_dataset,
    stack_direct_batch,
)
from .direct_rehab_metrics import aggregate_ranking_metrics, scene_level_metrics
from .models.top_aware_direct_scorer import (
    TopAwareDirectScorerConfig,
    TopAwareDirectScorerV3,
    checkpoint_payload,
    load_v3_checkpoint,
    selection_logit,
)
from .top_aware_losses import TopAwareLossConfig, top_aware_direct_loss


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Mapping[str, float]
    scene_rows: tuple[Mapping[str, Any], ...]
    selection_values: np.ndarray
    predicted_factors: np.ndarray
    hard_safety_logits: np.ndarray


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _forward(
    model: TopAwareDirectScorerV3,
    batch: Any,
    *,
    candidate_chunk: int,
) -> Mapping[str, Tensor]:
    return model(
        batch.trajectory,
        batch.ego_status,
        batch.current_bev_tokens,
        batch.candidate_current_feature,
        candidate_chunk=candidate_chunk,
    )


def _selection_values(
    outputs: Mapping[str, Tensor],
    *,
    objective: str,
    safety_lambda: float,
) -> Tensor:
    if objective == "O0":
        return outputs["factor_score"]
    if objective == "O3":
        return selection_logit(outputs, safety_lambda)
    return outputs["utility_logit"]


def evaluate_model(
    model: TopAwareDirectScorerV3,
    dataset: DirectDataset,
    *,
    device: torch.device,
    objective: str,
    safety_lambda: float,
    batch_scenes: int,
    candidate_chunk: int,
) -> EvaluationResult:
    model.eval()
    selections: list[np.ndarray] = []
    factors: list[np.ndarray] = []
    safety: list[np.ndarray] = []
    true_factors: list[np.ndarray] = []
    true_scores: list[np.ndarray] = []
    tokens: list[str] = []
    with torch.inference_mode():
        for raw_batch in iter_direct_batches(
            dataset, batch_scenes=batch_scenes, seed=0, epoch=0, shuffle=False
        ):
            batch = raw_batch.to(device)
            output = _forward(model, batch, candidate_chunk=candidate_chunk)
            selections.append(
                _selection_values(
                    output, objective=objective, safety_lambda=safety_lambda
                )
                .detach()
                .cpu()
                .numpy()
            )
            factors.append(output["factors"].detach().cpu().numpy())
            safety.append(output["hard_safety_logit"].detach().cpu().numpy())
            true_factors.append(raw_batch.factor_labels.numpy())
            true_scores.append(raw_batch.score_labels.numpy())
            tokens.extend(raw_batch.tokens)
    selection_array = np.concatenate(selections, axis=0)
    factor_array = np.concatenate(factors, axis=0)
    safety_array = np.concatenate(safety, axis=0)
    label_array = np.concatenate(true_factors, axis=0)
    score_array = np.concatenate(true_scores, axis=0)
    rows = scene_level_metrics(
        tokens,
        selection_array,
        factor_array,
        label_array,
        score_array,
        predicted_hard_safety=safety_array,
    )
    metrics = aggregate_ranking_metrics(selection_array, score_array, rows)
    return EvaluationResult(
        metrics=metrics,
        scene_rows=tuple(rows),
        selection_values=selection_array,
        predicted_factors=factor_array,
        hard_safety_logits=safety_array,
    )


def reference_selector_metrics(dataset: DirectDataset) -> Mapping[str, float]:
    if any(scene.wote_selected_index is None for scene in dataset.scenes):
        raise ValueError("WoTE selector references were not loaded")
    selected_scores: list[float] = []
    regrets: list[float] = []
    ranks: list[int] = []
    false_safe: list[bool] = []
    direction: list[bool] = []
    zeros: list[bool] = []
    for scene in dataset.scenes:
        selected = int(scene.wote_selected_index)
        scores = np.asarray(scene.score_labels)
        factors = np.asarray(scene.factor_labels)
        oracle_score = float(np.max(scores))
        score = float(scores[selected])
        order = np.argsort(-scores, kind="stable")
        rank = int(np.nonzero(order == selected)[0][0]) + 1
        selected_scores.append(score)
        regrets.append(oracle_score - score)
        ranks.append(rank)
        false_safe.append(bool((factors[selected, (0, 1, 2, 4)] <= 0.0).any()))
        direction.append(bool(factors[selected, 2] < 1.0))
        zeros.append(bool(score <= 0.0))
    return {
        "selected_score": float(np.mean(selected_scores)),
        "top1_regret": float(np.mean(regrets)),
        "mean_selected_candidate_rank": float(np.mean(ranks)),
        "hard_false_safe": float(np.mean(false_safe)),
        "direction_non_compliance": float(np.mean(direction)),
        "zero_score_selection": float(np.mean(zeros)),
    }


def train_trial(
    *,
    train_dataset: DirectDataset,
    val_dataset: DirectDataset,
    representation: str,
    objective_config: TopAwareLossConfig,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_scenes: int,
    candidate_chunk: int,
    max_epochs: int,
    patience: int,
    safety_lambda: float,
    gradient_clip_norm: float,
    device: torch.device,
    output: Path,
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    if output.exists():
        raise FileExistsError(f"refusing existing trial output: {output}")
    _seed_everything(seed)
    model = TopAwareDirectScorerV3(
        TopAwareDirectScorerConfig(representation=representation)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_regret = float("inf")
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        sums: dict[str, float] = {}
        batches = 0
        for raw_batch in iter_direct_batches(
            train_dataset,
            batch_scenes=batch_scenes,
            seed=seed,
            epoch=epoch,
            shuffle=True,
        ):
            batch = raw_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward(model, batch, candidate_chunk=candidate_chunk)
            losses = top_aware_direct_loss(
                outputs, batch.factor_labels, batch.score_labels, objective_config
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("non-finite Direct V3 training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
            batches += 1
        validation = evaluate_model(
            model,
            val_dataset,
            device=device,
            objective=objective_config.objective,
            safety_lambda=safety_lambda,
            batch_scenes=batch_scenes,
            candidate_chunk=candidate_chunk,
        )
        row = {
            "epoch": epoch + 1,
            "train": {key: value / batches for key, value in sums.items()},
            "validation": dict(validation.metrics),
        }
        history.append(row)
        regret = float(validation.metrics["top1_regret"])
        print(
            f"[direct-v3] rep={representation} objective={objective_config.objective} "
            f"seed={seed} epoch={epoch + 1} loss={row['train']['total']:.6f} "
            f"val_score={validation.metrics['selected_score']:.6f} "
            f"val_regret={regret:.6f} "
            f"false_safe={validation.metrics['hard_false_safe']:.6f}",
            flush=True,
        )
        if regret < best_regret - 1.0e-8:
            best_regret = regret
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("early stopping did not capture a finite Direct V3 model")
    model.load_state_dict(best_state, strict=True)
    final_validation = evaluate_model(
        model,
        val_dataset,
        device=device,
        objective=objective_config.objective,
        safety_lambda=safety_lambda,
        batch_scenes=batch_scenes,
        candidate_chunk=candidate_chunk,
    )
    payload = checkpoint_payload(
        model,
        seed=seed,
        objective=objective_config.as_dict(),
        selection={"safety_lambda": float(safety_lambda)},
        metadata={
            **dict(metadata),
            "best_epoch": best_epoch,
            "best_validation_metrics": dict(final_validation.metrics),
            "history": history,
        },
    )
    _atomic_torch_save(output, payload)
    result = {
        "checkpoint": str(output),
        "representation": representation,
        "objective": objective_config.objective,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation": dict(final_validation.metrics),
        "wote_validation": dict(reference_selector_metrics(val_dataset)),
        "history": history,
    }
    _atomic_json(output.with_suffix(".json"), result)
    return result


def smoke_overfit(
    dataset: DirectDataset,
    *,
    representation: str,
    objective_config: TopAwareLossConfig,
    steps: int,
    learning_rate: float,
    candidate_chunk: int,
    device: torch.device,
) -> Mapping[str, Any]:
    if len(dataset) != 2 or steps <= 1:
        raise ValueError("smoke overfit requires exactly two scenes and >1 steps")
    _seed_everything(0)
    model = TopAwareDirectScorerV3(
        TopAwareDirectScorerConfig(representation=representation)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    batch = stack_direct_batch(dataset.scenes).to(device)
    values: list[float] = []
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, batch, candidate_chunk=candidate_chunk)
        losses = top_aware_direct_loss(
            outputs, batch.factor_labels, batch.score_labels, objective_config
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        values.append(float(losses["total"].detach().cpu()))
    if not np.isfinite(values).all() or values[-1] >= values[0]:
        raise RuntimeError(f"two-scene overfit failed: {values[0]} -> {values[-1]}")
    return {
        "status": "PASS",
        "representation": representation,
        "objective": objective_config.objective,
        "steps": steps,
        "initial_loss": values[0],
        "final_loss": values[-1],
        "minimum_loss": min(values),
    }


def _read_dataset(
    *,
    feature_root: Path,
    label_root: Path,
    token_path: Path,
    limit: int | None,
    policy_path: Path,
    phase: str,
    access_log: Path,
) -> DirectDataset:
    policy = AccessPolicy.load(policy_path)
    tokens = policy.read_token_file(token_path, phase)
    if limit is not None:
        tokens = tokens[:limit]
    return load_direct_dataset(
        feature_root=feature_root,
        label_root=label_root,
        expected_tokens=tokens,
        access_policy=policy,
        phase=phase,
        access_log=access_log,
        require_selector_reference=True,
    )


def _loss_config(args: argparse.Namespace) -> TopAwareLossConfig:
    return TopAwareLossConfig(
        objective=args.objective,
        factor_weight=args.factor_weight,
        score_weight=args.score_weight,
        listwise_weight=args.listwise_weight,
        top_pair_weight=args.top_pair_weight,
        safety_weight=args.safety_weight,
        target_temperature=args.target_temperature,
        prediction_temperature=args.prediction_temperature,
    )


def _add_data(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    option = f"{prefix}-" if prefix else ""
    destination = f"{prefix}_" if prefix else ""
    parser.add_argument(f"--{option}feature-root", dest=f"{destination}feature_root", type=Path, required=True)
    parser.add_argument(f"--{option}label-root", dest=f"{destination}label_root", type=Path, required=True)
    parser.add_argument(f"--{option}tokens", dest=f"{destination}tokens", type=Path, required=True)
    parser.add_argument(f"--{option}limit", dest=f"{destination}limit", type=int)


def _add_objective(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--objective", choices=("O0", "O1", "O2", "O3"), default="O3")
    parser.add_argument("--factor-weight", type=float, default=0.5)
    parser.add_argument("--score-weight", type=float, default=0.5)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--top-pair-weight", type=float, default=0.5)
    parser.add_argument("--safety-weight", type=float, default=0.25)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--prediction-temperature", type=float, default=1.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    _add_data(smoke)
    _add_objective(smoke)
    smoke.add_argument("--representation", default="hybrid_current")
    smoke.add_argument("--steps", type=int, default=40)
    smoke.add_argument("--learning-rate", type=float, default=3.0e-4)
    smoke.add_argument("--candidate-chunk", type=int, default=64)
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
    _add_data(train, "train")
    _add_data(train, "val")
    _add_objective(train)
    train.add_argument("--representation", required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--learning-rate", type=float, default=3.0e-4)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--batch-scenes", type=int, default=4)
    train.add_argument("--candidate-chunk", type=int, default=64)
    train.add_argument("--max-epochs", type=int, default=20)
    train.add_argument("--patience", type=int, default=4)
    train.add_argument("--safety-lambda", type=float, default=0.5)
    train.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train.add_argument("--device", default="cuda")
    train.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    _add_data(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--batch-scenes", type=int, default=4)
    evaluate.add_argument("--candidate-chunk", type=int, default=64)
    evaluate.add_argument("--phase", default="development")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--output", type=Path, required=True)
    for command in (smoke, train, evaluate):
        command.add_argument("--access-policy", type=Path, required=True)
        command.add_argument("--access-log", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = torch.device(args.device)
    if args.command == "smoke":
        dataset = _read_dataset(
            feature_root=args.feature_root,
            label_root=args.label_root,
            token_path=args.tokens,
            limit=2,
            policy_path=args.access_policy,
            phase="development",
            access_log=args.access_log,
        )
        result = smoke_overfit(
            dataset,
            representation=args.representation,
            objective_config=_loss_config(args),
            steps=args.steps,
            learning_rate=args.learning_rate,
            candidate_chunk=args.candidate_chunk,
            device=device,
        )
        _atomic_json(args.output, result)
    elif args.command == "train":
        train_dataset = _read_dataset(
            feature_root=args.train_feature_root,
            label_root=args.train_label_root,
            token_path=args.train_tokens,
            limit=args.train_limit,
            policy_path=args.access_policy,
            phase="development",
            access_log=args.access_log,
        )
        val_dataset = _read_dataset(
            feature_root=args.val_feature_root,
            label_root=args.val_label_root,
            token_path=args.val_tokens,
            limit=args.val_limit,
            policy_path=args.access_policy,
            phase="development",
            access_log=args.access_log,
        )
        result = train_trial(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            representation=args.representation,
            objective_config=_loss_config(args),
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_scenes=args.batch_scenes,
            candidate_chunk=args.candidate_chunk,
            max_epochs=args.max_epochs,
            patience=args.patience,
            safety_lambda=args.safety_lambda,
            gradient_clip_norm=args.gradient_clip_norm,
            device=device,
            output=args.output,
            metadata={
                "train_tokens": str(args.train_tokens),
                "val_tokens": str(args.val_tokens),
                "train_scenes": len(train_dataset),
                "val_scenes": len(val_dataset),
            },
        )
    else:
        dataset = _read_dataset(
            feature_root=args.feature_root,
            label_root=args.label_root,
            token_path=args.tokens,
            limit=args.limit,
            policy_path=args.access_policy,
            phase=args.phase,
            access_log=args.access_log,
        )
        model, payload = load_v3_checkpoint(args.checkpoint, map_location=device)
        model.to(device)
        objective = str(payload["objective"]["objective"])
        safety_lambda = float(payload["selection"]["safety_lambda"])
        evaluation = evaluate_model(
            model,
            dataset,
            device=device,
            objective=objective,
            safety_lambda=safety_lambda,
            batch_scenes=args.batch_scenes,
            candidate_chunk=args.candidate_chunk,
        )
        result = {
            "checkpoint": str(args.checkpoint),
            "phase": args.phase,
            "scene_count": len(dataset),
            "metrics": dict(evaluation.metrics),
            "wote_reference": dict(reference_selector_metrics(dataset)),
            "scene_rows": list(evaluation.scene_rows),
        }
        _atomic_json(args.output, result)
    display = result
    if args.command == "train":
        display = {
            key: result[key]
            for key in (
                "checkpoint",
                "representation",
                "objective",
                "seed",
                "best_epoch",
                "validation",
                "wote_validation",
            )
        }
    elif args.command == "evaluate":
        display = {
            key: result[key]
            for key in ("checkpoint", "phase", "scene_count", "metrics", "wote_reference")
        }
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
