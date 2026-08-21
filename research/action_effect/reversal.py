"""Cross-scene action-order reversal cases for scene/action binding tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


def perturbation_signature(row: Mapping[str, Any]) -> str:
    """Return a deterministic perturbation family/parameter signature."""

    parameters = json.dumps(row.get("perturbation_parameters", {}), sort_keys=True, separators=(",", ":"))
    return f"{row.get('perturbation_type')}:{parameters}"


def build_reversal_cases(
    consequence_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    selected_scene_ids: Sequence[str],
    maximum_cases: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Match identical local action comparisons whose logged score order reverses."""

    selected = set(str(value) for value in selected_scene_ids)
    candidate = {str(row["candidate_id"]): row for row in consequence_rows}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pair in pair_rows:
        if str(pair["scene_id"]) not in selected or int(pair.get("log_replay_order", 0)) == 0:
            continue
        left = candidate[str(pair["candidate_i"])]
        right = candidate[str(pair["candidate_j"])]
        left_signature = perturbation_signature(left)
        right_signature = perturbation_signature(right)
        if left_signature == right_signature:
            continue
        if left_signature < right_signature:
            first, second = left, right
            order = int(pair["log_replay_order"])
            key = (left_signature, right_signature)
        else:
            first, second = right, left
            order = -int(pair["log_replay_order"])
            key = (right_signature, left_signature)
        grouped.setdefault(key, []).append(
            {
                "scene_id": str(pair["scene_id"]),
                "candidate_a": str(first["candidate_id"]),
                "candidate_b": str(second["candidate_id"]),
                "order": order,
            }
        )
    cases: list[dict[str, Any]] = []
    for signatures, comparisons in sorted(grouped.items()):
        positive = sorted(
            [row for row in comparisons if row["order"] > 0], key=lambda row: row["scene_id"]
        )
        negative = sorted(
            [row for row in comparisons if row["order"] < 0], key=lambda row: row["scene_id"]
        )
        count = min(len(positive), len(negative))
        for index in range(count):
            cases.append(
                {
                    "signature_a": signatures[0],
                    "signature_b": signatures[1],
                    "positive": positive[index],
                    "negative": negative[index],
                }
            )
    ranked = sorted(
        cases,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['positive']['scene_id']}:{row['negative']['scene_id']}:{row['signature_a']}:{row['signature_b']}".encode(
                "utf-8"
            )
        ).hexdigest(),
    )
    return ranked[:maximum_cases]


def reversal_accuracy(
    cases: Sequence[Mapping[str, Any]],
    *,
    candidate_ids: np.ndarray,
    predicted_utility: np.ndarray,
    tolerance: float = 1.0e-6,
) -> tuple[float, np.ndarray]:
    """Require both sides of every reversal case to have the correct order."""

    lookup = {str(candidate): index for index, candidate in enumerate(candidate_ids)}
    correct = np.zeros(len(cases), dtype=bool)
    for index, case in enumerate(cases):
        decisions = []
        for side in (case["positive"], case["negative"]):
            delta = float(
                predicted_utility[lookup[str(side["candidate_a"])]]
                - predicted_utility[lookup[str(side["candidate_b"])]]
            )
            predicted_order = 0 if abs(delta) <= tolerance else (1 if delta > 0 else -1)
            decisions.append(predicted_order == int(side["order"]))
        correct[index] = all(decisions)
    return (float(np.mean(correct)) if len(correct) else float("nan")), correct
