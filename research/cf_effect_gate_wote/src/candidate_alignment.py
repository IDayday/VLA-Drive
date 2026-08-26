"""Candidate/label alignment audit, fixed split construction, and G1 oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .feature_store import FeatureShardReader, atomic_write_json, stable_array_hash
from .metrics import FACTOR_NAMES, candidate_ranks, paired_scene_bootstrap, pdms_from_factors


SCORE_FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
)
PUBLISHED_SCORE_PROPOSAL_NUM_POSES = 80
DEFAULT_METRIC_CACHE_PROPOSAL_NUM_POSES = 40
DEFAULT_METRIC_CACHE_FUTURE_NUM_POSES = 50


class CandidateAlignmentError(RuntimeError):
    """A candidate tensor cannot be legally paired with its labels."""


@dataclass(frozen=True)
class CandidateLabels:
    factors: npt.NDArray[np.float32]
    score: npt.NDArray[np.float32]

    def validate(self, token: str, expected_candidates: int = 256) -> None:
        if self.factors.shape != (expected_candidates, 5):
            raise CandidateAlignmentError(
                f"{token}: expected factors [{expected_candidates},5], got {self.factors.shape}"
            )
        if self.score.shape != (expected_candidates,):
            raise CandidateAlignmentError(
                f"{token}: expected scores [{expected_candidates}], got {self.score.shape}"
            )
        if not np.isfinite(self.factors).all() or not np.isfinite(self.score).all():
            raise CandidateAlignmentError(f"{token}: candidate labels contain NaN/Inf")


class CandidateScoreTable:
    """Strict reader for WoTE's published candidate-indexed score dictionary."""

    def __init__(self, path: Path, expected_candidates: int = 256):
        if not path.is_file():
            raise FileNotFoundError(f"candidate score file does not exist: {path}")
        payload = np.load(path, allow_pickle=True).item()
        if not isinstance(payload, dict) or not payload:
            raise CandidateAlignmentError(f"score file is not a non-empty dictionary: {path}")
        self.path = path
        self.expected_candidates = expected_candidates
        self._payload: Mapping[str, Any] = payload

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(self._payload.keys())

    def labels(self, token: str) -> CandidateLabels:
        if token not in self._payload:
            raise CandidateAlignmentError(f"token missing from score table: {token}")
        entry = self._payload[token]
        try:
            table = entry["trajectory_scores"][0]
        except (KeyError, IndexError, TypeError) as error:
            raise CandidateAlignmentError(
                f"{token}: malformed trajectory_scores entry"
            ) from error
        missing = [key for key in SCORE_FACTOR_KEYS if key not in table]
        if missing:
            raise CandidateAlignmentError(f"{token}: missing factor keys {missing}")
        factors = np.stack(
            [np.asarray(table[key], dtype=np.float32) for key in SCORE_FACTOR_KEYS], axis=-1
        )
        if "score" in table:
            score = np.asarray(table["score"], dtype=np.float32)
        else:
            score = pdms_from_factors(factors).astype(np.float32)
        labels = CandidateLabels(factors=factors, score=score)
        labels.validate(token, self.expected_candidates)
        return labels


def load_anchors(path: Path, expected_candidates: int = 256) -> npt.NDArray[np.float32]:
    if not path.is_file():
        raise FileNotFoundError(f"trajectory anchor file does not exist: {path}")
    anchors = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if anchors.shape != (expected_candidates, 8, 3):
        raise CandidateAlignmentError(
            f"expected anchor shape [{expected_candidates},8,3], got {anchors.shape}"
        )
    if not np.isfinite(anchors).all():
        raise CandidateAlignmentError("trajectory anchors contain NaN/Inf")
    if len({stable_array_hash(anchor) for anchor in anchors}) != expected_candidates:
        raise CandidateAlignmentError("trajectory anchor bank contains duplicate candidates")
    return anchors


def source_alignment_audit(wote_root: Path) -> dict[str, Any]:
    """Audit the exact source statements that bind score indices to anchors."""

    targets_path = wote_root / "navsim/agents/WoTE/WoTE_targets.py"
    model_path = wote_root / "navsim/agents/WoTE/WoTE_model.py"
    loss_path = wote_root / "navsim/agents/WoTE/WoTE_loss.py"
    for path in (targets_path, model_path, loss_path):
        if not path.is_file():
            raise CandidateAlignmentError(f"missing audited WoTE source: {path}")
    targets = targets_path.read_text(encoding="utf-8")
    model = model_path.read_text(encoding="utf-8")
    loss = loss_path.read_text(encoding="utf-8")
    generator_path = wote_root / "scripts/miscs/gen_multi_trajs_pdm_score.py"
    cache_processor_path = (
        wote_root / "navsim/planning/metric_caching/metric_cache_processor.py"
    )
    generator = (
        generator_path.read_text(encoding="utf-8")
        if generator_path.is_file()
        else ""
    )
    cache_processor = (
        cache_processor_path.read_text(encoding="utf-8")
        if cache_processor_path.is_file()
        else ""
    )
    evidence = {
        "target_reads_candidate_index_directly": bool(
            re.search(r"sim_reward_dict\[token\]\['trajectory_scores'\]\[0\]", targets)
        ),
        "target_stacks_factor_arrays_without_reindex": "np.vstack([sim_reward_dict_single[key] for key in self.sim_keys])"
        in targets,
        "training_scores_base_anchors": 'result["trajectory_anchors"] = self.trajectory_anchors'
        in model,
        "loss_compares_labels_to_base_anchor_axis": 'trajectory_anchors = predictions["trajectory_anchors"]'
        in loss,
        "test_outputs_offsets": "trajectory_anchors = self.trajectory_anchors + offset" in model,
    }
    base_alignment = all(
        evidence[key]
        for key in (
            "target_reads_candidate_index_directly",
            "target_stacks_factor_arrays_without_reindex",
            "training_scores_base_anchors",
            "loss_compares_labels_to_base_anchor_axis",
        )
    )
    result = {
        "wote_root": str(wote_root),
        "score_alignment_domain": "base_anchors" if base_alignment else "unresolved",
        "offset_label_mismatch_risk": bool(base_alignment and evidence["test_outputs_offsets"]),
        "gate_requires_base_anchors": bool(base_alignment),
        "evidence": evidence,
        "score_generation_horizon_audit": {
            "published_generator_source_present": generator_path.is_file(),
            "published_generator_sets_eight_second_horizon": all(
                snippet in generator
                for snippet in (
                    "num_horizons = 8",
                    "proposal_sampling.num_poses = int(num_horizons * 10)",
                )
            ),
            "published_generator_proposal_num_poses": PUBLISHED_SCORE_PROPOSAL_NUM_POSES,
            "default_cache_source_present": cache_processor_path.is_file(),
            "default_cache_proposal_num_poses": DEFAULT_METRIC_CACHE_PROPOSAL_NUM_POSES,
            "default_cache_future_num_poses": DEFAULT_METRIC_CACHE_FUTURE_NUM_POSES,
            "published_generator_default_cache_conflict": all(
                snippet in generator or snippet in cache_processor
                for snippet in (
                    "num_horizons = 8",
                    "proposal_traj_num_poses: int = 40",
                    "future_traj_num_poses: int = 50",
                )
            ),
            "upstream_issue": "https://github.com/liyingyanUCAS/WoTE/issues/16",
        },
        "pass": bool(base_alignment and evidence["test_outputs_offsets"]),
    }
    if not result["pass"]:
        raise CandidateAlignmentError(f"WoTE source alignment is unresolved: {result}")
    return result


def sha1_sorted_tokens(tokens: Iterable[str]) -> list[str]:
    unique = set(tokens)
    if not unique or "" in unique:
        raise ValueError("scene tokens must be unique, non-empty strings")
    return sorted(unique, key=lambda token: (hashlib.sha1(token.encode("utf-8")).hexdigest(), token))


def build_fixed_splits(
    tokens: Iterable[str],
    requested: tuple[int, int, int] = (8192, 1024, 2048),
) -> dict[str, list[str]]:
    ordered = sha1_sorted_tokens(tokens)
    requested_total = sum(requested)
    if len(ordered) >= requested_total:
        train_count, val_count, test_count = requested
    else:
        train_count = int(np.floor(len(ordered) * 0.70))
        val_count = int(np.floor(len(ordered) * 0.10))
        test_count = len(ordered) - train_count - val_count
        if min(train_count, val_count, test_count) <= 0:
            raise ValueError(f"insufficient scenes for 70/10/20 fallback: {len(ordered)}")
    train_end = train_count
    val_end = train_end + val_count
    test_end = val_end + test_count
    splits = {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:test_end],
    }
    flattened = [token for name in ("train", "val", "test") for token in splits[name]]
    if len(flattened) != len(set(flattened)):
        raise AssertionError("fixed splits overlap")
    return splits


def write_fixed_splits(splits: Mapping[str, Sequence[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False) if not output_dir.exists() else None
    for name in ("train", "val", "test"):
        path = output_dir / f"{name}_tokens.txt"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite split: {path}")
        tokens = list(splits[name])
        path.write_text("".join(f"{token}\n" for token in tokens), encoding="utf-8")


@dataclass(frozen=True)
class AlignmentRow:
    scene_token: str
    candidate_index: int
    factor: str
    precomputed: float
    recomputed: float
    absolute_error: float
    trajectory_hash: str


def sample_audit_indices(
    tokens: Sequence[str], scenes: int = 20, candidates: int = 10, seed: int = 20260827
) -> dict[str, npt.NDArray[np.int64]]:
    if len(tokens) < scenes:
        raise CandidateAlignmentError(
            f"alignment requires at least {scenes} scenes, got {len(tokens)}"
        )
    ordered = sha1_sorted_tokens(tokens)[:scenes]
    result: dict[str, npt.NDArray[np.int64]] = {}
    for token in ordered:
        token_seed = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) ^ seed
        rng = np.random.default_rng(token_seed)
        result[token] = np.sort(rng.choice(256, size=candidates, replace=False))
    return result


def audit_alignment(
    anchors: npt.NDArray[np.float32],
    score_table: CandidateScoreTable,
    tokens: Sequence[str],
    recompute: Callable[[str, npt.NDArray[np.float32]], CandidateLabels],
    tolerance: float = 1e-6,
) -> tuple[list[AlignmentRow], dict[str, Any]]:
    sampled = sample_audit_indices(tokens)
    rows: list[AlignmentRow] = []
    mismatched_candidates: set[tuple[str, int]] = set()
    candidate_errors: list[float] = []
    for token, indices in sampled.items():
        precomputed = score_table.labels(token)
        recomputed = recompute(token, anchors)
        recomputed.validate(token)
        for candidate_index in indices:
            per_candidate_errors: list[float] = []
            values = list(zip(FACTOR_NAMES, precomputed.factors[candidate_index], recomputed.factors[candidate_index]))
            values.append(
                ("score", precomputed.score[candidate_index], recomputed.score[candidate_index])
            )
            for factor, left, right in values:
                error = abs(float(left) - float(right))
                rows.append(
                    AlignmentRow(
                        scene_token=token,
                        candidate_index=int(candidate_index),
                        factor=factor,
                        precomputed=float(left),
                        recomputed=float(right),
                        absolute_error=error,
                        trajectory_hash=stable_array_hash(anchors[candidate_index]),
                    )
                )
                per_candidate_errors.append(error)
            maximum = max(per_candidate_errors)
            candidate_errors.append(maximum)
            if maximum > tolerance:
                mismatched_candidates.add((token, int(candidate_index)))
    errors = np.asarray([row.absolute_error for row in rows], dtype=np.float64)
    summary = {
        "audited_scenes": len(sampled),
        "audited_candidates_per_scene": len(next(iter(sampled.values()))),
        "audited_factor_values": len(rows),
        "maximum_absolute_error": float(errors.max()),
        "mean_absolute_error": float(errors.mean()),
        "mismatched_candidate_fraction": len(mismatched_candidates) / len(candidate_errors),
        "tolerance": tolerance,
        "pass": len(mismatched_candidates) == 0,
        "alignment_domain": "base_anchors",
    }
    return rows, summary


def official_recompute_factory(
    wote_root: Path,
    metric_cache_root: Path,
    proposal_num_poses: int = DEFAULT_METRIC_CACHE_PROPOSAL_NUM_POSES,
) -> Callable[[str, npt.NDArray[np.float32]], CandidateLabels]:
    if proposal_num_poses <= 0:
        raise ValueError("proposal_num_poses must be positive")
    sys.path.insert(0, str(wote_root))
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score_multi_trajs
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cache_loader = MetricCacheLoader(metric_cache_root)
    sampling = TrajectorySampling(num_poses=proposal_num_poses, interval_length=0.1)
    simulator = PDMSimulator(sampling)
    scorer = PDMScorer(sampling)

    def recompute(token: str, anchors: npt.NDArray[np.float32]) -> CandidateLabels:
        if token not in cache_loader.metric_cache_paths:
            raise CandidateAlignmentError(f"metric cache missing audited token: {token}")
        with lzma.open(cache_loader.metric_cache_paths[token], "rb") as stream:
            metric_cache = pickle.load(stream)
        result = pdm_score_multi_trajs(
            metric_cache=metric_cache,
            model_trajectory_list=anchors,
            future_sampling=sampling,
            simulator=simulator,
            scorer=scorer,
        )
        factors = np.stack(
            [
                result.no_at_fault_collisions,
                result.drivable_area_compliance,
                result.ego_progress,
                result.time_to_collision_within_bound,
                result.comfort,
            ],
            axis=-1,
        ).astype(np.float32)
        return CandidateLabels(factors=factors, score=np.asarray(result.score, dtype=np.float32))

    return recompute


def extract_selected_indices(cache_root: Path) -> dict[str, int]:
    selected: dict[str, int] = {}
    reader = FeatureShardReader(cache_root)
    for sidecar, arrays in reader.iter_shards():
        values = np.asarray(arrays["selected_index"], dtype=np.int64)
        records = sidecar["records"]
        if values.shape != (len(records),):
            raise CandidateAlignmentError(
                f"selected_index shape mismatch in shard {sidecar['shard_index']}: {values.shape}"
            )
        for record, value in zip(records, values):
            index = int(value)
            if not 0 <= index < 256:
                raise CandidateAlignmentError(f"invalid selected candidate {index}")
            selected[record["scene_token"]] = index
    return selected


def candidate_oracle(
    score_table: CandidateScoreTable,
    selected_indices: Mapping[str, int],
    tokens: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for token in tokens:
        if token not in selected_indices:
            raise CandidateAlignmentError(f"selected index missing token: {token}")
        labels = score_table.labels(token)
        selected_index = int(selected_indices[token])
        if not 0 <= selected_index < 256:
            raise CandidateAlignmentError(f"{token}: invalid selected index {selected_index}")
        oracle_index = int(np.argmax(labels.score))
        selected_score = float(labels.score[selected_index])
        oracle_score = float(labels.score[oracle_index])
        regret = oracle_score - selected_score
        rank = int(candidate_ranks(labels.score[None])[0, selected_index])
        row: dict[str, Any] = {
            "scene_token": token,
            "selected_index": selected_index,
            "oracle_index": oracle_index,
            "selected_score_raw": selected_score,
            "selected_score_points": selected_score * 100.0,
            "oracle_score_raw": oracle_score,
            "oracle_score_points": oracle_score * 100.0,
            "regret_raw": regret,
            "regret_points": regret * 100.0,
            "selected_rank": rank,
            "has_better_candidate": regret > 0,
            "improvement_gt_0_01": regret > 0.01,
            "improvement_gt_0_02": regret > 0.02,
            "improvement_gt_0_05": regret > 0.05,
            "failed_selected": selected_score == 0.0,
            "recovered_positive": selected_score == 0.0 and oracle_score > 0.0,
        }
        for factor_index, factor_name in enumerate(FACTOR_NAMES):
            row[f"selected_{factor_name}"] = float(labels.factors[selected_index, factor_index])
            row[f"oracle_ceiling_{factor_name}"] = float(labels.factors[:, factor_index].max())
        rows.append(row)
    frame = pd.DataFrame(rows)
    failed_count = int(frame["failed_selected"].sum())
    recovery_rate = (
        float(frame["recovered_positive"].sum() / failed_count) if failed_count else 0.0
    )
    mean_gap = float(frame["regret_raw"].mean())
    gt_001 = float(frame["improvement_gt_0_01"].mean())
    g1_pass = (mean_gap >= 0.020 or gt_001 >= 0.20) and recovery_rate >= 0.15
    interval = paired_scene_bootstrap(
        frame["oracle_score_raw"].to_numpy(),
        frame["selected_score_raw"].to_numpy(),
    )
    summary: dict[str, Any] = {
        "scene_count": len(frame),
        "selected_score_raw": float(frame["selected_score_raw"].mean()),
        "selected_score_points": float(frame["selected_score_points"].mean()),
        "oracle_score_raw": float(frame["oracle_score_raw"].mean()),
        "oracle_score_points": float(frame["oracle_score_points"].mean()),
        "oracle_gap_raw": mean_gap,
        "oracle_gap_points": mean_gap * 100.0,
        "mean_selected_rank": float(frame["selected_rank"].mean()),
        "scenes_with_better_candidate_fraction": float(frame["has_better_candidate"].mean()),
        "improvement_gt_0_01_fraction": gt_001,
        "improvement_gt_0_02_fraction": float(frame["improvement_gt_0_02"].mean()),
        "improvement_gt_0_05_fraction": float(frame["improvement_gt_0_05"].mean()),
        "failed_scene_count": failed_count,
        "recovered_positive_count": int(frame["recovered_positive"].sum()),
        "failed_scene_recovery_fraction": recovery_rate,
        "paired_scene_bootstrap_95ci": asdict(interval),
        "gate_g1_pass": g1_pass,
    }
    for factor_name in FACTOR_NAMES:
        summary[f"selected_{factor_name}"] = float(frame[f"selected_{factor_name}"].mean())
        summary[f"oracle_ceiling_{factor_name}"] = float(
            frame[f"oracle_ceiling_{factor_name}"].mean()
        )
    return frame, summary


def _read_tokens(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError(f"token file must be unique and non-empty: {path}")
    return tokens


def _write_alignment_csv(path: Path, rows: Sequence[AlignmentRow]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite alignment CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("source-audit")
    source.add_argument("--wote-root", type=Path, required=True)
    source.add_argument("--output", type=Path)

    splits = commands.add_parser("build-splits")
    splits.add_argument("--score-path", type=Path, required=True)
    splits.add_argument("--output-dir", type=Path, required=True)

    alignment = commands.add_parser("audit")
    alignment.add_argument("--wote-root", type=Path, required=True)
    alignment.add_argument("--anchor-path", type=Path, required=True)
    alignment.add_argument("--score-path", type=Path, required=True)
    alignment.add_argument("--metric-cache-root", type=Path, required=True)
    alignment.add_argument("--tokens", type=Path, required=True)
    alignment.add_argument("--output-csv", type=Path, required=True)
    alignment.add_argument("--output-summary", type=Path, required=True)
    alignment.add_argument("--tolerance", type=float, default=1e-6)
    alignment.add_argument(
        "--proposal-num-poses",
        type=int,
        default=DEFAULT_METRIC_CACHE_PROPOSAL_NUM_POSES,
        help="Explicit evaluator horizon at 10 Hz; no automatic horizon fallback.",
    )

    selected = commands.add_parser("selected-from-cache")
    selected.add_argument("--cache-root", type=Path, required=True)
    selected.add_argument("--output", type=Path, required=True)

    oracle = commands.add_parser("oracle")
    oracle.add_argument("--score-path", type=Path, required=True)
    oracle.add_argument("--selected-json", type=Path, required=True)
    oracle.add_argument("--tokens", type=Path, required=True)
    oracle.add_argument("--output-csv", type=Path, required=True)
    oracle.add_argument("--output-summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "source-audit":
        result = source_alignment_audit(args.wote_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.output:
            atomic_write_json(args.output, result)
        return 0
    if args.command == "build-splits":
        table = CandidateScoreTable(args.score_path)
        splits = build_fixed_splits(table.tokens)
        write_fixed_splits(splits, args.output_dir)
        print(json.dumps({name: len(tokens) for name, tokens in splits.items()}, sort_keys=True))
        return 0
    if args.command == "audit":
        source_audit = source_alignment_audit(args.wote_root)
        horizon_audit = source_audit["score_generation_horizon_audit"]
        anchors = load_anchors(args.anchor_path)
        table = CandidateScoreTable(args.score_path)
        tokens = _read_tokens(args.tokens)
        recompute = official_recompute_factory(
            args.wote_root,
            args.metric_cache_root,
            proposal_num_poses=args.proposal_num_poses,
        )
        rows, summary = audit_alignment(
            anchors, table, tokens, recompute, tolerance=args.tolerance
        )
        summary.update(
            {
                "recompute_proposal_num_poses": args.proposal_num_poses,
                "recompute_interval_seconds": 0.1,
                "published_score_generator_proposal_num_poses": horizon_audit[
                    "published_generator_proposal_num_poses"
                ],
                "default_metric_cache_proposal_num_poses": horizon_audit[
                    "default_cache_proposal_num_poses"
                ],
                "default_metric_cache_future_num_poses": horizon_audit[
                    "default_cache_future_num_poses"
                ],
                "published_generator_default_cache_conflict": horizon_audit[
                    "published_generator_default_cache_conflict"
                ],
                "upstream_horizon_issue": horizon_audit["upstream_issue"],
            }
        )
        _write_alignment_csv(args.output_csv, rows)
        atomic_write_json(args.output_summary, summary)
        if not summary["pass"]:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 4
        return 0
    if args.command == "selected-from-cache":
        atomic_write_json(args.output, extract_selected_indices(args.cache_root))
        return 0
    if args.command == "oracle":
        table = CandidateScoreTable(args.score_path)
        selected = json.loads(args.selected_json.read_text(encoding="utf-8"))
        tokens = _read_tokens(args.tokens)
        frame, summary = candidate_oracle(table, selected, tokens)
        if args.output_csv.exists():
            raise FileExistsError(f"refusing to overwrite oracle CSV: {args.output_csv}")
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_csv, index=False)
        atomic_write_json(args.output_summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
