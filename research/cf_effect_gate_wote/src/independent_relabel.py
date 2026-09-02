"""Independently recompute fixed-bank NAVSIM labels under one 4-second contract.

This module deliberately has no dependency on WoTE's published candidate-score
table.  The only candidate labels it can emit are fresh outputs from the pinned
NAVSIM evaluator.
"""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from .direct_rehab_contracts import AccessAuditLog, AccessPolicy
from .feature_store import atomic_write_json, sha256_file, stable_array_hash
from .independent_label_store import (
    CANDIDATE_COUNT,
    IndependentCandidateLabelWriter,
    IndependentLabelRecord,
    IndependentLabelScene,
    IndependentLabelStoreError,
    ScoreReconstructionError,
    SixFactorIndependentCandidateLabelWriter,
    SixFactorIndependentLabelScene,
    SixFactorScoreReconstructionError,
)
from .six_factor_metrics import SIX_FACTOR_ORDER


WOTE_COMMIT = "298957c128a91d41a1c6075bd0bb6e7e845e093f"
CHECKPOINT_SHA256 = "f5e73261cc55220d681bdfe2ce306a2f8e8cd555b10be51034e9b20e2967e53b"
CANDIDATE_BANK_SHA256 = (
    "44f64a763473c3a80482aaa3f78669445f56af40a1c00741a351c6c0650e758b"
)
PROPOSAL_NUM_POSES = 40
PROPOSAL_INTERVAL_SECONDS = 0.1
METRIC_CACHE_FUTURE_NUM_POSES = 50
CANDIDATE_WAYPOINTS = 8
CANDIDATE_INTERVAL_SECONDS = 0.5
EVALUATOR_FILES = {
    "pdm_score.py": Path("navsim/evaluate/pdm_score.py"),
    "pdm_scorer.py": Path(
        "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py"
    ),
    "pdm_simulator.py": Path(
        "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py"
    ),
    "metric_cache_processor.py": Path(
        "navsim/planning/metric_caching/metric_cache_processor.py"
    ),
}


class RelabelContractError(RuntimeError):
    """The fixed evaluator, asset, token, or candidate contract was violated."""


def read_fixed_tokens(path: Path, expected_count: int | None = None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"fixed token file does not exist: {path}")
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not tokens or any(not token for token in tokens):
        raise RelabelContractError(
            f"fixed token file is empty or has blank rows: {path}"
        )
    if len(tokens) != len(set(tokens)):
        raise RelabelContractError(f"fixed token file contains duplicates: {path}")
    if expected_count is not None and len(tokens) != expected_count:
        raise RelabelContractError(
            f"fixed token count mismatch: expected {expected_count}, got {len(tokens)}"
        )
    return tokens


def load_base_anchor_bank(path: Path) -> npt.NDArray[np.float32]:
    if not path.is_file():
        raise FileNotFoundError(f"base anchor bank does not exist: {path}")
    anchors = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    expected = (CANDIDATE_COUNT, CANDIDATE_WAYPOINTS, 3)
    if anchors.shape != expected:
        raise RelabelContractError(
            f"base anchor bank expected {expected}, got {anchors.shape}"
        )
    if not np.isfinite(anchors).all():
        raise RelabelContractError("base anchor bank contains NaN/Inf")
    return np.ascontiguousarray(anchors)


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_evaluator_contract(
    wote_root: Path,
    checkpoint_path: Path,
    anchor_path: Path,
) -> dict[str, Any]:
    """Build the immutable, path-portable evaluator identity."""

    actual_wote_commit = _git_revision(wote_root)
    if actual_wote_commit != WOTE_COMMIT:
        raise RelabelContractError(
            f"WoTE commit mismatch: expected {WOTE_COMMIT}, got {actual_wote_commit}"
        )
    actual_checkpoint = sha256_file(checkpoint_path)
    if actual_checkpoint != CHECKPOINT_SHA256:
        raise RelabelContractError(
            "checkpoint SHA256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, got {actual_checkpoint}"
        )
    anchors = load_base_anchor_bank(anchor_path)
    source_hashes: dict[str, str] = {}
    for name, relative_path in EVALUATOR_FILES.items():
        source_path = wote_root / relative_path
        if not source_path.is_file():
            raise RelabelContractError(f"missing evaluator source: {relative_path}")
        source_hashes[name] = sha256_file(source_path)
    return {
        "candidate_domain": "wote_base_anchors",
        "candidate_count": CANDIDATE_COUNT,
        "candidate_waypoints": CANDIDATE_WAYPOINTS,
        "candidate_interval_seconds": CANDIDATE_INTERVAL_SECONDS,
        "candidate_horizon_seconds": 4.0,
        "proposal_num_poses": PROPOSAL_NUM_POSES,
        "proposal_interval_seconds": PROPOSAL_INTERVAL_SECONDS,
        "metric_cache_future_num_poses": METRIC_CACHE_FUTURE_NUM_POSES,
        "trajectory_offsets": False,
        "label_source": "independent_navsim_recompute",
        "wote_commit": actual_wote_commit,
        "checkpoint_sha256": actual_checkpoint,
        "candidate_bank_sha256": sha256_file(anchor_path),
        "candidate_bank_logical_sha256": stable_array_hash(anchors),
        "navsim_commit_or_tree_hash": actual_wote_commit,
        "evaluator_source_hashes": source_hashes,
    }


def write_evaluator_contract(
    output: Path,
    wote_root: Path,
    checkpoint_path: Path,
    anchor_path: Path,
) -> Path:
    contract = build_evaluator_contract(wote_root, checkpoint_path, anchor_path)
    atomic_write_json(output, contract)
    return output


def build_six_factor_evaluator_contract(
    wote_root: Path,
    checkpoint_path: Path,
    anchor_path: Path,
) -> dict[str, Any]:
    """Build the immutable v2 evaluator identity without altering the v1 contract."""

    contract = build_evaluator_contract(wote_root, checkpoint_path, anchor_path)
    if contract["candidate_bank_sha256"] != CANDIDATE_BANK_SHA256:
        raise RelabelContractError(
            "candidate bank SHA256 mismatch: "
            f"expected {CANDIDATE_BANK_SHA256}, got {contract['candidate_bank_sha256']}"
        )
    contract.update(
        {
            "factor_order": list(SIX_FACTOR_ORDER),
            "score_formula": "NC*DAC*DDC*(5*EP+5*TTC+2*Comfort)/12",
            "score_reconstruction_tolerance": 1e-6,
            "raw_progress_saved": True,
            "raw_progress_in_score_formula": False,
            "progress_normalization_scope": "all_256_candidates_within_scene",
            "candidate_set_dependent_ep": True,
            "label_schema_version": "independent_wote_labels_4s_six_factor.v2",
        }
    )
    return contract


def write_six_factor_evaluator_contract(
    output: Path,
    wote_root: Path,
    checkpoint_path: Path,
    anchor_path: Path,
) -> Path:
    contract = build_six_factor_evaluator_contract(
        wote_root, checkpoint_path, anchor_path
    )
    atomic_write_json(output, contract)
    return output


def _result_arrays(
    result: Any,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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
    score = np.asarray(result.score, dtype=np.float32)
    return factors, score


def _six_factor_result_arrays(
    result: Any,
    scorer: Any,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32],]:
    """Extract DDC and normalized EP plus diagnostic, non-factor raw progress."""

    factors = np.stack(
        [
            result.no_at_fault_collisions,
            result.drivable_area_compliance,
            result.driving_direction_compliance,
            result.ego_progress,
            result.time_to_collision_within_bound,
            result.comfort,
        ],
        axis=-1,
    ).astype(np.float32)
    score = np.asarray(result.score, dtype=np.float32)
    if scorer._progress_raw is None:
        raise RelabelContractError("PDMScorer raw progress was not populated")
    raw_progress = np.asarray(scorer._progress_raw, dtype=np.float32).copy()
    if raw_progress.shape != (CANDIDATE_COUNT,):
        raise RelabelContractError(
            f"raw progress expected ({CANDIDATE_COUNT},), got {raw_progress.shape}"
        )
    if not np.isfinite(raw_progress).all():
        raise RelabelContractError("raw progress contains NaN/Inf")
    return factors, score, raw_progress


def _failure_payload(
    token: str,
    error: Exception,
    result: Any | None,
    metric_cache_path: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "FAIL",
        "gate": "G0-R",
        "final_verdict": "RELABEL_CONTRACT_FAILED",
        "scene_token": token,
        "error_type": type(error).__name__,
        "error": str(error),
        "completed_scenes": 0,
        "attempted_scenes": 1 if token != "<not-started>" else 0,
        "candidates_evaluated": (CANDIDATE_COUNT if result is not None else 0),
    }
    if metric_cache_path is not None and metric_cache_path.is_file():
        payload["metric_cache_sha256"] = sha256_file(metric_cache_path)
    if isinstance(error, ScoreReconstructionError):
        bad = error.mismatched_indices
        payload.update(
            {
                "reason": "required_five_factor_score_reconstruction_exceeds_tolerance",
                "reconstruction_tolerance": 1e-6,
                "max_absolute_error": error.max_absolute_error,
                "mean_absolute_error": error.mean_absolute_error,
                "mismatched_candidate_count": int(len(bad)),
                "mismatched_candidate_indices": bad.tolist(),
                "candidate_differences": [
                    {
                        "candidate_index": int(index),
                        "evaluator_score": float(error.evaluator_score[index]),
                        "reassembled_score": float(error.reassembled_score[index]),
                        "absolute_error": float(
                            abs(
                                error.reassembled_score[index]
                                - error.evaluator_score[index]
                            )
                        ),
                        "driving_direction_compliance": (
                            float(result.driving_direction_compliance[index])
                            if result is not None
                            else None
                        ),
                    }
                    for index in bad
                ],
            }
        )
    return payload


def _six_factor_failure_payload(
    token: str,
    error: Exception,
    metric_cache_path: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "FAIL",
        "gate": "G0-R2",
        "scene_token": token,
        "error_type": type(error).__name__,
        "error": str(error),
        "completed_scenes": 0,
        "attempted_scenes": 1 if token != "<not-started>" else 0,
    }
    if metric_cache_path is not None and metric_cache_path.is_file():
        payload["metric_cache_sha256"] = sha256_file(metric_cache_path)
    if isinstance(error, SixFactorScoreReconstructionError):
        bad = error.mismatched_indices
        payload.update(
            {
                "reason": "six_factor_score_reconstruction_exceeds_tolerance",
                "reconstruction_tolerance": 1e-6,
                "max_absolute_error": error.max_absolute_error,
                "mean_absolute_error": error.mean_absolute_error,
                "mismatched_candidate_count": int(len(bad)),
                "mismatched_candidate_indices": bad.tolist(),
                "candidate_differences": [
                    {
                        "candidate_index": int(index),
                        **{
                            name: float(error.factors[index, factor_index])
                            for factor_index, name in enumerate(SIX_FACTOR_ORDER)
                        },
                        "raw_progress": float(error.raw_progress[index]),
                        "evaluator_score": float(error.evaluator_score[index]),
                        "six_factor_score": float(error.reassembled_score[index]),
                        "absolute_error": float(
                            abs(
                                error.reassembled_score[index]
                                - error.evaluator_score[index]
                            )
                        ),
                    }
                    for index in bad
                ],
            }
        )
    return payload


def run_independent_relabel(
    *,
    wote_root: Path,
    metric_cache_root: Path,
    token_path: Path,
    anchor_path: Path,
    evaluator_contract_path: Path,
    output: Path,
    expected_scenes: int | None,
    shard_scenes: int,
) -> Path:
    """Evaluate every scene exactly once and finalize a deterministic label store."""

    if output.exists():
        raise FileExistsError(f"refusing existing relabel output: {output}")
    contract = json.loads(evaluator_contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("proposal_num_poses") != PROPOSAL_NUM_POSES
        or contract.get("proposal_interval_seconds") != PROPOSAL_INTERVAL_SECONDS
    ):
        raise RelabelContractError(
            "evaluator contract is not fixed at 40 poses x 0.1 s"
        )
    if contract.get("metric_cache_future_num_poses") != METRIC_CACHE_FUTURE_NUM_POSES:
        raise RelabelContractError("metric-cache future contract is not 50 poses")
    if contract.get("trajectory_offsets") is not False:
        raise RelabelContractError("trajectory offsets must be disabled")

    tokens = read_fixed_tokens(token_path, expected_count=expected_scenes)
    anchors = load_base_anchor_bank(anchor_path)
    candidate_bank_hash = stable_array_hash(anchors)
    if contract.get("candidate_bank_logical_sha256") != candidate_bank_hash:
        raise RelabelContractError("candidate bank differs from evaluator contract")
    contract_sha = sha256_file(evaluator_contract_path)

    sys.path.insert(0, str(wote_root))
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score_multi_trajs
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
        PDMSimulator,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    cache_loader = MetricCacheLoader(metric_cache_root)
    missing = [
        token for token in tokens if token not in cache_loader.metric_cache_paths
    ]
    if missing:
        raise RelabelContractError(
            f"metric cache is missing {len(missing)} fixed tokens; first={missing[0]}"
        )
    sampling = TrajectorySampling(
        num_poses=PROPOSAL_NUM_POSES,
        interval_length=PROPOSAL_INTERVAL_SECONDS,
    )
    simulator = PDMSimulator(sampling)
    scorer = PDMScorer(sampling)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise RelabelContractError("simulator/scorer sampling mismatch")

    scenes: list[IndependentLabelScene] = []
    current_token = "<not-started>"
    current_result: Any | None = None
    current_cache_path: Path | None = None
    try:
        for scene_number, token in enumerate(tokens, start=1):
            current_token = token
            current_result = None
            current_cache_path = Path(cache_loader.metric_cache_paths[token])
            if not current_cache_path.is_file():
                raise RelabelContractError(
                    f"metric-cache metadata points to a missing file: {current_cache_path}"
                )
            with lzma.open(current_cache_path, "rb") as stream:
                metric_cache = pickle.load(stream)
            current_result = pdm_score_multi_trajs(
                metric_cache=metric_cache,
                model_trajectory_list=anchors,
                future_sampling=sampling,
                simulator=simulator,
                scorer=scorer,
            )
            factors, score = _result_arrays(current_result)
            scene = IndependentLabelScene(
                record=IndependentLabelRecord(
                    scene_token=token,
                    candidate_bank_hash=candidate_bank_hash,
                    trajectory_hash=stable_array_hash(anchors),
                    metric_cache_sha256=sha256_file(current_cache_path),
                ),
                factors=factors,
                score=score,
                oracle_index=int(np.argmax(score)),
                candidate_indices=np.arange(CANDIDATE_COUNT, dtype=np.int64),
            )
            scene.validate(reconstruction_tolerance=1e-6)
            scenes.append(scene)
            print(
                f"[independent-relabel] {scene_number}/{len(tokens)} {token}",
                flush=True,
            )
    except Exception as error:
        output.mkdir(parents=True, exist_ok=False)
        payload = _failure_payload(
            current_token, error, current_result, current_cache_path
        )
        payload["completed_scenes"] = len(scenes)
        atomic_write_json(output / "failure.json", payload)
        raise

    writer = IndependentCandidateLabelWriter(
        output,
        candidate_bank_hash=candidate_bank_hash,
        evaluator_contract_sha256=contract_sha,
        shard_scenes=shard_scenes,
    )
    return writer.write(scenes)


def run_six_factor_relabel(
    *,
    wote_root: Path,
    metric_cache_root: Path,
    token_path: Path,
    anchor_path: Path,
    evaluator_contract_path: Path,
    output: Path,
    expected_scenes: int | None,
    shard_scenes: int,
    access_policy_path: Path | None = None,
    access_log_path: Path | None = None,
    access_phase: str = "legacy",
) -> Path:
    """Evaluate every complete 256-anchor set and write explicit v2 labels."""

    if output.exists():
        raise FileExistsError(f"refusing existing six-factor relabel output: {output}")
    contract = json.loads(evaluator_contract_path.read_text(encoding="utf-8"))
    required_contract = {
        "proposal_num_poses": PROPOSAL_NUM_POSES,
        "proposal_interval_seconds": PROPOSAL_INTERVAL_SECONDS,
        "metric_cache_future_num_poses": METRIC_CACHE_FUTURE_NUM_POSES,
        "trajectory_offsets": False,
        "factor_order": list(SIX_FACTOR_ORDER),
        "score_formula": "NC*DAC*DDC*(5*EP+5*TTC+2*Comfort)/12",
        "raw_progress_saved": True,
        "raw_progress_in_score_formula": False,
        "progress_normalization_scope": "all_256_candidates_within_scene",
        "candidate_set_dependent_ep": True,
        "label_schema_version": "independent_wote_labels_4s_six_factor.v2",
    }
    changed = {
        name: (contract.get(name), expected)
        for name, expected in required_contract.items()
        if contract.get(name) != expected
    }
    if changed:
        raise RelabelContractError(f"six-factor evaluator contract changed: {changed}")

    policy = AccessPolicy.load(access_policy_path) if access_policy_path else None
    tokens = (
        list(policy.read_token_file(token_path, access_phase))
        if policy is not None
        else read_fixed_tokens(token_path, expected_count=expected_scenes)
    )
    if expected_scenes is not None and len(tokens) != expected_scenes:
        raise RelabelContractError(
            f"fixed token count mismatch: expected {expected_scenes}, got {len(tokens)}"
        )
    if policy is not None and access_log_path is None:
        raise RelabelContractError("--access-log is required with --access-policy")
    access_audit = (
        AccessAuditLog(access_log_path, policy, access_phase)
        if policy is not None and access_log_path is not None
        else None
    )
    anchors = load_base_anchor_bank(anchor_path)
    physical_anchor_sha = sha256_file(anchor_path)
    if physical_anchor_sha != CANDIDATE_BANK_SHA256:
        raise RelabelContractError(
            "candidate bank SHA256 mismatch: "
            f"expected {CANDIDATE_BANK_SHA256}, got {physical_anchor_sha}"
        )
    candidate_bank_hash = stable_array_hash(anchors)
    if (
        contract.get("candidate_bank_logical_sha256") != candidate_bank_hash
        or contract.get("candidate_bank_sha256") != physical_anchor_sha
    ):
        raise RelabelContractError("candidate bank differs from six-factor contract")
    contract_sha = sha256_file(evaluator_contract_path)

    sys.path.insert(0, str(wote_root))
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score_multi_trajs
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
        PDMSimulator,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    cache_loader = MetricCacheLoader(metric_cache_root)
    missing = [
        token for token in tokens if token not in cache_loader.metric_cache_paths
    ]
    if missing:
        raise RelabelContractError(
            f"metric cache is missing {len(missing)} fixed tokens; first={missing[0]}"
        )
    sampling = TrajectorySampling(
        num_poses=PROPOSAL_NUM_POSES,
        interval_length=PROPOSAL_INTERVAL_SECONDS,
    )
    simulator = PDMSimulator(sampling)
    scorer = PDMScorer(sampling)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise RelabelContractError("simulator/scorer sampling mismatch")

    scenes: list[SixFactorIndependentLabelScene] = []
    current_token = "<not-started>"
    current_cache_path: Path | None = None
    try:
        for scene_number, token in enumerate(tokens, start=1):
            current_token = token
            if access_audit is not None:
                access_audit.record(token, "six_factor_label_generation")
            current_cache_path = Path(cache_loader.metric_cache_paths[token])
            if not current_cache_path.is_file():
                raise RelabelContractError(
                    f"metric-cache metadata points to a missing file: {current_cache_path}"
                )
            with lzma.open(current_cache_path, "rb") as stream:
                metric_cache = pickle.load(stream)

            # EP is candidate-set dependent: this call must always receive all 256
            # anchors together. Candidate chunking is intentionally unsupported.
            result = pdm_score_multi_trajs(
                metric_cache=metric_cache,
                model_trajectory_list=anchors,
                future_sampling=sampling,
                simulator=simulator,
                scorer=scorer,
            )
            factors, score, raw_progress = _six_factor_result_arrays(result, scorer)
            scene = SixFactorIndependentLabelScene(
                record=IndependentLabelRecord(
                    scene_token=token,
                    candidate_bank_hash=candidate_bank_hash,
                    trajectory_hash=stable_array_hash(anchors),
                    metric_cache_sha256=sha256_file(current_cache_path),
                ),
                factors=factors,
                score=score,
                raw_progress=raw_progress,
                oracle_index=int(np.argmax(score)),
                candidate_indices=np.arange(CANDIDATE_COUNT, dtype=np.int64),
            )
            scene.validate(reconstruction_tolerance=1e-6)
            scenes.append(scene)
            print(
                f"[six-factor-relabel] {scene_number}/{len(tokens)} {token}",
                flush=True,
            )
    except Exception as error:
        output.mkdir(parents=True, exist_ok=False)
        payload = _six_factor_failure_payload(current_token, error, current_cache_path)
        payload["completed_scenes"] = len(scenes)
        payload["candidates_evaluated"] = len(scenes) * CANDIDATE_COUNT
        atomic_write_json(output / "failure.json", payload)
        raise

    writer = SixFactorIndependentCandidateLabelWriter(
        output,
        candidate_bank_hash=candidate_bank_hash,
        evaluator_contract_sha256=contract_sha,
        shard_scenes=shard_scenes,
    )
    return writer.write(scenes)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    contract = commands.add_parser("write-contract")
    contract.add_argument("--wote-root", type=Path, required=True)
    contract.add_argument("--checkpoint", type=Path, required=True)
    contract.add_argument("--anchors", type=Path, required=True)
    contract.add_argument("--output", type=Path, required=True)

    six_contract = commands.add_parser("write-six-factor-contract")
    six_contract.add_argument("--wote-root", type=Path, required=True)
    six_contract.add_argument("--checkpoint", type=Path, required=True)
    six_contract.add_argument("--anchors", type=Path, required=True)
    six_contract.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--wote-root", type=Path, required=True)
    run.add_argument("--metric-cache-root", type=Path, required=True)
    run.add_argument("--tokens", type=Path, required=True)
    run.add_argument("--anchors", type=Path, required=True)
    run.add_argument("--evaluator-contract", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--expected-scenes", type=int)
    run.add_argument("--shard-scenes", type=int, default=16)

    six_run = commands.add_parser("run-six-factor")
    six_run.add_argument("--wote-root", type=Path, required=True)
    six_run.add_argument("--metric-cache-root", type=Path, required=True)
    six_run.add_argument("--tokens", type=Path, required=True)
    six_run.add_argument("--anchors", type=Path, required=True)
    six_run.add_argument("--evaluator-contract", type=Path, required=True)
    six_run.add_argument("--output", type=Path, required=True)
    six_run.add_argument("--expected-scenes", type=int)
    six_run.add_argument("--shard-scenes", type=int, default=16)
    six_run.add_argument("--access-policy", type=Path)
    six_run.add_argument("--access-log", type=Path)
    six_run.add_argument("--access-phase", default="legacy")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write-contract":
        path = write_evaluator_contract(
            args.output, args.wote_root, args.checkpoint, args.anchors
        )
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
        return 0
    if args.command == "write-six-factor-contract":
        path = write_six_factor_evaluator_contract(
            args.output, args.wote_root, args.checkpoint, args.anchors
        )
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
        return 0
    if args.expected_scenes is not None and args.expected_scenes <= 0:
        raise ValueError("--expected-scenes must be positive")
    if args.shard_scenes <= 0:
        raise ValueError("--shard-scenes must be positive")
    try:
        arguments = {
            "wote_root": args.wote_root,
            "metric_cache_root": args.metric_cache_root,
            "token_path": args.tokens,
            "anchor_path": args.anchors,
            "evaluator_contract_path": args.evaluator_contract,
            "output": args.output,
            "expected_scenes": args.expected_scenes,
            "shard_scenes": args.shard_scenes,
        }
        if args.command == "run-six-factor":
            arguments.update(
                {
                    "access_policy_path": args.access_policy,
                    "access_log_path": args.access_log,
                    "access_phase": args.access_phase,
                }
            )
        manifest = (
            run_six_factor_relabel(**arguments)
            if args.command == "run-six-factor"
            else run_independent_relabel(**arguments)
        )
    except (IndependentLabelStoreError, RelabelContractError) as error:
        gate = "G0-R2" if args.command == "run-six-factor" else "G0-R"
        print(f"{gate} relabel failed: {error}", file=sys.stderr)
        return 4
    print(json.dumps(json.loads(manifest.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
