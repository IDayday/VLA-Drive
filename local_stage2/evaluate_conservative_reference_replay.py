"""Re-evaluate a conservative-reference checkpoint on locked log-heldout replay."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from local_stage2.train_conservative_reference_scorer import (
    _split_indices,
    _threshold_specs,
    collect_reference_predictions,
    evaluate_reference_predictions,
)
from local_stage2.train_independent_scorer import (
    ReplaySource,
    _atomic_json_dump,
    _sha256,
    load_replay_sources,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    ConservativeReferenceConfig,
    IndependentConservativeReferenceRanker,
    IndependentRankerConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("NAME", "FEATURE_ROOT", "LABEL_ROOT"),
        required=True,
    )
    parser.add_argument("--private-observation-root", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--gain-quantile-grid", default="0,1")
    parser.add_argument("--lcb-gain-grid", default="-0.01,0.0,0.0025,0.005,0.01")
    parser.add_argument("--safety-probability-grid", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--improvement-probability-grid", default="0.3,0.5,0.7")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("architecture") != "IndependentConservativeReferenceRanker":
        raise RuntimeError("artifact architecture mismatch")
    sources = [
        ReplaySource(name, Path(feature_root), Path(label_root))
        for name, feature_root, label_root in args.source
    ]
    data, source_lineage = load_replay_sources(
        sources, private_observation_root=args.private_observation_root
    )
    _train, validation, split_lineage = _split_indices(data, args)
    ranker_config = IndependentRankerConfig(**artifact["ranker_config"])
    reference_config = ConservativeReferenceConfig(**artifact["reference_config"])
    model = IndependentConservativeReferenceRanker(
        ranker_config, reference_config
    )
    model.load_state_dict(artifact["state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    prediction = collect_reference_predictions(
        model, data, validation, device, args.eval_batch_size
    )
    logs = [data.physical_logs[index] for index in validation]
    metrics = evaluate_reference_predictions(
        prediction,
        logs,
        _threshold_specs(args),
        args.seed,
        args.bootstrap_replicates,
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": _sha256(args.artifact),
        "scene_count": len(validation),
        "physical_log_count": len(set(logs)),
        "split_lineage": split_lineage,
        "source_lineage": source_lineage,
        "numeric_base_score_used_as_model_input": False,
        "base_selection_index_used_as_reference": True,
        "future_or_evaluator_input": False,
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json_dump(payload, args.output_dir / "evaluation.json")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
