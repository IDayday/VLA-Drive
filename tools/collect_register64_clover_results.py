#!/usr/bin/env python3
"""Collect the complete CLOVER-PDMS training and official navtest result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.training.navtest_score_io import validate_score_directory


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--cycles-root", required=True)
    parser.add_argument("--num-cycles", type=int, required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--drivor-checkpoint", required=True)
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--model-selection", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--pdms-results-dir", required=True)
    parser.add_argument("--navtest-datalist", required=True)
    parser.add_argument("--expected-scenarios", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    args = _args()
    run_root = Path(args.run_root).expanduser().resolve()
    stage1_dir = Path(args.stage1_dir).expanduser().resolve()
    cycles_root = Path(args.cycles_root).expanduser().resolve()
    generator = Path(args.generator_checkpoint).expanduser().resolve()
    drivor = Path(args.drivor_checkpoint).expanduser().resolve()
    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    prediction_manifest = _json(prediction_dir / "prediction_manifest.json")
    model_selection = _json(Path(args.model_selection).expanduser().resolve())
    selected_pair = model_selection.get("selected", {})
    selected_generator = Path(
        selected_pair.get("generator_checkpoint", "")
    ).expanduser().resolve()
    selected_scorer = Path(
        selected_pair.get("scorer_checkpoint", "")
    ).expanduser().resolve()
    selected_bank_parent = Path(
        selected_pair.get("train_bank_root", "")
    ).expanduser().resolve().parent
    bank_root = Path(args.bank_root).expanduser().resolve()
    if selected_generator != generator or selected_scorer != drivor:
        raise RuntimeError("reported checkpoints differ from model selection")
    if selected_bank_parent != bank_root:
        raise RuntimeError("reported bank root differs from model selection")
    checkpoints = {
        "generator": _checkpoint(generator),
        "drivor_pdms": _checkpoint(drivor),
    }
    for name, path in (("generator", generator), ("drivor", drivor)):
        actual = prediction_manifest.get("checkpoints", {}).get(name, {}).get("sha256")
        if actual != _sha256(path):
            raise RuntimeError(f"prediction/checkpoint mismatch for {name}")
    if prediction_manifest.get("num_predictions") != args.expected_scenarios:
        raise RuntimeError("prediction manifest is not complete navtest")

    tokens = _json(Path(args.navtest_datalist).expanduser().resolve())
    pdms = validate_score_directory(
        args.pdms_results_dir,
        protocol="pdms",
        expected_scenarios=args.expected_scenarios,
        expected_tokens=tokens,
    )
    stage1_complete = _json(stage1_dir / "training_complete.json")
    stage1_metrics = _jsonl(stage1_dir / "metrics.jsonl")
    cycles = []
    for cycle in range(1, args.num_cycles + 1):
        stem = f"cycle_{cycle:02d}"
        cycle_root = cycles_root / stem
        scorer_dir = cycle_root / "scorer"
        generator_dir = cycle_root / "generator"
        scorer_complete = _json(scorer_dir / "training_complete.json")
        generator_complete = _json(generator_dir / "training_complete.json")
        scorer_metrics = _jsonl(scorer_dir / "metrics.jsonl")
        cycles.append(
            {
                "cycle": cycle,
                "bank_train_report": _json(
                    cycle_root / "candidate_bank" / "train" / "candidate_bank_report.json"
                ),
                "bank_val_report": _json(
                    cycle_root / "candidate_bank" / "val" / "candidate_bank_report.json"
                ),
                "bank_selection_report": _json(
                    cycle_root
                    / "candidate_bank"
                    / "selection"
                    / "candidate_bank_report.json"
                ),
                "scorer": scorer_complete,
                "scorer_validation": scorer_metrics[-1] if scorer_metrics else None,
                "generator": generator_complete,
            }
        )
    closing_root = cycles_root / "closing_critic"
    closing = {
        "bank_train_report": _json(
            closing_root / "candidate_bank" / "train" / "candidate_bank_report.json"
        ),
        "bank_val_report": _json(
            closing_root / "candidate_bank" / "val" / "candidate_bank_report.json"
        ),
        "bank_selection_report": _json(
            closing_root
            / "candidate_bank"
            / "selection"
            / "candidate_bank_report.json"
        ),
        "scorer": _json(closing_root / "scorer" / "training_complete.json"),
        "scorer_validation": _jsonl(closing_root / "scorer" / "metrics.jsonl")[-1],
    }
    project_root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    result = {
        "schema_version": 1,
        "method": "register64_pdms_first_closed_loop",
        "repository_commit": commit,
        "run_root": str(run_root),
        "architecture": (
            "Qwen -> Global Q-Former16 -> Register64 -> detached DrivoR "
            "submetrics + direct PDMS value -> calibrated hybrid argmax"
        ),
        "training": {
            "stage1": {
                "completion": stage1_complete,
                "epochs": stage1_metrics,
            },
            "alternating_cycles": cycles,
            "closing_critic": closing,
        },
        "checkpoints": checkpoints,
        "model_selection": model_selection,
        "final_candidate_bank": str(bank_root),
        "predictions": prediction_manifest,
        "official_navtest": {"pdms_v1_1": pdms},
        "drivesuprim_promoted": False,
        "drivesuprim_reason": (
            "The prior dynamic Top-32 arm had negative true-PDMS refinement gain; "
            "it remains an ablation and is not on the promoted best-PDMS path."
        ),
    }
    output = Path(args.output_dir).expanduser().resolve()
    _atomic_text(
        output / "summary.json", json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "method",
            "protocol",
            "score",
            "score_percent",
            "num_scenarios",
            "official_csv",
            "repository_commit",
        ),
    )
    writer.writeheader()
    writer.writerow(
        {
            "method": "register64_pdms_first_closed_loop",
            "protocol": "pdms_v1_1",
            "score": pdms["score"],
            "score_percent": pdms["score_percent"],
            "num_scenarios": pdms["num_scenarios"],
            "official_csv": pdms["csv"],
            "repository_commit": commit,
        }
    )
    _atomic_text(output / "summary.csv", buffer.getvalue())
    best_stage1 = stage1_complete.get("selection_metric_value")
    final_val = model_selection["selected"]
    lines = [
        "# Register64 PDMS-First Closed-Loop result",
        "",
        f"- Commit: `{commit}`",
        f"- Official navtest PDMS: **{pdms['score_percent']:.3f}**",
        f"- Evaluated scenes: {pdms['num_scenarios']}",
        f"- Stage-1 best selected validation PDMS: `{best_stage1}`",
        f"- Selected cycle/pair: `{final_val.get('label')}`",
        f"- Selected-pair untouched-holdout PDMS: `{final_val.get('selected_true_pdms')}`",
        f"- Selected-pair PDMS 95% LCB: `{final_val.get('selected_true_pdms_lcb95')}`",
        f"- Paired selection decision: `{final_val.get('paired_selection')}`",
        f"- Selected-pair untouched-holdout regret: `{final_val.get('regret')}`",
        f"- Calibrated selector alpha: `{final_val.get('selector_alpha')}`",
        f"- Final generator: `{generator}`",
        f"- Final PDMS scorer: `{drivor}`",
        "- DriveSuprim: not promoted; retained as an isolated ablation.",
        "",
        "The machine-readable cycle-by-cycle bank/scorer/generator records are in "
        f"`{output / 'summary.json'}`.",
        "",
    ]
    _atomic_text(output / "summary.md", "\n".join(lines))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
