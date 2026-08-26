#!/usr/bin/env python3
"""Collect trained components, bank reports, and official navtest scores."""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--arm", choices=("off", "on"), required=True)
    parser.add_argument(
        "--generator-variant",
        choices=("frozen", "visual_unfrozen"),
        default="frozen",
    )
    parser.add_argument(
        "--generator-stage-id", default="qwen_register64_generator"
    )
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--drivor-checkpoint", required=True)
    parser.add_argument("--suprim-checkpoint")
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--pdms-results-dir", required=True)
    parser.add_argument("--epdms-results-dir", required=True)
    parser.add_argument("--expected-scenarios", type=int, required=True)
    parser.add_argument("--navtest-datalist", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return None
    return {
        "epochs_recorded": len(rows),
        "wall_time_seconds": sum(float(row.get("epoch_seconds", 0.0)) for row in rows),
        "last": rows[-1],
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    generator = Path(args.generator_checkpoint).expanduser().resolve()
    drivor = Path(args.drivor_checkpoint).expanduser().resolve()
    checkpoints = {"generator": generator, "drivor": drivor}
    if args.arm == "on":
        if not args.suprim_checkpoint:
            raise ValueError("ON arm requires --suprim-checkpoint")
        checkpoints["suprim_dynamic"] = (
            Path(args.suprim_checkpoint).expanduser().resolve()
        )
    elif args.suprim_checkpoint:
        raise ValueError("OFF arm must not provide a DriveSuprim checkpoint")
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    prediction_manifest_path = (
        Path(args.prediction_dir).expanduser().resolve() / "prediction_manifest.json"
    )
    if not prediction_manifest_path.is_file():
        raise FileNotFoundError(prediction_manifest_path)
    prediction_manifest = _json(prediction_manifest_path)
    if prediction_manifest.get("num_predictions") != args.expected_scenarios:
        raise RuntimeError(
            "prediction manifest does not cover the complete navtest set"
        )
    for component_name, checkpoint_path in checkpoints.items():
        manifest_name = (
            "suprim" if component_name == "suprim_dynamic" else component_name
        )
        manifest_checkpoint = prediction_manifest.get("checkpoints", {}).get(
            manifest_name, {}
        )
        if manifest_checkpoint.get("sha256") != _sha256(checkpoint_path):
            raise RuntimeError(
                f"prediction manifest {manifest_name} checkpoint hash mismatch"
            )

    bank_root = Path(args.bank_root).expanduser().resolve()
    bank_reports = {}
    for split in ("train", "val"):
        report = bank_root / split / "candidate_bank_report.json"
        manifest = bank_root / split / "manifest.json"
        if not report.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"completed {split} candidate bank is missing")
        bank_reports[split] = {"report": _json(report), "manifest": _json(manifest)}

    navtest_tokens = _json(Path(args.navtest_datalist).expanduser().resolve())
    pdms = validate_score_directory(
        args.pdms_results_dir,
        protocol="pdms",
        expected_scenarios=args.expected_scenarios,
        expected_tokens=navtest_tokens,
    )
    epdms = validate_score_directory(
        args.epdms_results_dir,
        protocol="epdms",
        expected_scenarios=args.expected_scenarios,
        expected_tokens=navtest_tokens,
    )
    project_root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    stages_root = run_root / "stages"
    training = {
        "generator": _jsonl_summary(
            stages_root / args.generator_stage_id / "metrics.jsonl"
        ),
        "drivor": _jsonl_summary(
            stages_root / "register64_drivor_scorer" / "metrics.jsonl"
        ),
        "suprim_dynamic": (
            _jsonl_summary(
                stages_root / "register64_drivor_suprim_dynamic" / "metrics.jsonl"
            )
            if args.arm == "on"
            else None
        ),
    }
    result = {
        "schema_version": 1,
        "arm": args.arm,
        "generator_variant": args.generator_variant,
        "repository_commit": commit,
        "run_root": str(run_root),
        "architecture": (
            "Qwen -> Q-Former16 -> Register64 -> DrivoR"
            + (" -> DriveSuprim dynamic Top-32" if args.arm == "on" else "")
        ),
        "selector_training_target": "single NAVSIM-v2 candidate-bank metric schema",
        "separate_pdms_epdms_scorers": False,
        "checkpoints": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in checkpoints.items()
        },
        "candidate_banks": bank_reports,
        "predictions": prediction_manifest,
        "training": training,
        "official_navtest": {"pdms_v1_1": pdms, "epdms_v2": epdms},
    }
    _atomic_write(
        output_dir / "summary.json", json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=(
            "arm",
            "generator_variant",
            "protocol",
            "score",
            "score_percent",
            "num_scenarios",
            "official_csv",
            "repository_commit",
        ),
    )
    writer.writeheader()
    for protocol, values in (("pdms_v1_1", pdms), ("epdms_v2_navtest", epdms)):
        writer.writerow(
            {
                "arm": args.arm,
                "generator_variant": args.generator_variant,
                "protocol": protocol,
                "score": values["score"],
                "score_percent": values["score_percent"],
                "num_scenarios": values["num_scenarios"],
                "official_csv": values["csv"],
                "repository_commit": commit,
            }
        )
    _atomic_write(output_dir / "summary.csv", csv_buffer.getvalue())
    lines = [
        f"# Register64 {args.arm.upper()} complete-pipeline result",
        "",
        f"- Commit: `{commit}`",
        f"- Generator variant: `{args.generator_variant}`",
        f"- Architecture: {result['architecture']}",
        f"- Official navtest PDMS (v1.1): **{pdms['score_percent']:.3f}**",
        f"- Official navtest EPDMS (v2): **{epdms['score_percent']:.3f}**",
        f"- Evaluated scenarios: {args.expected_scenarios}",
        "- Scorer policy: one NAVSIM-v2-trained selector checkpoint is frozen and evaluated under both protocols.",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(
        f"- {name}: `{entry['path']}`" for name, entry in result["checkpoints"].items()
    )
    lines.extend(
        [
            f"- PDMS CSV: `{pdms['csv']}`",
            f"- EPDMS CSV: `{epdms['csv']}`",
            f"- Machine-readable summary: `{output_dir / 'summary.json'}`",
            f"- Stable score table: `{output_dir / 'summary.csv'}`",
            "",
        ]
    )
    _atomic_write(output_dir / "summary.md", "\n".join(lines))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
