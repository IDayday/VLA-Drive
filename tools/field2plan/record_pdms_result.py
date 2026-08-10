#!/usr/bin/env python3
"""Atomically record one NAVSIM v1.1 PDMS result and rebuild live summaries.

The evaluator writes one CSV containing per-scenario rows followed by an
``average`` row.  This utility converts that artifact into a small immutable
JSON record, then regenerates long-form CSV and human-readable Markdown
summaries under an advisory file lock.  Multiple DLC evaluator processes may
call it concurrently.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
from typing import Iterable, Optional


SUMMARY_COLUMNS = (
    "updated_at_utc",
    "status",
    "protocol",
    "experiment",
    "checkpoint_step",
    "pdms",
    "scenario_count",
    "valid_scenarios",
    "failed_scenarios",
    "inference_world_size",
    "inference_seed",
    "checkpoint",
    "checkpoint_bytes",
    "prediction_dir",
    "result_csv",
    "evaluator_log",
    "error",
)


@dataclass(frozen=True)
class PdmsRecord:
    """One checkpoint evaluation record; PDMS is stored on the [0, 1] scale."""

    updated_at_utc: str
    status: str
    protocol: str
    experiment: str
    checkpoint_step: int
    pdms: Optional[float]
    scenario_count: int
    valid_scenarios: int
    failed_scenarios: int
    inference_world_size: int
    inference_seed: int
    checkpoint: str
    checkpoint_bytes: int
    prediction_dir: str
    result_csv: str
    evaluator_log: str
    error: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_key(experiment: str, checkpoint_step: int) -> str:
    safe_experiment = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment)
    return f"{safe_experiment}-step{checkpoint_step}"


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as output_file:
        output_file.write(value)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary, path)


def parse_pdms_csv(path: Path) -> tuple[float, int, int, int]:
    """Return ``(pdms, total, valid, failed)`` from a NAVSIM score CSV."""

    if not path.is_file():
        raise FileNotFoundError(f"PDMS result CSV does not exist: {path}")
    average_score: Optional[float] = None
    scenario_count = 0
    valid_scenarios = 0
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"token", "valid", "score"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"PDMS CSV lacks required columns {sorted(required)}: {path}"
            )
        for row in reader:
            token = row["token"]
            if token == "average":
                average_score = float(row["score"])
                continue
            scenario_count += 1
            if row["valid"].strip().lower() in {"true", "1"}:
                valid_scenarios += 1
    if average_score is None:
        raise ValueError(f"PDMS CSV has no average row: {path}")
    failed_scenarios = scenario_count - valid_scenarios
    return average_score, scenario_count, valid_scenarios, failed_scenarios


def _load_records(records_dir: Path) -> list[PdmsRecord]:
    records = []
    for path in records_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as input_file:
            records.append(PdmsRecord(**json.load(input_file)))
    return sorted(records, key=lambda item: (item.checkpoint_step, item.experiment))


def _format_csv(records: Iterable[PdmsRecord]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SUMMARY_COLUMNS)
    writer.writeheader()
    for record in records:
        row = asdict(record)
        row["pdms"] = "" if record.pdms is None else f"{record.pdms:.12f}"
        writer.writerow(row)
    return buffer.getvalue()


def _format_markdown(records: list[PdmsRecord]) -> str:
    completed = sum(record.status == "complete" for record in records)
    failed = sum(record.status == "failed" for record in records)
    steps = sorted({record.checkpoint_step for record in records})
    experiments = sorted({record.experiment for record in records})
    lookup = {
        (record.experiment, record.checkpoint_step): record for record in records
    }
    lines = [
        "# Field2Plan live NAVSIM v1.1 PDMS summary",
        "",
        f"Updated: {_utc_now()}",
        "",
        f"Completed: **{completed}** | Failed: **{failed}**",
        "",
    ]
    if not records:
        lines.extend(["No evaluation records yet.", ""])
        return "\n".join(lines)

    lines.append("All displayed scores are NAVSIM v1.1 **PDMS** on the 0–100 scale.")
    lines.append("")
    header = ["experiment", *[f"{step // 1000}k" for step in steps]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---", *["---:" for _ in steps]]) + " |")
    for experiment in experiments:
        values = []
        for step in steps:
            record = lookup.get((experiment, step))
            if record is None:
                values.append("—")
            elif record.status == "complete" and record.pdms is not None:
                values.append(f"{record.pdms * 100:.6f}")
            else:
                values.append("FAILED")
        lines.append("| " + " | ".join([experiment, *values]) + " |")

    lines.extend(
        [
            "",
            "## Records",
            "",
            "| status | experiment | step | PDMS | valid/total | result CSV |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for record in records:
        pdms = "—" if record.pdms is None else f"{record.pdms * 100:.6f}"
        result_csv = record.result_csv or "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    record.status,
                    record.experiment,
                    str(record.checkpoint_step),
                    pdms,
                    f"{record.valid_scenarios}/{record.scenario_count}",
                    result_csv,
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def record_result(
    *,
    summary_root: Path,
    status: str,
    protocol: str,
    experiment: str,
    checkpoint_step: int,
    inference_world_size: int,
    inference_seed: int,
    checkpoint: Path,
    prediction_dir: Path,
    evaluator_log: Path,
    result_csv: Optional[Path] = None,
    error: str = "",
) -> PdmsRecord:
    """Persist one result and atomically refresh ``summary.csv``/``.md``."""

    if status not in {"complete", "failed"}:
        raise ValueError("status must be 'complete' or 'failed'")
    if checkpoint_step <= 0 or inference_world_size <= 0 or inference_seed < 0:
        raise ValueError("step/world size must be positive and seed non-negative")

    if status == "complete":
        if result_csv is None:
            raise ValueError("complete records require result_csv")
        pdms, total, valid, failed = parse_pdms_csv(result_csv)
    else:
        pdms, total, valid, failed = None, 0, 0, 0

    checkpoint_bytes = checkpoint.stat().st_size if checkpoint.is_file() else 0
    record = PdmsRecord(
        updated_at_utc=_utc_now(),
        status=status,
        protocol=protocol,
        experiment=experiment,
        checkpoint_step=int(checkpoint_step),
        pdms=pdms,
        scenario_count=total,
        valid_scenarios=valid,
        failed_scenarios=failed,
        inference_world_size=int(inference_world_size),
        inference_seed=int(inference_seed),
        checkpoint=str(checkpoint.resolve()),
        checkpoint_bytes=checkpoint_bytes,
        prediction_dir=str(prediction_dir.resolve()),
        result_csv="" if result_csv is None else str(result_csv.resolve()),
        evaluator_log=str(evaluator_log.resolve()),
        error=error,
    )

    summary_root.mkdir(parents=True, exist_ok=True)
    records_dir = summary_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    lock_path = summary_root / ".summary.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        record_path = records_dir / f"{_safe_key(experiment, checkpoint_step)}.json"
        _atomic_write_text(record_path, json.dumps(asdict(record), indent=2) + "\n")
        records = _load_records(records_dir)
        _atomic_write_text(summary_root / "summary.csv", _format_csv(records))
        _atomic_write_text(summary_root / "summary.md", _format_markdown(records))
        _atomic_write_text(summary_root / "last_update_utc.txt", _utc_now() + "\n")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--status", choices=("complete", "failed"), required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--inference-world-size", type=int, required=True)
    parser.add_argument("--inference-seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--evaluator-log", type=Path, required=True)
    parser.add_argument("--result-csv", type=Path)
    parser.add_argument("--error", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = record_result(**vars(args))
    score = "NA" if record.pdms is None else f"{record.pdms * 100:.6f}"
    print(
        f"recorded status={record.status} experiment={record.experiment} "
        f"step={record.checkpoint_step} pdms={score}"
    )


if __name__ == "__main__":
    main()
