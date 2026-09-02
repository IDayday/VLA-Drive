#!/usr/bin/env python3
"""Audit the two standalone VLM initializations used by formal PlanReg-WM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.EpisodeDrive.formal_initialization import (
    audit_vlm_checkpoint,
    compare_formal_vlm_audits,
)


def _compact(audit: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(audit)
    result.pop("token_id_map", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Prepared Base InternVL directory")
    parser.add_argument("--vqa", required=True, help="Driving-VQA InternVL directory")
    parser.add_argument("--output", required=True, help="JSON audit report")
    parser.add_argument(
        "--load-runtime-classes",
        action="store_true",
        help="Map each complete VLM on CPU and record concrete Python classes",
    )
    args = parser.parse_args()

    base = audit_vlm_checkpoint(
        args.base, variant="base", load_runtime_classes=args.load_runtime_classes
    )
    vqa = audit_vlm_checkpoint(
        args.vqa,
        variant="driving_vqa",
        load_runtime_classes=args.load_runtime_classes,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        pair = compare_formal_vlm_audits(base, vqa)
    except RuntimeError as error:
        report = {
            "schema_version": 1,
            "agent_checkpoint_loaded": False,
            "base": _compact(base),
            "driving_vqa": _compact(vqa),
            "pair": {"formal_pair_compatible": False, "error": str(error)},
        }
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    report = {
        "schema_version": 1,
        "agent_checkpoint_loaded": False,
        "base": _compact(base),
        "driving_vqa": _compact(vqa),
        "pair": pair,
    }
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
