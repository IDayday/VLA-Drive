"""Validate small lossless feedback/pair JSON artifacts, without simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tools.action_effect.feedback_schema import FEEDBACK_SCHEMA, PAIR_SCHEMA, validate_feedback, validate_pair


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.max_bytes <= 64 * 1024 * 1024 or args.input.stat().st_size > args.max_bytes:
            raise ValueError("Explicit validation budget exceeded")
        value = json.loads(args.input.read_text())
        if value.get("schema") == FEEDBACK_SCHEMA:
            validate_feedback(value)
        elif value.get("schema") == PAIR_SCHEMA:
            validate_pair(value)
        else:
            raise ValueError("Unsupported feedback/pair schema")
        print(json.dumps({"status": "PASS", "schema": value["schema"], "identity": value["identity"],
                          "simulation_calls": 0}))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc),
                          "simulation_calls": 0}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
