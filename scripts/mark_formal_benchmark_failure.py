#!/usr/bin/env python3
"""Write a machine-readable failed formal-layout benchmark record."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--global-batch", required=True, type=int)
    parser.add_argument("--gpu-count", required=True, type=int)
    parser.add_argument("--per-gpu-batch", required=True, type=int)
    parser.add_argument("--scorer-processes-per-rank", required=True, type=int)
    parser.add_argument("--num-workers-per-rank", required=True, type=int)
    parser.add_argument("--deadlock", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        return
    payload = {
        "schema_version": 1,
        "layout": args.layout,
        "status": "failed",
        "exit_code": args.exit_code,
        "timed_optimizer_steps": 0,
        "gpu_count": args.gpu_count,
        "per_gpu_batch_size": args.per_gpu_batch,
        "global_batch_size": args.global_batch,
        "scorer_processes_per_rank": args.scorer_processes_per_rank,
        "num_workers_per_rank": args.num_workers_per_rank,
        "nonfinite_count": 0,
        "oom": False,
        "deadlock": bool(args.deadlock),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
