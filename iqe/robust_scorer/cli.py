import argparse
import json
import sys
from .config import load, CELLS


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded from-scratch fixed-pool IQE Scorer experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("bind", "audit-data", "train"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "train":
            p.add_argument("--cell", required=True, choices=CELLS)
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--resume", nargs="?", const="auto")
            p.add_argument("--stop-after", type=int)
    args = parser.parse_args(argv)
    try:
        config = load(args.config)
        from . import runner
        if args.command == "bind":
            result = runner.binding(config)
        elif args.command == "audit-data":
            result = runner.data_audit(config)
        else:
            result = runner.train_cell(config, args.cell, args.device, args.resume, args.stop_after)
        print(json.dumps({"status": "OK", "command": args.command, "result": result}, allow_nan=False))
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
