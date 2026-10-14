import argparse
import json
import sys
from .config import load_config


def main():
    p = argparse.ArgumentParser(description="Own expert + randomly initialized internal W/A scorer; no navtest or release")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("bind-base", "cache-features", "build-hard-set", "revise-hard-set", "train-expert", "build-pair-bank", "train-scorer", "evaluate", "diagnostics", "audit-online", "export-bundle", "benchmark", "run"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        if name == "revise-hard-set":
            s.add_argument("--threshold", type=float, choices=(.8, .85, .9), required=True)
            s.add_argument("--revision-id", required=True)
        if name == "cache-features":
            s.add_argument("--shard", type=int, default=0)
            s.add_argument("--shards", type=int, default=1)
            s.add_argument("--device", default="cuda:0")
            s.add_argument("--max-samples", type=int)
        elif name in {"train-expert", "build-pair-bank", "train-scorer", "evaluate", "diagnostics", "audit-online", "export-bundle", "benchmark", "run"}:
            s.add_argument("--device", default="cuda:0")
            if name.startswith("train-"):
                s.add_argument("--resume", nargs="?", const="auto")
                s.add_argument("--stop-after", type=int)
            if name == "run":
                s.add_argument("--wait-cache-seconds", type=int, default=0)
                s.add_argument("--from-stage", choices=("beginning", "scorer"), default="beginning",
                               help="Resume after an already complete, hash-verified pair bank")
    args = p.parse_args()
    try:
        c = load_config(args.config)
        if args.command != "bind-base":
            from iqe.io import read_json, file_hash
            from iqe.contracts import require
            from iqe.data.manifests import load_scenes
            from iqe.data.splits import validate_isolation
            b = read_json(c.root/"BASE_BINDING.json")
            require(b["config_hash"] == c.identity and b["scene_manifest_hash"] == file_hash(c.root/"scenes.json"), "stale experiment inputs")
            audit = validate_isolation(load_scenes(c.root/"scenes.json"))
            require(not set(audit["roles"])-{"incremental_fit", "stage_val", "selector_cal", "dev_report"}, "forbidden evaluation role")
        if args.command == "bind-base":
            from .binding import bind
            result = bind(c)
        elif args.command == "cache-features":
            from .features import build_cache
            result = build_cache(c, args.shard, args.shards, args.device, args.max_samples)
        elif args.command == "run":
            from .run import run
            result = run(c, args.device, args.wait_cache_seconds, args.from_stage)
        elif args.command == "build-hard-set":
            from .bank import build_initial
            result = build_initial(c)
        elif args.command == "revise-hard-set":
            from .expert_data_revision import revise
            result = revise(c, args.threshold, args.revision_id)
        elif args.command == "benchmark":
            from .latency import benchmark
            result = benchmark(c, args.device)
        elif args.command in {"train-expert", "train-scorer", "build-pair-bank"}:
            from .training import train_expert, train_scorer, build_pair_bank
            if args.command == "build-pair-bank":
                result = build_pair_bank(c, args.device)
            else:
                runner = train_expert if args.command == "train-expert" else train_scorer
                result = runner(c, args.device, args.resume, args.stop_after)
        else:
            from .reporting import evaluate, diagnostics, audit_online, export_bundle
            result = {"evaluate": evaluate, "diagnostics": diagnostics, "audit-online": audit_online,
                      "export-bundle": export_bundle}[args.command](c, args.device)
        print(json.dumps(result, allow_nan=False), flush=True)
    except Exception as e:
        print(json.dumps({"status": "FAILED", "error": repr(e)}), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
