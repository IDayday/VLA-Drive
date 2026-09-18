"""Parallel calls to the original evaluator, keeping every log on one worker.

NAVSIM's adjacency is per log and its final normalization is per scene. Complete
logs may therefore be evaluated independently. Token-keyed inference noise and
the original CLI are preserved; whole-result publication uses its transaction.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import numpy as np
import pandas as pd
from scripts.cluster_flow_grpo.cluster import ROOT, PYTHON, base_env, write_json
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.contracts import digest
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.reward import build_cache_index
from starVLA.rl.flow_grpo.evaluation import (
    evaluation_identity, validate_evaluation_tokens, METRIC_PROTOCOL,
)
from starVLA.rl.flow_grpo.evaluation_transaction import (
    evaluation_transaction, completed_evaluation,
)
from starVLA.rl.flow_grpo.reproducibility import configure_numerics


@lru_cache(maxsize=4)
def cache_index_snapshot(root):
    # Index paths once per controller. Content is still rehashed for every
    # evaluation identity and every reusable completed result.
    return build_cache_index(root)


def selected_cache_view(index, tokens, root):
    """A shallow directory view avoids rescanning103k unrelated scene folders.

    Only directory links are created; every cache byte remains at its original
    path and is verified by the unchanged evaluator's content identity.
    """
    mapping = {t: str(Path(index[t]).resolve()) for t in sorted(tokens)}
    view = Path(root)/digest(mapping)
    if not view.exists():
        temporary = view.with_name(view.name+f".tmp-{os.getpid()}-{time.time_ns()}")
        temporary.mkdir(parents=True)
        for token, target in mapping.items():
            source = Path(target)
            dest = temporary/source.parents[2].name/source.parents[1].name/token
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to(source.parent,target_is_directory=True)
        write_json(temporary/"view_identity.json",mapping)
        view.parent.mkdir(parents=True,exist_ok=True)
        try:
            temporary.rename(view)
        except FileExistsError:
            import shutil
            shutil.rmtree(temporary)
    if json.loads((view/"view_identity.json").read_text()) != mapping:
        raise ValueError("cache view identity conflict")
    actual = build_cache_index(view)
    if set(actual) != set(mapping) or any(str(Path(actual[t]).resolve()) != mapping[t] for t in mapping):
        raise ValueError("cache view link/content inventory changed")
    return view


def plan_shards(tokens, token_logs, workers):
    if not tokens or len(tokens) != len(set(tokens)) or workers < 1:
        raise ValueError("unique nonempty tokens and positive workers required")
    groups = defaultdict(list)
    for token in tokens:
        if not token_logs.get(token):
            raise ValueError(f"missing log identity: {token}")
        groups[token_logs[token]].append(token)
    bins = [[] for _ in range(min(workers, len(groups)))]
    for log in sorted(groups, key=lambda log: (-len(groups[log]), log)):
        dest = min(range(len(bins)), key=lambda i: (len(bins[i]), i))
        bins[dest].extend(groups[log])
    # Retain the original order within each shard. Seeds are token-keyed anyway.
    sets = [set(values) for values in bins]
    return [[t for t in tokens if t in values] for values in sets]


def merge_results(root, identity, tokens, shards, split, complete, checkpoint,
                  tokens_file, seed, elapsed):
    frames, arrays, caches, adjacent = [], {}, {}, {}
    seen_logs = set()
    reports = []
    for directory, shard_identity, shard_tokens in shards:
        report = completed_evaluation(directory, shard_identity, shard_tokens)
        if report is None:
            raise ValueError("incomplete evaluation shard")
        frame = pd.read_csv(Path(directory) / "original_protocol_scores.csv",
                            dtype={"token": str, "log_name": str}, float_precision="round_trip")
        logs = set(frame.log_name)
        if logs & seen_logs:
            raise ValueError("a log spans shards; adjacency would be incomplete")
        seen_logs.update(logs)
        if set(frame.token) & set(arrays):
            raise ValueError("duplicate scene across evaluation shards")
        frames.append(frame)
        with np.load(Path(directory) / "trajectories.npz", allow_pickle=False) as archive:
            for i, token in enumerate(archive["tokens"].tolist()):
                arrays[token] = (archive["normalized"][i], archive["physical"][i])
        caches.update(json.loads((Path(directory) / "metric_cache_identity.json").read_text()))
        mapping = report.get("aggregation", {}).get("adjacent_mapping", {})
        if set(mapping) & set(adjacent) or any(t not in shard_tokens for pair in mapping.items() for t in pair):
            raise ValueError("invalid shard adjacency mapping")
        adjacent.update(mapping)
        reports.append(report)
    if set(arrays) != set(tokens) or set(caches) != set(tokens):
        raise ValueError("evaluation shards do not exactly cover requested scenes")
    if digest(caches) != identity["metric_assets_sha256"]:
        raise ValueError("sharded metric assets differ from full evaluation identity")
    frame = pd.concat(frames, ignore_index=True).set_index("token").loc[tokens].reset_index()
    frame.to_csv(root / "original_protocol_scores.csv", index=False)
    physical = np.asarray([arrays[t][1] for t in tokens])
    np.savez(root / "trajectories.npz", tokens=np.asarray(tokens),
             normalized=np.asarray([arrays[t][0] for t in tokens]), physical=physical)
    predictions = root / "predictions" / split
    predictions.mkdir(parents=True)
    for token, trajectory in zip(tokens, physical):
        np.save(predictions / f"{token}.npy", trajectory)
    write_json(root / "metric_cache_identity.json", caches)
    available = int(frame.two_frame_extended_comfort.notna().sum()) if "two_frame_extended_comfort" in frame else 0
    aggregation = deepcopy(reports[0].get("aggregation", {}))
    aggregation.update(adjacent_pairs=len(adjacent), adjacent_mapping=adjacent,
                       two_frame_available=available, two_frame_coverage=available/len(tokens))
    report = deepcopy(reports[0])
    for key in ("artifacts", "status", "schema_version"):
        report.pop(key, None)
    report.update(identity=identity, metric_cache_identity=digest(caches),
                  checkpoint=str(checkpoint), split=split, complete_split=complete,
                  tokens_sha256=file_sha(tokens_file), seed=seed,
                  metric_name=f"{split}_v2_EPDMS" if complete else f"partial_{split}_v2_EPDMS",
                  aggregation=aggregation, scene_count=len(tokens), valid=int(frame.valid.sum()),
                  epdms=float(frame.score.mean()), seconds=elapsed,
                  parallel_execution={"strategy": "complete_logs", "shards": len(shards),
                                      "shard_inference_scoring_seconds": [r.get("seconds") for r in reports],
                                      "timing_note": "parent seconds include merge and actual requests; completed shards can be reused, so this value alone is not a GPU throughput measurement",
                                      "source_sha256": file_sha(__file__)})
    write_json(root / "shard_receipts.json", [
        {"path": str(p), "report_sha256": file_sha(Path(p)/"evaluation.json")}
        for p, _, _ in shards
    ])
    return report


def evaluate_parallel(config, checkpoint, output, split, tokens_file, data_root,
                      metric_cache, seed, slots, executor=None, cancel_event=None):
    # The parent computes exactly the numerical identity used by the CLI child.
    os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    configure_numerics()
    cfg, sft = resolve_config(config)
    tokens = json.loads(Path(tokens_file).read_text())
    complete = validate_evaluation_tokens(cfg, split, tokens)
    index = cache_index_snapshot(str(metric_cache))
    view_root = Path(str(output)+".cache_views")
    full_view = selected_cache_view(index,tokens,view_root)
    identity = evaluation_identity(cfg, sft, checkpoint, tokens_file, seed, split,
                                   data_root, full_view)
    saved = completed_evaluation(output, identity, tokens)
    if saved is not None:
        return saved
    if not slots or len({(s["host"], s["gpu"]) for s in slots}) != len(slots):
        raise ValueError("unique explicit evaluation GPU slots required")
    # The trusted cache layout is root/log/scene_type/token/metric_cache.pkl.
    token_logs = {t: Path(index[t]).parents[2].name for t in tokens}
    assignments = plan_shards(tokens, token_logs, len(slots))
    shard_root = Path(str(output) + ".shards")
    shard_root.mkdir(parents=True, exist_ok=True)
    plan = {"identity": identity, "assignments": assignments}
    plan_path = shard_root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("conflicting sharded evaluation plan")
    write_json(plan_path, plan)

    def execute_slot(i):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("evaluation cancelled")
        selected = assignments[i]
        token_path = shard_root / f"tokens_{i}.json"
        write_json(token_path, selected)
        dest = shard_root / f"shard_{i}"
        selected_view = selected_cache_view(index,selected,view_root)
        sid = evaluation_identity(cfg, sft, checkpoint, token_path, seed, split,
                                  data_root, selected_view)
        saved = completed_evaluation(dest, sid, selected)
        if saved is None:
            command = [PYTHON, "-m", "starVLA.rl.flow_grpo.cli", "evaluate",
                       "--config", str(config), "--checkpoint", str(checkpoint),
                       "--output-dir", str(dest), "--split", split,
                       "--tokens", str(token_path), "--data-root", str(data_root),
                       "--metric-cache", str(selected_view), "--seed", str(seed),
                       "--metric-protocol", METRIC_PROTOCOL]
            if executor:
                executor(command, slots[i], shard_root / f"shard_{i}.log")
            else:
                run_evaluator(command, slots[i], shard_root / f"shard_{i}.log",
                              cancel_event=cancel_event)
        if completed_evaluation(dest, sid, selected) is None:
            raise RuntimeError("evaluation worker exited without complete publication")
        return dest, sid, selected

    def execute(root):
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
            shards = list(pool.map(execute_slot, range(len(assignments))))
        return merge_results(root, identity, tokens, shards, split, complete,
                             checkpoint, tokens_file, seed, time.monotonic()-started)

    return evaluation_transaction(output, identity, tokens, execute)


def run_evaluator(command, slot, log, cancel_event=None):
    from scripts.cluster_flow_grpo.cluster import run
    affinity = slot.get("cpu_affinity", list(range(int(slot["gpu"])*8, int(slot["gpu"])*8+8)))
    log = Path(log)
    if log.exists():
        log = log.with_name(log.name + f".attempt_{time.time_ns()}")
    spec = {"job_id": str(log), "nodes": [{"host": slot["host"],
            "devices": [slot["gpu"]], "cpu_affinity": affinity}],
            "direct_command": command, "control_dir": str(log)+".control",
            "require_idle_gpus": slot.get("require_idle", False),
            "timeout_seconds": 14400}
    spec_path = Path(str(log)+".spec.json")
    write_json(spec_path, spec)
    write_json(log, {"supervision": str(log)+".control"})
    return run(spec_path, cancel_event=cancel_event)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    for key in ("config", "checkpoint", "output", "split", "tokens", "data-root", "metric-cache", "slots"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    report=evaluate_parallel(a.config, a.checkpoint, a.output, a.split,
          a.tokens, a.data_root, a.metric_cache, a.seed, json.loads(Path(a.slots).read_text()))
    print(json.dumps({k:report[k] for k in ("status","scene_count","epdms","seconds")}))
