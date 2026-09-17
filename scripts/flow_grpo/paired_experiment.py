"""Fixed-budget paired experiments. Requires matching real-profile release records.

Run each pair concurrently only on explicitly supplied disjoint idle GPUs. Each
phase uses torchrun max-restarts=0; a failed peer experiment is not relaunched.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
import subprocess
import signal
import threading
import time
import sys
import numpy as np
import pandas as pd
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.orchestration import advance_target
from starVLA.rl.flow_grpo.checkpoint import validate_checkpoint, validate_export
from starVLA.rl.flow_grpo.evaluation_transaction import request_evaluation
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.flow_grpo.reproducibility import training_provenance
from starVLA.rl.flow_grpo.reproducibility import configure_numerics, resume_assets
from starVLA.rl.flow_grpo.acceptance import acceptance_context, enforce_training_budget
from starVLA.rl.flow_grpo.evaluation import (
    EVALUATION_SEEDS,
    METRIC_PROTOCOL,
    paired_scores,
    evaluation_identity,
)


def idle_devices(devices):
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    values = {
        parts[0].strip(): [int(x) for x in parts[1:]]
        for parts in (line.split(",") for line in rows.splitlines())
    }
    for device in devices:
        if device not in values or values[device][0] > 1024 or values[device][1] > 10:
            raise RuntimeError(f"GPU {device} is occupied; no experiment launched")


_CANCELLED = threading.Event()
_PROCESSES = set()
_PROCESS_LOCK = threading.Lock()


def stop_owned_processes():
    _CANCELLED.set()
    with _PROCESS_LOCK:
        for process in list(_PROCESSES):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def invoke(command, env, log):
    if _CANCELLED.is_set():
        raise RuntimeError("paired experiment stopped after peer failure")
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        log = log.with_name(log.name + f".attempt_{time.time_ns()}")
    with log.open("x") as stream:
        with _PROCESS_LOCK:
            if _CANCELLED.is_set():
                raise RuntimeError("paired experiment cancelled")
            process = subprocess.Popen(
                command,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _PROCESSES.add(process)
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if _CANCELLED.is_set():
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                        break
            if process.wait() != 0:
                stop_owned_processes()
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            with _PROCESS_LOCK:
                _PROCESSES.discard(process)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices-f", required=True)
    parser.add_argument("--devices-u", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--navtest-cache", required=True)
    parser.add_argument(
        "--config-f",
        default="runs/paired_full_assets_v1/published_v2/configs/paired_frozen_visual.yaml",
    )
    parser.add_argument(
        "--config-u",
        default="runs/paired_full_assets_v1/published_v2/configs/paired_unfrozen_visual.yaml",
    )
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    devices = {
        "frozen_visual": args.devices_f.split(","),
        "unfrozen_visual": args.devices_u.split(","),
    }
    world = len(devices["frozen_visual"])
    if world != len(devices["unfrozen_visual"]) or 16 % world:
        raise ValueError("both experiments require equal world size dividing 16")
    if not args.sequential and set(devices["frozen_visual"]) & set(
        devices["unfrozen_visual"]
    ):
        raise ValueError("parallel experiments require disjoint GPUs")
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    configure_numerics()
    configurations = {}
    for variant in devices:
        config = args.config_f if variant == "frozen_visual" else args.config_u
        cfg, sft = resolve_config(config)
        if cfg["runtime"]["accumulation_steps"] != 16 // world:
            raise ValueError(
                "configured accumulation does not match selected world size"
            )
        enforce_training_budget(cfg)
        assets = resume_assets(cfg, sft)
        enforce_training_budget(cfg, acceptance_context(cfg, assets))
        configurations[variant] = (
            config,
            cfg,
            sft,
            training_provenance(cfg, sft, assets),
        )
    # Do not create an apparent run until both releases and GPU availability pass.
    idle_devices(set(sum(devices.values(), [])))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=args.resume)

    def _variant_run(variant):
        config, cfg, sft, provenance = configurations[variant]
        run = root / variant
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(devices[variant]),
            "NUM_GPUS": str(world),
            "OUTPUT_DIR": str(run),
            "CONFIG": config,
            "PYTHON_BIN": sys.executable,
            "TRITON_CACHE_DIR": str(run / "triton"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        # WORLD_SIZE/RANK belong to torchrun; avoid stale parent metadata in eval.
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        data_manifest = json.loads(Path(cfg["paths"]["split_manifest"]).read_text())
        dev_tokens = Path(cfg["paths"]["split_manifest"]).parent / "dev_tokens.json"
        eval_root = root / (variant + "_evaluation")
        eval_root.mkdir(exist_ok=True)

        def evaluate(checkpoint, label, split, seeds):
            results = {}
            for seed in seeds:
                dest = eval_root / f"{label}_{split}_seed{seed}"
                token_file = (
                    dev_tokens if split == "rl_dev" else cfg["paths"]["test_list"]
                )
                cache = (
                    cfg["paths"]["metric_cache"]
                    if split == "rl_dev"
                    else args.navtest_cache
                )
                identity = evaluation_identity(
                    cfg,
                    sft,
                    checkpoint,
                    token_file,
                    seed,
                    split,
                    cfg["paths"]["data_root"],
                    cache,
                )
                tokens = json.loads(Path(token_file).read_text())
                command = [
                    sys.executable,
                    "-m",
                    "starVLA.rl.flow_grpo.cli",
                    "evaluate",
                    "--config",
                    config,
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(dest),
                    "--split",
                    split,
                    "--tokens",
                    str(dev_tokens if split == "rl_dev" else cfg["paths"]["test_list"]),
                    "--data-root",
                    cfg["paths"]["data_root"],
                    "--metric-cache",
                    cfg["paths"]["metric_cache"]
                    if split == "rl_dev"
                    else args.navtest_cache,
                    "--seed",
                    str(seed),
                    "--metric-protocol",
                    METRIC_PROTOCOL,
                ]
                results[seed] = request_evaluation(
                    dest,
                    identity,
                    tokens,
                    lambda: invoke(
                        command, env, eval_root / f"{label}_{split}_seed{seed}.log"
                    ),
                )
                progress_path = eval_root / "evaluation_progress.json"
                progress = (
                    json.loads(progress_path.read_text())
                    if progress_path.exists()
                    else {}
                )
                progress[f"{label}/{split}/{seed}"] = str(dest)
                atomic_json(progress_path, progress)
            return results

        evaluate(cfg["sft_checkpoint"], "sft", "rl_dev", [42])
        best = None

        def train_to(resume, target):
            command = ["scripts/flow_grpo/launch.sh", "train"]
            if resume is not None:
                command += ["--resume", str(resume)]
            command += [
                "--set",
                "runtime.run_mode=" + ("paired_short" if target == 100 else "formal"),
            ]
            invoke(
                command,
                {**env, "MAX_UPDATES": str(target)},
                root / f"{variant}_to{target}.log",
            )

        def export_at(checkpoint, target):
            export = run / f"export_update{target}"
            if not validate_export(checkpoint, export):
                invoke(
                    [
                        sys.executable,
                        "-m",
                        "starVLA.rl.flow_grpo.cli",
                        "export",
                        "--checkpoint",
                        str(checkpoint),
                        "--output-dir",
                        str(export),
                    ],
                    env,
                    root / f"{variant}_export{target}.log",
                )
            if not validate_export(checkpoint, export):
                raise RuntimeError("export executor did not publish a complete result")
            return export

        for target in [100] + list(range(200, 2001, 200)):
            checkpoint, export, dev = advance_target(
                run,
                target,
                validate=lambda path: validate_checkpoint(path, cfg, provenance, world),
                train=train_to,
                export=export_at,
                evaluate=lambda exported, update: evaluate(
                    exported, f"step{update}", "rl_dev", [42]
                )[42],
            )
            if not dev["complete_split"]:
                raise ValueError("incomplete dev evaluation cannot select best")
            if target % 200 == 0 and (best is None or dev["epdms"] > best["epdms"]):
                best = {
                    "update": target,
                    "epdms": dev["epdms"],
                    "checkpoint": str(checkpoint),
                    "export": str(export),
                }
            (run / "selection.json").write_text(
                json.dumps(
                    {
                        "last": str(
                            run
                            / "checkpoints"
                            / f"update_{json.loads((run / 'orchestration_progress.json').read_text())['training_latest_update']:06d}"
                        ),
                        "last_evaluated_checkpoint": str(checkpoint),
                        "best": best,
                        "rule": "max full dev seed42 EPDMS every 200 updates, earliest tie; no navtest",
                        "dev_scenes": len(data_manifest["dev_tokens"]),
                    },
                    indent=2,
                )
            )
        # Fixed five seeds for the same SFT, last and dev-selected best policies.
        for split in ("rl_dev", "navtest"):
            evaluate(cfg["sft_checkpoint"], "sft", split, EVALUATION_SEEDS)
            for label, checkpoint in [
                ("last", run / "export_update2000"),
                ("best", Path(best["export"])),
            ]:
                evaluate(checkpoint, label, split, EVALUATION_SEEDS)
                results = []
                for seed in EVALUATION_SEEDS:
                    left = eval_root / f"sft_{split}_seed{seed}"
                    right = eval_root / f"{label}_{split}_seed{seed}"
                    if not all(
                        json.loads((p / "evaluation.json").read_text())[
                            "complete_split"
                        ]
                        for p in (left, right)
                    ):
                        raise ValueError("partial split is not a final result")
                    rows, report = paired_scores(
                        pd.read_csv(left / "original_protocol_scores.csv"),
                        pd.read_csv(right / "original_protocol_scores.csv"),
                    )
                    rows.to_csv(
                        eval_root / f"paired_{label}_{split}_seed{seed}.csv",
                        index=False,
                    )
                    results.append({"seed": seed, **report})
                deltas = [r["mean_paired_delta"] for r in results]
                (eval_root / f"paired_{label}_{split}.json").write_text(
                    json.dumps(
                        {
                            "seeds": results,
                            "mean_delta": float(np.mean(deltas)),
                            "seed_std": float(np.std(deltas, ddof=1)),
                        },
                        indent=2,
                    )
                )

    def variant_run(variant):
        try:
            return _variant_run(variant)
        except BaseException:
            stop_owned_processes()
            raise

    if args.sequential:
        for variant in devices:
            idle_devices(devices[variant])
            variant_run(variant)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(variant_run, variant) for variant in devices]
            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
