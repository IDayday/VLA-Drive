#!/usr/bin/env python3
"""Finish one already-locked cell's registered predictions on a spare GPU.

Shares the existing prediction locks/receipts with the serial finalizer. No new
training, selection, samples, or labels are introduced by parallel execution.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from iqe.contracts import require
from iqe.io import atomic_json, file_hash, read_json, lock
from iqe.robust_scorer.config import load, CELLS
from iqe.robust_scorer.evaluation import predictions
from iqe.specialist_study.training import runtime
from scripts.iqe.finish_robust_scorer import available_gpu, ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--cell', required=True, choices=tuple(CELLS))
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--expect-host', required=True)
    args = parser.parse_args()
    config = load(args.config)
    host = socket.gethostname().removesuffix('-worker-0')
    require(host == args.expect_host and host != 'training-vla-zt2', 'wrong/protected host')
    policy = Path('/mnt/project/server_dispatch_policy.json')
    require(host in read_json(policy)['allowed_hosts'], 'permanent host required')
    require(ready(config), 'all fixed-budget training must be complete')
    choice = config.root / 'MODEL_CHOICE.json'
    evaluation = read_json(config.root / 'FINAL_EVALUATION_LOCK.json')
    require(evaluation['choice_hash'] == file_hash(choice), 'final choice must already be locked')
    folder = config.root / 'runtime/parallel_predictions' / args.cell
    with lock(folder / 'RUN.lock'):
        available, capacity = available_gpu(args.gpu)
        require(available, 'assigned GPU is no longer free; preserve its current owner')
        sources = {name: file_hash(ROOT / 'iqe/robust_scorer' / name)
                   for name in ('evaluation.py', 'runner.py', 'model.py')}
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
        device = runtime('cuda:0')
        started = time.monotonic()
        receipt = dict(host=host, gpu=args.gpu, capacity_at_start=capacity, pid=os.getpid(),
                       start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                       started_utc=datetime.now(timezone.utc).isoformat(), config_hash=config.identity,
                       choice_hash=file_hash(choice), source_hashes=sources, new_optimizer_steps=0,
                       roles=['dev_report', 'scorer_fit'], no_other_process_signals=True)
        atomic_json(folder / 'LAUNCH.json', receipt)
        for role in receipt['roles']:
            require(host in read_json(policy)['allowed_hosts'], 'host permission withdrawn')
            require(all(file_hash(ROOT / 'iqe/robust_scorer' / name) == checksum
                        for name, checksum in sources.items()), 'prediction implementation changed')
            atomic_json(folder / 'STATUS.json', dict(status='RUNNING', role=role, cell=args.cell))
            predictions(config, args.cell, role, device)
        result = dict(status='COMPLETE', cell=args.cell, elapsed_seconds=time.monotonic()-started,
                      finished_utc=datetime.now(timezone.utc).isoformat(), new_optimizer_steps=0,
                      dev_used_to_select_model=False)
        atomic_json(folder / 'STATUS.json', result)
        print(result)


if __name__ == '__main__':
    main()
