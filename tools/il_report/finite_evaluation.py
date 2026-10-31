"""Coordinate pinned requests through the already qualified evaluator unchanged.

The operator only limits placement/concurrency and publishes receipts. Export,
native parity, actor forward, scoring and all per-scene data use the pinned
upstream repository; no automatic Git mutation happens in an evaluator tree.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write(path, data):
    path = Path(path)
    tmp = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('w') as stream:
        json.dump(data, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evaluator-source', type=Path, required=True)
    args = parser.parse_args()
    root, source = args.root.resolve(), args.evaluator_source.resolve()
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'owner.json').exists():
            raise FileExistsError('Pinned evaluator already has an invocation; review before retry')
        # Imports and all spawned scientific worker cwd refer to this immutable source.
        sys.path.insert(0, str(source))
        from tools.qwen_world_compare.interim_owner import InterimRequest
        pins = json.loads((root/'PINNED.json').read_text())
        owner = dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            at=time.time(), source=str(source), source_sha=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
            operator=str(Path(__file__).resolve()), operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            status='RUNNING', physical_trainers_modified=False)
        write(root/'owner.json', owner)
        requests = {}
        for label, pin in pins['models'].items():
            directory = root/label; directory.mkdir(exist_ok=False)
            requests[label] = InterimRequest(directory, pin)
        completed, failed = {}, {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(request.run): label for label, request in requests.items()}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    completed[label] = future.result()
                except Exception:
                    failed[label] = traceback.format_exc()
                write(root/'status.json', dict(completed=completed, failed=failed, at=time.time()))
        owner.update(status='FAILED' if failed else 'COMPLETE', ended_unix=time.time())
        write(root/'owner.json', owner)
        if failed:
            write(root/'failure.json', dict(failed=failed, no_automatic_algorithm_retry=True))
            raise RuntimeError('Pinned evaluation failed; inspect retained receipts')
        write(root/'COMPLETE.json', dict(status='COMPLETE', models=list(completed),
            all_owned_workers_exited=True, at=time.time(), evaluation_source_sha=owner['source_sha'],
            operator_sha256=owner['operator_sha256'], no_training_changes=True))


if __name__ == '__main__':
    main()
