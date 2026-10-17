"""Wait for the single reviewed GPU, then export and score once. No retries."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    spec_bytes = (here/'EVALUATION_SPEC.json').read_bytes()
    if hashlib.sha256(spec_bytes).hexdigest() != '59cdf8a3ab692994f926ce460a4a353bb16ffba1a3bb21d1ff6c4beacb936452':
        raise ValueError('Fixed evaluation specification changed')
    spec = json.loads(spec_bytes)
    control = Path(spec['registration_root'])
    control.mkdir(parents=True, exist_ok=True)
    lock = (control/'WAIT_EVALUATE.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = control/'WAIT_EVALUATE.json'
    if status.exists():
        raise RuntimeError('One-shot waiter already attempted; no automatic retry')
    started = time.time()
    record = {'status': 'WAITING_RESOURCE', 'pid': os.getpid(), 'started_unix': started,
              'resource_wait_limit_seconds': 3600, 'maximum_total_seconds': 16200,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'review': args.review, 'review_sha256': args.review_sha256,
              'GPU_signals_sent': False, 'training_started': False, 'stages': []}
    def save():
        temp = status.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2)+'\n')
        os.replace(temp, status)
    save()
    try:
        wait_end = time.monotonic()+3600
        while True:
            remaining = wait_end-time.monotonic()
            if remaining <= 0:
                record['status'] = 'BLOCKED_RESOURCE'; save(); return 3
            row = subprocess.check_output(['nvidia-smi', '--id='+str(spec['gpu']),
                '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                text=True, timeout=min(15, remaining)).strip().split(',')
            if time.monotonic() >= wait_end:
                record['status'] = 'BLOCKED_RESOURCE'; save(); return 3
            if row[0].strip() != spec['gpu_uuid']:
                raise ValueError('GPU identity changed')
            if int(row[1]) < 64 and int(row[2]) < 5:
                break
            time.sleep(min(30, max(0, wait_end-time.monotonic())))
        for stage in ('export', 'score'):
            command = [sys.executable, str(here/'launch_evaluation.py'), '--stage', stage,
                       '--review', args.review, '--review-sha256', args.review_sha256]
            result = subprocess.run(command, text=True, capture_output=True, timeout=120)
            if result.returncode:
                raise RuntimeError('No retry after launch refusal: '+result.stderr[-3000:])
            launch = json.loads(result.stdout.strip().splitlines()[-1])
            record['status'] = stage.upper()+'_RUNNING'; record['stages'].append(launch); save()
            deadline = time.monotonic()+(8000 if stage == 'export' else 4300)
            pid = launch['pid']; proc = Path('/proc')/str(pid)
            try:
                initial = proc.joinpath('stat').read_text().split(') ', 1)[1].split()[19]
            except FileNotFoundError:
                initial = None
            while initial is not None:
                try:
                    fields = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
                except FileNotFoundError:
                    break
                if fields[0] == 'Z' or fields[19] != initial:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Child exceeds original bounded timeout; do not signal unrelated PID')
                time.sleep(15)
            out = Path(spec['output_root'])
            required = out/('export/COMPLETE_rank0.json' if stage == 'export' else 'official_score/COMPLETE.json')
            if not required.is_file():
                raise RuntimeError('Stage did not complete; preserve output and do not retry')
            completed = json.loads(required.read_text())
            if stage == 'score' and completed.get('status') != 'COMPLETE':
                raise RuntimeError('Official scoring incomplete')
        record['status'] = 'COMPLETE'; record['ended_unix'] = time.time(); save(); return 0
    except BaseException as exc:
        record['status'] = 'FAILED_NO_AUTO_RETRY'; record['error'] = repr(exc); save(); raise


if __name__ == '__main__':
    raise SystemExit(main())
