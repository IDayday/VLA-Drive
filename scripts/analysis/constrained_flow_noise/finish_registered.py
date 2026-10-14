"""Wait for the two existing formal runs, then perform their locked confirmation.

Never starts/restarts training, changes a budget, or resumes a safety pause.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from .evaluate_policy import child_environment
from .lock_final import collect, make_lock, read, sealed


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--population', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    host = require_dispatch()
    root, output = Path(args.run_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    guard = (output/'controller.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    population = read(args.population)
    lock_path = output/'final_lock.json'
    try:
        while not lock_path.exists():
            require_dispatch()
            evidence = collect(root, population)
            atomic_json(output/'status.json', dict(status=evidence['status'], pending=evidence['pending'],
                        host=host, utc_unix=time.time(), source_sha256=file_sha(__file__)))
            if evidence['status'] == 'READY_TO_LOCK':
                make_lock(root, population, evidence, lock_path)
                break
            if not args.wait:
                print(json.dumps(dict(status=evidence['status'], pending=evidence['pending'])))
                return
            time.sleep(60)
        lock = sealed(lock_path)
        require_dispatch()
        env = child_environment()
        command = [sys.executable, '-m', 'scripts.analysis.constrained_flow_noise.evaluate_final',
                   '--final-lock', str(lock_path), '--population', args.population,
                   '--output', str(output/'Navtest'), '--workers', '16']
        atomic_json(output/'confirmation_command.json', dict(args=command, host=host, final_lock_sha256=lock['sha256']))
        subprocess.run(command, env=env, check=True)
        summary = output/'formal_curves'
        if not (summary/'summary.json').exists():
            require_dispatch()
            subprocess.run([sys.executable, '-m', 'scripts.analysis.constrained_flow_noise.summarize_runs',
                            '--run-root', str(root), '--output', str(summary), '--plot'], env=env, check=True)
        atomic_json(output/'status.json', dict(status='CONFIRMATION_AND_CURVES_COMPLETE_ANALYSIS_REQUIRED',
                    final_lock_sha256=lock['sha256'], host=host, utc_unix=time.time(),
                    research_goal_complete=False, human_readable_result_analysis_still_required=True))
    except Exception as error:
        atomic_json(output/f'failure_{time.time_ns()}.json', dict(type=type(error).__name__, message=str(error),
                    automatic_training_restart=False, host=host))
        raise


if __name__ == '__main__':
    main()
