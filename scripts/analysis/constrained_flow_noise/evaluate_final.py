"""Evaluate only sealed two-line endpoints/selections using the original ODE.

Reuses the existing locked Navtest actor and canonical single-candidate worker.
No training, selection, search or noise-head code executes in the actor.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.analysis.flow_safe_improvement.sample_locked_navtest import locked_contract
from starVLA.rl.constrained_flow_noise.contracts import file_sha, require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json
from .analyze_ode import analyze
from .evaluate_policy import child_environment
from .lock_final import read, sealed


def evaluate_one(lock_path, model, population_path, output, baseline_scores, workers):
    host = require_dispatch()
    population = read(population_path)
    lock, lock_sha, selected = locked_contract(lock_path, model, population)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    guard = (output/'evaluation.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(final_lock_sha256=lock_sha, model=model, weight_sha256=selected['weight_sha256'],
                    seed=2101, population_identity=population['identity_sha256'], world=8,
                    source_sha256=file_sha(__file__), analysis_sha256=file_sha(Path(__file__).with_name('analyze_ode.py')))
    if (output/'identity.json').exists() and read(output/'identity.json') != identity:
        raise ValueError('Final evaluation recovery identity changed')
    atomic_json(output/'identity.json', identity)
    if (output/'report.json').exists():
        return sealed(output/'report.json')
    processes, handles = [], []
    started = time.monotonic()
    env = child_environment()
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    def start(args, name):
        require_dispatch()
        handle = (output/(name+'.log')).open('a')
        handles.append(handle)
        process = subprocess.Popen(args, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        atomic_json(output/(name+'_launch.json'), dict(args=args, pid=process.pid, host=host, started_unix=time.time()))
        return process

    try:
        args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', '-m',
                'scripts.analysis.flow_safe_improvement.sample_locked_navtest', '--final-lock', str(lock_path),
                '--model', model, '--population', str(population_path), '--output', str(output/'samples'),
                '--shard-offset', '0', '--shard-count', '8']
        if (output/'samples/manifest.json').exists():
            args.append('--resume')
        inference = start(args, 'inference')
        while not (output/'samples/manifest.json').exists():
            if inference.poll() is not None:
                raise RuntimeError('Original Navtest actor exited before publishing its identity')
            time.sleep(2)
        args = [sys.executable, '-m', 'scripts.analysis.flow_safe_improvement.score_samples',
                '--samples', str(output/'samples'), '--population', str(population_path), '--cache-root', lock['v1_cache'],
                '--output', str(output/'scores'), '--workers', str(workers)]
        if (output/'scores/manifest.json').exists():
            args.append('--resume')
        start(args, 'scoring')
        while any(p.poll() is None for p in processes):
            require_dispatch()
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Final actor/scorer failed; all missing scenes remain required')
            time.sleep(5)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('Final actor/scorer did not complete')
        manifest = sealed(output/'scores/manifest.json')
        if manifest['protocol_sha256'] != lock['v1_protocol_sha256']:
            raise ValueError('Navtest scoring protocol changed after selection lock')
        report = analyze(output/'scores', baseline_scores, population)
        if report['scenes'] != 12146 or report['behavior_sha256'] != selected['weight_sha256']:
            raise ValueError('Incomplete or wrong final model')
        atomic_json(output/'report.json', report)
        atomic_json(output/'cost.json', dict(status='COMPLETE', host=host, wall_seconds=time.monotonic()-started,
                    scoring=read(output/'scores/status.json'),
                    inference_shards=[read(p) for p in sorted((output/'samples').glob('shard*.json'))]))
        return report
    except Exception as error:
        atomic_json(output/f'failure_{time.time_ns()}.json',
                    dict(type=type(error).__name__, message=str(error), missing_scenes_excluded=False))
        # Only groups created by this invocation are signalled; unknown jobs and
        # unrelated cache/training workers are never examined or preempted.
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            if process.poll() is None:
                process.wait()
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--final-lock', required=True)
    parser.add_argument('--population', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    require_dispatch()
    if not 1 <= args.workers <= 32:
        raise ValueError('Bounded CPU worker count required')
    lock = sealed(args.final_lock)
    if (lock['schema'] != 'cfn_two_line_final_confirmation_v1'
            or lock['unique_export_count'] > 5 or lock['inference_seeds'] != [2101]
            or lock['training_seeds'] != [42]):
        raise ValueError('Not the registered two-line single-seed confirmation')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    # JSON insertion order is not relied on: the permanent SFT must finish first.
    for name in ['SFT'] + [x for x in lock['models'] if x != 'SFT']:
        selected = lock['models'][name]
        alias = selected.get('reuse_identical_model')
        if alias:
            if alias not in results:
                raise ValueError('Alias must refer to an already evaluated identical export')
            results[name] = dict(results[alias], role=selected['role'], reuse_identical_model=alias)
        else:
            while True:
                require_dispatch()
                rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader'], text=True).splitlines()
                if len(rows) == 8 and len({x.split(',')[0] for x in rows}) == 8 and min(int(x.split(',')[1].split()[0]) for x in rows) >= 35000:
                    break
                atomic_json(root/'status.json', dict(status='WAITING_FOR_CAPACITY', model=name, utc_unix=time.time()))
                time.sleep(30)
            report = evaluate_one(args.final_lock, name, args.population, root/name, root/'SFT/scores', args.workers)
            results[name] = dict(report=str(root/name/'report.json'), report_sha256=report['sha256'],
                                 PDMS=100*report['metrics']['score']['current'], role=selected['role'])
        atomic_json(root/'status.json', dict(status='EVALUATING_LOCKED_MODELS', completed=results, utc_unix=time.time()))
    atomic_json(root/'COMPLETE.json', dict(status='COMPLETE_LOCKED_FINAL_NAVTEST', final_lock_sha256=lock['sha256'],
                models=results, seed=2101, unique_exports=lock['unique_export_count'], new_model_selection_performed=False))
    print(json.dumps(dict(status='COMPLETE_LOCKED_FINAL_NAVTEST', models=results)))


if __name__ == '__main__':
    main()
