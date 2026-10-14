#!/usr/bin/env python3
"""Finish the fixed campaign after the bounded expert matrix; never starts Base/navtest."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from iqe.contracts import require
from iqe.io import atomic_json, read_json, file_hash, lock
from iqe.specialist_study.config import load_config
from iqe.specialist_study.resource_guard import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpus', type=int, nargs=4, required=True)
    parser.add_argument('--deadline-hours', type=float, default=24.)
    args = parser.parse_args()
    require(len(set(args.gpus)) == 4 and 0 < args.deadline_hours <= 36, 'four explicit GPUs and bounded deadline required')
    config = load_config(args.config)
    repo = Path(__file__).resolve().parents[2]
    output = config.root / 'runtime/completion_driver'
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.deadline_hours * 3600
    env = dict(os.environ, PYTHONPATH=str(repo), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def tick(stage, detail):
        require(time.monotonic() < deadline, 'bounded orchestration deadline reached; preserve existing checkpoints')
        atomic_json(output / 'STATUS.json', {'stage': stage, 'detail': detail, 'host': os.uname().nodename,
                                             'pid': os.getpid(), 'updated_unix': time.time()})

    def launch(command, gpu, *extra):
        check(config, command)
        from iqe.specialist_study.capacity import probe
        while True:
            capacity = probe(gpu)
            atomic_json(output / ('CAPACITY_gpu_'+str(gpu)+'.json'), capacity)
            if capacity['ready']:
                break
            tick('WAITING_CAPACITY', {'command': command, 'gpu': gpu, 'capacity': capacity})
            time.sleep(20)
        name = command + ('_' + extra[1] if extra and extra[0] == '--cell' else '')
        previous = sorted(output.glob(name + '_attempt_*.json'))
        attempt = len(previous) + 1
        command_line = [sys.executable, '-u', '-m', 'iqe.specialist_study.cli', command,
                        '--config', args.config, '--device', 'cuda:0', *extra]
        log = open(output / (name + '.log'), 'ab')
        process = subprocess.Popen(command_line, cwd=repo, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        atomic_json(output / f'{name}_attempt_{attempt:04d}.json', {'pid': process.pid, 'GPU': gpu,
            'command': command_line, 'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'config_hash': config.identity, 'Base_binding': file_hash(config.root / 'BASE_BINDING.json'),
            'started_unix': time.time()}, immutable=True)
        return process

    def wait(process, stage):
        while process.poll() is None:
            tick(stage, {'pid': process.pid})
            time.sleep(30)
        require(process.returncode == 0, f'{stage} failed ({process.returncode}); no automatic experiment replacement')

    def run(command, gpu, *extra):
        wait(launch(command, gpu, *extra), command)

    def resume_args(folder):
        return ['--resume', 'auto'] if list(folder.glob('step_*.pt.COMPLETE.json')) else []

    with lock(output / 'RUN.lock'):
        check(config, 'run-specialist-scorer-study')
        while True:
            matrix_path = config.root / 'EXPERT_MATRIX_DATA.json'
            if matrix_path.exists():
                matrix = read_json(matrix_path)
                required = [cell for cell, spec in matrix['cells'].items() if spec.get('status', 'READY') == 'READY']
                complete = [cell for cell in required if (config.root / 'experts' / cell / 'result.json').exists()
                            and read_json(config.root / 'experts' / cell / 'result.json')['status'] == 'COMPLETE']
                if len(complete) == len(required):
                    break
                status = read_json(config.root / 'runtime/expert_matrix_driver/STATUS.json')
                require(status.get('status') not in ('FINISHED', 'FAILED'), 'expert matrix ended with incomplete cells')
            tick('WAITING_EXPERT_MATRIX', {'dependency': str(matrix_path)})
            time.sleep(30)
        run('evaluate-expert-matrix', args.gpus[0])

        from iqe.specialist_study.data import SPECIALTIES
        from iqe.specialist_study.experiments import extension_manifest
        selection = read_json(config.root / 'EXPERT_RECIPE_SELECTION.json')
        active = []
        if selection['scientific_status'] != 'NO_HELDOUT_ORACLE_GAIN':
            for specialty in SPECIALTIES:
                if specialty == selection['pilot']:
                    continue
                manifest = extension_manifest(config, specialty, selection['selected'])
                if not manifest['scene_ids']:
                    continue
                cell = 'EXT_' + specialty
                folder = config.root / 'experts' / cell
                if (folder / 'result.json').exists() and read_json(folder / 'result.json')['status'] == 'COMPLETE':
                    continue
                active.append(launch('train-expert-cell', args.gpus[len(active)], '--cell', cell, *resume_args(folder)))
            for process in active:
                wait(process, 'SPECIALIST_EXTENSION')
        run('freeze-candidate-pool', args.gpus[0])
        pool = read_json(config.root / 'SERVING_CANDIDATE_POOL.json')
        run('build-deployment-bank', args.gpus[0])
        if pool['research_only_no_oracle_gain']:
            run('train-scorer-cell', args.gpus[0], '--cell', 'S00_DIAGNOSTIC', '--stop-after', '300',
                *resume_args(config.root / 'diagnostics/S00_DIAGNOSTIC'))
            tick('COMPLETE_NO_HELDOUT_ORACLE_GAIN', {'extra_Scorer_matrix_skipped': True,
                                                   'diagnostic': str(config.root / 'diagnostics/S00_DIAGNOSTIC')})
            return

        rich = launch('build-rich-bank', args.gpus[1])
        semantic = launch('train-scorer-cell', args.gpus[0], '--cell', 'S00', *resume_args(config.root / 'scorers/S00'))
        while not (config.root / 'SPATIAL_CONTEXT_COMPLETE.json').exists():
            require(rich.poll() in (None, 0) and semantic.poll() in (None, 0), 'bank/semantic training failed')
            tick('WAITING_SPATIAL_CACHE', {'semantic_pid': semantic.pid, 'rich_bank_pid': rich.pid})
            time.sleep(30)
        from iqe.specialist_study.context_parallel import verify as verify_spatial
        topology = read_json(config.root / 'runtime/ACTIVE_SPATIAL_TOPOLOGY.json')
        plan = read_json(topology['plan'])
        verify_spatial(config, plan['topology'])
        spatial = launch('train-scorer-cell', args.gpus[2], '--cell', 'S10', *resume_args(config.root / 'scorers/S10'))
        wait(rich, 'RICH_BANK')
        semantic_rich = launch('train-scorer-cell', args.gpus[1], '--cell', 'S01', *resume_args(config.root / 'scorers/S01'))
        spatial_rich = launch('train-scorer-cell', args.gpus[3], '--cell', 'S11', *resume_args(config.root / 'scorers/S11'))
        for process in (semantic, semantic_rich, spatial, spatial_rich):
            wait(process, 'SCORER_MATRIX')
        for command in ('evaluate-scorer-matrix', 'calibrate-selection', 'evaluate-final-system', 'benchmark-system'):
            run(command, args.gpus[0])
        tick('COMPLETE', {'verdict': str(config.root / 'VERDICT.json'), 'ACTIVE_updated': False,
                         'Base_training_untouched': True})


if __name__ == '__main__':
    main()
