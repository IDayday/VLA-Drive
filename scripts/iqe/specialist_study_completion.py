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


class FinishedProcess:
    pid=0
    returncode=0
    def poll(self):return 0


class AdoptedProcess:
    """Observe a verified existing child without signalling or relaunching it."""
    def __init__(self, record, completion):
        self.record=record;self.pid=record['pid'];self.completion=completion;self.returncode=None
        self.poll()

    def poll(self):
        path=Path('/proc')/str(self.pid)
        if path.exists():
            fields=(path/'stat').read_text().split()
            require(fields[21]==self.record['start_ticks'], 'adopted PID was reused')
            if fields[2]!='Z':
                actual=(path/'cmdline').read_bytes().split(b'\0')
                require([v.decode() for v in actual if v]==self.record['command'], 'adopted process command changed')
                require(str((path/'cwd').resolve())==self.record['cwd'], 'adopted process cwd changed')
                return None
        self.returncode=0 if self.completion() else 1
        return self.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpus', type=int, nargs=4, required=True)
    parser.add_argument('--deadline-hours', type=float, default=24.)
    parser.add_argument('--phase', choices=('all','post_expert'), default='all')
    parser.add_argument('--adopt-existing', help='Verified local PID/start-time receipt; never kills child processes')
    parser.add_argument('--spatial-reader', choices=('legacy','mmap_v1'), default='legacy')
    parser.add_argument('--spatial-records', type=int, default=32768)
    args = parser.parse_args()
    require(len(set(args.gpus)) == 4 and 0 < args.deadline_hours <= 36, 'four explicit GPUs and bounded deadline required')
    config = load_config(args.config)
    repo = Path(__file__).resolve().parents[2]
    output = config.root / 'runtime/completion_driver'
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.deadline_hours * 3600
    env = dict(os.environ, PYTHONPATH=str(repo), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    require(1<=args.spatial_records<=32768,'bounded mapped spatial records')
    if args.spatial_reader=='mmap_v1':
        qualification=read_json(config.root/'qualifications/spatial_reader_mmap_v1/REPORT.json')
        require(qualification['status']=='PASS' and qualification['all_input_tensor_bytes_equal']
                and qualification['RNG_unchanged'] and qualification['gradient_Adam_parameter_update_equal']
                and qualification['reader_code_hash']==file_hash(repo/'iqe/specialist_study/context_reader.py')
                and qualification['Base_binding_hash']==file_hash(config.root/'BASE_BINDING.json')
                and qualification['deployment_bank_hash']==file_hash(config.root/'banks/deploy/scorer_fit/BANK.json'),
                'mmap reader qualification missing or stale')
        env.update(IQE_SPATIAL_READER_MODE='mmap_v1',IQE_SPATIAL_MMAP_RECORDS=str(args.spatial_records))
    adopted=read_json(args.adopt_existing) if args.adopt_existing else None
    if adopted:
        require(adopted['host']==os.uname().nodename and adopted['config_identity']==config.identity
                and adopted['Base_binding_hash']==file_hash(config.root/'BASE_BINDING.json'),
                'controller adoption host/config/Base changed')

    def tick(stage, detail):
        require(time.monotonic() < deadline, 'bounded orchestration deadline reached; preserve existing checkpoints')
        atomic_json(output / 'STATUS.json', {'stage': stage, 'detail': detail, 'host': os.uname().nodename,
                                             'pid': os.getpid(), 'updated_unix': time.time()})

    def launch(command, gpu, *extra):
        check(config, command)
        name = command + ('_' + extra[1] if extra and extra[0] == '--cell' else '')
        if command=='train-scorer-cell':
            cell=extra[1];result=config.root/'scorers'/cell/'result.json'
            if result.exists() and read_json(result)['status']=='COMPLETE':
                return FinishedProcess()
        if command=='build-rich-bank' and (config.root/'banks/rich/scorer_fit/BANK.json').exists():
            return FinishedProcess()
        if adopted and name in adopted['processes']:
            record=adopted['processes'][name]
            def complete():
                if command=='train-scorer-cell':
                    result=config.root/'scorers'/extra[1]/'result.json'
                    return result.exists() and read_json(result)['status']=='COMPLETE'
                return (config.root/'banks/rich/scorer_fit/BANK.json').exists()
            process=AdoptedProcess(record,complete)
            atomic_json(output/(name+'_ADOPTED_'+str(os.getpid())+'.json'),
                        {'process':record,'adoption_receipt_hash':file_hash(args.adopt_existing),
                         'new_controller_pid':os.getpid(),'updated_unix':time.time()},immutable=True)
            return process
        from iqe.specialist_study.capacity import probe
        while True:
            capacity = probe(gpu)
            atomic_json(output / ('CAPACITY_gpu_'+str(gpu)+'.json'), capacity)
            if capacity['ready']:
                break
            tick('WAITING_CAPACITY', {'command': command, 'gpu': gpu, 'capacity': capacity})
            time.sleep(20)
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
            'config_hash': config.identity, 'config_file_sha256':file_hash(args.config),
            'Base_binding': file_hash(config.root / 'BASE_BINDING.json'),
            'host':os.uname().nodename,'start_ticks':Path(f'/proc/{process.pid}/stat').read_text().split()[21],
            'spatial_reader':args.spatial_reader,'spatial_record_capacity':args.spatial_records,
            'started_unix': time.time()}, immutable=True)
        return process

    def wait(process, stage):
        if isinstance(process,FinishedProcess):
            require(process.returncode==0,'completed stage failure');return
        while process.poll() is None:
            tick(stage, {'pid': process.pid})
            time.sleep(30)
        if process.returncode != 0:
            tick('FAILED', {'failed_stage': stage, 'pid': process.pid,
                            'returncode': process.returncode, 'artifacts_preserved': True})
        require(process.returncode == 0, f'{stage} failed ({process.returncode}); no automatic experiment replacement')

    def run(command, gpu, *extra):
        wait(launch(command, gpu, *extra), command)

    def resume_args(folder):
        return ['--resume', 'auto'] if list(folder.glob('step_*.pt.COMPLETE.json')) else []

    with lock(output / 'RUN.lock'):
        check(config, 'run-specialist-scorer-study')
        if args.phase=='post_expert':
            require((config.root/'SERVING_CANDIDATE_POOL.json').exists(), 'post-expert phase needs frozen pool')
            from iqe.specialist_study.bank import load_bank
            for role in ('scorer_fit','stage_val','selector_cal','dev_report'):
                load_bank(config,'deploy',role)
        if args.phase=='all':
            pre_scorer(config,args,run,launch,wait,resume_args,tick)
        pool = read_json(config.root / 'SERVING_CANDIDATE_POOL.json')
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


def pre_scorer(config,args,run,launch,wait,resume_args,tick):
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
        run('build-deployment-bank', args.gpus[0])


if __name__ == '__main__':
    main()
