"""Execute one already locked model with streaming v1 and auxiliary v2 scoring.

Resource placement is supplied after a live inventory. This wrapper never selects
a model, changes its inference path, or creates another evaluation seed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.analysis.flow_safe_improvement.run_learning_window import finish
from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import authorized
from scripts.analysis.flow_safe_improvement.sample_locked_navtest import locked_contract
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('run-root', 'final-lock', 'model'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--gpu-count', type=int, required=True)
    parser.add_argument('--cpu-workers', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args(); root = Path(a.run_root); host = authorized(require_dispatch())
    if not 1 <= a.gpu_count <= 8 or not 1 <= a.cpu_workers <= 16:
        raise ValueError('Explicit bounded resources are required')
    population = root/'population/manifest.json'
    lock, lock_sha, selected = locked_contract(a.final_lock, a.model, read(population))
    if selected.get('reuse_identical_model'):
        raise ValueError('Exact model aliases must reuse the sealed result, not rerun it')
    out = root/'evaluations/final_navtest'/a.model
    out.mkdir(parents=True, exist_ok=a.resume)
    guard = (out/'operator.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = ['run_locked_navtest', 'sample_locked_navtest', 'score_samples',
               'evaluate_locked_navtest', 'score_auxiliary_v2']
    identity = dict(final_lock_sha256=lock_sha, model=a.model, host=host,
                    gpu_count=a.gpu_count, cpu_workers=a.cpu_workers,
                    sources={s: file_sha(Path(__file__).with_name(s+'.py')) for s in sources})
    identity['sha256'] = fingerprint(identity)
    if (out/'identity.json').exists() and read(out/'identity.json') != identity:
        raise ValueError('Recovery changed the locked evaluation or worker identity')
    atomic_json(out/'identity.json', identity)
    if (out/'COMPLETE.json').exists():
        return
    snapshot = out/'source_snapshot'; snapshot.mkdir(exist_ok=True)
    for name in sources:
        source = Path(__file__).with_name(name+'.py'); saved = snapshot/source.name
        if saved.exists() and file_sha(saved) != file_sha(source):
            raise ValueError('Frozen evaluation source changed')
        if not saved.exists():
            saved.write_bytes(source.read_bytes())
    records = read(out/'phase_records.json') if (out/'phase_records.json').exists() else []
    for record in records:
        path = Path('/proc')/str(record['pid'])/'stat'
        if record.get('finished_unix') is None and path.exists():
            fields = path.read_text().rsplit(') ', 1)[1].split()
            if fields[19] == record.get('process_start_ticks') and fields[0] != 'Z':
                raise RuntimeError('Preserve the active owned child; do not duplicate this evaluation')
    state = dict(status='PREPARING_LOCKED_MODEL', model=a.model, host=host, pid=os.getpid())

    def save(**kwargs):
        state.update(kwargs); atomic_json(out/'state.json', state)
        atomic_json(out/'phase_records.json', records)

    def start(name, arguments, phase, distributed=False):
        authorized(host)
        command = [sys.executable]
        if distributed:
            command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={a.gpu_count}']
        command += ['-m', 'scripts.analysis.flow_safe_improvement.'+name, *map(str, arguments)]
        with (out/(phase+'.log')).open('a') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        record = dict(host=host, pid=process.pid, command=command, log=str(out/(phase+'.log')),
            started_unix=time.time(), phase=phase,
            process_start_ticks=(Path('/proc')/str(process.pid)/'stat').read_text().rsplit(') ', 1)[1].split()[19])
        records.append(record); save(status=phase)
        return process, record

    def complete(process, record):
        try:
            finish(process, record)
        finally:
            save()

    samples, scores, auxiliary, summary = [out/n for n in ('samples', 'v1', 'v2', 'summary')]
    common = ['--final-lock', a.final_lock, '--population', population]
    try:
        save()
        if not (scores/'COMPLETE').exists():
            complete_shards = [samples/f'shard{i}.json' for i in range(a.gpu_count)]
            sampler = sr = None
            if not all(p.exists() and read(p)['status'] == 'ASSIGNED_POPULATION_COMPLETE' for p in complete_shards):
                admissions = []
                visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
                if len(visible) != a.gpu_count or any(not x.isdigit() for x in visible):
                    raise ValueError('Bind the inventoried physical devices explicitly')
                for _ in range(2):
                    authorized(host)
                    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                        '--format=csv,noheader,nounits'], text=True).splitlines()
                    selected_rows = [r for r in rows if r.split(',')[0].strip() in visible]
                    if len(selected_rows) != a.gpu_count or min(float(r.split(',')[-1]) for r in selected_rows) < 16384:
                        raise RuntimeError('Headroom below the measured 7.40-GiB peak plus conservative reserve')
                    admissions.append(dict(rows=selected_rows, unix=time.time(), minimum_free_MiB=16384))
                    atomic_json(out/'GPU_admission.json', admissions); time.sleep(2)
                args = [*common, '--model', a.model, '--output', samples, '--shard-offset', 0,
                        '--shard-count', a.gpu_count]
                if (samples/'manifest.json').exists():
                    args += ['--resume']
                sampler, sr = start('sample_locked_navtest', args, 'sample', True)
                deadline = time.monotonic()+1800
                while not (samples/'manifest.json').exists():
                    if sampler.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Locked sampler failed before publishing its identity')
                    time.sleep(2)
            args = ['--samples', samples, '--population', population, '--cache-root', lock['v1_cache'],
                    '--workers', a.cpu_workers, '--output', scores]
            if (scores/'manifest.json').exists():
                args += ['--resume']
            scorer, cr = start('score_samples', args, 'v1')
            if sampler is not None:
                complete(sampler, sr)
            complete(scorer, cr)
        if not (auxiliary/'report.json').exists():
            args = [*common, '--samples', samples, '--cache-root', lock['v2_cache'], '--workers', a.cpu_workers,
                    '--evaluation-override', root/'resources/single_seed_override/REGISTRATION.json', '--output', auxiliary]
            if (auxiliary/'manifest.json').exists():
                args += ['--resume']
            proc, rec = start('score_auxiliary_v2', args, 'v2'); complete(proc, rec)
        baseline = root/'evaluations/final_navtest/SFT/v1'
        deadline = time.monotonic()+14400
        while not (baseline/'COMPLETE').exists():
            authorized(host); save(status='WAITING_FOR_MATCHED_SFT_SCORES')
            if time.monotonic() > deadline:
                raise TimeoutError('Matched baseline remains incomplete; retain all valid outputs')
            time.sleep(10)
        if not (summary/'report.json').exists():
            proc, rec = start('evaluate_locked_navtest', [*common, '--model', a.model, '--scores', scores,
                '--baseline-scores', baseline, '--output', summary], 'summary'); complete(proc, rec)
        result = read(summary/'report.json')
        if result['status'] != 'COMPLETE_LOCKED_FINAL_NAVTEST' or result['scenes'] != 12146:
            raise ValueError('A shard or partial result is not a completed Navtest')
        atomic_json(out/'COMPLETE.json', dict(identity_sha256=identity['sha256'], report_sha256=result['sha256'],
            auxiliary_report_sha256=file_sha(auxiliary/'report.json'), unique_scenes=12146, inference_seeds=[2101]))
        save(status='COMPLETE_LOCKED_FINAL_NAVTEST')
    except BaseException as error:
        save(status='FAILED_PRESERVE_SEALED_OUTPUTS_AND_OWNED_CHILDREN', error=repr(error))
        raise


if __name__ == '__main__':
    main()
