"""Run fixed user-requested snapshots through the existing native evaluator.

Reuse the qualified spare-GPU lease and original export/parity/inference entries.
This operator changes routing only; the registered immutable model source runs
all model operations. It neither changes training nor selects from new scores.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--registration', required=True)
    args = parser.parse_args()
    control = Path(args.registration).resolve().parent
    plan = json.loads(Path(args.registration).read_text())
    source = Path(plan['source'])
    sys.path.insert(0, str(source))
    from starVLA.rl.qwen_sourceflow.resources import available, launch, MODEL_PYTHON
    from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha, require_dispatch
    from starVLA.rl.flow_grpo.transactions import atomic_json
    body = dict(plan)
    assert body.pop('sha256') == fingerprint(body)
    assert plan['seed'] == 2101 and plan['scenes_each'] == 12146
    assert plan['metric'] == 'canonical_NAVSIM_v1.1'
    assert plan['inference'] == 'single_candidate_source_shift5_ODE_K10'
    require_dispatch()
    lock = (control/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (control/'owner.json').exists(), 'Never duplicate/retry an invocation silently'
    for path, sha in [(source, plan['source_sha']), (Path(plan['analysis_source']), plan['analysis_source_sha'])]:
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip() == sha
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=path, text=True).strip()
    lease_path = Path(plan['lease_adapter'])
    assert file_sha(lease_path) == plan['lease_adapter_sha256']
    module_spec = importlib.util.spec_from_file_location('qualified_native_spare_lease', lease_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    lease = module.SpareNavtestLease(control)
    atomic_json(control/'owner.json', dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]), at=time.time()))
    launched = []

    def submit(name, module, arguments, route=None, rank=None, world=None):
        job = control/'jobs'/name
        job.mkdir(parents=True, exist_ok=False)
        command = [MODEL_PYTHON, '-u', '-m', module, *arguments]
        if rank is not None:
            command += ['--rank', str(rank), '--world', str(world)]
        spec = dict(source=str(source), source_sha=plan['source_sha'],
                    host=route['host'] if route else 'training-vla-zt',
                    gpus=[route['gpu']] if route else [], gpu_uuids=[route['uuid']] if route else [],
                    min_free_MiB=route['minimum_free_MiB'] if route else 0,
                    module=module, command=command, receipt_dir=str(job))
        atomic_json(job/'spec.json', spec)
        return job, spec

    def dispatch(jobs):
        for job, spec in jobs:
            atomic_json(job/'operator.json', launch(job/'spec.json', source))
            launched.append((job, spec))

    def wait(jobs, stage, *, fail=True):
        while True:
            done, failures = 0, []
            for job, _ in jobs:
                result = job/'result.json'
                if not result.exists():
                    continue
                row = json.loads(result.read_text())
                if row['status'] == 'PROCESS_COMPLETE' and row['exit_code'] == 0:
                    done += 1
                else:
                    failures.append(str(result))
            counts = {key: sum(1 for _ in (Path(value['output'])/'scenes').glob('*.json')) for key, value in plan['models'].items()}
            atomic_json(control/'status.json', dict(at=time.time(), stage=stage, scenes=counts,
                        total_each=12146, completed_stage_workers=done, stage_workers=len(jobs), failed_workers=failures))
            if done+len(failures) == len(jobs):
                if failures and fail:
                    raise RuntimeError('Failure retained; no automatic retry: '+str(failures))
                return
            time.sleep(30)

    def analysis(module, arguments):
        subprocess.run([MODEL_PYTHON, '-m', module, *arguments], cwd=plan['analysis_source'], check=True)

    try:
        lease.enter()
        rl = [label for label in plan['models'] if label != 'IL']
        exports = [submit(label+'_export', 'scripts.analysis.qwen_sourceflow.export_policy',
                   ['--checkpoint', plan['models'][label]['checkpoint'], '--output', plan['models'][label]['bundle']]) for label in rl]
        atomic_json(control/'export_dispatch.json', dict(jobs=[spec for _, spec in exports], at=time.time()))
        dispatch(exports)
        wait(exports, 'CPU_EXPORT')
        routes, inventory = available(min_free_MiB=70000)
        spare = [row for row in routes if row['host'] in lease.hosts and row['gpu'] in lease.hosts[row['host']]]
        assert len(spare) >= len(rl), 'Insufficient eligible parity capacity; preserve fixed snapshots'
        parity = []
        for label, route in zip(rl, spare):
            route = dict(route, minimum_free_MiB=70000)
            output = control/'parity'/f'{label}.json'
            output.parent.mkdir(exist_ok=True)
            parity.append(submit(label+'_export_parity', 'scripts.analysis.qwen_sourceflow.validate_rl_deployment',
                          ['--checkpoint', plan['models'][label]['checkpoint'], '--bundle', plan['models'][label]['bundle'], '--output', str(output)], route))
        atomic_json(control/'parity_dispatch.json', dict(jobs=[spec for _, spec in parity], at=time.time()))
        dispatch(parity)
        wait(parity, 'EXPORT_ODE_PARITY')
        for label in rl:
            proof = json.loads((control/'parity'/f'{label}.json').read_text())
            assert proof['status'] == 'PASS' and proof['updates'] == plan['models'][label]['updates']
        routes, inventory = available(min_free_MiB=plan['resources']['minimum_shared_free_MiB'])
        spare = sorted([row for row in routes if row['host'] in lease.hosts and row['gpu'] in lease.hosts[row['host']] and row['free_MiB'] >= 70000], key=lambda row: (row['host'], row['gpu']))
        shared = [row for row in routes if row['host'] == plan['resources']['additional_shared_host']]
        assert spare and len({row['uuid'] for row in spare+shared}) == len(spare+shared)
        # Only spare devices get a second inference process. Shared devices
        # already carry one preserved training process and get exactly one.
        slots = [dict(row, wave=0, minimum_free_MiB=70000) for row in spare]
        slots += [dict(row, wave=0, minimum_free_MiB=plan['resources']['minimum_shared_free_MiB']) for row in shared]
        slots += [dict(row, wave=1, minimum_free_MiB=45000) for row in spare]
        atomic_json(control/'evaluation_admission.json', dict(at=time.time(), routes=slots, inventory=inventory,
                    physical_GPUs=len(spare+shared), worker_slots=len(slots), preserve_all_training=True,
                    shared_GPU_worker_time_is_not_exclusive_compute=True))
        jobs, layouts = [], {}
        labels = list(plan['models'])
        parent = json.loads(Path(plan['parent_selection']).read_text())['selected']['checkpoint']
        for index, label in enumerate(labels):
            selected = slots[index::len(labels)]
            assert selected
            layouts[label] = len(selected)
            model = plan['models'][label]
            arguments = ['--checkpoint', model['checkpoint'] if label == 'IL' else parent,
                         '--output', model['output'], '--cohort', 'NAVTEST', '--data', plan['navtest_data']]
            if label == 'IL':
                arguments += ['--interim-navtest-registration', model['interim_registration']]
            else:
                arguments += ['--parent-selection', plan['parent_selection'], '--rl-export', model['bundle']]
            for rank, route in enumerate(selected):
                jobs.append(submit(f'eval_{label}_rank{rank:03d}', 'scripts.analysis.qwen_sourceflow.evaluate_policy', arguments, route, rank, len(selected)))
        atomic_json(control/'evaluation_dispatch.json', dict(at=time.time(), workers=[spec for _, spec in jobs], layouts=layouts,
                    unique_physical_GPUs=len(spare+shared), worker_slots=len(slots), maximum_total_workers_per_GPU=2,
                    fixed_single_candidates_each=12146, interim_user_request=True))
        first = [item for item in jobs if any(row['wave'] == 0 and row['uuid'] == item[1]['gpu_uuids'][0] and row['minimum_free_MiB'] == item[1]['min_free_MiB'] for row in slots)]
        second = [item for item in jobs if item not in first]
        dispatch(first)
        deadline = time.time()+900
        while True:
            ready, errors = 0, []
            for job, spec in first:
                result = job/'result.json'
                if result.exists() and json.loads(result.read_text()).get('exit_code') != 0:
                    errors.append(str(result))
                command = spec['command']
                output = Path(command[command.index('--output')+1])
                rank = int(command[command.index('--rank')+1])
                if any(json.loads(path.read_text()).get('rank') == rank for path in (output/'invocations').glob('*.json')):
                    ready += 1
            atomic_json(control/'loading_status.json', dict(at=time.time(), first_wave_ready=ready, total_first_wave=len(first), errors=errors))
            if errors:
                raise RuntimeError('First-wave failure: '+str(errors))
            if ready == len(first):
                break
            if time.time() > deadline:
                raise TimeoutError('First-wave loading deadline; retain all outputs')
            time.sleep(15)
        dispatch(second)
        atomic_json(control/'DISPATCH_COMPLETE.json', dict(at=time.time(), workers=len(jobs), layouts=layouts,
                    models={key: value['updates'] for key, value in plan['models'].items()}))
        wait(jobs, 'NAVTEST_INFERENCE_AND_CANONICAL_SCORING')
        summaries = {}
        for label, model in plan['models'].items():
            summary = control/'analysis'/f'{label}_summary.json'
            summary.parent.mkdir(exist_ok=True)
            analysis('scripts.analysis.qwen_sourceflow.summarize_results', ['--evaluation', model['output'], '--output', str(summary)])
            paired = control/'analysis'/f'{label}_paired_vs_IL53k.json'
            module = 'scripts.analysis.qwen_sourceflow.analyze_il_milestones' if label == 'IL' else 'scripts.analysis.qwen_sourceflow.analyze_paired'
            analysis(module, ['--evaluation', model['output'], '--reference', plan['baseline_evaluation'], '--output', str(paired)])
            summaries[label] = json.loads(summary.read_text())
        atomic_json(control/'worker_exit_verification.json', dict(status='PASS', workers=len(jobs),
                    all_native_children_reaped_with_exit0=True, result_receipts=[str(job/'result.json') for job, _ in jobs]))
        atomic_json(control/'COMPLETE.json', dict(status='COMPLETE', at=time.time(), interim_only=True,
                    analysis_source_sha=plan['analysis_source_sha'], models={key: dict(updates=plan['models'][key]['updates'],
                    PDMS=value['PDMS'], scenes=value['scenes'], logs=value['logs'], summary=str(control/'analysis'/f'{key}_summary.json')) for key, value in summaries.items()}))
    except BaseException:
        atomic_json(control/'failure.json', dict(at=time.time(), error=traceback.format_exc(), automatic_retry=False))
        # A policy lease cannot close while its launched native children still
        # need admission. Drain successful work without deleting scenes.
        wait(launched, 'FAILURE_DRAIN', fail=False)
        raise
    finally:
        lease.close()


if __name__ == '__main__':
    main()
