"""Finite, registered follow-up evaluation of sealed original-format exports.

This composes existing inference/scoring commands. It does not train, select
candidates, change a checkpoint, or expand the registered evaluation population.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.analysis.flow_safe_improvement.run_learning_window import module, finish
from scripts.analysis.flow_safe_improvement.sample_original_export import PROFILE
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch, safety_decision


def authorized(expected):
    host = require_dispatch()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if host != expected:
        raise RuntimeError('Evaluation queue moved to an unregistered host')
    for item in policy.get('global_dispatch_prohibitions', {}).values():
        if (host == item.get('host') or host in item.get('aliases', [])) and item.get('new_task_dispatch_allowed') is False:
            raise RuntimeError('Latest exact-host withdrawal overrides older allowlists')
    return host


def launch(name, args, log, records, host, distributed=False):
    authorized(host)
    command = module(name, args, distributed=distributed)
    with Path(log).open('a') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    records.append(dict(host=host, pid=process.pid, command=command, log=str(log), started_unix=time.time()))
    return process, records[-1]


def checked_report(base, name, weight_sha):
    sample = base/f'{name}_DEV5_samples'
    report_path = base/f'{name}_DEV5_summary'/'report.json'
    auxiliary = base/f'{name}_DEV5_v2'/'report.json'
    if not report_path.exists() or not auxiliary.exists():
        return None
    manifest = json.loads((sample/'manifest.json').read_text())
    report = json.loads(report_path.read_text())
    if (manifest['behavior_sha256'] != weight_sha or manifest['execution_profile']['name'] != PROFILE
            or report['status'] != 'COMPLETE_MATCHED_EVALUATION' or report['scenes'] != 1696
            or not report['no_best_of_N'] or manifest['seeds'] != [2101, 2102, 2103, 2104, 2105]):
        raise ValueError('Completed evaluation has the wrong deployment identity')
    return report


def summary_for(base, name):
    direct = base/f'{name}_DEV5_summary'/'report.json'
    alias = base/f'{name}_IDENTICAL_EXPORT.json'
    if not direct.exists() and alias.exists():
        receipt = json.loads(alias.read_text())
        direct = base/f"{receipt['source_name']}_DEV5_summary"/'report.json'
        if direct.exists() and json.loads(direct.read_text())['sha256'] != receipt['source_report_sha256']:
            raise ValueError('Reused report identity changed')
    return direct


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    registration = json.loads(Path(a.registration).read_text())
    host = authorized(registration['host'])
    jobs = registration['jobs']
    if not jobs or len(jobs) > 25 or len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('A finite unique list of at most 25 registered models is required')
    if any(not j['name'].replace('_', '').isalnum() for j in jobs):
        raise ValueError('Invalid evaluation name')
    base, out = Path(registration['evaluation_root']), Path(a.output)
    out.mkdir(parents=True, exist_ok=a.resume)
    lock = (out/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(registration_sha256=file_sha(a.registration), source_sha256=file_sha(__file__), host=host)
    identity['sha256'] = fingerprint(identity)
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text()) != identity:
        raise ValueError('Recovery changed the finite queue identity')
    atomic_json(out/'identity.json', identity)
    records = json.loads((out/'phase_records.json').read_text()) if (out/'phase_records.json').exists() else []
    state = json.loads((out/'state.json').read_text()) if (out/'state.json').exists() else dict(completed={})
    deadline = time.monotonic()+registration['max_wait_hours']*3600
    digests = {}

    def sealed_digest(path):
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        cached = digests.get(str(path))
        if cached and cached[0] != signature:
            raise ValueError('Previously verified immutable export changed')
        if cached is None:
            cached = (signature, file_sha(path))
            digests[str(path)] = cached
        return cached[1]

    def save(**kwargs):
        state.update(kwargs)
        atomic_json(out/'state.json', state)
        atomic_json(out/'phase_records.json', records)

    try:
        while len(state['completed']) < len(jobs):
            authorized(host)
            progressed = False
            for job in jobs:
                name, export = job['name'], Path(job['export'])
                if name in state['completed'] or not (export/'COMPLETE').exists():
                    continue
                weight_sha = json.loads((export/'export_state.json').read_text())['files']['pytorch_model.pt']
                if isinstance(weight_sha, dict):
                    weight_sha = weight_sha['sha256']
                if sealed_digest(export/'pytorch_model.pt') != weight_sha:
                    raise ValueError('Sealed export weights changed')
                previous = job.get('previous_name')
                previous_path = summary_for(base, previous) if previous else None
                if previous_path and not previous_path.exists():
                    continue
                alias = job.get('reuse_if_identical')
                if alias:
                    other_export = Path(alias['export'])
                    if not (other_export/'COMPLETE').exists():
                        continue
                    identical = (sealed_digest(other_export/'pytorch_model.pt') == weight_sha
                                 and sealed_digest(other_export/'config.yaml') == sealed_digest(export/'config.yaml'))
                    if identical:
                        report = checked_report(base, alias['name'], weight_sha)
                        if report is None:
                            continue
                        receipt = dict(status='REUSE_IDENTICAL_FULL_EXPORT_AND_CONFIG', name=name,
                                       source_name=alias['name'], export=str(export), source_export=str(other_export),
                                       weight_sha256=weight_sha, config_sha256=file_sha(export/'config.yaml'),
                                       source_report_sha256=report['sha256'], logical_scoring_calls_per_protocol=8480,
                                       actual_new_scoring_calls=0, actual_new_GPU_generation=0,
                                       actual_training_was_separate=True)
                        atomic_json(base/f'{name}_IDENTICAL_EXPORT.json', receipt)
                        state['completed'][name] = receipt
                        save(status='RUNNING'); progressed = True
                        continue
                existing = checked_report(base, name, weight_sha)
                if existing:
                    report = existing
                else:
                    save(status='EVALUATING_SEALED_ORIGINAL_EXPORT', current=name, weight_sha256=weight_sha)
                    # The source snapshot is small, immutable and contains no data or weights.
                    snapshot = out/'source_snapshot'
                    snapshot.mkdir(exist_ok=True)
                    for source in (Path(__file__), Path(__file__).with_name('sample_original_export.py'),
                                   Path(__file__).with_name('score_samples.py'), Path(__file__).with_name('evaluate_original_policy.py'),
                                   Path(__file__).with_name('score_auxiliary_v2.py')):
                        dest = snapshot/source.name
                        if dest.exists() and file_sha(dest) != file_sha(source):
                            raise ValueError('Evaluation source changed during the queue')
                        if not dest.exists():
                            dest.write_bytes(source.read_bytes())
                    samples, scores, summary, v2 = [base/f'{name}_DEV5_{suffix}' for suffix in ('samples', 'v1', 'summary', 'v2')]
                    if not (summary/'report.json').exists():
                        args = ['--export', export, '--export-sha256', weight_sha, '--population', registration['population'], '--output', samples]
                        if (samples/'manifest.json').exists():
                            args += ['--resume']
                        sampler, sr = launch('sample_original_export', args, base/f'{name}_DEV5_sample.log', records, host, True)
                        save()
                        start = time.monotonic()
                        while not (samples/'manifest.json').exists():
                            if sampler.poll() is not None or time.monotonic()-start > 900:
                                raise RuntimeError('Original sampler failed before manifest publication')
                            time.sleep(2)
                        args = ['--samples', samples, '--population', registration['population'], '--cache-root', registration['v1_cache'],
                                '--workers', 8, '--output', scores]
                        if (scores/'manifest.json').exists():
                            args += ['--resume']
                        scorer, cr = launch('score_samples', args, base/f'{name}_DEV5_v1.log', records, host)
                        save(); finish(sampler, sr); finish(scorer, cr)
                        args = ['--scores', scores, '--baseline-scores', base/'SFT_DEV5_v1',
                                '--reference-scores', registration['reference_scores'], '--population', registration['population'], '--output', summary]
                        proc, record = launch('evaluate_original_policy', args, base/f'{name}_DEV5_summary.log', records, host)
                        save(); finish(proc, record)
                    report = json.loads((summary/'report.json').read_text())
                    if not (v2/'report.json').exists():
                        args = ['--samples', samples, '--population', registration['population'], '--cache-root', registration['v2_cache'], '--workers', 8, '--output', v2]
                        if (v2/'manifest.json').exists():
                            args += ['--resume']
                        proc, record = launch('score_auxiliary_v2', args, base/f'{name}_DEV5_v2.log', records, host)
                        save(); finish(proc, record)
                    checked_report(base, name, weight_sha)
                previous_decision = None
                decision_path = base/f'{previous}_DEV5_safety.json' if previous else None
                if previous:
                    if decision_path.exists():
                        previous_decision = json.loads(decision_path.read_text())
                    else:
                        # Earlier already-completed correction evaluations predate this queue.
                        previous_decision = safety_decision(json.loads(previous_path.read_text()))
                decision = safety_decision(report, previous_decision)
                atomic_json(base/f'{name}_DEV5_safety.json', decision)
                state['completed'][name] = dict(status='COMPLETE', weight_sha256=weight_sha,
                                               report_sha256=report['sha256'], safety=decision['status'])
                save(status='RUNNING'); progressed = True
                if decision['status'] == 'PAUSE_VARIANT':
                    save(status='PAUSE_REQUIRED_PRESERVE_CHECKPOINTS', affected_job=name)
                    raise RuntimeError('Original-profile safety boundary triggered; inspect the associated training variant')
            if len(state['completed']) == len(jobs):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Finite registered queue still has missing exports; no missing result is imputed')
            if not progressed:
                save(status='WAITING_FOR_REGISTERED_EXPORTS')
                time.sleep(15)
        save(status='COMPLETE_FINITE_REGISTERED_QUEUE')
    except BaseException as error:
        save(status='FAILED_PRESERVE_ALL_EVIDENCE', error=repr(error))
        raise


if __name__ == '__main__':
    main()
