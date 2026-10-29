"""Finish pre-CUDA lease refusals; retain the original evaluation/source identity."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from tools.qwen_world_postcot.latest_snapshot_owner import SnapshotRequest, finalize
from tools.qwen_world_postcot.user_interim_owner import (
    read, write, sha, validate_complete_summary,
)


def admission_refusal(directory):
    """Only an unstarted shared-lease rejection is eligible for this recovery."""
    directory = Path(directory)
    row = read(directory/'result.json')
    return (row.get('status') == 'FAILED'
            and 'BlockingIOError: [Errno 11]' in row.get('error', '')
            and 'fcntl.flock(lease' in row.get('error', '')
            and not (directory/'process.json').exists()
            and not (directory/'worker.log').exists())


def alive(receipt):
    try:
        fields = (Path('/proc')/str(receipt['pid'])/'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and int(fields[19]) == receipt['start_ticks']
    except FileNotFoundError:
        return False


def finish_model(request, failures):
    for rank, original in failures:
        spec = read(original/'spec.json')
        if spec['source'] != str(request.source) or spec['source_sha'] != request.source_sha:
            raise ValueError('Changed worker source identity')
        command = spec['command']; module = spec['module']
        if (module != 'tools.qwen_world_postcot.evaluate'
                or command[command.index('--rank')+1] != str(rank)
                or command[command.index('--world')+1] != '8'):
            raise ValueError('Unexpected worker invocation')
        args = command[command.index(module)+1:]
        route = None
        while route is None:
            routes, inventory = request.resources.available(min_free_MiB=32768)
            route = request.route(routes, inventory)
            if route is None:
                time.sleep(15)
        name = f'rank{rank}_admission_recovery001'
        request.submit(name, module, args, route)
        while request.done(name) is None:
            time.sleep(15)
        if request.done(name) is not True:
            raise RuntimeError('Recovery failed; retain attempt and review, no automatic retry')
    summary = request.root/'SUMMARY.json'
    with (request.root/'admission_recovery_summary.log').open('x') as log:
        subprocess.run([request.resources.MODEL_PYTHON, '-m', 'tools.qwen_world_postcot.summarize',
                        '--evaluation', str(request.output), '--output', str(summary)],
                       cwd=request.source, stdout=log, stderr=subprocess.STDOUT, check=True)
    row = read(summary)
    population = read('/mnt/project/flow-safe-improvement-20261013/population/manifest.json')
    counts = validate_complete_summary(row, read(request.output/'identity.json'), population)
    write(request.root/'COMPLETE.json', dict(status='COMPLETE', PDMS=row['PDMS'], scenes=12146,
        logs=counts['manifest_logs'], physical_logs=counts['physical_logs'], all_owned_workers_exited=True,
        registration_sha256=read(request.reg)['sha256'], at=time.time(),
        recovery='Pre-CUDA lease refusal only; old invocations and successful scenes retained.'))
    request.publish(row)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--report-dir', required=True)
    args = parser.parse_args(); root = Path(args.root)
    lock = (root/'admission_recovery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = read(root/'owner.json'); source = Path(owner['source'])
    if alive(owner) or (root/'COMPLETE.json').exists():
        raise ValueError('Do not compete with a live/completed owner')
    for old_job in root.glob('*/jobs/*'):
        if not (old_job/'result.json').exists():
            raise ValueError('Original workers have not all drained: '+str(old_job))
    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=source, text=True).strip()
    if git('rev-parse', 'HEAD') != owner['source_sha'] or git('status', '--porcelain'):
        raise ValueError('Original runtime must remain clean and immutable until workers exit')
    # The new owner is scheduling/analysis only; every hash entering evaluator
    # identity, plus the evaluator and summarizer, remains the original file.
    requests = {}; retry = []
    for label, pin in read(root/'PINNED.json')['models'].items():
        request = SnapshotRequest(root/label, pin)
        request.source = source; request.source_sha = owner['source_sha']
        requests[label] = request
        if (request.root/'COMPLETE.json').exists():
            continue
        identity = read(request.output/'identity.json')
        for relative, expected in identity['source'].items():
            if sha(source/relative) != expected:
                raise ValueError('Changed original evaluator dependency: '+relative)
        failures = []
        for rank in range(8):
            job = request.root/'jobs'/f'rank{rank}'; result = read(job/'result.json')
            if result.get('status') == 'PROCESS_COMPLETE' and result.get('exit_code') == 0:
                continue
            if rank == 0 or not admission_refusal(job):
                raise ValueError('Failure is not a pre-CUDA admission refusal: '+str(job))
            failures.append((rank, job))
        retry.append((request, failures))
    write(root/'ADMISSION_RECOVERY_OWNER.json', dict(pid=os.getpid(), at=time.time(),
        source=str(source), source_sha=owner['source_sha'],
        start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        failed_invocations=[str(job) for _, failures in retry for _, job in failures],
        all_original_invocations_drained=True, original_failure_evidence_preserved=True))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: finish_model(*item), retry))
    if git('rev-parse', 'HEAD') != owner['source_sha'] or git('status', '--porcelain'):
        raise ValueError('Original source changed during recovery')
    report = Path(args.report_dir)
    finalize(root, requests, report)
    prefix = str(report.relative_to(source))
    for row in git('status', '--porcelain=v1', '-z', '--untracked-files=all').split('\0'):
        if row and not row[3:].startswith(prefix+'/'):
            raise ValueError('Unrelated publication change')
    subprocess.run(['git','add','--',prefix],cwd=source,check=True)
    subprocess.run(['git','commit','-m','Complete fixed RL midpoint Test after pre-CUDA admission refusals'],cwd=source,check=True)
    branch = git('branch','--show-current'); commit = git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=source,check=True)
    if git('ls-remote','origin','refs/heads/'+branch).split()[0] != commit:
        raise ValueError('Remote publication SHA mismatch')
    write(root/'PUBLICATION.json',dict(status='PUBLISHED',commit=commit,remote_SHA=commit,at=time.time()))
    write(root/'ADMISSION_RECOVERY_COMPLETE.json',dict(status='COMPLETE',at=time.time(),commit=commit))
    write(root/'COMPLETE.json',dict(status='COMPLETE',models=list(requests),at=time.time(),
        all_owned_workers_exited=True,recovery='ADMISSION_RECOVERY_COMPLETE.json'))


if __name__ == '__main__':
    main()
