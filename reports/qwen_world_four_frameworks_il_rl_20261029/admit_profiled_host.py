"""One-time measured host admission through the unchanged Campaign driver API.

Only the CPU-load proxy for the explicitly requested host is bypassed. GPU
ownership/leases, RAM, source, recipe and budgets remain checked. No monkeypatch
or training-controller source edit; the original owner resumes immediately.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def alive(record):
    try:
        fields = (Path('/proc')/str(record['pid'])/'stat').read_text().split()
        return fields[2] not in ('Z', 'X') and int(fields[21]) == record['start_ticks']
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--architecture', choices=list('ABCD'), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    runs = read(root/'RUNS.json');plan = runs['plan']
    controller = Path(plan['controller_source'])
    sys.path.insert(0, str(controller))
    os.environ.update(WORLD_COMPARE_ROOT=str(root), WORLD_COMPARE_TASK='qwen_world_compare_20261029')
    from tools.qwen_world_compare.campaign import Campaign, expected_source_files
    from tools.qwen_world_compare import dispatch
    from starVLA.rl.qwen_world_compare.resources import probe, gpu_activity_issue
    from starVLA.rl.qwen_world_compare.contracts import allowed_hosts, file_sha
    from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
    from starVLA.model.modules.qwen_world_compare.factory import load_config

    assert expected_source_files(plan['source']) == runs['source_files']
    identity = read(args.profile/'IDENTITY.json')
    assert identity['qualification'] is True and identity['architecture_id'] == args.architecture
    assert identity['config'] == load_config(plan['architectures'][args.architecture]['config'])
    assert read(args.profile/'INVOCATION_COMPLETE.json')['profile_no_save'] is True
    samples = []
    for rank in range(8):
        rows = [json.loads(line) for line in (args.profile/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
        assert [r['updates'] for r in rows] == list(range(1, 7))
        assert all(r['optimizer_skipped_steps'] == 0 and r['optimizer_global_steps'] == r['updates'] for r in rows)
        assert all(math.isfinite(v) for row in rows for v in row['losses'].values())
        samples.append([r['slowest_rank_seconds'] for r in rows[-4:]])
    critical = [max(row[i] for row in samples) for i in range(4)]
    # Resource admission only; never a reward-based architecture screen.
    assert statistics.median(critical) < 20., 'Actual CPU contention is too severe for this admission'
    args.receipt.mkdir(parents=True, exist_ok=False)
    old_owner = read(root/'campaign/OWNER.json')
    argv = read(root/'campaign/LAUNCH_fcd43eb_attempt001.json')['argv']
    assert alive(old_owner)
    command = (Path('/proc')/str(old_owner['pid'])/'cmdline').read_bytes().decode().split('\0')
    assert 'tools.qwen_world_compare.campaign' in command
    atomic_json(args.receipt/'REQUEST.json', dict(host=args.host, architecture=args.architecture,
        user_instruction='还有空闲资源，vla-zt tsinghua-zt5都空闲', prior_owner=old_owner,
        script=str(Path(__file__).resolve()), script_sha256=file_sha(__file__),
        qualification=str(args.profile), critical_joint_window_seconds=critical,
        median_joint_update_seconds=statistics.median(critical), admission_limit_seconds=20.,
        only_override='CPU-load proxy for this measured host; original whole-GPU and RAM admission retained',
        scientific_source=plan['source'], controller_source=str(controller),
        source_files_unchanged=True, qualification_weights_used=False))
    restart_needed = False
    try:
        os.kill(old_owner['pid'], signal.SIGTERM)
        print('Owned CPU campaign owner stopping; all training workers preserved', flush=True)
        deadline = time.monotonic()+150
        while alive(old_owner):
            if time.monotonic() > deadline:
                raise TimeoutError('Original CPU owner remains alive; no duplicate owner or dispatch')
            time.sleep(.25)
        restart_needed = True
        with (root/'campaign/owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(args.receipt/'STATE_BEFORE.json', read(root/'campaign/STATE.json'))

            class MeasuredHostDriver:
                launch = staticmethod(dispatch.launch)
                job_state = staticmethod(dispatch.job_state)

                @staticmethod
                def choose_host(*, excluded_uuids=(), minimum=65000):
                    assert args.host in allowed_hosts(admit_long_job=True)
                    before = probe(args.host);time.sleep(1);after = probe(args.host)
                    uuids = [g['uuid'] for g in after['gpus']]
                    assert len(uuids) == 8 and uuids == [g['uuid'] for g in before['gpus']]
                    assert not set(uuids)&set(excluded_uuids)
                    assert not before['owners'] and not after['owners']
                    assert all(not gpu_activity_issue(g, p['owners']) for p in (before, after) for g in p['gpus'])
                    assert min(g['free_MiB'] for p in (before, after) for g in p['gpus']) >= minimum
                    assert after['MemAvailable_bytes'] >= 250*2**30
                    assert dispatch.leases_available(uuids)
                    atomic_json(args.receipt/'ACTUAL_PRE_DISPATCH.json', dict(before=before, after=after))
                    return args.host, after

            campaign = Campaign(plan, runs['source_files'], driver=MeasuredHostDriver())
            campaign._poll()
            previous = set(campaign.state['jobs'])
            campaign._training(args.architecture, 'IL')
            created = set(campaign.state['jobs'])-previous
            assert len(created) <= 1
            for key in created:
                job = campaign.state['jobs'][key]
                assert job['host'] == args.host and '--qualification' not in job['command']
                job['measured_host_admission_receipt'] = str(args.receipt/'REQUEST.json')
                job['measured_host_admission_sha256'] = file_sha(args.receipt/'REQUEST.json')
            campaign.persist()
            atomic_json(args.receipt/'DISPATCH_COMMITTED.json', dict(jobs=list(created),
                status='DISPATCHED' if created else 'ALREADY_OWNED', original_recipe_and_budget=True))
    finally:
        if restart_needed:
            env = dict(os.environ, PYTHONPATH=str(controller), OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (args.receipt/'canonical_owner.log').open('xb') as log:
                process = subprocess.Popen(argv, cwd=controller, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(args.receipt/'CANONICAL_OWNER_RESTART.json', dict(pid=process.pid,
                start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21]), argv=argv,
                scientific_source=plan['source'], controller_source=str(controller), workers_not_signalled=True))
            print('Original immutable canonical owner resumed', process.pid, flush=True)


if __name__ == '__main__':
    main()
