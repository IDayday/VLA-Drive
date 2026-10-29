"""Read-only half-hour progress/cost snapshots; never dispatch or stop jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def tail_rows(path, limit=1000):
    if not path.exists():
        return []
    with path.open('rb') as stream:
        offset = max(0, path.stat().st_size-4*1024**2)
        stream.seek(offset)
        raw = stream.read()
    lines = raw.split(b'\n')
    # A live writer can have an unfinished final line. Preserve it on disk.
    if offset:
        lines = lines[1:]
    rows = [json.loads(line) for line in lines[:-1] if line][-limit:]
    steps = [row['updates'] for row in rows]
    if steps and steps != list(range(steps[0], steps[-1]+1)):
        raise ValueError('Duplicate/gapped effective update journal requires review: '+str(path))
    return rows


def snapshot(root):
    state_path = root/'campaign/STATE.json'
    state = read(state_path) if state_path.exists() else {'architectures': {}, 'jobs': {}}
    output = dict(as_of_unix=time.time(), complete=all(
        state['architectures'].get(a, {}).get('status') == 'COMPLETE' for a in 'ABCD'), architectures={},
        whole_campaign_ETA_seconds=None,
        ETA_limits='Training-only estimates need100 recent joint updates; exclude DEV/Navtest, queue and changing host capacity.',
        GPU_cost_semantics='Allocated worker occupancy, not exclusive GPU busy time; worker walls are not summed into end-to-end wall.')
    for architecture in 'ABCD':
        arm = dict(status=state['architectures'].get(architecture, {}).get('status', 'PENDING'))
        for stage, budget in [('IL', 100000), ('RL', 50000)]:
            run = root/'runs'/f'ARCH_{architecture}'/f'{stage}_seed42'
            rows = tail_rows(run/'metrics_rank0.jsonl')
            if not rows:
                arm[stage] = dict(updates=0, budget=budget, status='PENDING', training_only_ETA_seconds=None)
                continue
            last = rows[-1]
            recent = [r for r in rows if stage == 'RL' or r.get('phase') == 'joint'][-100:]
            seconds = [r['slowest_rank_seconds'] if stage == 'IL' else r['seconds_update'] for r in recent]
            point = dict(updates=last['updates'], budget=budget,
                scene_exposures=last['cursor'] if stage == 'IL' else last['raw_scene_exposures'],
                new_RL_candidates=last.get('new_candidates'),
                phase=last.get('phase'), recent_joint_window=len(recent),
                training_only_ETA_seconds=None, source_journal=str(run/'metrics_rank0.jsonl'))
            if len(seconds) == 100 and stage == 'IL':
                progress = read(run/'progress.json') if (run/'progress.json').exists() else {}
                overhead = progress.get('save_seconds', 0.)/1000
                point.update(recent_joint_mean_update_seconds=sum(seconds)/len(seconds),
                    checkpoint_overhead_seconds_per_update=overhead,
                    training_only_ETA_seconds=(budget-last['updates'])*(sum(seconds)/len(seconds)+overhead))
            # RL row timings exclude collection/scoring; do not invent a total ETA.
            arm[stage] = point
        allocated = 0.
        physical_jobs = 0
        for job in state.get('jobs', {}).values():
            if job['architecture'] != architecture:
                continue
            receipt = Path(job['receipt'])
            path = next((receipt/n for n in ['RESULT.json', 'RUNNING.json'] if (receipt/n).exists()), None)
            if path is None:
                continue
            value = read(path)
            if value.get('child_not_started') or 'start_unix' not in value:
                continue
            allocated += ((value.get('end_unix') or time.time())-value['start_unix'])*len(job.get('gpus', []))/3600
            physical_jobs += 1
        arm.update(allocated_worker_GPU_hours=allocated, physical_job_invocations=physical_jobs)
        output['architectures'][architecture] = arm
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.poll_seconds < 60:
        raise ValueError('Runtime reporting is bounded to infrequent read-only snapshots')
    directory = args.root/'campaign/runtime_snapshots'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        value = snapshot(args.root)
        path = directory/(str(time.time_ns())+'.json')
        path.write_text(json.dumps(value, indent=2)+'\n')
        latest = directory/'LATEST.json'
        temporary = latest.with_name(latest.name+'.tmp.'+str(os.getpid()))
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        os.replace(temporary, latest)
        if args.once or value['complete']:
            print(json.dumps(value))
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
