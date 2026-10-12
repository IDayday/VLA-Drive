"""Summarize completed G3 80k execution; never launches or signals jobs."""
import datetime
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
DISPATCH = ROOT.parent / 'navsim_navtest_dispatch_20261011T0225'
KEY = 'G3_EVENT_LOCAL_0080000'
JOB = DISPATCH / 'jobs' / KEY


def load(path):
    return json.loads(path.read_text())


def seconds(value):
    return datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def main():
    receipt = load(DISPATCH / 'completed' / (KEY + '.json'))
    assert receipt['valid'] is True and receipt['request']['scenes'] == 12146
    completion = load(JOB / 'COMPLETE.json')
    assert completion['full_population'] is True
    final_reports = [load(p) for p in (JOB / 'attempts' / completion['final_attempt'] / 'shards').glob('*.json')]
    assert len(final_reports) == completion['shards'] and all(r['status'] == 'COMPLETE' for r in final_reports)
    reports = [load(p) for p in (JOB / 'attempts').glob('*/shards/*.json')]
    start = min(seconds(r['started_utc']) for r in reports)
    end = max(seconds(r['completed_utc']) for r in final_reports)
    history = [json.loads(line) for line in (ROOT / 'TRAINING_INTERFERENCE.jsonl').read_text().splitlines()]
    baseline = {r['group']: r for r in load(ROOT / 'TRAINING_BASELINE_BEFORE_EVAL.json')['rows']}
    training = []
    windows = []
    for attempt in sorted({r['attempt'] for r in reports}):
        phase = [r for r in reports if r['attempt'] == attempt]
        lo_time = min(seconds(r['started_utc']) for r in phase)
        hi_time = max(seconds(r.get('completed_utc', r.get('failed_utc', r['observed_utc']))) for r in phase)
        windows.append((attempt, lo_time, hi_time))
    for label, lo_time, hi_time in [('overall_including_recovery_gap', start, end), *windows]:
      first = max((r for r in history if seconds(r['utc']) <= lo_time), key=lambda r: r['utc'])
      last = max((r for r in history if seconds(r['utc']) <= hi_time), key=lambda r: r['utc'])
      for group in ('G1_FULL_UNIFORM', 'G3_EVENT_LOCAL'):
        before = next(r for r in first['rows'] if r['group'] == group)
        after = next(r for r in last['rows'] if r['group'] == group)
        lo, hi = before['updates'][-1], after['updates'][-1]
        file = Path(baseline[group]['run']) / 'metrics.jsonl'
        values = {}
        with file.open('rb') as stream:
            stream.seek(max(0, stream.seek(0, 2) - 64 * 1024 * 1024))
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if lo < row['updates'] <= hi:
                    assert row['updates'] not in values
                    values[row['updates']] = float(row['seconds'])
        assert set(values) == set(range(lo + 1, hi + 1)), 'Incomplete training timing window'
        durations = list(values.values())
        mean = statistics.mean(durations)
        training.append({'window': label, 'utc': [first['utc'], last['utc']],
                         'group': group, 'updates': [lo + 1, hi], 'count': len(values),
                         'mean_step_seconds': mean, 'median_step_seconds': statistics.median(durations),
                         'p90_step_seconds': sorted(durations)[int(.9 * (len(durations) - 1))],
                         'baseline_mean_step_seconds': baseline[group]['mean_seconds'],
                         'mean_step_ratio_to_baseline': mean / baseline[group]['mean_seconds'],
                         'observed_wall_seconds_per_update': (seconds(last['utc']) - seconds(first['utc'])) / (hi - lo)})
    hosts = []
    for host in sorted({r['host'] for r in reports}):
        rows = [r for r in reports if r['host'] == host]
        wall = sum(r['inference_seconds'] for r in rows)
        idle = sum(r['scheduling_idle_seconds'] for r in rows)
        hosts.append({'host': host, 'worker_instances': len(rows),
                      'historical_reported_lease_positions_not_unique_scenes': sum(r['scenes_done'] for r in rows),
                      'summed_reported_process_wall_hours_including_sleep_lower_bound': wall / 3600,
                      'summed_reported_scheduling_sleep_hours_lower_bound': idle / 3600,
                      'summed_reported_non_sleep_scene_loop_wall_hours_lower_bound': (wall - idle) / 3600,
                      'summed_worker_startup_hours': sum(seconds(r['loaded_utc']) - seconds(r['started_utc']) for r in rows) / 3600,
                      'peak_reserved_GiB': max(r['peak_reserved_bytes'] for r in rows) / 1024**3,
                      'peak_allocated_GiB': max(r['peak_allocated_bytes'] for r in rows) / 1024**3})
    output = {'key': KEY, 'full_population': 12146, 'worker_instances': len(reports),
              'final_attempt_workers': len(final_reports),
              'unique_physical_GPUs': len({r['visible_gpu'] for r in reports}),
              'recovery_reused_unique_predictions': 8201, 'recovery_new_unique_predictions': 12146 - 8201,
              'first_worker_started_utc': datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat(),
              'last_worker_completed_utc': datetime.datetime.fromtimestamp(end, datetime.timezone.utc).isoformat(),
              'inference_wall_seconds_including_startup': end - start,
              'queue_valid_completed_utc': receipt['completed_utc'],
              'training': training, 'hosts': hosts,
              'CPU_budget': load(JOB / 'CPU_BUDGET.json'),
              'limitations': ['Aborted worker reports stop at their last publication: historical costs are lower bounds; no fabricated exact elapsed time for dead workers.',
                             'Historical lease-position counts include cached positions and cannot be summed as unique evaluated scenes; full membership is certified separately.',
                             'Process wall hours include cohost sleeps and are not exclusive GPU-hours or kernel time.',
                             'Non-sleep scene-loop wall time still includes CPU/IO, not pure GPU kernel time.',
                             'Training windows use sixty-second observations near worker start/end; original metrics supply every included update.',
                             'Baseline is a prior100-update observational window, not a controlled A/B benchmark; wall seconds/update also include checkpoint and other overhead.',
                             'Qualification/export and historical bank validation precede worker startup and are separate setup costs.']}
    path = ROOT / 'G3_80K_EXECUTION_COST.json'
    assert not path.exists(), 'Preserve an existing completed execution summary'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
