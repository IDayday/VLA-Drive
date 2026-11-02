"""Read completed own invocation receipts and published counters; never dispatch.

GPU slot hours include initialization and waiting inside an admitted invocation.
They are neither CUDA kernel time nor exclusive physical GPU ownership. Scorer
timings are candidate wall times; they cannot establish CPU core hours.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def merged_seconds(intervals):
    total = 0.0
    previous = None
    for start, end in sorted(intervals):
        if previous is None:
            previous = [start, end]
        elif start > previous[1]:
            total += previous[1] - previous[0]
            previous = [start, end]
        else:
            previous[1] = max(previous[1], end)
    return total + (previous[1] - previous[0] if previous else 0.0)


def collect(root):
    root = Path(root).resolve()
    inputs = []

    def read(path):
        raw = path.read_bytes()
        inputs.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)

    registry = read(root / 'RUNS.json')
    runs = {r['id']: r for r in registry['runs']}
    completed, pending, rejected, cpu_only, devices = [], [], [], [], {}
    # Only known formal/evaluation job trees, never recursive checkpoint scans.
    paths = set()
    for arm in runs:
        paths.update((root / 'jobs' / arm).glob('*/RESULT.json'))
    paths.update((root / 'jobs' / 'evaluation').glob('*/*/RESULT.json'))
    paths.update((root / 'jobs' / 'evaluation').glob('*/NAVTEST/*/RESULT.json'))
    for path in sorted({p.resolve() for p in paths}):
        try:
            row = read(path)
            if not row.get('child_pid'):
                pending.append(dict(path=str(path), reason='NO_ADMITTED_CHILD_NO_GPU_COST_INFERRED'))
                continue
            start, end = row['start_unix'], row['end_unix']
            uuids = row['gpu_uuids']
            if not uuids and row.get('gpus') == []:
                cpu_only.append(dict(path=str(path), reason='CPU_ONLY_EXPORT_NO_GPU_ALLOCATION'))
                continue
            if (not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end))
                    or end < start or not uuids or len(set(uuids)) != len(uuids)):
                raise ValueError('Invalid completed time/UUID interval')
            output = dict(path=str(path), host=row['host'], start_unix=start, end_unix=end,
                          GPU_UUIDs=uuids, returncode=row['returncode'],
                          admitted_invocation_GPU_slot_hours=(end-start)*len(uuids)/3600)
            completed.append(output)
            for uuid in uuids:
                devices.setdefault((row['host'], uuid), []).append((start, end))
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejected.append(dict(path=str(path), reason=str(error)))
    snapshots = []
    for arm, run in runs.items():
        base = Path(run['root']).resolve()
        latest = base / 'progress.json'
        row = read(latest) if latest.is_file() else {}
        rank_paths = [base / f'result_rank{i}.json' for i in range(8)]
        rank_rows = [read(f) for f in rank_paths if f.is_file()]
        consistent = (len(rank_rows) == 8 and
                      len({(r.get('updates'), r.get('next_inner'), r.get('checkpoint')) for r in rank_rows}) == 1)
        times = [r.get('seconds_scoring_cpu') for r in rank_rows]
        timing_known = consistent and all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in times)
        peaks = [r.get('peak_GPU_bytes') for r in rank_rows]
        snapshot = dict(arm=arm, latest_published_updates=row.get('updates'),
                        latest_published_valid_scene_exposures=row.get('valid_scene_exposures'),
                        latest_published_fresh_PDM_queries=row.get('new_candidates'),
                        latest_published_extra_B1_queries=row.get('extra_B1_queries'),
                        latest_published_adaptive_G_queries=row.get('adaptive_G_queries'),
                        latest_published_history_B1_queries=row.get('history_B1_queries'),
                        latest_published_structured_progress_queries=row.get('structured_progress', {}).get('global_queries'),
                        ended_rank_snapshot_consistent=consistent,
                        ended_rank_snapshot_updates=rank_rows[0].get('updates') if consistent else None,
                        committed_main_candidate_scorer_wall_hours=sum(times)/3600 if timing_known else None,
                        peak_reported_PyTorch_allocated_bytes=max(peaks) if peaks and all(isinstance(v,int) for v in peaks) else None,
                        interpretation='Published counters may lag an active invocation; no pending work or failed replay queries are invented.')
        snapshots.append(snapshot)
    return dict(schema='world_performance_actual_completed_invocation_cost_audit_v1',
                generated_UTC=datetime.now(timezone.utc).isoformat(),
                completed_admitted_invocations=completed,
                completed_admitted_invocation_GPU_slot_hours=sum(x['admitted_invocation_GPU_slot_hours'] for x in completed),
                completed_physical_device_interval_union_hours=sum(merged_seconds(v) for v in devices.values())/3600,
                published_training_snapshots=snapshots, unadmitted_receipts=pending,
                CPU_only_receipts=cpu_only, rejected_receipts=rejected,
                CPU_core_hours=None, input_files=inputs,
                limits=[
                    'Only completed own registered formal/evaluation invocation receipts are counted; active invocations, qualifications and TRAIN diagnostics are excluded.',
                    'GPU slot hours include admitted invocation startup and waits and may overlap across cohosted jobs. Device-union hours merge such intervals, but do not imply exclusive ownership or kernel utilization.',
                    'seconds_scoring_cpu is the sum of per-candidate scoring wall times, not CPU process_time; it covers committed main candidate scoring and excludes auxiliary/failed replay work.',
                    'Published snapshot counters are cumulative within one logical run and are never summed across its resumed invocations.',
                    'This is a measured completed-work lower bound, not the complete campaign bill or a cost limit. Missing values remain null.'
                ])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=Path('/mnt/project/qwen-world-performance-rl-20261103'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    value = collect(a.root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_name(a.output.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(a.output)
    print(json.dumps(dict(path=str(a.output), completed=len(value['completed_admitted_invocations']),
                          GPU_slot_hours=value['completed_admitted_invocation_GPU_slot_hours'],
                          rejected=len(value['rejected_receipts']))))


if __name__ == '__main__':
    main()
