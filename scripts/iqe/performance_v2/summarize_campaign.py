"""Summarize existing artifacts only; never starts training or modifies a run."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, receipts, loop_gpu_seconds, updates = [], [], 0.0, 0
    for path in sorted(args.root.glob('*/RESULT.json')):
        value = json.loads(path.read_text())
        loop_gpu_seconds += value.get('loop_allocated_gpu_seconds', 0)
        updates += value.get('updates', 0)
        stats = value.get('slowest_rank_per_step')
        if not stats or stats['n'] < 10 or value.get('trace_run_not_throughput') or value.get('mask_probe_not_throughput'):
            continue
        options = value['identity']['options']
        name = path.parent.name
        numeric_files = {
            'p1_fused': 'P1_FUSED_NUMERICS.json', 'p1_bucket': 'P1_BUCKET_NUMERICS.json',
            'p2_visual': 'P2_NUMERICS.json', 'p2b_': 'P2B_NUMERICS.json',
            'p3_alternating': 'P3_NUMERICS.json', 'p3_no_': 'P3_NONE_NUMERICS.json',
            'p6_': 'P6_NUMERICS.json', 'combined_fp32': 'FINAL_FP32_NUMERICS.json',
            'combined_visual': 'FINAL_BEST_NUMERICS.json',
        }
        numerical = 'REFERENCE' if name.startswith('p0') else 'UNTESTED'
        if options['query_training_semantics'] == 'native_repeat1':
            numerical = 'T2_DIFFERENT_ESTIMATOR_NOT_T1_EQUIVALENCE'
        for prefix, filename in numeric_files.items():
            candidate = args.root / filename
            if name.startswith(prefix) and candidate.exists():
                passed = json.loads(candidate.read_text())['sampled_numerical_criteria_passed']
                numerical = 'PASS_SAMPLED' if passed else 'FAIL_PREDECLARED_BUDGET'
        resume = 'UNTESTED_THIS_RUNTIME'
        resume_file = ('BEST_EXACT_RESUME.json' if name.startswith('combined_visual') else
                       'FP32_EXACT_RESUME.json' if name.startswith('combined_fp32') else None)
        if resume_file and (args.root / resume_file).exists():
            resume = json.loads((args.root / resume_file).read_text())['status']
        rows.append({
            'experiment': name,
            'tier': 'T2' if options['query_training_semantics'] == 'native_repeat1' else ('T0_reference' if path.parent.name.startswith('p0') else 'T1'),
            'execution_status': value['status'],
            'mean_seconds': stats['mean'], 'p50_seconds': stats['p50'], 'p95_seconds': stats['p95'],
            'timed_updates': stats['n'], 'total_updates': value['updates'], 'global_batch': 32,
            'samples_per_second': 32 / stats['mean'],
            'gpu_seconds_per_1024_samples': stats['mean'] * 256,
            'peak_allocated_GiB': value['peak_allocated_bytes'] / 2**30,
            'peak_reserved_GiB': value['peak_reserved_bytes'] / 2**30,
            'checkpoint_seconds': value['checkpoint_seconds'],
            'checkpoint_amortized_mean_at_1000': stats['mean'] + value['checkpoint_seconds'] / 1000 if value['checkpoint_seconds'] else '',
            'visual_precision': options['visual_precision'], 'query_semantics': options['query_training_semantics'],
            'execution_weights': options['execution_weights'], 'architecture_changed': False,
            'numerical_check': numerical, 'same_backend_exact_resume': resume,
            'source': str(path),
        })
    if rows:
        with (args.output / 'CANDIDATE_RESULTS.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    for path in sorted(args.root.glob('*.exit.json')):
        value = json.loads(path.read_text())
        seconds = value['end'] - value['start']
        receipts.append({'experiment': path.name.removesuffix('.exit.json'), 'process_wall_seconds': seconds,
                         'reserved_eight_GPU_hours': seconds * 8 / 3600,
                         'returncode': value['returncode'], 'receipt': str(path)})
    cost = {'receipted_processes': receipts,
            'receipted_reserved_GPU_hours': sum(r['reserved_eight_GPU_hours'] for r in receipts),
            'recorded_training_loop_GPU_hours': loop_gpu_seconds / 3600,
            'scope': 'Process costs include startup/save/audit; loop costs exclude most startup/save. They are overlapping accounting views, not additive. Missing exit receipts are not inferred. Dedicated-worker reservation includes additional idle/control time.',
            'long_runs': [r['experiment'] for r in rows if r['total_updates'] == 200],
            'real_Query_optimizer_updates': updates,
            'new_full_training_started': False}
    allocation = args.root / 'PRESSURE_STOP.json'
    if allocation.exists() and receipts:
        start = json.loads(allocation.read_text())['time']
        end = max(json.loads(Path(r['receipt']).read_text())['end'] for r in receipts)
        cost['dedicated_worker_reservation'] = {
            'start_utc': datetime.fromtimestamp(start, timezone.utc).isoformat(),
            'last_bounded_process_exit_utc': datetime.fromtimestamp(end, timezone.utc).isoformat(),
            'hours': (end - start) / 3600, 'eight_GPU_hours': (end - start) * 8 / 3600,
            'includes_idle_and_control_time': True, 'cap_GPU_hours': 16,
            'within_cap': (end - start) * 8 / 3600 <= 16,
        }
    (args.output / 'CAMPAIGN_COST.json').write_text(json.dumps(cost, indent=2) + '\n')
    print(json.dumps({'timing_rows': len(rows), 'receipts': len(receipts), 'reserved_GPU_hours': cost['receipted_reserved_GPU_hours']}))


if __name__ == '__main__':
    main()
