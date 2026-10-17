"""Summarize one completed bounded PROFILE using small receipts, never tensors."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path('/mnt/project/st-world-policy-20261016/profile/world_indexed_provider_fulltrain_updates64_world2_v1')
RUN = ROOT / 'runs/indexed_provider_fulltrain_updates64_world2_v1'


def read(path):
    assert path.stat().st_size < 32 << 20, path
    return json.loads(path.read_text())


def pin(path):
    assert path.stat().st_size < 32 << 20, path
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def lines(path):
    assert path.stat().st_size < 32 << 20, path
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def timing(values):
    values = sorted(values)
    assert values and all(math.isfinite(x) and x > 0 for x in values)
    def quantile(q):
        pos = (len(values) - 1) * q
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)
    return {'n': len(values), 'mean_seconds': statistics.mean(values),
            'p50_seconds': quantile(.5), 'p95_seconds': quantile(.95)}


def summarize():
    exit_result = read(ROOT / 'EXIT_initial.json')
    native = read(RUN / 'RESULT.json')
    monitor = read(ROOT / 'MONITOR_RESULT_initial.json')
    launch = read(ROOT / 'LAUNCH_initial.json')
    rows = lines(RUN / 'steps.jsonl')
    costs = lines(RUN / 'checkpoint_costs.jsonl')
    ranks = [read(RUN / f'teacher_provider_rank{rank}.json') for rank in range(2)]
    ids = [sample for row in rows for batch in row['sample_ids_by_rank'] for sample in batch]
    complete = (exit_result['returncode'] == 0 and monitor['reason'] is None
                and native['completed'] == 64 and native['exposure'] == 128
                and [row['step'] for row in rows] == list(range(1, 65)))
    assert len(ids) == native['exposure']
    assert all(r['tensor_disk_bytes'] == 0 for r in ranks)
    finite = all(math.isfinite(float(row['gradient_norm'])) and
                 all(math.isfinite(float(v)) for v in row['losses'].values()) for row in rows)
    total_cp = sum(row['seconds'] for row in costs)
    files = [p for p in (RUN / 'checkpoints').rglob('*') if p.is_file()]
    result = {
        'schema': 'st_full_population_profile_actual_v1',
        'status': 'BOUNDED_PROFILE_COMPLETE' if complete and finite else 'FAILED_OR_PARTIAL',
        'runtime_source': native['source_sha'],
        'population': 101592, 'world_size': 2, 'global_batch': 2,
        'completed_updates': native['completed'], 'scene_exposures': len(ids),
        'unique_scenes': len(set(ids)), 'epoch': native['epoch'], 'offset': native['offset'],
        'losses_and_gradient_norms_finite': finite,
        'internal_update_timer': timing([row['slowest_rank_seconds'] for row in rows]),
        'internal_update_timer_after_20_warmup': timing([row['slowest_rank_seconds'] for row in rows[20:]]),
        'timing_scope': 'Native slowest-rank update timer includes data/compute/sync; excludes subsequent ID/log/status writes and checkpoint. This is NOT full-loop steady throughput.',
        'allocation_wall_seconds_including_cold_start_and_checkpoint': native['allocation_wall_seconds'],
        'controller_wall_seconds_including_cleanup': monitor['wall_seconds'],
        'same_host_wall_clock_launch_to_training_ready_seconds': native['started_unix'] - launch['started_unix'],
        'same_host_wall_clock_training_ready_to_exit_seconds_including_save_and_cleanup': exit_result['ended_unix'] - native['started_unix'],
        'checkpoint_seconds': total_cp,
        'checkpoint_seconds_per_consumed_update': total_cp / len(rows),
        'checkpoint_actual_file_bytes': sum(p.stat().st_size for p in files),
        'checkpoint_file_count': len(files),
        'gpu_peak_allocated_bytes_by_rank': [max(row['per_rank_profile'][rank][2] for row in rows) for rank in range(2)],
        'gpu_peak_reserved_bytes_by_rank': [max(row['per_rank_profile'][rank][3] for row in rows) for rank in range(2)],
        'data_seconds_mean_by_rank': [statistics.mean(row['per_rank_profile'][rank][1] for row in rows) for rank in range(2)],
        'aggregate_process_tree_peak_rss_bytes': monitor['rss_peak_bytes'],
        'provider_by_rank': ranks,
        'actual_execution_by_rank': [read(RUN / 'official_runtime' / f'updated_engine_rank{rank}.json') for rank in range(2)],
        'new_teacher_tensor_disk_bytes': 0,
        'verified_owned_processes': monitor['verified_owned_processes'],
        'live_owned_remaining': monitor['live_owned_remaining'],
        'weights_usable_for_formal_init': False,
        'formal_global32_efficiency_qualified': False,
        'scientific_result': 'NOT_EVALUATED',
        'evidence': {p.name: pin(p) for p in [ROOT/'REGISTRATION.json', ROOT/'CONFIG.json',
            ROOT/'EXIT_initial.json', ROOT/'MONITOR_RESULT_initial.json', RUN/'RESULT.json',
            RUN/'steps.jsonl', RUN/'checkpoint_costs.jsonl']},
    }
    return result


if __name__ == '__main__':
    print(json.dumps(summarize(), indent=2, sort_keys=True))
