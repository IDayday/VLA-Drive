"""Receipt-based round costs without double-counting worker/parent summaries.

GPU process wall-hours can overlap on one physical GPU. This is not a power
measurement or an assertion that every import/administrative second was timed.
"""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, digest, read, source_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Cost ledgers are timestamped immutable snapshots')
    root = args.artifacts
    rows, used, missing = [], set(), []

    def add(path, keys, category, note='', multiplier=1.):
        path = Path(path)
        if not path.exists():
            missing.append(str(path.relative_to(root)))
            return
        if path in used:
            raise ValueError('Cost receipt selected twice: '+str(path))
        data = read(path)
        value = sum(float(data[key]) for key in keys)*multiplier
        if value < 0:
            raise ValueError('Negative runtime')
        used.add(path)
        rows.append({'category': category, 'GPU_process_hours': value,
                     'receipt': str(path.relative_to(root)), 'receipt_sha256': digest(path),
                     'fields': list(keys), 'multiplier': multiplier, 'note': note})

    add(root/'FULL_POPULATION_RECEIPT.json',
        ['old_12_worker_process_GPU_hours', 'new_21_worker_process_GPU_hours'],
        'train_anchor_cache', 'Includes retired workers, retained partial chunks and new loading')
    add(root/'final_navtest_v1/PREPARATION_COMPLETE_RECEIPT.json',
        ['GPU_hours_total_including_loading_and_retired_attempts'], 'navtest_anchor_cache')
    for bank, category in [('anchor_dev_full_v1', 'development_anchor_cache'),
                           ('anchor_diagnostic4096_v1', 'diagnostic_anchor_cache'),
                           ('anchor_profile64_fp16', 'initial_and_cache_validation')]:
        for path in sorted((root/bank/'workers').glob('*.json')):
            if 'GPU_hours' in read(path):
                add(path, ['GPU_hours'], category)
    add(root/'launches/anchor_diagnostic4096_retired.json', ['GPU_hours_until_stop'],
        'diagnostic_anchor_cache', 'Retired partial producer; retained work not charged twice elsewhere')

    # These parents aggregate worker hours already. Never add their workers a
    # second time, or a STATUS alongside its completed native accumulator.
    families = {
        'formal_seed42': 'primary_small_heads', 'formal_seed43': 'primary_small_heads',
        'conditional_oof_v1': 'OOF_training_and_evaluation',
        'G_diagnostic_dev75k_v1': 'G_readonly_fields', 'G_diagnostic_smoke4': 'G_readonly_validation',
        'diagnostic4096_v1': 'small_data_and_numerical_checks',
        'diagnostic4096_mmap_v1': 'small_data_and_numerical_checks',
        'smoke': 'small_data_and_numerical_checks', 'smoke_current_ground': 'small_data_and_numerical_checks',
        'smoke_kinv2': 'small_data_and_numerical_checks', 'resume_smoke': 'small_data_and_numerical_checks',
        'reader_same_gpu_v1': 'small_data_and_numerical_checks',
        'reader_training_pair_v1': 'small_data_and_numerical_checks',
        'relation_endpoints_v1': 'relationship_evaluation', 'relation_smoke64': 'relationship_evaluation',
        'relation_smoke64_complete_metrics': 'relationship_evaluation',
        'relation_observer_diagnostic1696': 'relationship_evaluation',
        'relation_observer_debug4': 'relationship_evaluation',
        'correction_diagnostics_v1': 'correction_diagnostics',
        'online_deploy_smoke4_kinv2': 'online_deployment',
        'instrumented_deployment_v1': 'online_deployment',
        'formal_deployment_v1': 'online_deployment',
        'planning_smoke_R_DIFF': 'planning_export', 'dev_score_smoke4': 'planning_export',
        'shard_validation1696': 'planning_export', 'stream_shard_validation4': 'planning_export',
        'stream_observer_debug4': 'planning_export', 'observer_debug4': 'planning_export',
        'development_observations_v1': 'planning_export', 'information_seed43_v1': 'planning_export',
        'final_navtest_v1': 'planning_export',
    }
    excluded = {'chunks', 'records', 'proposals', 'q0', 'q_final', 'scores', 'workers',
                'claims', 'checkpoints', 'envs', 'features', 'queries'}
    for family, category in families.items():
        for directory, dirs, files in os.walk(root/family):
            dirs[:] = [name for name in dirs if name not in excluded]
            directory = Path(directory)
            for name in files:
                if not name.endswith('.json'):
                    continue
                path = directory/name
                if name == 'STATUS.json' and (directory/'COMPLETE.json').exists():
                    continue
                # Sidecar registries, resume origins and failed old timer
                # reports are not new independent compute intervals.
                permitted = name in ('COMPLETE.json', 'STATUS.json', 'summary.json')
                if family == 'formal_deployment_v1' and directory == root/family:
                    permitted |= name.startswith('seed')
                if not permitted:
                    continue
                data = read(path)
                if 'GPU_hours' not in data:
                    continue
                add(path, ['GPU_hours'], category,
                    'Native cumulative cost, including resumes' if family.startswith('formal_seed') else '')
    add(root/'LOSS_SCALE_4096.json', ['GPU_hours'], 'small_data_and_numerical_checks')
    add(root/'deployment_queue_debug2/TIMING_CORRECTION.json',
        ['correct_whole_child_wall_GPU_hours_upper_bound'], 'online_deployment',
        'Corrected whole-process timing; original faulty timer is preserved, not counted')
    failed = root/'instrumented_deployment_v1/FAILED_OUTER_HOOK.json'
    if failed.exists():
        add(failed, ['GPU_process_hours_upper_bound_from_log_mtime'], 'online_deployment',
            'Failed instrumentation attempt; log-final-write elapsed estimate, not a wait() timer')
    pending = []
    for registry in ('HEAD_TASKS.json', 'information_seed43_v1/HEAD_TASKS.json'):
        for job in read(root/registry)['jobs']:
            # Registered formal jobs use their original output even after a
            # physical read-only cache-path migration.
            command = job['command']
            run = Path(command[command.index('--output')+1])
            if not (run/'COMPLETE.json').exists():
                pending.append(job['id'])
    if pending and not args.allow_partial:
        raise ValueError('Formal training unfinished: '+repr(pending))
    totals = defaultdict(float)
    for row in rows:
        totals[row['category']] += row['GPU_process_hours']
    cpu = []
    for bank in sorted(root.glob('labels_*')):
        for name in ('COMPLETE.json', 'STATUS.json'):
            path = bank/name
            if path.exists():
                data = read(path)
                if 'CPU_worker_wall_hours_including_wait' in data:
                    cpu.append({'receipt': str(path.relative_to(root)), 'sha256': digest(path),
                                'CPU_worker_wall_hours_including_wait': data['CPU_worker_wall_hours_including_wait'],
                                'wall_seconds': data['seconds'], 'scenes': data['scenes']})
                    break
    atomic_json(args.output, {'source_sha': source_sha(ROOT), 'tool_sha256': digest(__file__),
        'recorded_GPU_process_hours': sum(totals.values()), 'categories': dict(totals),
        'components': rows, 'pending_formal_training': pending, 'missing_cost_receipts': missing,
        'CPU_label_receipts': cpu,
        'CPU_label_worker_wall_hours_including_dependency_wait': sum(x['CPU_worker_wall_hours_including_wait'] for x in cpu),
        'historical_shared_S0_training_GPU_hours': read(root/'ANCHOR.json')['base_training_GPU_hours'],
        'operational_cap_GPU_hours': 100, 'unmetered_overhead_budget_reserve_GPU_hours': 2,
        'reserve_is_not_reported_as_measured_consumption': True,
        'limits': ['Process wall intervals can overlap on the same GPU; they are not physical-device busy hours or energy',
                   'Small imports, metadata commands, CPU-only analysis and some old failure startup intervals were not separately metered',
                   'Several native timers begin within the Python entry; do not claim exact allocation billing',
                   'CPU label worker wall-hours include time waiting for cache producers, not CPU kernel-hours',
                   'No base training, label-cache source or ancestor cost is counted as a fresh small-head experiment']})


if __name__ == '__main__':
    main()
