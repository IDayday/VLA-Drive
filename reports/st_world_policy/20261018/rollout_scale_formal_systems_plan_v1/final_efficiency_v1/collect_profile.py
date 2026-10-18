"""Read only the one completed PROFILE's timing/control records, never tensors.

This writes measurements, not launch approval. It refuses partial 240-step runs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

RUNTIME = Path('/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018')
ROOT = Path('/mnt/project/st-world-policy-20261016/profiles/scene_center_rollout_std_v1_fulltrain240')
RUN = ROOT / 'runs/C1_rollout_std_profile_v1'
SOURCE = '751fcfa2b41549ae16f7b3204aef8dd18b912016'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Immutable measurement output already exists')
    # Refuse before importing numerical libraries or scanning growing journals.
    result = json.loads((RUN / 'RESULT.json').read_text())
    exit_result = json.loads((ROOT / 'EXIT.json').read_text())
    assert result['completed'] == result['new_optimizer_updates'] == 240
    assert result['source_sha'] == SOURCE and result['frozen_encoder_unchanged']
    assert exit_result['exit_code'] == 0
    assert exit_result['resource_monitor']['stop_request'] is None
    assert not (RUN / 'STOP_REQUESTED').exists()

    sys.path.insert(0, str(RUNTIME))
    from tools.st_world_policy.profile_pipeline import TimingPlan, summarize_ranks
    import numpy as np

    pins = []

    def read(path):
        path = Path(path)
        raw = path.read_bytes()
        pins.append(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        return raw

    def obj(path):
        return json.loads(read(path))

    def lines(path):
        return [json.loads(line) for line in read(path).splitlines() if line]

    def distribution(values):
        assert values and all(math.isfinite(x) for x in values)
        return dict(n=len(values), mean=float(np.mean(values)), p50=float(np.quantile(values, .5)),
                    p95=float(np.quantile(values, .95)), max=max(values))

    ranks = [obj(RUN / 'profile' / f'rank{rank}.json') for rank in range(4)]
    summary = summarize_ranks(ranks, TimingPlan())
    assert summary['identity']['source_sha'] == SOURCE
    assert summary['world_size'] == 4
    steps = lines(RUN / 'steps.jsonl')
    assert [step['step'] for step in steps] == list(range(1, 241))
    assert all(len(step['per_rank']) == 4 for step in steps)
    assert all(step['exposure'] == step['step'] * 32 for step in steps)
    assert all(step['fp32_master_update_verified'] for step in steps)
    assert all(math.isfinite(step['gradient_norm']) for step in steps)
    assert all(math.isfinite(v) for step in steps for v in step['losses'].values())
    assert all(all(rank['official']['valid_candidates'] == 32 for rank in step['per_rank']) for step in steps)
    valid_scores = sum(rank['official']['valid_candidates'] for step in steps for rank in step['per_rank'])
    assert valid_scores == 30720
    assert all(sum(len(rank['sample_ids']) for rank in step['per_rank']) == 32 for step in steps)
    ids = [scene for step in steps for rank in step['per_rank'] for scene in rank['sample_ids']]
    assert len(ids) == len(set(ids)) == 7680
    assert all(len({rank['advantage_normalization']['common_sha256'] for rank in step['per_rank']}) == 1
               for step in steps)
    assert all(rank['advantage_normalization']['count'] == 128 for step in steps for rank in step['per_rank'])
    assert all(math.isfinite(rank['advantage_normalization'][key])
               for step in steps for rank in step['per_rank'] for key in ('mean', 'std', 'gain'))

    checkpoints = lines(RUN / 'checkpoint_costs.jsonl')
    monitor = lines(ROOT / 'RESOURCE_MONITOR.jsonl')
    samples = [row for row in monitor if 'rss_sum_bytes' in row]
    assert samples
    assert not any(row.get('monitor_error') or row.get('stop_requested') or
                   row.get('unmapped_selected_gpu_processes') for row in monitor)
    assert max(row['rss_sum_bytes'] for row in samples) <= 256 * 2**30
    assert max(max(row['owned_gpu_used_bytes'].values()) for row in samples) <= 48 * 2**30
    assert max(row['study_output_bytes'] for row in samples) <= 96 * 2**30
    assert max(row['study_native_directory_count'] for row in samples) <= 6

    cpu_path = ROOT / 'PROCESS_TREE_SAMPLES.jsonl'
    cpu_hash = hashlib.sha256()
    cpu_bytes = 0
    cpu_header = None
    cpu_footer = None
    cpu_samples = []
    with cpu_path.open('rb') as stream:
        for line in stream:
            cpu_bytes += len(line)
            assert cpu_bytes <= 256 * 2**20
            cpu_hash.update(line)
            row = json.loads(line)
            if 'schema' in row:
                cpu_header = row
            elif 'tree_cpu_seconds' in row:
                cpu_samples.append({k: row[k] for k in ('unix', 'tree_cpu_seconds', 'rss_sum_bytes')})
            elif row.get('status') == 'OBSERVATION_ENDED':
                cpu_footer = row
    pins.append(dict(path=str(cpu_path), bytes=cpu_bytes, sha256=cpu_hash.hexdigest()))
    assert cpu_header and cpu_samples and cpu_footer
    # This older bounded PROFILE observer has no reason field. Do not invent it:
    # require its footer and pair it with the launcher's successful child.wait().
    assert cpu_footer['status'] == 'OBSERVATION_ENDED'
    assert cpu_footer['unix'] - cpu_header['started_unix'] < 7400
    assert cpu_header['root_pid'] == 599215

    def sampled_cpu_at(stamp):
        previous = [row for row in cpu_samples if row['unix'] <= stamp]
        if not previous:
            raise ValueError('Missing CPU sample before steady boundary')
        return previous[-1]

    windows = []
    for window in summary['windows']:
        selected = [step for step in steps if step['step'] in window['steps']]
        region_names = list(selected[0]['per_rank'][0]['seconds'])
        regions = {name: distribution([max(rank['seconds'][name] for rank in step['per_rank'])
                                       for step in selected]) for name in region_names}
        first_step, last_step = window['steps'][0], window['steps'][-1]
        first = min(rank['updates'][first_step-1]['start_unix'] for rank in ranks)
        last = max(rank['updates'][last_step-1]['end_unix'] for rank in ranks)
        start_cpu, end_cpu = sampled_cpu_at(first), sampled_cpu_at(last)
        windows.append(dict(window=window['window'], slowest_rank_nested_region_seconds=regions,
                            normalizer_is_subset_of_official_feedback=True,
                            regions_are_not_additive_critical_path=True,
                            CPU_sampled_tree_seconds=end_cpu['tree_cpu_seconds']-start_cpu['tree_cpu_seconds'],
                            CPU_boundary_lags_seconds=[first-start_cpu['unix'],last-end_cpu['unix']],
                            CPU_scope='Live subtree self + waited children; interval-boundary approximation, not wall*workers'))

    launch = obj(ROOT / 'LAUNCH.json')
    obj(ROOT / 'EXIT.json'); obj(RUN / 'RESULT.json')
    obj(ROOT / 'REGISTRATION.json'); obj(ROOT / 'CONFIG.json')
    obj(RUN / 'identity.json'); obj(RUN / 'initialization.json')
    obj(ROOT / 'EXECUTION_BINDING.json'); obj(ROOT / 'PROCESS_OBSERVER_LAUNCH.json')
    normal_ids = [i for i in range(240) if all(rank['updates'][i]['checkpoint_seconds'] == 0 for rank in ranks)]
    steady = [i-1 for window in summary['windows'] for i in window['steps']]
    normal_steady = [i for i in steady if i in normal_ids]
    normal_distribution = distribution([max(rank['updates'][i]['seconds'] for rank in ranks) for i in normal_steady])
    all_checkpoint_rows = [[rank['updates'][i]['checkpoint_seconds'] for rank in ranks] for i in range(240)]
    periodic_cost = [max(values) for values in all_checkpoint_rows if max(values) > 0]
    checkpoint_mean = statistics.mean(periodic_cost) if periodic_cost else None
    driver_wall = exit_result['ended_unix'] - launch['started_unix']
    assert driver_wall > 0
    data = dict(schema='independent_complete_profile240_measurement_v1',status='MEASURED_NOT_AN_APPROVAL',
                source_sha=SOURCE,profile_root=str(ROOT),timing=summary,regions_and_live_CPU=windows,
                aggregate_normal_steady_update_seconds=normal_distribution,
                formal_checkpoint_amortization=dict(save_every=512,measured_periodic_checkpoint_seconds=periodic_cost,
                    measured_mean_checkpoint_seconds=checkpoint_mean,
                    projected_seconds_per_update=normal_distribution['mean']+checkpoint_mean/512 if checkpoint_mean else None,
                    scope='Projection from actual profile checkpoint; retains original512 schedule, excludes one-time init/final overhead'),
                checkpoint_cost_records=checkpoints,
                budgets=dict(completed_updates=240,unique_scenes=7680,scene_presentations=7680,
                    valid_official_scores=valid_scores,official_admitted_calls=steps[-1]['official_admitted_calls'],
                    BC_repeat8_presentations=result['counts']['BC_repeat8_presentations'],
                    policy_MC_points_each_old_and_current=result['counts']['policy_MC_points_per_old_and_current']),
                process_cost=dict(driver_wall_seconds=driver_wall,driver_allocated_GPU_seconds=4*driver_wall,
                    NCCL_ready_to_final_hash_wall_seconds=result['allocation_wall_seconds'],
                    NCCL_ready_to_final_hash_GPU_seconds=4*result['allocation_wall_seconds'],
                    waited_child_CPU_seconds=exit_result['waited_child_cpu_seconds'],
                    CPU_observed_last_tree_seconds=cpu_samples[-1]['tree_cpu_seconds'],
                    CPU_observed_max_tree_seconds=max(row['tree_cpu_seconds'] for row in cpu_samples),
                    CPU_sample_count=len(cpu_samples),CPU_footer=cpu_footer,
                    note='CPU scopes overlap and must not be added; allocated GPU seconds are reservation cost, not GPU-active kernel time'),
                resource_monitor=dict(samples=len(samples),rss_sum_peak_bytes=max(row['rss_sum_bytes'] for row in samples),
                    owned_driver_GPU_peak_bytes=max(max(row['owned_gpu_used_bytes'].values()) for row in samples),
                    minimum_free_bytes=min(row['free_bytes'] for row in samples),
                    cumulative_smoke_profile_output_peak_bytes=max(row['study_output_bytes'] for row in samples),
                    combined_native_directory_peak=max(row['study_native_directory_count'] for row in samples),
                    log_peak_bytes=max(row['log_bytes'] for row in samples),
                    errors=[],stops=[],unmapped_selected_GPU_processes=[]),
                frozen_encoder_unchanged=True,evidence_artifacts=pins,
                limitations=['No new 8-vs4 topology matrix; historical zt4 timings are reference, not pure normalizer causal difference.',
                            'No per-device physical I/O counters were collected; data-region wall and measured output occupancy are reported, not invented read throughput.',
                            'Normalizer timing includes preparation/hash/vote/collective wait inside official-feedback wall; it is not a standalone GPU kernel benchmark.',
                            'Scientific/development results are not evaluated by this resource review.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=data['status'], output=str(output), normal=normal_distribution)))


if __name__ == '__main__':
    main()
