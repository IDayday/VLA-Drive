"""Read recorded timings; never launch or alter a training/metric job."""
import argparse
from collections import deque
import json
from pathlib import Path
import statistics


def timing(rows, field):
    values = [row[field] for row in rows]
    return {'updates': len(values), 'mean_s': statistics.mean(values),
            'median_s': statistics.median(values), 'minimum_s': min(values), 'maximum_s': max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--historical-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'schema': 'recorded_training_step_time_diagnosis_v2', 'historical_S': {}, 'structured_world': {},
        'same_effective_batch': {'eight_A800': True, 'global_batch': 32, 'microbatch': 4, 'FM_repeat': 8},
        'timing_scope': 'Data read/collation, full forward/backward and optimizer update; checkpoint saves separately recorded',
        'limitation': 'Historical steady timings and small fixed-initialization profiles are not paired final planning experiments; CUDA current-stream stage intervals overlap inclusive parents and must not be blindly summed.'}
    for group in ('S0', 'S3'):
        run = args.historical_root/'students'/('formal_'+group+'_seed42_full100k_v2')
        first, last, total = [], deque(maxlen=1000), 0
        with (run/'steps.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line); total += 1; last.append(row)
                if 101 <= row['update'] <= 500: first.append(row)
        identity = json.loads((run/'identity.json').read_text())
        report['historical_S'][group] = {'recorded_updates': total, 'source_sha': identity['source_sha'],
            'precision': identity['precision'], 'deterministic_training_flag': identity['deterministic'],
            'updates_101_500': timing(first, 'seconds'), 'last_1000': timing(last, 'seconds'),
            'first_window_rankmax_data_median_s': statistics.median(max(rank[1] for rank in row['per_rank_profile']) for row in first),
            'first_window_sequence_length_median': statistics.median(max(rank[4] for rank in row['per_rank_profile']) for row in first)}
    for group, old_queue in [('G1_FULL_UNIFORM', 'formal_navsim_zt2_G1_G0_v1'), ('G3_EVENT_LOCAL', 'formal_navsim_local_G3_G2_v1')]:
        old = args.artifacts/old_queue/group
        prior = [json.loads(line) for line in (old/'metrics.jsonl').read_text().splitlines()]
        profile = args.artifacts/('navsim_vision_amp_profile_'+group+'_v2')
        rows = [json.loads(line) for line in (profile/'metrics.jsonl').read_text().splitlines()]
        complete = json.loads((profile/'checkpoints'/(profile/'checkpoints/latest').read_text().strip()/'COMPLETE.json').read_text())
        assert len(rows) == 8 and complete['completed'] == 8 and complete['status'] == 'COMPLETE'
        steady = rows[2:]
        stages = {}
        for name in steady[0]['per_rank_stage_timing'][0]:
            if name in ('data_load_and_collate_seconds', 'observed_Qwen_vision_precision'): continue
            stages[name] = {'rank0_CUDA_stream_median_s': statistics.median(row['per_rank_stage_timing'][0][name]['CUDA_stream_ms']/1000 for row in steady),
                'rankmax_CUDA_stream_median_s': statistics.median(max(rank[name]['CUDA_stream_ms'] for rank in row['per_rank_stage_timing'])/1000 for row in steady)}
        assert all(item['autocast_enabled'] and item['first_output_dtype'] == 'torch.bfloat16'
            for row in rows for rank in row['per_rank_stage_timing'] for item in rank['observed_Qwen_vision_precision'])
        measured = timing(steady, 'seconds')
        report['structured_world'][group] = {'original_last_100_updates': timing(prior[-100:], 'seconds'),
            'old_complete_recovery_checkpoint': json.loads((old/'PRECISION_DISCLOSURE_AMENDMENT.json').read_text()),
            'corrected_profile_source_sha': json.loads((profile/'identity.json').read_text())['training_source_sha'],
            'corrected_full_chain_profile_all_eight_updates': timing(rows, 'seconds'),
            'corrected_full_chain_steady_profile_updates_3_8': measured,
            'steady_rankmax_data_median_s': statistics.median(max(rank['data_load_and_collate_seconds'] for rank in row['per_rank_stage_timing']) for row in steady),
            'observed_vision_autocast_BF16_all_eight_ranks': True,
            'stages': stages,
            'checkpoint_complete': complete['completed'],
            'training_update_only_100k_GPU_hours_at_steady_median': measured['median_s']*100000*8/3600}
    report['independent_CPU_real_data_probe'] = json.loads((args.artifacts/'STEP_TIME_REAL_DATALOADER64_CPU_v1.json').read_text())
    report['original_live_mainthread_sample_not_CUDA_partition'] = json.loads((args.artifacts/'G3_LIVE_MAINTHREAD_SAMPLING_SUMMARY_v1.json').read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({group: value['corrected_full_chain_steady_profile_updates_3_8'] for group, value in report['structured_world'].items()}))


if __name__ == '__main__':
    main()
