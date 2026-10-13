"""Collect recorded D/P/O/N costs; distinguish allocations, reuse and missing timing."""
import argparse
import datetime
import json
import re
from pathlib import Path

import numpy as np

from scripts.analysis.flow_safe_improvement.summarize_direction import training_cost
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint


def read(path):
    return json.loads(Path(path).read_text())


def ranks_at(root):
    paths = sorted(root.glob('rank[0-9]*.json')) + sorted(root.glob('shard[0-9]*.json'))
    return [read(p) for p in paths]


def rank_cost(root):
    ranks = ranks_at(root)
    if not ranks:
        return dict(status='NO_RECORDED_WORKER_COUNTERS')
    names = ('new_scenes', 'reused_scenes', 'new_candidates', 'logical_proposals', 'actual_new_scoring_calls',
             'actual_new_scores', 'velocity_forward_calls', 'extra_qualification_velocity_calls',
             'invalid_physical_proposals', 'duplicate_proposals', 'GPU_generation_seconds', 'wall_seconds')
    totals = {key: sum(x.get(key, 0) for x in ranks) for key in names if any(key in x for x in ranks)}
    return dict(status='ALL_RECORDED_WORKERS_COMPLETE' if all(x['status'] == 'ASSIGNED_POPULATION_COMPLETE' for x in ranks)
                else 'PARTIAL_OR_INTERRUPTED_WORKER_COUNTERS', recorded_workers=len(ranks), counters=totals,
                allocated_worker_hours_excluding_initialization=totals.get('wall_seconds', 0)/3600,
                peak_cuda_allocated_bytes_max=max((x.get('peak_cuda_allocated_bytes', 0) for x in ranks), default=0) or None,
                mean_recorded_generation_seconds_per_new_scene=(totals.get('GPU_generation_seconds', 0)/totals['new_scenes']
                                                               if totals.get('new_scenes', 0) else None),
                counters_sha256=fingerprint(ranks), counters_may_exclude_unflushed_interrupted_work=True)


def union_seconds(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('Negative process duration')
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end-start for start, end in merged)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); root = Path(a.run_root)
    source = Path(__file__).resolve().parents[3]
    learning, banks = {}, {}
    for variant in ('P0_REWARD_FM', 'P1_SAFE_FM', 'P2_SEARCH_SAFE_FM'):
        learning[variant] = root/'training'/(variant+'_seed42')/'train'
    for variant in ('ODE_REWARD_FM', 'ODE_SAFE_FM', 'EXPLORE_SAFE_FM', 'ODE_NFT'):
        for iteration in (1, 2, 3):
            base = root/'online'/(variant+'_seed42')/f'round{iteration}'
            learning[f'{variant}_round{iteration}'] = base/'learning/train'
            banks[f'{variant}_round{iteration}'] = base/'bank'
    banks.update({name: root/'banks'/name for name in ('BANK_ODE', 'BANK_EXPLORE_BEHAVIOR')})
    training = {}
    for name, path in learning.items():
        item = training_cost(path)
        if item['updates']:
            rank_rows = [[json.loads(x) for x in p.read_text().splitlines()] for p in sorted(path.glob('training_rank[0-9]*.jsonl'))]
            item.update(recorded_rank_optimizer_loop_hours=sum(x['seconds'] for rows in rank_rows for x in rows)/3600,
                        recorded_rank_count=len(rank_rows),
                        velocity_forward_noise_instances=item['velocity_forward_scene_calls']*8,
                        velocity_backward_noise_instances=item['velocity_backward_scene_calls']*8,
                        GT_replay_scene_exposures=item['scene_exposures'],
                        reference_safe_retention_scene_exposures=item['scene_exposures'])
            values=[x['seconds'] for rows in rank_rows for x in rows]
            item['rank_update_seconds_quantiles_10_50_90'] = np.quantile(values, [.1,.5,.9]).tolist()
            result=path/'result_rank0.json'
            item['window_status']=read(result)['status'] if result.exists() else 'IN_PROGRESS'
        training[name]=item
    qualification_learning = {}
    for path in sorted((root/'validation').glob('full_head_*')):
        if not path.is_dir() or not (path/'training_rank0.jsonl').exists():
            continue
        rows = [json.loads(line) for line in (path/'training_rank0.jsonl').read_text().splitlines()]
        all_rows = [json.loads(line) for p in path.glob('training_rank[0-9]*.jsonl') for line in p.read_text().splitlines()]
        qualification_learning[path.name] = dict(executed_updates_this_invocation=len(rows),
            update_numbers=[x['update'] for x in rows], global_scene_exposures_this_invocation=16*len(rows),
            recorded_rank_optimizer_loop_hours=sum(x['seconds'] for x in all_rows)/3600,
            no_additional_research_seed=True, not_formal_method_efficacy_result=True,
            result=read(path/'result_rank0.json') if (path/'result_rank0.json').exists() else None)
    candidates={name:rank_cost(path) for name,path in banks.items() if (path/'manifest.json').exists()}
    for name,item in candidates.items():
        manifest=read(banks[name]/'manifest.json')
        item.update(identity_sha256=manifest['sha256'], independent_scene_population=len(manifest['tokens']),
                    common_reference_logical_calls=4*len(manifest['tokens']),
                    reference_scoring_paid_once_in_reference_screen=True)
    d1={name:rank_cost(root/'diagnostics'/name) for name in ('D1_SFT_attempt2','D1_RL8700','D1_RL12912')}
    d3=read(root/'diagnostics/D3_all_methods_summary/report.json')['costs']
    reference=read(source/'reports/flow_safe_improvement_20261013/evidence/reference_screen_execution.json')
    # Each distinct output directory is real work, even for identical policies.
    # Pure report aliases have no sample manifest and incur no new rollout cost.
    patterns=['evaluations/*/manifest.json','evaluations/original_export/*/manifest.json',
        'evaluations/original_export_single_seed/*/manifest.json',
        'evaluations/final_navtest/*/*/manifest.json',
        'training/*/DEV*/*/manifest.json','online/*/round*/learning/DEV*/*/manifest.json']
    sample_costs, score_costs = {}, {}
    for path in sorted({p for pattern in patterns for p in root.glob(pattern)}):
        manifest=read(path); folder=path.parent; name=str(folder.relative_to(root))
        if manifest.get('schema') == 'flow_safe_frozen_samples_v1':
            sample_costs[name]=dict(identity_sha256=manifest['sha256'],seeds=manifest['seeds'],
                                   cohort=manifest['cohort'],profile=manifest.get('execution_profile'),**rank_cost(folder))
        elif manifest.get('schema') == 'flow_safe_scored_samples_v1' and (folder/'status.json').exists():
            score_costs[name]=read(folder/'status.json')
    auxiliary={str(p.parent.relative_to(root)):read(p) for p in root.glob('evaluations/single_seed_auxiliary/*/report.json')}
    auxiliary.update({str(p.parent.relative_to(root)):read(p) for p in root.glob('evaluations/final_navtest/*/v2/report.json')})
    historical_v2_paths=list(root.glob('evaluations/original_export/*DEV5_v2*/report.json'))
    historical_v2_paths+=list(root.glob('evaluations/*DEV5_v2*/report.json'))
    historical_v2={str(p.parent.relative_to(root)):read(p) for p in historical_v2_paths}
    shared_queues={str(p.parent.relative_to(root)):read(p) for p in
        [root/'scoring_queue_D/status.json',root/'scoring_queue_P/status.json',
         *root.glob('online/*/round1/scoring_queue/status.json')] if p.exists()}
    # Whole-process intervals include model/optimizer loading and checkpoint saves.
    # Their union avoids counting overlapping eight-GPU jobs twice on one host.
    phase_patterns=['training/*/*phase_records.json','training/*/DEV*/*phase_records.json',
        'online/*/*phase_records.json','online/*/round*/*phase_records.json',
        'online/*/round*/learning/*phase_records.json','online/*/round*/learning/DEV*/*phase_records.json',
        'evaluations/final_navtest/*/phase_records.json',
        'evaluations/original_export/*/phase_records.json','evaluations/original_export_single_seed/*commands.json']
    unique, by_host, missing_ends = {}, {}, []
    for path in sorted({p for pattern in phase_patterns for p in root.glob(pattern)}):
        for record in read(path):
            command=record.get('command',[])
            if 'torch.distributed.run' not in command:
                continue
            match=re.search(r'--nproc_per_node(?:=|\s+)(\d+)', ' '.join(command))
            if match is None or int(match[1]) != 8:
                continue
            host=record.get('host','UNKNOWN').removesuffix('-worker-0')
            if not host.startswith('training-vlawm-tsinghua-zt'):
                continue
            key=fingerprint(dict(host=host,pid=record['pid'],command=command,started=record['started_unix']))
            previous=unique.get(key)
            if previous and previous.get('finished_unix') is not None:
                continue
            unique[key]=record
    for key,record in unique.items():
        if record.get('finished_unix') is None:
            missing_ends.append(dict(invocation_sha256=key,host=record.get('host'),pid=record['pid']))
            continue
        host=record['host'].removesuffix('-worker-0')
        by_host.setdefault(host,[]).append((record['started_unix'],record['finished_unix']))
    intervals={host:dict(recorded_eight_GPU_processes=len(xs), allocated_GPU_hours_sum=8*sum(b-a for a,b in xs)/3600,
                        distinct_GPU_occupancy_hours_union=8*union_seconds(xs)/3600) for host,xs in by_host.items()}
    archives={str(p.parent.relative_to(root)):read(p) for p in root.glob('scoring_queue_archives/*/round*/report.json')}
    historical={'source':'reports/ddp_flow_grpo_final_report_20261013/README.md',
        'single_host_8GPU_saved_transition_GRPO_seconds_per_update_short_window':5.86,
        'includes_rollout_and_official_reward_amortized_over_two_inner_epochs':True,
        'optimizer_only_seconds_approx':[1.2,1.5],
        'same_current_reward_population_and_execution_budget':False}
    report=dict(schema='flow_safe_recorded_cost_ledger_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='RECORDED_COST_SNAPSHOT_NOT_AN_ESTIMATED_SPEEDUP',training=training,
        qualification_learning=qualification_learning,candidate_banks=candidates,
        D1=d1,D3=d3,reference_and_cache=reference,evaluation_sampling=sample_costs,evaluation_scoring=score_costs,
        auxiliary_v2=auxiliary,historical_multiseed_v2=historical_v2,
        shared_scoring_service_receipts=shared_queues,completed_local_scoring_archives=archives,
        recorded_complete_eight_GPU_intervals=intervals,intervals_without_recorded_end=missing_ends,
        historical_GRPO_profile=historical,source_sha256=file_sha(__file__),
        limitations=[
            'Worker GPU_generation_seconds includes host/cache waiting and is not measured CUDA-kernel busy time.',
            'Per-scene generation time includes input IO, encoding, ODE and the first-scene parity check; it is not isolated model latency.',
            'Allocated worker hours overlap when processes share GPUs; never add them as unique physical GPU-hours.',
            'Interval union covers only saved eight-GPU child invocations with both timestamps; it is a lower bound on all research occupancy.',
            'Interrupted old multi-seed output counters can exclude unflushed work; preserved files and failure records are not discarded.',
            'Optimizer-loop timing excludes model initialization, export and checkpoint IO; complete child intervals include those costs.',
            'Training velocity noise-instance counts multiply scene-call counters by FM repeat8; batched candidate call counts are separate units.',
            'Existing upstream feature-cache construction and SFT pretraining were not re-executed or charged as new work; historical cost is not inferred as zero.',
            'P0/P1 share BANK_ODE historical construction cost; do not charge its new scoring twice.',
            'Reference logical slots are charged equally per method; their real historical scoring is counted once.',
            'Single-seed slices of historical v2 receipts incur zero new scoring; their inherited historical CPU time must not be added twice.',
            'A partial bank counter is observed work to date, not its eventual complete cost.'
        ])
    report['sha256']=fingerprint(report);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'report.json',report)
    print(json.dumps(dict(status=report['status'],recorded_optimizer_updates=sum(x['updates'] for x in training.values()),
                         bank_records=len(candidates),sample_roots=len(sample_costs),score_roots=len(score_costs))))


if __name__=='__main__':
    main()
