"""Read-only cost accounting; resumed segments and reused labels stay explicit."""
import json
from pathlib import Path

from iqe.io import file_hash, read_json


def training_cost(root, kind, cell):
    root = Path(root)
    folder = root/kind/cell
    steps = {}
    path = folder/'steps.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            steps[row['optimizer_step']] = row
    # The native writer reports the current invocation only. Preserve explicit
    # completed invocation receipts before migration; never multiply the final
    # segment's elapsed time as though it covered the earlier optimizer steps.
    sources = sorted((root/'runtime/training_segments'/cell).glob('result_*.json'))
    if (folder/'result.json').exists():
        sources.append(folder/'result.json')
    segments, seen = [], set()
    for source in sources:
        checksum = file_hash(source)
        if checksum in seen:
            continue
        seen.add(checksum)
        result = read_json(source)
        end = result['optimizer_steps']
        added = result.get('new_optimizer_steps')
        segments.append({'path': str(source), 'hash': checksum,
            'start_step': end-added if added is not None else None,
            'end_step': end, 'new_steps': added, 'status': result['status'],
            'trainer_wall_seconds': result.get('elapsed_seconds'),
            'allocated_GPU_seconds': result.get('GPU_seconds')})
    ranges = sorted((s['start_step'], s['end_step']) for s in segments
                    if s['start_step'] is not None and s['new_steps'] > 0)
    covered = 0
    contiguous = True
    for start, end in ranges:
        contiguous &= start == covered
        covered = end
    final_step = max(steps, default=0)
    return {'cell': cell, 'kind': kind, 'optimizer_updates_logged': final_step,
        'unique_update_records': len(steps),
        'sample_presentations': sum(r['global_batch_actual'] for r in steps.values()),
        'optimizer_update_loop_seconds': sum(r['compute_update_seconds'] for r in steps.values()),
        'completed_invocation_trainer_wall_seconds': sum(s['trainer_wall_seconds'] or 0. for s in segments),
        'completed_invocation_allocated_GPU_seconds': sum(s['allocated_GPU_seconds'] or 0. for s in segments),
        'invocation_coverage_complete': contiguous and covered == final_step,
        'invocation_receipts': segments,
        'wall_includes': ['validation', 'checkpoint', 'CPU_scoring_wait'],
        'GPU_seconds_are_allocated_process_wall_not_measured_SM_seconds': True}


def summarize(root):
    root = Path(root)
    cells = [training_cost(root, kind, p.name) for kind in ('experts', 'scorers')
             for p in sorted((root/kind).glob('*')) if (p/'steps.jsonl').exists()]
    scoring = []
    for path in sorted((root/'scoring_jobs').glob('*/STATUS.json')):
        record = read_json(path)
        if 'candidate_count' not in record:
            continue
        workers = [read_json(p) for p in path.parent.glob('worker_*.json')]
        scoring.append({'job': path.parent.name, 'status': record['status'],
            'requested_candidates_including_cache_reuse': record['candidate_count'],
            'coordinator_wall_seconds': record['wall_seconds'],
            'primary_worker_wall_seconds': sum(w.get('seconds', 0.) for w in workers),
            'primary_workers': record['workers']})
    helpers = []
    for folder in (root/'runtime/cooperative_scoring').glob('*'):
        if not folder.is_dir():
            continue
        for path in folder.glob('helper_*.json'):
            if path.name.endswith(('.LAUNCH.json', '.STATUS.json')):
                continue
            record = read_json(path)
            if 'job' in record and 'seconds' in record:
                helpers.append({'receipt': str(path), 'job': record['job'],
                    'worker_wall_seconds': record['seconds'],
                    'records_scored_or_reused': record['done'], 'failed': record['failed']})
    return {'schema': 'observed_campaign_cost_v1', 'training': cells,
        'official_scoring_jobs': scoring, 'cooperative_helpers': helpers,
        'sum_scoring_job_wall_is_not_elapsed_campaign_wall': True,
        'worker_wall_includes_IO_and_record_lock_wait': True,
        'CPU_process_time_not_instrumented': True,
        'diagnostic_training_excluded_from_main_model_update_counts': True,
        'inherited_context_extraction_cost_not_charged_as_new_extraction': True}
