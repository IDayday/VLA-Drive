"""Bounded CPU pre-scoring of the fixed, already generated rich endpoint columns.

Only the timing of official scoring changes. The permanent rich-bank builder
keeps the same source order, duplicate screening, twelve-candidate limit and
original score requests. No optimizer, new candidate or model forward is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import os
from pathlib import Path
import socket
import time

from iqe.contracts import ScoreRecord, require, strict_record
from iqe.io import atomic_json, digest, file_hash, lock, read_json


def folder(config):
    return config.root/'runtime/rich_official_precompute'


def binding(config, scenes, candidates, job_name):
    from iqe.internal_baseline.bank import protocol
    require(len(scenes) == len(candidates), 'rich official scene join')
    return {'schema': 'canonical_rich_official_column_v1', 'job_name': job_name,
            'config_hash': config.identity, 'pool_hash': read_json(config.root/'SERVING_CANDIDATE_POOL.json')['hash'],
            'Base_binding_hash': file_hash(config.root/'BASE_BINDING.json'),
            'scene_hash': digest([asdict(s) for s in scenes]),
            'candidate_keys': [c.key for c in candidates],
            'protocol_hash': protocol(config)['protocol_hash']}


def pending_indices(candidates, hashes, counts, maximum):
    """Apply exactly the original source-priority rule before any score is read."""
    require(len(candidates) == len(hashes) == len(counts), 'rich priority population')
    pending = []
    for index, candidate in enumerate(candidates):
        if candidate.trajectory_hash not in hashes[index] and counts[index] < maximum:
            pending.append(index)
            hashes[index].add(candidate.trajectory_hash)
            counts[index] += 1
    return pending


def prepared_or_score(config, scenes, candidates, job_name):
    from iqe.internal_baseline.bank import score_candidates
    plan_path = folder(config)/'PLAN.json'
    if not plan_path.exists():
        return score_candidates(config, scenes, candidates, job_name)
    plan = read_json(plan_path)
    require(plan['config_hash'] == config.identity and
            plan['pool_hash'] == read_json(config.root/'SERVING_CANDIDATE_POOL.json')['hash'] and
            plan['rich_plan_hash'] == file_hash(config.root/'runtime/rich_precompute/PLAN.json'),
            'rich official precompute plan changed')
    require(job_name in plan['job_names'], 'rich score column outside fixed precompute plan')
    expected = binding(config, scenes, candidates, job_name)
    output = folder(config)/(job_name+'.json')
    receipt = output.with_suffix('.COMPLETE.json')
    while not receipt.exists():
        status_path = folder(config)/'STATUS.json'
        status = read_json(status_path) if status_path.exists() else {}
        require(status.get('status') != 'FAILED', 'rich official precompute failed; preserve records and resume')
        require(time.time() < plan['deadline_unix'], 'bounded rich official precompute expired; inspect and resume')
        time.sleep(5)
    proof = read_json(receipt)
    require(proof['checksum'] == file_hash(output) and proof['binding_hash'] == digest(expected),
            'rich precomputed official response checksum/identity mismatch')
    result = read_json(output)
    require(result['binding'] == expected and len(result['scores']) == len(candidates),
            'rich precomputed official column changed')
    rows = [strict_record(ScoreRecord, row) for row in result['scores']]
    for scene, candidate, score in zip(scenes, candidates, rows):
        require(score.score_valid and score.candidate_key == candidate.key and
                score.metric_context_hash == scene.metric_context_hash and
                score.score_protocol_hash == expected['protocol_hash'], 'invalid rich precomputed score join')
    return rows


def run(config_path, parallel_columns=2, score_workers=2, hours=6.):
    from .config import load_config
    from .resource_guard import check
    from .bank import pool, load_bank
    from .binding import scenes_by_role
    from .experiments import endpoint
    from .rich_precompute import collect_completed
    from iqe.internal_baseline.bank import score_candidates
    require(1 <= parallel_columns <= 2 and 1 <= score_workers <= 2 and 0 < hours <= 8,
            'bounded rich official CPU allocation')
    config = load_config(config_path)
    check(config, 'precompute-rich-official-labels')
    require(score_workers <= config.scoring_workers, 'official worker budget')
    os.environ['IQE_OFFICIAL_SCORE_WORKERS'] = str(score_workers)
    p = pool(config)
    columns = []
    for item in p['experts']:
        for step in (1000, 3000):
            e = endpoint(config, item['cell'], step, 'raw')
            e['expert_id'] = item['cell']+f'_raw_{step}'
            e['job_name'] = item['cell']+f'_raw_{step}_unique_'+e['model_hash'][:12]
            columns.append(e)
    plan = {'schema': 'fixed_rich_column_cpu_schedule_v1', 'config_hash': config.identity,
            'pool_hash': p['hash'], 'rich_plan_hash': file_hash(config.root/'runtime/rich_precompute/PLAN.json'),
            'job_names': [e['job_name'] for e in columns], 'parallel_columns': parallel_columns,
            'score_workers_per_column': score_workers, 'created_unix': time.time(),
            'deadline_unix': time.time()+hours*3600, 'new_candidates': False, 'model_forwards': 0}
    target = folder(config)
    with lock(target/'RUN.lock'):
        if (target/'PLAN.json').exists():
            old = read_json(target/'PLAN.json')
            for key in ('schema', 'config_hash', 'pool_hash', 'rich_plan_hash', 'job_names'):
                require(old[key] == plan[key], 'rich official continuation identity changed')
            # An expired execution lease does not change the scientific plan.
            if old['deadline_unix'] <= time.time():
                atomic_json(target/f'PLAN_EXPIRED_{int(old["deadline_unix"])}.json', old, immutable=True)
                old = {**old, 'deadline_unix': plan['deadline_unix']}
                atomic_json(target/'PLAN.json', old)
            plan = old
        else:
            atomic_json(target/'PLAN.json', plan, immutable=True)
        def status(state, **details):
            atomic_json(target/'STATUS.json', {'status': state, 'pid': os.getpid(), 'host': socket.gethostname(),
                        'updated_unix': time.time(), **details})
        try:
            while not (config.root/'banks/deploy/scorer_fit/BANK.json').exists():
                require(time.time() < plan['deadline_unix'], 'deployment bank wait expired')
                status('WAITING_DEPLOY_BANK'); time.sleep(10)
            meta, _ = load_bank(config, 'deploy', 'scorer_fit')
            scenes = scenes_by_role(config)['scorer_fit']
            require(meta['scene_ids'] == [s.scene_id for s in scenes], 'rich official D_S order changed')
            hashes = [{row['candidate']['trajectory_hash'] for row in rows} for rows in meta['records']]
            counts = [len(rows) for rows in meta['records']]
            def score(item, selected, candidates):
                name = item['job_name']; expected = binding(config, selected, candidates, name)
                output = target/(name+'.json'); receipt = output.with_suffix('.COMPLETE.json')
                if receipt.exists():
                    return {'column': name, 'records': len(prepared_or_score(config, selected, candidates, name)), 'reused': True}
                labels = score_candidates(config, selected, candidates, name)
                atomic_json(output, {'binding': expected, 'scores': [asdict(s) for s in labels]}, immutable=True)
                atomic_json(receipt, {'status': 'COMPLETE', 'checksum': file_hash(output),
                            'binding_hash': digest(expected), 'officially_validated_count': len(labels)}, immutable=True)
                return {'column': name, 'records': len(labels), 'reused': False}
            futures = []; results = []
            with ThreadPoolExecutor(max_workers=parallel_columns) as executor:
                for item in columns:
                    require(time.time() < plan['deadline_unix'], 'rich official column admission expired')
                    candidates = collect_completed(config, scenes, item['expert_id'], item['model_hash'])
                    require(candidates is not None, 'rich source records incomplete; no hidden model fallback')
                    indices = pending_indices(candidates, hashes, counts, config.max_rich_candidates)
                    if indices:
                        futures.append(executor.submit(score, item, [scenes[i] for i in indices], [candidates[i] for i in indices]))
                    status('RUNNING', submitted_columns=len(futures), current_column=item['expert_id'])
                for future in futures:
                    results.append(future.result()); status('RUNNING', completed=results)
            status('COMPLETE', columns=results, model_forwards=0, new_candidates=False)
        except BaseException as error:
            status('FAILED', error=repr(error), committed_records_retained=True)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--parallel-columns', type=int, default=2)
    p.add_argument('--score-workers', type=int, default=2); p.add_argument('--hours', type=float, default=6.)
    a = p.parse_args(); run(a.config, a.parallel_columns, a.score_workers, a.hours)


if __name__ == '__main__':
    main()
