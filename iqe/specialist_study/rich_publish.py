"""Publish every fixed rich endpoint job for existing record-locked helpers.

The original coordinator remains the only column/whole-bank completion owner.
This process only brings forward immutable CPU requests for already generated
candidates. Source priority, deduplication, candidate budget and labels do not
depend on the worker schedule.
"""
import argparse
import os
import socket
import time

from iqe.contracts import require
from iqe.io import atomic_json, file_hash, lock, read_json


def publish(config_path):
    from .config import load_config
    from .resource_guard import check
    from .bank import load_bank, pool
    from .binding import scenes_by_role
    from .experiments import endpoint
    from .rich_precompute import collect_completed
    from .rich_labels import pending_indices, folder
    from iqe.internal_baseline.bank import prepare_score_job
    from iqe.internal_baseline import score_worker
    config = load_config(config_path)
    check(config, 'precompute-rich-official-labels')
    target = folder(config)
    with lock(target/'PUBLISH.lock'):
        plan = read_json(target/'PLAN.json')
        require(plan['config_hash'] == config.identity and plan['pool_hash'] == pool(config)['hash']
                and plan['rich_plan_hash'] == file_hash(config.root/'runtime/rich_precompute/PLAN.json'),
                'rich request publication identity changed')
        meta, _ = load_bank(config, 'deploy', 'scorer_fit')
        scenes = scenes_by_role(config)['scorer_fit']
        require(meta['scene_ids'] == [s.scene_id for s in scenes], 'rich publication scene order')
        hashes = [{row['candidate']['trajectory_hash'] for row in rows} for rows in meta['records']]
        counts = [len(rows) for rows in meta['records']]
        published = []
        for item in pool(config)['experts']:
            for step in (1000, 3000):
                require(time.time() < plan['deadline_unix'], 'rich publication deadline expired')
                e = endpoint(config, item['cell'], step, 'raw')
                expert_id = item['cell']+f'_raw_{step}'
                name = expert_id+'_unique_'+e['model_hash'][:12]
                require(name in plan['job_names'], 'unregistered rich source')
                candidates = collect_completed(config, scenes, expert_id, e['model_hash'])
                require(candidates is not None, 'rich source incomplete; no forward fallback')
                indices = pending_indices(candidates, hashes, counts, config.max_rich_candidates)
                if indices:
                    path, job = prepare_score_job(config, [scenes[i] for i in indices],
                                                   [candidates[i] for i in indices], name)
                    # Same protocol the primary worker writes before its loop.
                    atomic_json(path.parent/'COOPERATIVE_PROTOCOL.json', {
                        'schema': 'record_locked_official_score_v1', 'job_hash': file_hash(path),
                        'worker_code_hash': file_hash(score_worker.__file__), 'lock_suffix': '.SCORING.lock',
                        'official_numeric_function_changed': False}, immutable=True)
                    published.append({'column': name, 'records': len(job['items']),
                                      'job': str(path), 'job_hash': file_hash(path)})
                else:
                    published.append({'column': name, 'records': 0, 'duplicate_or_full': True})
                atomic_json(target/'PUBLICATION_STATUS.json', {
                    'status': 'RUNNING', 'pid': os.getpid(), 'host': socket.gethostname(),
                    'updated_unix': time.time(), 'columns': published,
                    'additional_primary_workers': 0, 'new_candidates': False})
        atomic_json(target/'PUBLICATION_COMPLETE.json', {
            'status': 'COMPLETE', 'columns': published, 'config_identity': config.identity,
            'config_file_sha256': file_hash(config_path), 'plan_file_sha256': file_hash(target/'PLAN.json'),
            'additional_primary_workers': 0, 'new_candidates': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    publish(p.parse_args().config)


if __name__ == '__main__':
    main()
