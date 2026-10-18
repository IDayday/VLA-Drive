"""Stream sealed single-candidate Navtest requests through another official release.

This process never loads a policy or generates/selects a trajectory. Incomplete
input predictions remain pending; errors retain the full evaluation denominator.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from scripts.analysis.fm_transfer.build_versioned_pdms_cache import make_spec
from starVLA.rl.fm_transfer.versioned_pdms import initialize, score_requests
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json, publication_lock
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch


def authenticated(path):
    value = json.loads(Path(path).read_text()); body = dict(value); digest = body.pop('sha256')
    if fingerprint(body) != digest: raise ValueError('Changed manifest: ' + str(path))
    return value


def validate_input_manifest(manifest, population):
    tokens = population['navtest_tokens']
    if (manifest['cohort'] != 'NAVTEST' or manifest['seed'] != 2101 or manifest['qualification_only'] or
            manifest['scenes'] != 12146 or manifest['tokens_sha256'] != fingerprint(tokens) or
            manifest['modes'] != ['E_NATIVE', 'E_SHIFT5'] or
            manifest['population_identity'] != population['identity_sha256']):
        raise ValueError('Only complete-population fixed-seed single-candidate Navtest input is allowed')
    lock = manifest['final_lock']
    fixed_stage = (lock.get('schema') == 'source_process_fixed_milestone_navtest_v1' and
        lock.get('observed_update') in (20000, 170000) and lock.get('evaluation_budget_complete') is True and
        lock['formal_budget_complete'] == (lock['observed_update'] == 170000) and
        lock.get('Navtest_scores_read_for_selection') is False)
    if (lock['status'] != 'LOCKED_BEFORE_FINAL_NAVTEST' or not (lock['formal_budget_complete'] or fixed_stage) or
            manifest['weight_sha256'] not in lock['weight_sha256']):
        raise ValueError('Input model was not locked before final evaluation')


def rescore_one(job):
    evaluation, output, token, log, manifest_sha, identity_sha = job
    source = Path(evaluation) / 'scenes' / token[:2] / (token + '.pt')
    source_seal = json.loads(source.with_suffix('.json').read_text())
    if (source_seal['token'] != token or source_seal['log'] != log or
            source_seal['identity_sha256'] != manifest_sha or source_seal['payload_sha256'] != file_sha(source)):
        raise ValueError('Source prediction seal changed')
    data = torch.load(source, map_location='cpu', weights_only=False)
    if data['token'] != token or data['log'] != log or data['identity_sha256'] != manifest_sha:
        raise ValueError('Wrong original prediction identity')
    modes = ['E_NATIVE', 'E_SHIFT5']
    for mode in modes:
        if data['records'][mode]['status'] != 'OK' or data['records'][mode]['metrics'] != source_seal['metrics'][mode]:
            raise ValueError('Invalid original canonical score')
    requests = [data['records'][mode]['requested_trajectory'] for mode in modes]
    records = score_requests((token, log, requests))
    payload = dict(token=token, log=log, identity_sha256=identity_sha,
        original_prediction_sha256=source_seal['payload_sha256'],
        original_manifest_identity=manifest_sha, records=dict(zip(modes, records)),
        original_v1_1_metrics=source_seal['metrics'])
    path = Path(output) / 'scenes' / token[:2] / (token + '.pt')
    path.parent.mkdir(parents=True, exist_ok=True)
    with publication_lock(path.with_suffix('.lock')):
        if path.with_suffix('.json').exists():
            raise RuntimeError('Duplicate score owner; do not overwrite a committed score')
        if path.exists(): path.rename(path.with_name(path.name + '.unsealed-' + uuid.uuid4().hex))
        atomic_torch(path, payload)
        atomic_json(path.with_suffix('.json'), dict(token=token, log=log, identity_sha256=identity_sha,
            payload_sha256=file_sha(path), original_prediction_sha256=source_seal['payload_sha256'],
            metrics={mode: records[i]['metrics'] for i, mode in enumerate(modes)}))
    return dict(token=token, scoring_seconds=sum(r['scoring_seconds'] for r in records), physical_scoring_calls=2)


def reusable(output, evaluation, token, identity_sha):
    path = Path(output) / 'scenes' / token[:2] / (token + '.pt')
    if not path.with_suffix('.json').exists(): return False
    seal = json.loads(path.with_suffix('.json').read_text())
    original = Path(evaluation) / 'scenes' / token[:2] / (token + '.pt')
    old = json.loads(original.with_suffix('.json').read_text())
    if (seal['token'] != token or seal['identity_sha256'] != identity_sha or
            seal['payload_sha256'] != file_sha(path) or
            seal['original_prediction_sha256'] != old['payload_sha256'] or
            file_sha(original) != old['payload_sha256']):
        raise ValueError('Changed rescore or original prediction; no silent reuse')
    return True


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--evaluation', required=True); p.add_argument('--output', required=True)
    p.add_argument('--source', required=True); p.add_argument('--version', choices=['v1.0', 'v1.1'], required=True)
    p.add_argument('--cache-root', required=True); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    p.add_argument('--wait-seconds', type=float, default=14400)
    a = p.parse_args(); host = require_dispatch()
    if not 1 <= a.workers <= 32 or not 0 <= a.shard < a.shards <= 64 or not 0 < a.wait_seconds <= 86400:
        raise ValueError('Invalid bounded dispatch')
    pop, cache_identity, spec = make_spec(a.source, a.version, a.cache_root, POPULATION,
        '/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/test', '/mnt/navsim/maps')
    if authenticated(Path(a.cache_root) / 'manifest.json') != cache_identity:
        raise ValueError('Wrong version-specific cache')
    evaluation, output = Path(a.evaluation), Path(a.output)
    manifest = authenticated(evaluation / 'manifest.json'); validate_input_manifest(manifest, pop)
    identity = dict(schema='saved_navtest_official_release_rescore_v1',
        original_evaluation=str(evaluation.resolve()), original_manifest_sha256=file_sha(evaluation / 'manifest.json'),
        original_manifest_identity=manifest['sha256'], weight_sha256=manifest['weight_sha256'],
        original_scoring_identity=manifest['scoring'], population_identity=pop['identity_sha256'],
        tokens_sha256=manifest['tokens_sha256'], scenes=12146, logs=136, seed=2101,
        modes=manifest['modes'], protocol=spec['protocol'], cache_identity=cache_identity['sha256'],
        scorer_adapter_sha256=file_sha(Path(__file__).resolve().parents[3] / 'starVLA/rl/fm_transfer/versioned_pdms.py'),
        worker_sha256=file_sha(__file__), additional_model_predictions=0)
    identity['sha256'] = fingerprint(identity)
    output.mkdir(parents=True, exist_ok=True)
    with publication_lock(output / '.manifest.lock'):
        path = output / 'manifest.json'
        if path.exists() and json.loads(path.read_text()) != identity: raise ValueError('Score identity changed')
        if not path.exists(): atomic_json(path, identity)
    invocation = uuid.uuid4().hex
    attempts = output / 'attempts'; attempts.mkdir(exist_ok=True)
    receipt = attempts / (invocation + '.json')
    status = dict(status='RUNNING', invocation=invocation, host=host, shard=a.shard, shards=a.shards,
        started_utc=datetime.now(timezone.utc).isoformat(), identity_sha256=identity['sha256'],
        new=0, reused=0, physical_scoring_calls=0, scoring_worker_seconds=0., additional_model_predictions=0)
    assigned = pop['navtest_tokens'][a.shard::a.shards]
    pending = set(assigned); status['assigned_scenes'] = len(assigned)
    started = time.monotonic(); atomic_json(receipt, status)
    try:
        with publication_lock(output / '.owners' / f'shard{a.shard}-of-{a.shards}.lock'):
            with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize, initargs=(spec,)) as pool:
                futures = {}
                while pending or futures:
                    require_dispatch()
                    if time.monotonic() - started > a.wait_seconds: raise TimeoutError('Full denominator remains incomplete')
                    for token in assigned:
                        if token not in pending: continue
                        if reusable(output, evaluation, token, identity['sha256']):
                            pending.remove(token); status['reused'] += 1; continue
                        if len(futures) >= a.workers * 2: break
                        log = pop['token_logs'][token]
                        if not (evaluation / 'scenes' / token[:2] / (token + '.json')).exists(): continue
                        if not (Path(a.cache_root) / log / 'unknown' / token / 'metric_cache.json').exists(): continue
                        job = (str(evaluation), str(output), token, log, manifest['sha256'], identity['sha256'])
                        futures[pool.submit(rescore_one, job)] = token; pending.remove(token)
                    if futures:
                        completed, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                        for future in completed:
                            row = future.result(); futures.pop(future)
                            status['new'] += 1; status['physical_scoring_calls'] += row['physical_scoring_calls']
                            status['scoring_worker_seconds'] += row['scoring_seconds']
                    else:
                        if pending: time.sleep(30)
                    status['wall_seconds'] = time.monotonic() - started
                    status['pending_scenes'] = len(pending) + len(futures)
                    atomic_json(receipt, status)
        if status['new'] + status['reused'] != len(assigned): raise ValueError('Incomplete score denominator')
        status['status'] = 'ASSIGNED_COMPLETE'
    except BaseException as error:
        status.update(status='FAILED_PRESERVED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status['wall_seconds'] = time.monotonic() - started
        status['ended_utc'] = datetime.now(timezone.utc).isoformat(); atomic_json(receipt, status)
    print(json.dumps(dict(status=status['status'], receipt=str(receipt), scenes=len(assigned))), flush=True)


if __name__ == '__main__': main()
