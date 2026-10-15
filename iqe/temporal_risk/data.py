"""Bounded official trace extraction on the already fixed 64-scene D_S diagnostic."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from iqe.contracts import require
from iqe.io import atomic_bytes, atomic_json, digest, file_hash, lock, read_json
from iqe.internal_baseline.bank import protocol
from iqe.specialist_study.bank import load_bank
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.config import StudyConfig
from iqe.specialist_study.resource_guard import check


def resource(root):
    policy = read_json('/mnt/project/server_dispatch_policy.json')
    allowed = set(policy['allowed_hosts'])
    for item in policy['allowed_hosts'].values():
        allowed.update(item.get('aliases', []))
    require(socket.gethostname() in allowed, 'permanent authorized host required')
    return check(SimpleNamespace(root=Path(root)), 'research-preflight')


def prepare(source_root, control_root, output):
    root = Path(output).resolve(); resource(root)
    source = StudyConfig(**read_json(Path(source_root)/'resolved_config.json'))
    require(not root.is_relative_to(source.root), 'new isolated output; never mutate original banks')
    control = Path(control_root)
    selection = read_json(control/'DIAGNOSTIC_DATA.json')
    ids = selection['scene_ids']
    require(len(ids) == len(set(ids)) == 64 and not selection['final_test_or_dev_access'], 'fixed64 training diagnostic')
    dm, deploy = load_bank(source, 'deploy', 'scorer_fit')
    rm, rich = load_bank(source, 'rich', 'scorer_fit')
    require(rm['hash'] == selection['rich_bank_hash'] and rm['scene_ids'] == dm['scene_ids'], 'changed source bank')
    scene_map = {s.scene_id:s for s in scenes_by_role(source)['scorer_fit']}
    require(all(sid in scene_map and scene_map[sid].split_role == 'incremental_fit' for sid in ids), 'only D_S input')
    proto = protocol(source)
    indices = [rm['scene_ids'].index(sid) for sid in ids]
    source_hashes = {name:file_hash(Path(__file__).with_name(name)) for name in ('data.py','labels.py','worker.py')}
    items, seen = [], set()
    for sid, row in zip(ids, indices):
        scene = scene_map[sid]
        for record in rm['records'][row]+dm['records'][row]:
            key = record['label']['candidate_key']
            if key in seen: continue
            seen.add(key)
            require(record['candidate']['scene_id'] == sid and record['label']['score_valid'], 'candidate join')
            request = dict(candidate=record['candidate'], scene=asdict(scene), expected_label=record['label'],
                navsim_root=proto['navsim_root'], nuplan_root=proto['nuplan_root'],
                protocol_hash=proto['protocol_hash'], config=proto['config'],
                seed=int(digest(dict(scene_id=sid, seed=42, repetition=0))[:8], 16),
                source_hashes=source_hashes)
            items.append(dict(request=request, response=str(root/'records'/(key+'.json'))))
    require(len(items) <= 64*12, 'diagnostic label cap768; no extra candidates')
    job = dict(schema='official_temporal_label_diagnostic_v1', source_campaign=str(source.root),
        control_root=str(control.resolve()), scene_ids=ids, source_logs=[scene_map[s].source_log_id for s in ids],
        deployment_bank_hash=dm['hash'], rich_bank_hash=rm['hash'],
        score_protocol=proto, source_hashes=source_hashes, items=items, worker_limit=4,
        attempts_per_invocation=2, max_labels=768, no_dev_or_test=True,
        no_new_candidates=True, no_base_updates=True)
    atomic_json(root/'JOB.json', job, immutable=True)
    plan = dict(registered_utc=datetime.now(timezone.utc).isoformat(), job_hash=digest(job),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scenes=64, candidates=len(items), workers=4, maximum_new_optimizer_updates=300,
        training_runs=1, control='existing matched geometric300',
        changed_factor='auxiliary cumulative official event-time BCE, weight1, same total score head',
        inherited=['Base','experts','rich candidates','random common initialization','scene order','LR','loss terms'],
        scientific_generalization_result=False, no_automatic_full_training=True,
        stopping_rule='stop on data/numeric failure; no threshold or budget adaptation')
    if not (root/'PLAN.json').exists(): atomic_json(root/'PLAN.json', plan, immutable=True)
    require(read_json(root/'PLAN.json')['job_hash'] == digest(job), 'diagnostic registration changed')
    return job


def collect(root, job):
    lookup, errors = {}, []
    for item in job['items']:
        path = Path(item['response']); receipt = path.with_suffix('.COMPLETE.json')
        require(receipt.exists(), 'temporal label incomplete; resume build')
        rec = read_json(receipt)
        require(rec['checksum'] == file_hash(path) and rec['request_hash'] == digest(item['request']), 'trace receipt changed')
        result = read_json(path)
        require(result['score_valid'] and result['supervision_only'] and not result['inference_input'], 'invalid trace')
        lookup[result['candidate_key']] = result
        errors.append(result['original_bank_score_max_abs_error'])
    source = StudyConfig(**read_json(Path(job['source_campaign'])/'resolved_config.json'))
    stats, arrays = {}, {}
    for kind in ('deploy', 'rich'):
        meta, bank = load_bank(source, kind, 'scorer_fit')
        require(meta['hash'] == job['deployment_bank_hash' if kind=='deploy' else 'rich_bank_hash'], 'bank changed')
        ix = [meta['scene_ids'].index(sid) for sid in job['scene_ids']]
        steps = bank['trajectories'].shape[-2]
        shape = (len(ix),bank['scores'].shape[1],steps,3)
        event, valid = np.zeros(shape, bool), np.zeros(shape, bool)
        first = []
        for x, old in enumerate(ix):
            for k, ref in enumerate(meta['records'][old]):
                result = lookup[ref['label']['candidate_key']]
                event[x,k] = result['labels']['cumulative_event']
                valid[x,k] = result['labels']['valid']
                first.append(result['labels']['first_event_s'])
        require(np.array_equal(valid.all((-2,-1)),bank['valid'][ix]), 'missing or padded events')
        arrays[kind+'_events'], arrays[kind+'_valid'] = event, valid
        stats[kind] = dict(valid_candidates=int(bank['valid'][ix].sum()),
            event_candidates=event[:,:,-1].sum((0,1)).tolist(),
            cumulative_events_by_time=event.sum((0,1)).tolist(),
            first_event_seconds=[sorted(t[j] for t in first if t[j] is not None) for j in range(3)])
    stream = io.BytesIO(); np.savez(stream, **arrays)
    atomic_bytes(root/'events.npz', stream.getvalue(), immutable=True)
    result = dict(status='COMPLETE', job_hash=digest(job), scenes=64,
        logs=len(set(job['source_logs'])), official_candidate_calls=len(lookup),
        original_bank_score_max_abs_error=max(errors), array_hash=file_hash(root/'events.npz'),
        banks=stats, no_dev_or_test=True, no_new_candidates=True,
        event_semantics='first at-fault event including NC0.5; first offroad; first official TTC infraction')
    atomic_json(root/'COMPLETE.json', result, immutable=True)
    return result


def build(root, job):
    root = Path(root); resource(root)
    with lock(root/'BUILD.lock'):
        if (root/'COMPLETE.json').exists():
            return collect(root,job)
        source = StudyConfig(**read_json(Path(job['source_campaign'])/'resolved_config.json'))
        protocol(source)
        repo = Path(__file__).resolve().parents[2]
        env = dict(os.environ, PYTHONPATH=str(repo), CUDA_VISIBLE_DEVICES='',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        def worker(shard):
            with open(root/f'worker_{shard}.log','a') as log:
                return subprocess.run([source.scoring_python,'-m','iqe.temporal_risk.worker',
                    str(root/'JOB.json'),'--shard',str(shard),'--shards','4'],cwd=repo,env=env,
                    stdout=log,stderr=log,timeout=1800).returncode
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            codes=list(pool.map(worker,range(4)))
        atomic_json(root/'BUILD_STATUS.json',dict(codes=codes,seconds=time.perf_counter()-start,
            status='COMPLETE' if not any(codes) else 'PARTIAL'))
        require(not any(codes),'temporal score worker failed; inspect retained evidence')
        return collect(root,job)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('command',choices=('prepare','build'))
    p.add_argument('--source-campaign',required=True);p.add_argument('--control-root',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    job=prepare(a.source_campaign,a.control_root,a.output)
    result=build(a.output,job) if a.command=='build' else dict(status='PREPARED',scenes=64,candidates=len(job['items']))
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
