"""Finite CPU-only TRAIN process-label replay; no new trajectories or actor updates."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time
import traceback
import uuid

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement import v1_reward
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.simwam_recipe.contracts import CACHE_ROOTS
from starVLA.rl.trajectory_repair.feedback import read_feedback
from starVLA.rl.trajectory_repair import official_events


SPEC = None


def prepare_attempt(out, attempt):
    for sub in ('scenes', 'invocations', 'attempts/' + attempt):
        (Path(out) / sub).mkdir(parents=True, exist_ok=True)


def initialize(spec):
    global SPEC
    SPEC = spec
    if spec['sources'] != dict(exporter=file_sha(__file__),trace=file_sha(official_events.__file__)):
        raise ValueError('Source changed before CPU worker initialization')
    torch.set_num_threads(1)
    v1_reward.initialize(spec['scorer'])


def export_one(path):
    path = Path(path)
    out = Path(SPEC['output'])
    progress = out / 'attempts' / SPEC['attempt'] / (path.stem + '.json')
    ledger = dict(status='STARTED',attempt=SPEC['attempt'],started_unix=time.time(),
                  candidate_scoring_calls_started=0,candidate_scoring_calls_completed=0)
    atomic_json(progress, ledger)
    try:
        row = json.loads(path.read_text()); body = dict(row)
        if (body.pop('sha256') != fingerprint(body) or row['status'] != 'COMPLETE' or
            row['split'] != 'TRAIN' or row['scene_token'] != path.stem or row['base_model_sha256'] != SPEC['base_model_sha256'] or
            row['common_protocol_sha256'] != SPEC['common_protocol_sha256']):
            raise ValueError('Changed or wrong-split input search record')
        dest = out / 'scenes' / (path.stem + '.pt')
        receipt = dest.with_suffix('.json')
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if (old['identity'] != SPEC['identity'] or old['input_sha256'] != file_sha(path) or
                old['payload_sha256'] != file_sha(dest) or old['status'] != 'COMPLETE'):
                raise ValueError('Cannot reuse changed process labels')
            ledger.update(status='REUSED',ended_unix=time.time(),records=old['records'])
            atomic_json(progress, ledger)
            return ledger
        if dest.exists():
            raise ValueError('Unsealed earlier payload retained; review before retrying')
        candidates = row['result']['candidates']
        lookup = {r['candidate_id']: r for r in candidates}
        parents = {r['parent_candidate_id'] for r in candidates if r.get('external_proposal')}
        selected = [lookup[k] for k in sorted(parents)]
        selected += [r for r in candidates if r.get('external_proposal') and r.get('raw_reward') is not None]
        if len(selected) != 25:
            raise ValueError('Locked replay expects one parent and24 existing scored edits')
        records = []
        for candidate in selected:
            old = candidate['feedback']
            if old['protocol_sha256'] != SPEC['scorer']['protocol_sha256']:
                raise ValueError('Original scorer identity changed')
            ledger['candidate_scoring_calls_started'] += 1
            atomic_json(progress, ledger)
            fresh = official_events.score_scene_with_events((row['scene_token'], row['log_id'],
                [np.asarray(old['requested_trajectory'])]))[0]
            ledger['candidate_scoring_calls_completed'] += 1
            read_feedback(fresh)
            if any(abs(fresh['metrics'][k]-old['metrics'][k])>2e-12 for k in old['metrics']):
                raise ValueError('Replay changed an existing canonical submetric')
            if float(np.abs(np.asarray(fresh['simulated_states'])-old['simulated_states']).max())>1e-8:
                raise ValueError('Replay changed simulated execution')
            if not np.array_equal(fresh['non_drivable_body_by_time'],old['non_drivable_body_by_time']):
                raise ValueError('Replay changed official DAC timeline')
            records.append(dict(candidate_id=candidate['candidate_id'],parent_candidate_id=candidate.get('parent_candidate_id'),
                                edit_parameters=candidate.get('parameters'),feedback=fresh))
        atomic_torch(dest,dict(split='TRAIN',scene_token=row['scene_token'],log_id=row['log_id'],
                              identity=SPEC['identity'],input_sha256=file_sha(path),candidates=records))
        atomic_json(receipt,dict(status='COMPLETE',identity=SPEC['identity'],input_sha256=file_sha(path),
                                payload_sha256=file_sha(dest),records=len(records)))
        ledger.update(status='COMPLETE',ended_unix=time.time(),records=len(records))
        atomic_json(progress, ledger)
        return ledger
    except BaseException:
        ledger.update(status='FAILED',ended_unix=time.time(),error=traceback.format_exc())
        atomic_json(progress, ledger)
        raise


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--train-search-root',required=True)
    parser.add_argument('--bank-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--qualification-scenes',type=int,choices=[1,4])
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not 1<=args.workers<=4:
        raise PermissionError('Explicit CPU-only replay, maximum four workers')
    policy=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host=socket.gethostname().removesuffix('-worker-0')
    grant=policy['task_authorizations']['qwen_flow_trajectory_repair_20261025']
    if host not in policy['allowed_hosts'] or host not in grant['CPU_preparation_hosts']:
        raise PermissionError('CPU replay host outside permanent task authorization')
    source,bank,out=Path(args.train_search_root).resolve(),Path(args.bank_root).resolve(),Path(args.output)
    if source.name!='TRAIN':raise ValueError('TRAIN-only input required')
    index=json.loads((bank/'index_TRAIN.json').read_text());body=dict(index)
    if body.pop('sha256')!=fingerprint(body):raise ValueError('Changed TRAIN bank index')
    paths=sorted((source/'scenes').glob('*.json'))
    if {p.stem for p in paths}!={Path(p).stem for p in index['TRAIN']} or len(paths)!=512:
        raise ValueError('Locked512-scene label population differs')
    if args.qualification_scenes:paths=paths[:args.qualification_scenes]
    scorer=dict(repo=str(Path.cwd()),maps_root='/mnt/navsim/maps',cache_roots=CACHE_ROOTS,
                protocol_sha256=v1_reward.protocol_identity(Path.cwd())['sha256'])
    identity=dict(schema='trajectory_repair_TRAIN_process_replay_v1',base_model_sha256=index['base_model_sha256'],
                  common_protocol_sha256=index['protocol_sha256'],scorer=scorer,
                  sources=dict(exporter=file_sha(__file__),trace=file_sha(official_events.__file__)),
                  input_files_sha256=fingerprint([file_sha(p) for p in paths]),scenes=len(paths),
                  qualification=bool(args.qualification_scenes),
                  existing_requests_per_scene=25,maximum_candidate_scoring_calls=25*len(paths),
                  new_policy_trajectories=0,optimizer_updates=0,workers=args.workers)
    identity['sha256']=fingerprint(identity);out.mkdir(parents=True,exist_ok=True)
    owner=(out/'owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'FAILED.json').exists():raise ValueError('Failed replay requires reviewed recovery and budget, no automatic retry')
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Changed replay identity; use a new output')
    else:atomic_json(out/'identity.json',identity)
    attempt=uuid.uuid4().hex; started=time.time()
    prepare_attempt(out, attempt)
    spec=dict(output=str(out),attempt=attempt,identity=identity['sha256'],scorer=scorer,
              sources=identity['sources'],base_model_sha256=index['base_model_sha256'],common_protocol_sha256=index['protocol_sha256'])
    atomic_json(out/'invocations'/f'{attempt}.json',dict(status='RUNNING',pid=os.getpid(),host=host,
                started_unix=started,identity=identity['sha256']))
    result=[]
    try:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize,initargs=(spec,)) as pool:
            futures=[pool.submit(export_one,str(p)) for p in paths]
            for future in as_completed(futures):
                try:result.append(future.result())
                except BaseException:
                    for pending in futures:pending.cancel()
                    raise
                atomic_json(out/'progress.json',dict(status='RUNNING',completed_scenes=len(result),total_scenes=len(paths),
                    actual_completed_scoring_calls=sum(r['candidate_scoring_calls_completed'] for r in result)))
        historical=[json.loads(p.read_text()) for p in (out/'attempts').glob('*/*.json')]
        atomic_json(out/'COMPLETE.json',dict(status='COMPLETE',identity=identity['sha256'],scenes=len(result),
                    records=sum(r['records'] for r in result),new_scoring_calls=sum(r['candidate_scoring_calls_completed'] for r in result),
                    cumulative_scoring_calls_started=sum(r['candidate_scoring_calls_started'] for r in historical),
                    cumulative_scoring_calls_completed=sum(r['candidate_scoring_calls_completed'] for r in historical),
                    wall_seconds=time.time()-started,no_actor_or_optimizer_updates=True))
    except BaseException:
        atomic_json(out/'FAILED.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',attempt=attempt,error=traceback.format_exc()))
        raise
    finally:
        atomic_json(out/'invocations'/f'{attempt}.json',dict(status='COMPLETE' if len(result)==len(paths) else 'FAILED',
            started_unix=started,ended_unix=time.time(),host=host,identity=identity['sha256'],
            completed_scenes=len(result),per_candidate_attempt_costs_retained_in=str(out/'attempts'/attempt)))


if __name__=='__main__':main()
