"""Validate/build fixed-bank native event evidence using CPU scene shards."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import time
import traceback

import numpy as np

from starVLA.model.modules.qwen_world_postcot.candidate_bank import CandidateBank, file_sha, fingerprint
from starVLA.rl.qwen_world_compare.contracts import require_dispatch, boundary_stop_due
from starVLA.rl.qwen_world_compare.evidence import (
    EvidenceWriter, EvidenceBank, SCHEMA, EVENT_RULE, atomic_json,
    make_evidence_record, merge_evidence, read_manifest,
)
from starVLA.rl.safe_flow_improvement import v1_reward
from starVLA.rl.simwam_recipe.contracts import CACHE_ROOTS

_BANK = _IDENTITY = _STOP = None


def source_files(repo):
    names = ['starVLA/rl/qwen_world_compare/evidence.py',
             'starVLA/model/modules/qwen_world_compare/contracts.py',
             'scripts/analysis/constrained_flow_noise/batched_canonical_score.py',
             'starVLA/rl/safe_flow_improvement/v1_reward.py',
             'tools/qwen_world_compare/build_evidence_bank.py']
    return {name: file_sha(Path(repo)/name) for name in names}


def evaluator_identity(repo):
    protocol = v1_reward.protocol_identity(repo)
    root = Path(repo)/'navsim_v1.1/navsim/navsim'
    names = ['planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py',
             'planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py',
             'evaluate/pdm_score.py']
    return dict(protocol_sha256=protocol['sha256'], native_sources={name: file_sha(root/name) for name in names},
                batched_adapter_sha256=file_sha(Path(repo)/'scripts/analysis/constrained_flow_noise/batched_canonical_score.py'),
                native_event_rule=EVENT_RULE)


def identity_for(bank, repo):
    evaluator = evaluator_identity(repo)
    if bank.manifest['identity']['score_protocol_sha256'] != evaluator['protocol_sha256']:
        raise ValueError('Fixed bank and actual native scorer protocol differ')
    return dict(schema=SCHEMA, source_bank_hash=bank.bank_hash,
                source_bank_manifest_sha256=file_sha(Path(bank.manifest_path)),
                source_data_manifest_sha256=bank.manifest['identity']['source_data_manifest_sha256'],
                score_protocol_sha256=evaluator['protocol_sha256'], evaluator_sha256=fingerprint(evaluator),
                evaluator_identity=evaluator, event_rule_sha256=fingerprint(EVENT_RULE),
                code_files=source_files(repo), split='TRAIN', maximum_candidates_per_scene=16,
                immutable_candidate_refresh=False)


def initialize_worker(bank_root, repo, identity, stop_file):
    global _BANK, _IDENTITY, _STOP
    require_dispatch()
    if source_files(repo) != identity['code_files']:
        raise ValueError('Evidence producer source files changed after registration')
    _BANK = CandidateBank(bank_root, allowed_splits=('TRAIN',), max_cached_scenes=2)
    _IDENTITY, _STOP = identity, stop_file
    if _BANK.bank_hash != identity['source_bank_hash']:
        raise ValueError('Fixed candidate bank changed')
    spec = dict(repo=str(repo), maps_root='/mnt/navsim/maps', cache_roots=CACHE_ROOTS,
                protocol_sha256=identity['score_protocol_sha256'])
    v1_reward.initialize(spec)


def build_part(job):
    from scripts.analysis.constrained_flow_noise.batched_canonical_score import score_scene
    part_root, scenes, resume, chunk = job
    part_root = Path(part_root)
    if (part_root/'COMPLETE.json').exists():
        m = read_manifest(part_root)
        if m['identity'] != _IDENTITY or m['scenes'] != scenes:
            raise ValueError('Completed evidence part differs from this plan')
        return dict(status='REUSED_COMPLETE', path=str(part_root), counts=m['counts'])
    started = time.monotonic()
    with EvidenceWriter(part_root, _IDENTITY, scenes, resume=resume) as writer:
        atomic_json(part_root/'WORKER.json', dict(pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            host=socket.gethostname(), part=str(part_root), started_unix=time.time(),
            source_files=_IDENTITY['code_files']))
        for token in scenes:
            if boundary_stop_due() or (_STOP and Path(_STOP).exists()):
                result = dict(status='PAUSED', counts=writer.counts(), accounting=writer.accounting())
                atomic_json(part_root/'PAUSED.json', result); return result
            require_dispatch()
            if writer.has_scene(token):
                continue
            records = _BANK.records(token)
            if not 1 <= len(records) <= 16:
                raise ValueError('Fixed evidence candidate count violates registered bound')
            if len({r.source_log for r in records}) != 1:
                raise ValueError('Mixed logs in a candidate scene')
            requests = np.asarray([r.trajectory_fp32 for r in records], dtype=np.float32)
            attempt = writer.begin_attempt(token, len(records)); then = time.monotonic()
            try:
                feedback = score_scene((token, records[0].source_log, requests), candidate_chunk=chunk)
                if len(feedback) != len(records):
                    raise ValueError('Native scorer omitted a candidate')
                values = [make_evidence_record(r, f, source_bank_hash=_BANK.bank_hash,
                          evaluator_identity=_IDENTITY['evaluator_identity']) for r, f in zip(records, feedback)]
                writer.finish_attempt(attempt, status='VALIDATED_SCORED', returned_candidates=len(feedback),
                    wall_seconds=time.monotonic()-then, simulator_batches=math.ceil(len(records)/chunk))
                writer.append_scene(token, values)
            except BaseException:
                writer.finish_attempt(attempt, status='FAILED_UNKNOWN_PARTIAL', returned_candidates=0,
                    wall_seconds=time.monotonic()-then, simulator_batches=None)
                atomic_json(part_root/('FAILURE_'+attempt+'.json'), dict(status='FAILED', token=token,
                    old_record_ids=[r.record_id for r in records], error=traceback.format_exc(),
                    candidate_count=len(records), reward_zero_substitution=False))
                raise
            counts = writer.counts()
            if counts['scenes'] % 25 == 0 or counts['scenes'] == len(scenes):
                atomic_json(part_root/'PROGRESS.json', dict(status='RUNNING', at=time.time(),
                    counts=counts, accounting=writer.accounting(),
                    invocation_wall_seconds=time.monotonic()-started))
        manifest = writer.finish()
    return dict(status='COMPLETE', path=str(part_root), counts=manifest['counts'])


def build(args):
    require_dispatch(); root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'COMPLETE.json').exists():
            print(json.dumps(dict(status='REUSED_COMPLETE', manifest=read_manifest(root)['sha256']))); return 0
        bank = CandidateBank(args.bank)
        try:
            identity = identity_for(bank, Path.cwd()); scenes = list(bank.manifest['scene_ids'])
        finally:
            bank.close()
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]
        if not 1 <= args.workers <= len(scenes) or args.candidate_chunk < 1:
            raise ValueError('Invalid evidence worker/scene/chunk configuration')
        jobs = [(str(root/'parts'/f'rank{i:05d}'), scenes[i::args.workers], args.resume, args.candidate_chunk)
                for i in range(args.workers)]
        plan = dict(schema=SCHEMA, identity=identity, scenes=scenes, workers=args.workers,
                    chunk=args.candidate_chunk, full_TRAIN=not bool(args.max_scenes))
        p = root/'PLAN.json'
        if p.exists():
            if not args.resume or json.loads(p.read_text()) != plan:
                raise ValueError('Explicit resume requires the exact registered evidence plan')
        else:
            atomic_json(p, plan)
        invocation = root/'invocations'/str(time.time_ns()); invocation.mkdir(parents=True)
        atomic_json(invocation/'OWNER.json', dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            started_unix=time.time(), plan_sha256=fingerprint(plan), source_files=identity['code_files']))
        stop_file = str(Path(args.stop_file) if args.stop_file else root/'STOP_REQUESTED')
        def request_stop(signum, _frame):
            atomic_json(stop_file, dict(reason='Owned CPU evidence stop requested', signal=signum, at=time.time()))
        for signum in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            signal.signal(signum, request_stop)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[key] = '1'  # inherited before spawned workers import NumPy
        try:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize_worker, initargs=(args.bank, str(Path.cwd()), identity, stop_file)) as pool:
                pending = {pool.submit(build_part, job) for job in jobs};results=[]
                while pending:
                    done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                    for future in done:
                        try:
                            results.append(future.result())
                        except BaseException:
                            atomic_json(stop_file, dict(reason='A native evidence part failed; other parts stop at scene boundary',at=time.time()))
                            raise
                    atomic_json(root/'STATUS.json', dict(status='RUNNING', at=time.time(),
                        completed_parts=len(results), parts=args.workers,
                        progress=[json.loads(p.read_text()) for p in sorted(root.glob('parts/*/PROGRESS.json'))]))
            if any(r['status'] == 'PAUSED' for r in results):
                atomic_json(invocation/'RESULT.json', dict(status='PAUSED', parts=results));return 75
            manifest = merge_evidence(root, [job[0] for job in jobs], expected_scenes=scenes)
            atomic_json(invocation/'RESULT.json', dict(status='COMPLETE', bank_hash=manifest['sha256'], counts=manifest['counts']))
            atomic_json(root/'STATUS.json', dict(status='COMPLETE', bank_hash=manifest['sha256'], counts=manifest['counts']))
            print(json.dumps(dict(status='COMPLETE', bank_hash=manifest['sha256'], counts=manifest['counts'])));return 0
        except BaseException:
            atomic_json(invocation/'RESULT.json', dict(status='FAILED', error=traceback.format_exc()))
            atomic_json(root/'FAILURE_'+invocation.name+'.json', dict(status='FAILED', error=traceback.format_exc()))
            raise


def validate(args):
    require_dispatch()
    root = Path(args.output);root.mkdir(parents=True, exist_ok=True)
    if (root/'COMPLETE.json').exists():
        raise FileExistsError('Qualification output is immutable; use a new output')
    bank = CandidateBank(args.bank)
    identity = identity_for(bank, Path.cwd())
    # Two fixed scenes x16 candidates=32; no choosing based on observed rewards.
    scenes = list(bank.manifest['scene_ids'])[:2]
    initialize_worker(args.bank, str(Path.cwd()), identity, None)
    from scripts.analysis.constrained_flow_noise.batched_canonical_score import score_scene
    results=[];max_metric_error=max_state_error=0.;candidate_scores=0;native_batches=0
    for token in scenes:
        records = bank.records(token);requests=np.asarray([r.trajectory_fp32 for r in records], dtype=np.float32)
        job=(token, records[0].source_log, requests)
        canonical = v1_reward.score_scene(job);candidate_scores += len(records);native_batches += len(records)
        outputs=[]
        for chunk, order in [(1,np.arange(len(records))), (4,np.arange(len(records))[::-1]), (16,np.arange(len(records)))]:
            got = score_scene((token,records[0].source_log,requests[order]),candidate_chunk=chunk)
            candidate_scores += len(records);native_batches += math.ceil(len(records)/chunk)
            inverse=np.argsort(order);got=[got[i]for i in inverse]
            for r,a,b in zip(records,canonical,got):
                ea=make_evidence_record(r,a,source_bank_hash=bank.bank_hash,evaluator_identity=identity['evaluator_identity'])
                eb=make_evidence_record(r,b,source_bank_hash=bank.bank_hash,evaluator_identity=identity['evaluator_identity'])
                for key in a['metrics']:
                    error=abs(a['metrics'][key]-b['metrics'][key]);max_metric_error=max(max_metric_error,error)
                    if error>5e-12:raise AssertionError('Batch/chunk/order changed native '+key)
                err=float(np.max(np.abs(a['simulated_states']-b['simulated_states'])));max_state_error=max(max_state_error,err)
                if err>1e-10 or ea['event_class']!=eb['event_class'] or ea['event_first_seconds']!=eb['event_first_seconds']:
                    raise AssertionError('Native executed states or first events changed under batching')
            outputs=got
        values=[make_evidence_record(r,f,source_bank_hash=bank.bank_hash,evaluator_identity=identity['evaluator_identity'])for r,f in zip(records,outputs)]
        results.extend(values)
    bank.close()
    atomic_json(root/'native_records.json',results)
    receipt=dict(status='PASS',qualification_only=True,scenes=len(scenes),unique_candidates=len(results),
        source_bank_hash=identity['source_bank_hash'],identity=identity,
        comparisons=['native one-candidate pairs','chunk1','chunk4 reversed','chunk16'],
        maximum_metric_error=max_metric_error,maximum_simulated_state_error=max_state_error,
        candidate_scoring_attempts=candidate_scores,simulator_batches=native_batches,
        single_candidate_EP_contract_preserved=True,events_from_native_arrays=True)
    atomic_json(root/'COMPLETE.json',receipt);print(json.dumps(receipt));return 0


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='mode',required=True)
    for mode in ['validate','build','finalize']:
        q=sub.add_parser(mode);q.add_argument('--bank',required=True);q.add_argument('--output',required=True)
        if mode=='build':
            q.add_argument('--workers',type=int,default=8);q.add_argument('--candidate-chunk',type=int,default=16)
            q.add_argument('--max-scenes',type=int,default=0);q.add_argument('--resume',action='store_true');q.add_argument('--stop-file')
    a=p.parse_args()
    if a.mode=='validate':return validate(a)
    if a.mode=='build':return build(a)
    require_dispatch();b=CandidateBank(a.bank)
    try:
        plan=json.loads((Path(a.output)/'PLAN.json').read_text())
        if b.bank_hash != plan['identity']['source_bank_hash']:raise ValueError('Finalize fixed-bank mismatch')
        m=merge_evidence(a.output,sorted((Path(a.output)/'parts').glob('rank*')),expected_scenes=plan['scenes'])
        print(json.dumps(dict(status='COMPLETE',bank_hash=m['sha256'],counts=m['counts'])));return 0
    finally:b.close()


if __name__=='__main__':
    raise SystemExit(main())
