"""Bounded native-trainer replay from a complete checkpoint after an abort.

Reuse already scored, unchanged chains from the SAME historical behavior
sequence. Verify every optimizer-step numerical log against the original and
periodically regenerate an identical chain. This is recovery evidence, not an
additional formal experiment. Source training/scoring/optimizer code is intact.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.distributed as dist

from starVLA.rl.constrained_flow_noise.contracts import file_sha, require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.constrained_flow_noise import train


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source-run', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--target-update', type=int, required=True)
    a = p.parse_args()
    host = require_dispatch()
    source, checkpoint, output = map(Path, (a.source_run, a.checkpoint, a.output))
    boundary = json.loads((checkpoint/'boundary.json').read_text())
    identity = json.loads((checkpoint/'identity.json').read_text())
    rank = int(os.environ['RANK'])
    if not (checkpoint/'COMPLETE').exists() or boundary['phase'] != 'ready':
        raise ValueError('A complete collection-boundary checkpoint is required')
    if identity['variant'] != 'C_SDE' or a.target_update != 1850 or boundary['update'] != 1024:
        raise ValueError('This recovery is bounded to the recorded C_SDE failure')
    if output.resolve() == source.resolve():
        raise ValueError('Preserve the entire failed attempt in its original directory')
    expected = {}
    for line in (source/f'training_rank{rank}.jsonl').read_text().splitlines():
        row = json.loads(line)
        if boundary['update'] < row['update'] <= a.target_update:
            if row['update'] in expected: raise ValueError('Ambiguous historical update')
            expected[row['update']] = row
    if set(expected) != set(range(boundary['update']+1, a.target_update+1)):
        raise ValueError('Missing original optimizer-step evidence')
    startup = json.loads(source.with_suffix('.launcher.json').read_text())['args']
    index = startup.index('scripts.analysis.constrained_flow_noise.train')
    native_args = startup[index+1:]
    native_args[native_args.index('--output')+1] = str(output)
    native_args.extend(['--resume', str(checkpoint)])
    record = dict(schema='cfn_bounded_replay_recovery_v1', status='INITIALIZING',
                  source_run=str(source), checkpoint=str(checkpoint), target_update=a.target_update,
                  host=host, rank=rank, historical_identity=identity['sha256'],
                  adapter_sha256=file_sha(__file__), actual_new_score_calls=0,
                  verified_updates=0, last_verified_update=boundary['update'],
                  chain_checks=[], replayed_collections=0, replayed_candidate_scores=0,
                  scope='Exact historical recovery verification, not new formal exposure',
                  no_formal_continuation_until_verified=True)
    progress = output/f'recovery_rank{rank}.json'
    cached_collection, archive = None, None
    started = time.monotonic()
    def replay_collect(behavior, reference, store, service, stream, pop, collector,
                       raw_start, count, calibration, local_rank, world, **kwargs):
        nonlocal cached_collection, archive
        collection = collector.behavior_identity['collection']
        if collection != cached_collection:
            path = source/'behavior_collections'/f'{collection:06d}.pt'
            seal = json.loads(path.with_suffix('.json').read_text())
            if file_sha(path) != seal['sha256']:
                raise ValueError('Changed historical behavior archive')
            archive = torch.load(path, map_location='cpu', weights_only=False)
            if (archive['behavior_identity'] != collector.behavior_identity or
                    not torch.equal(archive['lambda_snapshot'], collector.lambda_snapshot)):
                raise ValueError('Replay behavior or lambda is no longer historical')
            cached_collection = collection
            record['replayed_collections'] += 1
            record['replayed_candidate_scores'] += len(archive['rows'])*16
        offset = len(collector.rows)
        rows = archive['rows'][offset:offset+count]
        if len(rows) != count or any(r['position'] != raw_start+i or r['token'] != stream.at(raw_start+i)
                                     for i,r in enumerate(rows)):
            raise ValueError('Recovery changed the original data cursor')
        if offset == 0 and (collection % 64 == 0 or collection == 924):
            # Same current weights / observations / root and innovation RNG;
            # regenerate only as a consistency check, never rescore or replace.
            row = rows[local_rank]
            actual = behavior.sample_chain(row['chain']['raw'].cuda(), [row['token']],
                                           [row['position']], version=collector.behavior_identity)
            for key in ('chain', 'old_mean', 'old_std', 'old_elementwise_logp'):
                if not torch.equal(actual[key].cpu(), row['chain'][key]):
                    raise ValueError('Historical actor replay diverged at '+key)
            record['chain_checks'].append(dict(collection=collection,position=row['position'],exact=True))
        record.update(status='REPLAYING_VERIFIED_HISTORICAL_CHAIN',wall_seconds=time.monotonic()-started)
        atomic_json(progress, record)
        return [(offset+i,row) for i,row in enumerate(rows) if (offset+i)%world == local_rank]

    # Verify after each original per-rank log was written and before proceeding.
    # The trace only watches this native main frame; it never edits gradients,
    # parameters, optimizer state, reward, advantages, or random generators.
    trigger = next(i for i,line in enumerate(Path(train.__file__).read_text().splitlines(),1)
                   if "if rank==0:print(json.dumps(log,allow_nan=False),flush=True)" in line)
    fields = ('PG', 'reference_mean_KL', 'reference_variance_KL', 'reference_KL_weighted',
              'GT_replay', 'loss', 'ratio_mean', 'ratio_max_abs_from_one', 'clip_fraction',
              'joint_transition_logratio_abs_max', 'sigma_mean', 'gradient_tensors',
              'pre_clip_grad_norm', 'clip_scale', 'collection', 'inner', 'raw_exposure',
              'accepted_exposure', 'raw_groups', 'accepted_groups', 'LR', 'KL_coefficient',
              'lambda_snapshot', 'dual')
    def trace(frame, event, argument):
        if frame.f_code is not train.main.__code__:
            return None
        if event == 'line' and frame.f_lineno == trigger:
            current = frame.f_locals['log']; update = current['update']
            old = expected[update]
            mismatches = {k:dict(original=old[k],replayed=current[k])
                          for k in fields if old[k] != current[k]}
            failed = torch.tensor(bool(mismatches),device=frame.f_locals['accelerator'].device,dtype=torch.int32)
            dist.all_reduce(failed,op=dist.ReduceOp.MAX)
            if int(failed):
                record.update(status='REPLAY_DIVERGED_NATIVE_STOP_REQUESTED',divergence_update=update,
                              local_mismatches=mismatches)
                train.signal_stop()
            else:
                record['verified_updates'] += 1
                record['last_verified_update'] = update
                if update == a.target_update:
                    record['status'] = 'TARGET_VERIFIED_NATIVE_STOP_REQUESTED'
                    train.signal_stop()
            atomic_json(progress,record)
        return trace

    train.collect_batch = replay_collect
    sys.argv = [train.__file__, *native_args]
    sys.settrace(trace)
    try:
        train.main()
    except BaseException:
        record.update(status='RECOVERY_ABORTED',wall_seconds=time.monotonic()-started)
        atomic_json(progress,record)
        raise
    finally:
        sys.settrace(None)
    result = json.loads((output/f'result_rank{rank}.json').read_text())
    record.update(status=('RECOVERED_EXACT_OBSERVED_SEQUENCE' if record['last_verified_update']==a.target_update
                          else 'REPLAY_DIVERGED_SAVED_FOR_DIAGNOSIS'),
                  result_checkpoint=result['checkpoint'],wall_seconds=time.monotonic()-started,
                  original_historical_timing_fields_reused=True,
                  extra_regenerated_candidates=len(record['chain_checks'])*16)
    atomic_json(progress,record)


if __name__ == '__main__': main()
