"""Audit completed SFT exposure and observed cost without loading GPU models.

Overlapping update journals are rejected, never silently deduplicated. If a
future rollback creates overlap, an explicit ancestry audit is required first.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
from starVLA.rl.fm_transfer.contracts import ROOT,require_dispatch
from starVLA.rl.simwam_recipe.contracts import POPULATION,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.fm_transfer.dispatch_screen import completed_sft


def scan_metrics(paths, *, rank, updates, per_rank_batch, global_batch, allowed_tokens):
    seen=set();tokens=set();data=[];update_time=[];journals=[]
    for path in paths:
        count=0
        with Path(path).open() as stream:
            for line in stream:
                row=json.loads(line);u=row['updates']
                if (row['rank']!=rank or u in seen or not 1<=u<=updates or
                        row['cursor']!=u*global_batch or len(row['scene_tokens'])!=per_rank_batch):
                    raise ValueError('Duplicate/out-of-bound/misassigned SFT update; resolve ancestry explicitly')
                if not set(row['scene_tokens'])<=allowed_tokens:raise ValueError('Non-TRAIN token in SFT update')
                for key in ('loss','seconds_data','seconds_update','wall_seconds_this_invocation'):
                    if not math.isfinite(row[key]):raise ValueError('Nonfinite training log')
                if row['seconds_data']<0 or row['seconds_update']<0:raise ValueError('Negative duration')
                seen.add(u);tokens.update(row['scene_tokens']);count+=1
                data.append(row['seconds_data']);update_time.append(row['seconds_update'])
        journals.append(dict(path=str(path),rows=count,sha256=file_sha(path)))
    if seen!=set(range(1,updates+1)):raise ValueError('Incomplete logical optimizer update sequence')
    return dict(updates=updates,rank_scene_exposures=updates*per_rank_batch,
        data_seconds_sum=sum(data),update_seconds_sum=sum(update_time),
        data_seconds_median=statistics.median(data),update_seconds_median=statistics.median(update_time),
        journals=journals),tokens


def summarize(run):
    run=Path(run);identity=json.loads((run/'identity.json').read_text());cfg=identity['config']
    registration=json.loads((ROOT/'experiment_registration.json').read_text())
    if cfg!=registration['SFT'][cfg['variant']]:raise ValueError('SFT configuration differs from registration')
    checkpoint=run/'checkpoints'/f"update{cfg['max_updates']:06d}"
    parent,files=completed_sft(checkpoint)
    if parent!=identity:raise ValueError('Endpoint does not belong to this run')
    pop=json.loads(Path(POPULATION).read_text());train=set(pop['train_tokens'])
    if train&(set(pop['dev_tokens'])|set(pop['navtest_tokens'])):raise ValueError('Population split leakage')
    ranks=[];all_tokens=set();invocations=[]
    for rank in range(cfg['world']):
        finish=json.loads((run/f'result_rank{rank}.json').read_text())
        if (finish['status']!='COMPLETE' or finish['end_update']!=cfg['max_updates'] or finish['qualification'] or
                finish['rank']!=rank or finish['identity']!=identity['sha256'] or
                finish['cursor']!=cfg['max_updates']*cfg['global_batch']):
            raise ValueError('Requires all formal SFT ranks to finish')
        result,tokens=scan_metrics(sorted(run.glob(f'invocations/*/metrics_rank{rank}.jsonl')),
            rank=rank,updates=cfg['max_updates'],per_rank_batch=cfg['per_rank_batch'],
            global_batch=cfg['global_batch'],allowed_tokens=train)
        records=[]
        for path in sorted(run.glob(f'invocations/*/rank{rank}.json')):
            record=json.loads(path.read_text())
            if record['rank']!=rank or record['identity']!=identity['sha256']:
                raise ValueError('Foreign invocation identity/rank')
            records.append(dict(record,invocation=path.parent.name))
        if any(r['status'] not in ('COMPLETE','STOPPED_AT_SAVED_BOUNDARY','FAILED') or 'wall_seconds' not in r for r in records):
            raise ValueError('An invocation has no terminal cost record; do not erase missing wall time')
        for r in records:
            invocations.append({k:r[k] for k in ('rank','invocation','status','start_update','end_update',
                'updates_this_invocation','wall_seconds','start_unix') if k in r})
        result.update(rank=rank,peak_GPU_allocated_bytes=finish['peak_GPU_bytes'],
            cumulative_training_worker_wall_seconds=sum(r['wall_seconds'] for r in records))
        ranks.append(result);all_tokens.update(tokens)
    if cfg['world']*cfg['per_rank_batch']!=cfg['global_batch']:raise ValueError('Changed scene reduction')
    dispatches=[json.loads(p.read_text()) for p in sorted(run.glob('dispatches/*.json'))]
    earliest=min(d['start_unix'] for d in dispatches)
    # The trainer records start after initialization; its measured loop wall
    # includes checkpointing, but not the earlier model/optimizer setup.
    last_end=max(r['start_unix']+r['wall_seconds'] for r in invocations)
    nonloop={r['rank']:r['cumulative_training_worker_wall_seconds']-r['data_seconds_sum']-r['update_seconds_sum'] for r in ranks}
    rank0=[r for r in invocations if r['rank']==0]
    return dict(schema='fm_transfer_completed_SFT_budget_v1',status='COMPLETE_UNAMBIGUOUS_LOGICAL_EXPOSURE',
        run=str(run),identity=identity['sha256'],variant=cfg['variant'],source_sha256=file_sha(__file__),
        endpoint=str(checkpoint),endpoint_files_sha256=file_sha(checkpoint/'files.json'),
        logical_optimizer_updates=cfg['max_updates'],logical_scene_exposures=cfg['max_updates']*cfg['global_batch'],
        unique_TRAIN_scenes=len(all_tokens),TRAIN_population=len(train),no_DEV_or_Navtest=True,
        FM_noise_repeats=cfg['FM_repeats'],FM_interpolated_training_examples=cfg['max_updates']*cfg['global_batch']*cfg['FM_repeats'],
        RL_candidates_or_reward_calls=0,shared_GPU_time_is_not_exclusive_compute=True,
        cumulative_training_worker_wall_seconds=sum(r['cumulative_training_worker_wall_seconds'] for r in ranks),
        rank0_training_wall_seconds=sum(r['wall_seconds'] for r in rank0),
        elapsed_dispatch_to_final_training_boundary_seconds_approx=last_end-earliest,
        initial_dispatch_to_rank0_start_seconds_approx=min(r['start_unix'] for r in rank0)-earliest,
        non_data_update_wall_by_rank=nonloop,
        non_data_update_note='Includes checkpoint writes/hashing, synchronization and logging; not isolated checkpoint time.',
        cost_limitations=['Per-save durations were overwritten in progress.json; do not reconstruct exact checkpoint seconds.',
            'Model loading is included with rendezvous/setup before recorded loop start; not separately instrumented.',
            'Final process teardown after the last recorded boundary is not measured.',
            'These are SFT costs only; screening, evaluation, qualification and RL have separate ledgers.'],
        ranks=ranks,invocations=invocations)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    require_dispatch();output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve previous cost summaries')
    result=summarize(a.run);output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result)
    print(json.dumps({k:result[k] for k in ('status','variant','logical_optimizer_updates','logical_scene_exposures',
        'unique_TRAIN_scenes','rank0_training_wall_seconds')}))


if __name__=='__main__':main()
