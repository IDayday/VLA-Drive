"""Analysis-only accounting. Logical samples and physical attempts are distinct."""
import json
from pathlib import Path


def invocation_cost(run, *, done, latest_wall=None):
    rows=[json.loads(p.read_text()) for p in sorted((Path(run)/'invocations').glob('*/rank*.json'))]
    if not rows:
        return dict(schema=2,provenance='legacy rank summaries; no invocation ledger',
            cumulative_logical_updates=done,invocation_start_update=None,
            invocation_effective_updates=None,invocation_updates_per_second=None,
            linear_remaining_training_hours=None,cumulative_worker_wall_seconds=None,
            rollback_or_failure_worker_wall_seconds=None,save_worker_wall_seconds=None,
            last_reported_invocation_wall_seconds=latest_wall,
            missing=['invocation start counters','failed/rolled-back time','model loading','checkpoint save time'],
            note='Do not divide cumulative updates by a possibly resumed invocation wall. Legacy phase timers remain valid lower bounds.')
    rank0=sorted((r for r in rows if r['rank']==0),key=lambda r:r['start_unix'])
    last=rank0[-1];delta=last.get('end_update',last['start_update'])-last['start_update']
    wall=last.get('worker_wall_seconds',last.get('observed_worker_wall_seconds',0))
    rate=delta/wall if delta>=4 and wall>0 and 'end_update' in last else None
    # Sum worker time explicitly, never relabel it as end-to-end/exclusive GPU time.
    return dict(schema=2,provenance='immutable per-invocation ledger',cumulative_logical_updates=done,
        invocation_start_update=last['start_update'],invocation_effective_updates=delta,
        invocation_updates_per_second=rate,
        linear_remaining_training_hours=(20000-done)/rate/3600 if rate else None,
        cumulative_worker_wall_seconds=sum(r.get('worker_wall_seconds',r.get('observed_worker_wall_seconds',0)) for r in rows),
        rollback_or_failure_worker_wall_seconds=sum(r.get('worker_wall_seconds',r.get('observed_worker_wall_seconds',0)) for r in rows if r['status'] in ('ABORTED','RUNNING')),
        save_worker_wall_seconds=sum(r['seconds_save'] for r in rows),
        physical_generated_candidates=sum(r['generated_candidates'] for r in rows),
        physical_scored_candidates=sum(r['scored_candidates'] for r in rows),
        worker_records=len(rows),incomplete_invocations=sum(r['status']=='RUNNING' for r in rows),
        missing=['unobserved tail after a hard process kill; counters are lower bounds when an invocation is incomplete'],
        note='Shared GPU worker wall is not exclusive GPU compute; failed attempt work is included. A resumed segment rate uses end-start updates.')


def evaluation_cost(root,identity,ranks):
    rows=[json.loads(p.read_text()) for p in sorted((Path(root)/'invocations').glob('*/rank*.json'))]
    base=dict(schema=2,logical_scenes=identity['scenes'],logical_single_candidate_evaluations=identity['scenes']*len(identity['modes']),
        candidate_evaluations_per_grid=identity['scenes'],shared_GPU=True,
        note='Two separately reported single-candidate grids. Summed worker time is not job wall or exclusive GPU compute.')
    if not rows:
        base.update(provenance='legacy last worker summary; prior invocations may have been overwritten',
            last_invocation_new_scenes=sum(r['new'] for r in ranks),last_invocation_reused_scenes=sum(r['reused'] for r in ranks),
            last_invocation_new_predictions=sum(r['new_candidates'] for r in ranks),
            observed_worker_wall_seconds=sum(r['elapsed'] for r in ranks),
            generation_wall_seconds_lower_bound=sum(r['generation_seconds'] for r in ranks),
            scoring_wait_seconds_lower_bound=sum(r['scoring_seconds'] for r in ranks),
            cumulative_physical_predictions=None,cumulative_worker_wall_seconds=None,job_wall_seconds=None,model_load_seconds=None,
            missing=['previous worker invocations','inference retries','model loading','job start/end'],
            complete_cost_accounting=False)
        return base
    finished=[r for r in rows if 'end_unix' in r]
    complete=len(finished)==len(rows)
    base.update(provenance='per-worker invocation ledger',worker_invocations=len(rows),complete_cost_accounting=complete,
        new_scenes=sum(r.get('new',0) for r in rows),reused_scenes=sum(r.get('reused',0) for r in rows),
        cumulative_physical_predictions=sum(r.get('physical_predictions',0) for r in rows),
        cumulative_worker_wall_seconds=sum(r.get('total_worker_wall_seconds',r.get('elapsed',0)) for r in rows),
        job_wall_seconds=max(r['end_unix'] for r in finished)-min(r['start_unix'] for r in rows) if complete else None,
        model_load_seconds=sum(r.get('model_load_seconds',0) for r in rows),
        generation_wall_seconds_sum=sum(r.get('generation_seconds',0) for r in rows),
        scoring_wait_seconds_sum=sum(r.get('scoring_seconds',0) for r in rows),
        missing=[] if complete else ['unobserved tail of interrupted worker; physical counts are lower bounds'])
    return base
