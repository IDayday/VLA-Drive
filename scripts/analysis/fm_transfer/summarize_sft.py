"""Read-only SFT progress, ancestry-aware logical counts and physical run cost."""
import argparse
import json
from pathlib import Path
import time
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import POPULATION


def summarize(root):
    root=Path(root);identity=json.loads((root/'identity.json').read_text());cfg=identity['config']
    invocations=[]
    for record in root.glob('invocations/*/rank0.json'):
        meta=json.loads(record.read_text());journal=record.with_name('metrics_rank0.jsonl')
        lines=journal.read_text().splitlines(keepends=True) if journal.exists() else []
        rows=[json.loads(line) for line in lines if line.endswith('\n')]
        updates=[r['updates'] for r in rows]
        if updates!=list(range(meta['start_update']+1,meta['start_update']+1+len(rows))):
            raise ValueError('Non-contiguous updates inside invocation '+str(record))
        if meta['identity']!=identity['sha256']:raise ValueError('Mixed SFT identities')
        invocations.append((meta,rows,str(record)))
    invocations.sort(key=lambda x:x[0]['start_unix'])
    logical={};aborted=[];physical=0;wall=0.;segments=[]
    for meta,rows,path in invocations:
        start=meta['start_update']
        if logical and start>max(logical):raise ValueError('Resume ancestry has a missing update segment')
        if start and not meta['resume']:raise ValueError('Nonzero start without a native parent')
        if meta['resume']:
            seal=json.loads((Path(meta['resume'])/'COMPLETE').read_text())
            if seal['updates']!=start:raise ValueError('Resume does not start at its parent boundary')
        superseded=[u for u in logical if u>start]
        if superseded:
            aborted.append(dict(parent=meta['resume'],superseded_updates=[min(superseded),max(superseded)],
                count=len(superseded),reason='explicit native checkpoint rollback; original journals retained'))
            logical={u:r for u,r in logical.items() if u<=start}
        for row in rows:
            if row['updates'] in logical:raise ValueError('Duplicate logical update, no last-row deduplication allowed')
            logical[row['updates']]=row
        physical+=len(rows)
        elapsed=meta.get('wall_seconds',rows[-1]['wall_seconds_this_invocation'] if rows else 0.)
        wall+=elapsed;segments.append(dict(record=path,start_update=start,executed_updates=len(rows),
            wall_seconds=elapsed,status=meta['status'],resume=meta['resume']))
    last=invocations[-1] if invocations else None;rate=None;eta=None
    if last and len(last[1])>=101:
        recent=last[1][-201:];delta=recent[-1]['wall_seconds_this_invocation']-recent[0]['wall_seconds_this_invocation']
        if delta>0:rate=(len(recent)-1)/delta;eta=(cfg['max_updates']-recent[-1]['updates'])/rate
    done=max(logical,default=0)
    return dict(schema='fm_transfer_SFT_progress_v1',snapshot_unix=time.time(),run=str(root),
        identity=identity['sha256'],variant=cfg['variant'],registered_updates=cfg['max_updates'],
        logical_updates=done,logical_scene_exposures=done*cfg['global_batch'],
        physical_completed_updates=physical,rolled_back_completed_updates=physical-len(logical),
        original_scene_count=len(json.loads(Path(POPULATION).read_text())['train_tokens']),FM_repeats=cfg['FM_repeats'],RL_candidates=0,
        accumulated_recorded_wall_seconds=wall,
        eight_GPU_residency_hours=8*wall/3600.,GPU_cost_label='allocated/shared device residency; not measured exclusive GPU compute',
        recent_updates_per_second=rate,ETA_seconds_excluding_future_eval_and_unobserved_failures=eta,
        current_invocation_updates=len(last[1]) if last else 0,segments=segments,rollback_receipts=aborted,
        latest_loss=logical[done]['loss'] if done else None,
        loss_is_not_planning_score=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=summarize(a.root);atomic_json(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
