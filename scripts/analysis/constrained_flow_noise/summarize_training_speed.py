"""Read-only all-rank completion and observed raw-exposure ETA, without torch."""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import time


def summarize(run,total):
    logs=[]
    for rank in range(8):
        with (run/f'training_rank{rank}.jsonl').open() as stream:
            logs.append({x['update']:x for line in stream if line.endswith('\n') for x in [json.loads(line)]})
    common=min(max(x) for x in logs)
    if logs[0][common]['inner']!=2:common-=1
    if not all(common in x for x in logs):raise ValueError('No common complete inner2 update')
    row=logs[0][common];end=row['collection']
    earliest=min(int(p.stem) for p in (run/'behavior_collections').glob('*.pt'))
    result=dict(run=str(run),all_rank_complete_update=common,collection=end,raw_exposure=row['raw_exposure'],
                accepted_exposure=row['accepted_exposure'],raw_budget=total,remaining_raw=total-row['raw_exposure'],
                completion_fraction=row['raw_exposure']/total,
                max_rank_log_age_seconds=max(time.time()-(run/f'training_rank{i}.jsonl').stat().st_mtime for i in range(8)),windows=[])
    for count in (16,32,64):
        start=end-count+1
        if start-1<earliest:continue
        before=run/'behavior_collections'/f'{start-1:06d}.pt';after=run/'behavior_collections'/f'{end:06d}.pt'
        wall=after.stat().st_mtime-before.stat().st_mtime
        if wall<=0:raise ValueError('Nonpositive archive interval')
        summaries=[json.loads((run/'collection_summaries'/f'{i:06d}.json').read_text()) for i in range(start,end+1)]
        raw=sum(x['scenes'] for x in summaries);candidates=sum(x['candidates'] for x in summaries)
        update_seconds=[max(x[u]['seconds'] for x in logs) for i in range(start,end+1) for u in (i*2+1,i*2+2)]
        result['windows'].append(dict(collections=count,first_collection=start,last_collection=end,raw_scenes=raw,
            candidate_scores=candidates,wall_seconds=wall,raw_scenes_per_second=raw/wall,
            candidate_scores_per_second=candidates/wall,median_critical_rank_update_seconds=statistics.median(update_seconds),
            projected_remaining_training_hours=(total-row['raw_exposure'])/(raw/wall)/3600))
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',action='append',required=True);p.add_argument('--budget',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    budget=json.loads(Path(a.budget).read_text())
    record=dict(schema='cfn_live_raw_exposure_speed_v1',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        runs=[summarize(Path(path),budget['raw_exposures_per_variant']) for path in a.run],
        limitations='Read-only throughput, not a matched speed guarantee. Uses archive-to-archive wall intervals, not summed overlapping timers. Intervals contain preceding updates and subsequent collection; boundary work differs by one collection. Future checkpoint/evaluation costs and safety/numerical pauses are not predicted by the pure-training ETA. Process liveness must be inspected separately; old result files are not a live status source.')
    with out.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
