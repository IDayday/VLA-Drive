"""Read sealed real collections to distinguish worker sums and wall bottlenecks."""
import argparse
import json
from pathlib import Path
import statistics

import torch

from starVLA.rl.constrained_flow_noise.contracts import file_sha, require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json


def profile(run, count):
    run = Path(run)
    logs = []
    for rank in range(8):
        with (run/f'training_rank{rank}.jsonl').open() as f:
            logs.append({r['update']: r for line in f if line.endswith('\n')
                         for r in [json.loads(line)]})
    last = min(max(x) for x in logs)
    complete = last//2
    indices = range(max(1, complete-count), complete)
    records = []
    for index in indices:
        path = run/'behavior_collections'/f'{index:06d}.pt'
        seal = json.loads(path.with_suffix('.json').read_text())
        if file_sha(path) != seal['sha256']: raise ValueError('Changed archive')
        batch = torch.load(path, map_location='cpu', weights_only=False)
        rows = batch['rows']
        if any('collection_execution' in row for row in rows):
            raise ValueError('Serial component decomposition cannot sum overlapping timers; use compare_execution_throughput')
        critical = dict(rollout_seconds=0., scoring_seconds=0., reference_seconds=0.)
        stages, pipeline_floor = [], 0.
        # Original collect_batch processes rank-owned rows sequentially. First
        # batch16, then refill8: every call ends in a collective barrier.
        for begin, end in [(0,min(16,len(rows))), *[(i,min(i+8,len(rows))) for i in range(16,len(rows),8)]]:
            ranks = []
            for rank in range(8):
                owned = [rows[i] for i in range(begin,end) if i%8==rank]
                values = {k:sum(r[k] for r in owned) for k in critical}
                ranks.append(values)
            slowest = max(ranks,key=lambda x:sum(x.values()))
            for k in critical: critical[k] += slowest[k]
            stages.append(dict(start=begin, stop=end, critical_rank=ranks.index(slowest), measured=slowest))
            # Idealized two-worker score overlap; ignores dispatch overhead.
            # This is a bound/estimate only, never a measured improvement.
            rank_finish=[]
            for rank in range(8):
                gpu=0.;cpu=[0.,0.]
                for i in range(begin,end):
                    if i%8!=rank:continue
                    r=rows[i];gpu+=r['rollout_seconds']
                    worker=min(range(2),key=lambda j:cpu[j])
                    cpu[worker]=max(cpu[worker],gpu)+r['scoring_seconds']
                    gpu+=r['reference_seconds']
                rank_finish.append(max([gpu,*cpu]))
            pipeline_floor += max(rank_finish)
        updates = sum(max(logs[r][u]['seconds'] for r in range(8))
                      for u in (index*2+1,index*2+2))
        previous = run/'behavior_collections'/f'{index-1:06d}.pt'
        archive_interval = path.stat().st_mtime-previous.stat().st_mtime
        previous_updates = sum(max(logs[r][u]['seconds'] for r in range(8))
                               for u in (index*2-1,index*2))
        worker={k:sum(row[k] for row in rows) for k in critical}
        canonical=sum(x['scoring_seconds'] for row in rows for x in row['records'])
        geometry=sum(x['geometry_seconds'] for row in rows for x in row['records'])
        records.append(dict(collection=index,raw_groups=len(rows),accepted_groups=len(batch['accepted']),
                            candidates=16*len(rows),critical_path_components=critical,
                            measured_update_seconds=updates,worker_sum_seconds=worker,
                            canonical_candidate_seconds=canonical,geometry_candidate_seconds=geometry,
                            score_CPU_seconds=sum(row['scoring_cpu_seconds'] for row in rows),
                            archive_interval_seconds=archive_interval,
                            interval_minus_previous_updates_and_collection=archive_interval-previous_updates-sum(critical.values()),
                            ideal_overlap_collection_seconds=pipeline_floor,archive_bytes=path.stat().st_size))
    totals={key:sum(r['critical_path_components'][key] for r in records) for key in critical}
    totals['update_seconds']=sum(r['measured_update_seconds'] for r in records)
    wall=sum(r['archive_interval_seconds'] for r in records)
    return dict(run_identity=json.loads((run/'run_identity.json').read_text())['sha256'],
                last_all_rank_update=last,collections=len(records),raw_groups=sum(r['raw_groups'] for r in records),
                critical_components_seconds=totals,archive_intervals_wall_seconds=wall,
                median_collection_interval_seconds=statistics.median(r['archive_interval_seconds'] for r in records),
                canonical_seconds=sum(r['canonical_candidate_seconds'] for r in records),
                geometry_seconds=sum(r['geometry_candidate_seconds'] for r in records),
                worker_score_CPU_seconds=sum(r['score_CPU_seconds'] for r in records),
                ideal_overlap_collection_seconds=sum(r['ideal_overlap_collection_seconds'] for r in records),
                scope='Recent completed formal collections; wall inferred from atomic archive mtime, includes IO/waits; no new samples',
                limitations='Component timers omit surrounding IO/collectives; filesystem timestamps have coarse resolution. Worker sums are not wall time. Overlap is only a scheduling estimate.',
                records=records)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',required=True)
    p.add_argument('--collections',type=int,default=64);p.add_argument('--output',required=True)
    a=p.parse_args();require_dispatch()
    if Path(a.output).exists():raise FileExistsError(a.output)
    result={v:profile(Path(a.run_root)/f'formal_{v}_seed42_v1',a.collections)
            for v in ('C_SDE','N_FLOW_NOISE')}
    atomic_json(a.output,dict(schema='cfn_real_training_cost_v1',variants=result,source_sha256=file_sha(__file__)))
    print(json.dumps({v:{k:x[k] for k in ('collections','raw_groups','critical_components_seconds','archive_intervals_wall_seconds','canonical_seconds','geometry_seconds','worker_score_CPU_seconds','ideal_overlap_collection_seconds')} for v,x in result.items()}),flush=True)


if __name__=='__main__':main()
