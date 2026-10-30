"""Finite CPU analysis of already-written matched IL journals; no model calls."""
import argparse
import json
from pathlib import Path
import statistics
import time
from .evidence import FOUR, fingerprint
from .render import atomic


def diagnose(root=FOUR, window=500, endpoint=5000):
    gathered={};sources=[]
    for a in 'ABCD':
        ranks=[]
        for rank in range(8):
            path=Path(root)/f'runs/ARCH_{a}/IL_seed42/metrics_rank{rank}.jsonl'
            rows={}
            with path.open()as stream:
                for line in stream:
                    try:row=json.loads(line)
                    except ValueError:continue
                    step=row['updates']
                    if step>endpoint:break
                    if step>1000 and step<=1000+window or step>endpoint-window:
                        if step in rows:raise ValueError('Duplicate valid update requires lineage audit: '+str(path))
                        rows[step]=row
            ranks.append(rows)
            sources.append(dict(path=str(path),through_update=endpoint,mtime=path.stat().st_mtime))
        gathered[a]=ranks
    common=set.intersection(*(set(rank)for ranks in gathered.values()for rank in ranks))
    windows=[range(1001,1001+window),range(endpoint-window+1,endpoint+1)]
    output=[]
    for a,ranks in gathered.items():
        for steps in windows:
            selected=[s for s in steps if s in common]
            if len(selected)!=window:raise ValueError('Matched window incomplete')
            losses={};seconds=[];data=[];grad=[]
            for step in selected:
                batch=[rank[step]for rank in ranks]
                for row in batch:
                    assert row['optimizer_global_steps']==step and not row['optimizer_skipped_steps']
                    for k,v in row['losses'].items():losses.setdefault(k,[]).append(v)
                seconds.append(max(r.get('slowest_rank_seconds',r['seconds'])for r in batch))
                data.append(max(r['data_seconds']for r in batch))
                grad.append(statistics.mean(r['grad_norm']for r in batch))
            output.append(dict(architecture=a,start=selected[0],end=selected[-1],updates=len(selected),
                scene_exposures=len(selected)*32,mean_losses={k:statistics.mean(v)for k,v in losses.items()},
                mean_slowest_rank_seconds=statistics.mean(seconds),mean_data_wait_seconds=statistics.mean(data),
                mean_total_gradient_norm=statistics.mean(grad),total_gradient_not_component_attribution=True))
    mismatches=[]
    for step in sorted(common):
        batches={a:[token for rank in ranks for micro in rank[step]['sample_ids']for token in micro]for a,ranks in gathered.items()}
        if any(batches[a]!=batches['A']for a in'BCD'):mismatches.append(step)
    result=dict(schema='qwen_world_four_matched_IL_journal_diagnostic_v1',status='COMPLETE',at=time.time(),
        endpoint=endpoint,window=window,rows=output,matched_update_count=len(common),
        global_scene_order_mismatch_updates=mismatches,full_32_scene_order_matches=not mismatches,
        sources=sources,scope=dict(new_GPU_forwards=0,new_updates=0,new_scores=0,
            inference='Training noisy reconstruction is not ODE deployment; future ceiling cannot be inferred',
            timing='Historical shared-device windows, not isolated architecture performance benchmark'))
    result['sha256']=fingerprint(result);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    result=diagnose();atomic(Path(a.output),result)
    print(json.dumps(dict(rows=result['rows'],scene_order_matches=result['full_32_scene_order_matches']),ensure_ascii=False))


if __name__=='__main__':main()
