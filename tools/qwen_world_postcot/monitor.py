"""Bounded read-only journals: per-stage throughput, memory and numeric health."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np

def tail(path,maximum=500):
    with path.open('rb') as f:
        f.seek(0,2);end=f.tell();f.seek(max(0,end-4*1024**2));chunk=f.read()
    lines=chunk.splitlines()
    if end>len(chunk):lines=lines[1:]
    rows=[]
    selected=lines[-maximum:]
    for index,line in enumerate(selected):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(selected)-1:raise ValueError('Interior journal corruption: '+str(path))
            continue  # Only a currently appended tail.
    return rows

def snapshot(run):
    root=Path(run);journals=sorted(root.glob('metrics_rank*.jsonl'));ranks={}
    for path in journals:
        rows=tail(path)
        if rows:ranks[path.stem]=rows
    value=dict(schema='world_postcot_live_monitor_v1',observed_unix=time.time(),run=str(root.resolve()),
        ranks=len(ranks),status='WAITING_START' if not ranks else 'PROGRESS_OBSERVED',stages={},
        live_process_verified=False,activity_scope='Read-only journals; process liveness is verified separately by the job controller.')
    if not ranks:return value
    value['latest_update_by_rank']={k:v[-1]['updates'] for k,v in ranks.items()}
    value['oldest_journal_age_seconds']=max(time.time()-p.stat().st_mtime for p in journals)
    common=set.intersection(*({r['updates'] for r in rows} for rows in ranks.values()))
    latest_by_rank={k:{r['updates']:r for r in rows} for k,rows in ranks.items()}
    if not common:return dict(value,status='WAITING_COMMON_UPDATE')
    windows={}
    for step in sorted(common):
        rows=[v[step] for v in latest_by_rank.values()];phase=rows[0].get('phase','RL')
        seconds=max(float(r.get('slowest_rank_seconds',r.get('seconds',r.get('seconds_update',0.)))) for r in rows)
        losses=[v for r in rows for v in r.get('losses',r.get('metrics',{})).values() if isinstance(v,(int,float))]
        if any(not math.isfinite(v) for v in losses):value['status']='NONFINITE_REPORTED'
        windows.setdefault(phase,[]).append(dict(update=step,seconds=seconds,
            allocated=max(r.get('peak_allocated',r.get('peak_GPU_bytes',0)) for r in rows),
            reserved=max(r.get('peak_reserved',0) for r in rows)))
    for phase,rows in windows.items():
        # Keep warm phases separate; four first observed updates are excluded.
        selected=rows[4:] if len(rows)>8 else []
        if not selected:value['stages'][phase]=dict(status='WARMING',updates=len(rows));continue
        sec=[r['seconds'] for r in selected];mean=float(np.mean(sec))
        value['stages'][phase]=dict(samples=len(sec),first_update=selected[0]['update'],last_update=selected[-1]['update'],
            mean_seconds=mean,p50=float(np.quantile(sec,.5)),p95=float(np.quantile(sec,.95)),
            global_scenes_per_second=32/mean if phase in ('interface','joint') else None,
            peak_allocated=max(r['allocated'] for r in selected),peak_reserved=max(r['reserved'] for r in selected),
            timing_scope='normal IL data+update excluding checkpoints' if phase in ('interface','joint') else 'RL optimizer journal; fresh collection cost is separate')
    value['progress']=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else None
    if value['progress'] and value['progress']['status'] in ('COMPLETE','STOPPED','PAUSED','DIAGNOSTIC_COMPLETE'):
        value['status']=value['progress']['status']
    elif value['oldest_journal_age_seconds']>600 and value['status']=='PROGRESS_OBSERVED':value['status']='STALE_REQUIRES_INVESTIGATION'
    return value

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--loop',action='store_true');p.add_argument('--interval',type=int,default=60);a=p.parse_args()
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    while True:
        result=snapshot(a.run);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(path)
        if not a.loop:break
        time.sleep(max(10,a.interval))
if __name__=='__main__':main()
