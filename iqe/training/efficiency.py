"""Read-only monitoring of this task's own optimizer logs and resource budget."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from ..contracts import require
from ..io import read_json,atomic_json,file_hash


def training_efficiency(directory, warmup=3, maximum_seconds=None):
    directory=Path(directory)
    rows=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines() if line.strip()]
    require(len(rows)>warmup,'insufficient optimizer updates for a steady-state timing estimate')
    by_step={r['optimizer_step']:r for r in rows}
    rows=[by_step[k] for k in sorted(by_step)]
    # Timestamps are per optimizer step; repeated/resumed records are kept in the
    # raw log but do not count as additional updates in the throughput estimate.
    samples=[]
    for row in rows[warmup:]:
        if 'compute_update_seconds' in row:samples.append(row['compute_update_seconds'])
        elif 'seconds_per_update' in row:samples.append(row['seconds_per_update'])
    require(samples and all(np.isfinite(samples)),'invalid timing evidence')
    p50,p95=map(float,np.quantile(samples,[.5,.95]))
    return {'status':'PROFILED','updates':len(rows),'warmup_excluded':warmup,'p50_update_seconds':p50,'p95_update_seconds':p95,
        'max_allowed_seconds':maximum_seconds,'throughput_gate_passed':maximum_seconds is not None and p95<=maximum_seconds,
        'raw_log':str(directory/'steps.jsonl'),'log_hash':file_hash(directory/'steps.jsonl'),
        'automatic_semantic_changes':False,'optimization_policy':['cache only frozen full FeatureBundles','bound evaluation activations by independent scene chunks',
        'stream checkpoint writes','retain all losses, batch exposures, schedule, input resolution and trajectory contract'],
        'scientific_gain':'UNTESTED'}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--directory',required=True);p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--max-seconds-per-update',type=float,required=True);p.add_argument('--watch-seconds',type=int,default=0)
    a=p.parse_args();require(a.watch_seconds>=0 and a.max_seconds_per_update>0,'monitor budgets')
    from ..resources import qualify
    qualify('cpu')
    end=time.monotonic()+a.watch_seconds
    while True:
        try:
            result=training_efficiency(a.directory,a.warmup,a.max_seconds_per_update)
            atomic_json(Path(a.directory)/'EFFICIENCY.json',result)
            print(json.dumps(result),flush=True)
        except (FileNotFoundError,ValueError) as e:
            result={'status':'WAITING_FOR_OWN_TRAINING_EVIDENCE','error':str(e)}
            print(json.dumps(result),flush=True)
        if time.monotonic()>=end:break
        time.sleep(min(30,end-time.monotonic()))
    return 0 if result.get('throughput_gate_passed') else 2


if __name__=='__main__':
    raise SystemExit(main())
