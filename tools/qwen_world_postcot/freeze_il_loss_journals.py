"""Seal a finite effective TRAIN journal prefix without touching the trainer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np


def freeze_rank(run, output, rank, last):
    source=run/f'metrics_rank{rank}.jsonl'; target=output/source.name
    digest=hashlib.sha256(); x=[]; expected=1; byte_count=0
    with source.open('rb') as stream, target.open('xb') as dest:
        for line in stream:
            row=json.loads(line); update=row['updates']
            if update>last:break
            if (update!=expected or row['cursor']!=32*update
                    or row['optimizer_global_steps']!=update or row['optimizer_skipped_steps']!=0):
                raise ValueError('Not a contiguous legal effective optimizer prefix')
            value=float(row['losses']['x'])
            if not np.isfinite(value) or value<0:raise ValueError('Invalid normalized trajectory L1')
            digest.update(line);dest.write(line);byte_count+=len(line);x.append(value);expected+=1
    if expected!=last+1:raise ValueError('Journal does not reach the frozen checkpoint')
    return np.asarray(x,dtype=np.float64),dict(rank=rank,source=str(source),frozen=str(target),
        through_update=last,prefix_sha256=digest.hexdigest(),bytes=byte_count)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--last-update',type=int,required=True);a=p.parse_args()
    run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    raw=(run/'IDENTITY.json').read_bytes();identity=json.loads(raw)
    if identity['config']['global_batch']!=32:raise ValueError('Unexpected scientific scene exposure')
    with ThreadPoolExecutor(max_workers=4) as pool:
        ranks=list(pool.map(lambda rank:freeze_rank(run,out,rank,a.last_update),range(8)))
    x=ranks[0][0]
    if any(not np.array_equal(x,other) for other,_ in ranks[1:]):
        raise ValueError('Already-global loss replicas differ; do not average them as independent runs')
    if raw!=(run/'IDENTITY.json').read_bytes():raise ValueError('Training identity changed during freeze')
    (out/'IDENTITY.json').write_bytes(raw)
    interface=identity['config']['interface_updates']; bins=[]
    for end in range(1000,a.last_update+1,1000):
        start=end-1000
        if start<interface:continue
        bins.append(dict(update=end,window_updates=1000,x_L1=float(x[start:end].mean())))
    def trend(rows):
        values=np.array([r['x_L1'] for r in rows]);differences=np.diff(values)
        return dict(points=len(rows),first=float(values[0]),last=float(values[-1]),
                    increases=int((differences>0).sum()),decreases=int((differences<0).sum()),
                    ties=int((differences==0).sum()))
    checkpoints={}
    for end in sorted({30000,35000,37000,40000,45000,50000,52000,53000,55000,60000,62000,63000}):
        if end<=a.last_update:
            checkpoints[str(end)]={str(w):float(x[end-w:end].mean()) for w in [100,500,1000]}
    result=dict(schema='world_IL_frozen_trajectory_loss_v1',last_update=a.last_update,
        definition='Already globally reduced normalized clean-x L1 BEFORE coefficient10; online TRAIN, not Test trajectory error',
        world=8,global_scene_batch=32,train_seed=42,identity_sha256=hashlib.sha256(raw).hexdigest(),
        log_prefix_receipts=[r for _,r in ranks],checkpoint_L1=checkpoints,bins1000=bins,
        full_joint_1k_trend=trend(bins),midlate_30k_1k_trend=trend([r for r in bins if r['update']>=30000]),
        late_50k_1k_trend=trend([r for r in bins if r['update']>=50000]))
    (out/'TRAJECTORY_LOSS_CURVE.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',last_update=a.last_update,
        all8_global_L1_replicas_equal=True,all_optimizer_boundaries_valid=True))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['checkpoint_L1','full_joint_1k_trend','midlate_30k_1k_trend','late_50k_1k_trend']},indent=2))


if __name__=='__main__':main()
