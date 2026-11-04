"""Read-only all-rank health snapshots, called by the existing report owner.

No new polling process; no test labels, priorities or trajectories enter TRAIN.
Gradient role norms in the native journal are AFTER global clipping.
"""
import json
import math
from pathlib import Path
import statistics
import time


def tail(path,limit=64):
    try:
        with Path(path).open('rb') as f:
            f.seek(max(0,Path(path).stat().st_size-2**20))
            lines=f.read().splitlines()
    except OSError:return []
    rows=[]
    for line in lines:
        try:
            row=json.loads(line)
            if isinstance(row,dict) and isinstance(row.get('updates'),int):rows.append(row)
        except ValueError:continue
    return rows[-limit:]


def snapshot(output,world,now=None):
    now=time.time() if now is None else now
    tails=[tail(Path(output)/f'metrics_rank{rank}.jsonl') for rank in range(world)]
    if not all(tails):return dict(status='WAITING_ALL_RANKS',at=now,alerts=['尚未取得全部rank日志'])
    last=[r[-1] for r in tails];common=min(r['updates'] for r in last)
    run_ids={r.get('run') for r in last}
    indices=[{r['updates']:r for r in rows if r.get('run')==last[i].get('run') and r.get('invocation')==last[i].get('invocation')} for i,rows in enumerate(tails)]
    common_steps=sorted(set.intersection(*(set(r) for r in indices)))[-64:]
    rows=[rank[u] for u in common_steps for rank in indices]
    alerts=[]
    if len(run_ids)!=1:alerts.append('rank身份不一致')
    if max(r['updates'] for r in last)-common>1:alerts.append('rank提交进度不一致')
    if now-min(r['at'] for r in last)>600:alerts.append('日志超过10分钟未提交')
    health_keys=('gradient_norm','global_advantage_std')
    if any(not math.isfinite(r[k]) for r in rows for k in health_keys):alerts.append('梯度或优势出现非有限值')
    if any(r['losses']['fresh_ratio_error']>1e-4 for r in rows if r['inner_completed']==1):alerts.append('fresh ratio不等于1')
    std_spread=max((max(rank[u]['global_advantage_std'] for rank in indices)-min(rank[u]['global_advantage_std'] for rank in indices) for u in common_steps),default=0.)
    if std_spread>1e-10:alerts.append('全局优势std不一致')
    def avg(values):return statistics.mean(values) if values else None
    inner={str(i):{k:avg([r['losses'][k] for r in rows if r['inner_completed']==i]) for k in ('reference_KL','ratio_mean','clip_fraction')} for i in (1,2)}
    speed=None
    if len(common_steps)>=3:
        first,end=common_steps[0],common_steps[-1]
        elapsed=max(rank[end]['at'] for rank in indices)-max(rank[first]['at'] for rank in indices)
        if elapsed>0:speed=dict(from_update=first,to_update=end,seconds_per_update=elapsed/(end-first),updates_per_hour=3600*(end-first)/elapsed)
    roles={}
    for r in rows:
        norms=r.get('role_gradient_norms',{});sq=sum(v*v for v in norms.values() if v is not None)
        if sq>0:
            for k,v in norms.items():
                if v is not None:roles.setdefault(k,[]).append(v*v/sq)
    fraction={k:avg(v) for k,v in roles.items()}
    exploration=[indices[0][u]['exploration'] for u in common_steps if indices[0][u]['inner_completed']==1]
    return dict(status='REVIEW' if alerts else 'FINITE_CONSISTENT_NOT_A_PERFORMANCE_PROOF',at=now,
        common_update=common,scene_exposures=min(r['scene_exposures'] for r in last),scene_epochs=min(r['scene_epochs'] for r in last),
        sampled_rank_update_rows=len(rows),global_advantage_std_rank_spread=std_spread,inner=inner,
        gradient_norm_mean=avg([r['gradient_norm'] for r in rows]),
        gradient_norm_max=max((r['gradient_norm'] for r in rows),default=None),
        clipped_update_fraction=avg([float(r['gradient_norm']>1) for r in rows]),
        role_gradient_squared_norm_fraction=fraction,role_norms_recorded_after_global_clip=True,
        exploration={k:avg([r[k] for r in exploration]) for k in ('effective_PG_group_fraction','feasible_candidate_fraction','all_zero_groups','all_one_groups')},
        end_to_end=speed,alerts=alerts,
        limits='Recent all-rank window; raw gradients and quantization require separate replay. No inference of Adam update contribution or PDMS from gradient fractions.')
