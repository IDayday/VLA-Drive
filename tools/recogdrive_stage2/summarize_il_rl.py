"""Complete-population paired analysis; log-cluster bootstrap, real scene weights."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .assets import atomic_json, read
from .evaluate import signature


def summarize(root, replicates=5000):
    root = Path(root); identity = read(root/'IDENTITY.json')
    if read(root/'COMPLETE.json')['diagnostic']:
        raise ValueError('A smoke bank cannot be summarized as full Navtest')
    manifest = read(identity['current_manifest'])['rows']
    if len(manifest)!=12146 or len({r['log'] for r in manifest})!=136:
        raise ValueError('Unexpected full test population')
    ident = signature(identity); rows=[]; per_time={k:[] for k in ['IL','RL']}
    bins={str(i):dict(count=0,IL_sum=0.,RL_sum=0.) for i in range(10)}
    score_names=['score','no_at_fault_collisions','drivable_area_compliance','time_to_collision',
                 'ego_progress','comfort','driving_direction_compliance']
    for item in manifest:
        d=read(root/'relations'/f"{item['token']}.json")
        if d['identity_sha256']!=ident or d['log']!=item['log']:raise ValueError('Result identity changed')
        row=dict(token=item['token'],log=item['log'])
        if d['losses']['IL']['times']!=d['losses']['RL']['times'] or d['losses']['IL']['noise_sha256']!=d['losses']['RL']['noise_sha256']:
            raise ValueError('Unpaired Monte Carlo samples')
        cache_hashes=[]
        for name in ['IL','RL']:
            s=read(root/(name+'_scores')/'records'/f"{item['token']}.json")
            if s['status']!='ok' or s['log']!=item['log']:raise ValueError('Missing/invalid canonical score')
            receipt=read(root/name/'predictions'/f"{item['token']}.json")
            if receipt['identity_sha256']!=ident or s['proposal_sha256']!=receipt['proposal_sha256']:
                raise ValueError('Score belongs to a different sampled trajectory')
            cache_hashes.append(s['metric_cache_sha256'])
            loss=d['losses'][name]
            if len(loss['draws'])!=identity['draws'] or len(loss['times'])!=identity['draws']:
                raise ValueError('Incomplete paired loss draws')
            if not np.isfinite(loss['draws']).all():raise ValueError('Nonfinite loss draws')
            row[name+'_IL_noise_MSE']=d['losses'][name]['mean']
            for k,v in d['metrics'][name].items():
                if isinstance(v,(int,float)):row[name+'_'+k]=v
            for k in score_names:row[name+'_'+k]=s[k]
            for c, v in zip(['x','y','yaw'],d['losses'][name]['per_channel']):row[name+'_noise_MSE_'+c]=v
            per_time[name].append(d['metrics'][name]['L2_by_time_m'])
        if cache_hashes[0]!=cache_hashes[1]:raise ValueError('IL/RL used different canonical environments')
        for t,il,rl in zip(d['losses']['IL']['times'],d['losses']['IL']['draws'],d['losses']['RL']['draws']):
            if not 0<=t<100:raise ValueError('Wrong diffusion time distribution')
            b=bins[str(t//10)];b['count']+=1;b['IL_sum']+=il;b['RL_sum']+=rl
        rows.append(row)
    if set(p.stem for p in (root/'relations').glob('*.json')) != {r['token'] for r in manifest}:
        raise ValueError('Duplicate/foreign scene results')
    logs=sorted({r['log'] for r in rows}); li={x:i for i,x in enumerate(logs)}
    group=np.array([li[r['log']] for r in rows]);counts=np.bincount(group,minlength=len(logs))
    rng=np.random.default_rng(20261018);samples=rng.integers(len(logs),size=(replicates,len(logs)))
    denominators=counts[samples].sum(1)
    names=[k[3:] for k in rows[0] if k.startswith('IL_')]
    comparison={}
    for key in names:
        il=np.array([r['IL_'+key] for r in rows]);rl=np.array([r['RL_'+key] for r in rows]);delta=rl-il
        if not np.isfinite(il).all() or not np.isfinite(rl).all():raise ValueError('Nonfinite metric')
        sums=np.bincount(group,weights=delta,minlength=len(logs))
        boot=sums[samples].sum(1)/denominators
        comparison[key]=dict(IL=float(il.mean()),RL=float(rl.mean()),RL_minus_IL=float(delta.mean()),
            paired_95CI=np.quantile(boot,[.025,.975]).tolist(),
            relative_change_percent=float(100*(rl.mean()/il.mean()-1)) if il.mean()!=0 else None,
            scenes_RL_lower=int((delta<0).sum()),scenes_equal=int((delta==0).sum()),scenes_RL_higher=int((delta>0).sum()))
    il=np.array([r['IL_score'] for r in rows]);rl=np.array([r['RL_score'] for r in rows])
    lossdiff=np.array([r['RL_IL_noise_MSE']-r['IL_IL_noise_MSE'] for r in rows])
    adediff=np.array([r['RL_ADE_m']-r['IL_ADE_m'] for r in rows])
    transitions=dict(IL_zero_RL_positive=int(((il==0)&(rl>0)).sum()),
        IL_positive_RL_zero=int(((il>0)&(rl==0)).sum()),both_zero=int(((il==0)&(rl==0)).sum()),
        both_positive=int(((il>0)&(rl>0)).sum()),
        PDMS_up_IL_loss_up=int(((rl>il)&(lossdiff>0)).sum()),
        PDMS_up_ADE_up=int(((rl>il)&(adediff>0)).sum()),PDMS_up=int((rl>il).sum()),
        PDMS_down=int((rl<il).sum()),PDMS_equal=int((rl==il).sum()))
    groups={}
    for key,mask in dict(zero_repaired=(il==0)&(rl>0),new_zero=(il>0)&(rl==0),
                         both_zero=(il==0)&(rl==0),both_positive=(il>0)&(rl>0)).items():
        groups[key]=dict(scenes=int(mask.sum()),logs=len({rows[i]['log'] for i in np.flatnonzero(mask)}),
            PDMS_delta=float((rl-il)[mask].mean()) if mask.any() else None,
            IL_loss_delta=float(lossdiff[mask].mean()) if mask.any() else None,
            ADE_delta_m=float(adediff[mask].mean()) if mask.any() else None)
    time_bins={f'{i*10}-{i*10+9}':dict(count=b['count'],IL=b['IL_sum']/b['count'],
        RL=b['RL_sum']/b['count'],RL_minus_IL=(b['RL_sum']-b['IL_sum'])/b['count'])
        for i,b in [(int(k),v) for k,v in bins.items()]}
    summary=dict(identity=identity,scenes=len(rows),logs=len(logs),valid=True,failed=0,
        comparison=comparison,zero_transitions=transitions,zero_transfer_groups=groups,time_bins=time_bins,
        L2_by_physical_time_m={k:np.mean(v,axis=0).tolist() for k,v in per_time.items()},
        bootstrap=dict(unit='log',replicates=replicates,seed=20261018,
            reduction='pooled scene sum / pooled scene count; not equal-weight log means'),
        limitations=['One released IL-to-RL lineage, not independent training replicates.',
            'IL loss is 8 paired uniform-noise/time draws per scene, not an exact integral.',
            'One shared trajectory sampling seed42; no best-of-N or GT selection.',
            'Navtest is used only for requested post-hoc evaluation, no training or method selection.',
            'XY ADE/FDE and epsilon MSE are different objectives; validation L1 mixes meters/radians.'])
    atomic_json(root/'PAIRED_SUMMARY.json',summary)
    with (root/'paired_scenes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:comparison[k] for k in ['IL_noise_MSE','ADE_m','FDE_m','score']},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    p.add_argument('--bootstrap-replicates',type=int,default=5000);a=p.parse_args()
    summarize(a.output,a.bootstrap_replicates)
