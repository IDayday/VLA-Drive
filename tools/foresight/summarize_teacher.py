"""Snapshot actual teacher learning; GT-conditioned diagnostics are not planning scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .score_pdms import atomic_json


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditional_summary(rows,bootstrap=10000):
    """Paired queries stay in their log cluster, including multiple targets per scene."""
    if not rows:return {'queries':0,'status':'NOT_APPLICABLE'}
    failures=sum(bool(r['failure']) for r in rows)
    result={'queries':len(rows),'scenes':len({r['token'] for r in rows}),
            'logs':len({r['log'] for r in rows}),'failures':failures,'valid':failures==0,
            'valid_xy_points':sum(int(r['valid_points']) for r in rows)}
    if failures:
        # Distance errors have no meaningful zero-on-failure convention. Keep
        # the denominator and invalidate means rather than dropping hard rows.
        result['status']='INVALID_FAILURES';return result
    names=('full_peer_ADE','current_only_ADE','stationary_ADE')
    values=np.asarray([[float(r[k]) for k in names] for r in rows])
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Illegal teacher distance metric')
    logs=sorted({r['log'] for r in rows});lookup={v:i for i,v in enumerate(logs)}
    ids=np.asarray([lookup[r['log']] for r in rows]);counts=np.bincount(ids,minlength=len(logs))
    delta=values[:,0]-values[:,1];sums=np.bincount(ids,weights=delta,minlength=len(logs))
    interval=None
    if len(logs)>1:
        weights=np.random.default_rng(20260928).multinomial(len(logs),[1/len(logs)]*len(logs),size=bootstrap)
        interval=np.quantile((weights@sums)/(weights@counts),[.025,.975]).tolist()
    result.update(status='OK',**dict(zip(names,values.mean(0).tolist())),
                  full_minus_current_ADE=float(delta.mean()),paired_log_95_interval=interval,
                  interval_scope='query-weighted paired mean, complete log clusters, one training seed')
    for mode in ('full_peer','current_only','stationary'):
        for kind in ('last_valid_FDE','endpoint_FDE'):
            selected=[float(r[mode+'_'+kind]) for r in rows if r[mode+'_'+kind]!='']
            if not np.isfinite(selected).all() or any(v<0 for v in selected):raise ValueError('Illegal teacher FDE')
            result[mode+'_'+kind]=float(np.mean(selected)) if selected else None
            result[kind+'_queries']=len(selected)
    return result


def fixed_groups(rows):
    groups={'all':rows}
    for role in ('ego','vehicle'):
        base=[r for r in rows if r['role']==role];groups[role]=base
        for value in ('stationary','moving'):
            groups[role+'_'+value]=[r for r in base if r['motion_group']==value]
        for value in ('complete','partial'):
            groups[role+'_'+value]=[r for r in base if r['label_group']==value]
        groups[role+'_with_peer']=[r for r in base if int(r['known_peer_points'])>0]
        groups[role+'_without_peer']=[r for r in base if int(r['known_peer_points'])==0]
    return groups


def write_csv(path,rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run=Path(a.run);out=Path(a.output)
    identity=json.loads((run/'identity.json').read_text());status=json.loads((run/'status.json').read_text())
    if identity['schema']!='gt_mae_training_v1':raise ValueError('Not a new GT-MAE teacher')
    # A live append may have an incomplete last line; snapshot only complete records.
    raw=(run/'steps.jsonl').read_bytes();complete=raw[:raw.rfind(b'\n')+1]
    steps=[json.loads(line) for line in complete.splitlines()]
    if not steps or [s['update'] for s in steps]!=list(range(1,len(steps)+1)):
        raise ValueError('Training steps are missing or duplicated')
    metrics=[];groups={};expected=None;sources={};scope=None
    for path in sorted(run.glob('metrics_*.json'),key=lambda p:int(p.stem.split('_')[-1])):
        metric=json.loads(path.read_text());epoch=metric['epoch'];csv_path=run/f'queries_{epoch}.csv'
        with csv_path.open() as f:rows=list(csv.DictReader(f))
        keys=[(r['token'],r['target'],r['log']) for r in rows]
        if len(set(keys))!=len(keys) or len(rows)!=metric['queries'] or len({r['token'] for r in rows})!=metric['scenes']:
            raise ValueError('Teacher query population is incomplete')
        if expected is not None and keys!=expected:raise ValueError('Fixed teacher queries changed across milestones')
        if scope is not None and metric['scope']!=scope:raise ValueError('Mixed training/development teacher results')
        expected=keys;scope=metric['scope'];metrics.append(metric)
        groups[str(epoch)]={name:conditional_summary(part) for name,part in fixed_groups(rows).items()}
        sources[path.name]=file_hash(path);sources[csv_path.name]=file_hash(csv_path)
    if not metrics:raise ValueError('No teacher evaluation evidence')
    losses=np.asarray([s['loss'] for s in steps])
    if not np.isfinite(losses).all() or (losses<0).any():raise ValueError('Invalid training loss')
    out.mkdir(parents=True,exist_ok=False)
    last=steps[-1];roles=last['roles'];nominal=roles['nominal_ego']+roles['nominal_neighbor']
    report={'schema':'foresight_teacher_learning_report_v1','teacher_identity':identity['identity'],
            'training_source_sha':identity['source_sha'],'data_identity':identity['data_identity'],
            'scope':scope,'full_training_status':status['status'],'latest_complete_logged_update':last['update'],
            'sample_presentations':last['exposure'],'last_checkpoint_update':status['completed'],
            'valid_xy_coordinates_total':sum(s['valid_xy_coordinates'] for s in steps),
            'roles':roles,'nominal_ego_fraction':roles['nominal_ego']/nominal,
            'actual_ego_fraction':roles['actual_ego']/nominal,'fallback_fraction':roles['ego_fallback']/nominal,
            'logged_steps_sha256':hashlib.sha256(complete).hexdigest(),'source_metrics_sha256':sources,
            'milestones':groups,'interpretation':'Teacher sees GT peer futures. Removal may be out of distribution; these are mechanism diagnostics, not causal proof or camera-only planning performance.'}
    atomic_json(out/'summary.json',report)
    write_csv(out/'training_curve.csv',[{k:s[k] for k in ('update','epoch','exposure','loss','lr','grad_norm','valid_xy_coordinates')} for s in steps])
    write_csv(out/'validation_curve.csv',[{'epoch':m['epoch'],'update':m['completed'],
        **{role+'_'+condition:m['groups'][role+'_with_peer'][condition+'_ADE'] for role in ('ego','vehicle') for condition in ('full_peer','current_only')}} for m in metrics])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    window=min(100,len(losses));smooth=np.convolve(losses,np.ones(window)/window,mode='valid')
    axes[0].plot(np.arange(window,len(losses)+1),smooth);axes[0].set(title=f'Train Smooth L1 ({window}-step mean)',xlabel='Optimizer updates',ylabel='Loss',yscale='log')
    for axis,role in zip(axes[1:],('ego','vehicle')):
        for mode,label in [('full_peer','GT peers'),('current_only','Peers removed')]:
            axis.plot([m['epoch'] for m in metrics],[m['groups'][role+'_with_peer'][mode+'_ADE'] for m in metrics],marker='o',label=label)
        axis.set(title=role+' / queries with peer futures',xlabel='Completed epochs',ylabel='ADE (m)',yscale='log');axis.legend()
    fig.suptitle('GT-MAE teacher: '+scope+'; no student planning claim')
    fig.savefig(out/'learning_curves.png',dpi=160);fig.savefig(out/'learning_curves.pdf');plt.close(fig)


if __name__=='__main__':main()
