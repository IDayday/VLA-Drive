"""Canonical scoring of fixed legacy plans for the finite coverage audit.

No model, optimizer, solver, score selection or change to official scoring.
Primary seed42 and one outcome-independent recovery variant per scene only.
"""
import argparse,csv,hashlib,json,multiprocessing,resource,sys,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,atomic_json,digest,identity,check_host,source_sha
from tools.local_interaction_mask_v2.score_async import initialize,score_chunk,python_tree_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('labels','metric-index','devkit','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();check_host();start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    if not 1<=a.workers<=16:raise ValueError('Bound CPU concurrency')
    index=read(a.metric_index);label=read(a.labels/'identity.json')
    if {r['token'] for r in index}!={r['token'] for r in label['index']}:raise ValueError('Complete fixed audit population required')
    a.output.mkdir(parents=True,exist_ok=True)
    contract=dict(source_sha=source_sha(ROOT),label_identity=label['identity'],metric_index_sha256=digest(a.metric_index),
        navsim_tree_sha256=python_tree_digest(a.devkit/'navsim'),primary_seed=42,
        selection='NATURAL plan0; RECOVERY hash(token:fixed-recovery-v1)%4+1; both anchor_index0; no outcome selection',
        official_scoring_changed=False,index=[{'token':r['token'],'log':r['log']} for r in index])
    ident=identity(contract)
    if (a.output/'identity.json').exists() and read(a.output/'identity.json')['identity']!=ident:raise ValueError('Audit scorer identity changed')
    atomic_json(a.output/'identity.json',dict(identity=ident,**contract));tasks=[];allrows=[]
    for distribution in ('NATURAL','RECOVERY'):
        folder=a.output/distribution
        for part in ('proposals','records'):(folder/part).mkdir(parents=True,exist_ok=True)
        for row in index:
            record=folder/'records'/(row['token']+'.json');proposal=folder/'proposals'/(row['token']+'.npz')
            if record.exists():
                saved=read(record)
                if saved['status']!='ok' or saved['proposal_sha256']!=digest(proposal):raise ValueError('Failed/changed prior score requires explicit investigation')
                allrows.append(saved);continue
            path=a.labels/'scenes'/(row['token']+'.npz')
            receipt=read(a.labels/'records'/(row['token']+'.json'))
            if digest(path)!=receipt['sha256']:raise ValueError('Legacy plan changed')
            at=0 if distribution=='NATURAL' else 1+int(hashlib.sha256((row['token']+':fixed-recovery-v1').encode()).hexdigest()[:8],16)%4
            with np.load(path,allow_pickle=False) as f:poses=f['plan_in_physical'][0,at].copy()
            with proposal.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,trajectory=poses)
            proposal.with_suffix('.tmp').replace(proposal)
            tasks.append(dict(row,variant=distribution,proposal_path=str(proposal),proposal_sha256=digest(proposal)))
    initialize(str(a.devkit))
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(a.devkit),)) as pool:
        chunks=iter([tasks[i:i+8] for i in range(0,len(tasks),8)]);pending=set()
        def fill():
            while len(pending)<2*a.workers and time.monotonic()-start<a.max_seconds:
                chunk=next(chunks,None)
                if chunk is None:break
                pending.add(pool.submit(score_chunk,chunk))
        fill()
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for task in done:
                for row in task.result():
                    atomic_json(a.output/row['variant']/'records'/(row['token']+'.json'),row);allrows.append(row)
            atomic_json(a.output/'STATUS.json',dict(completed=len(allrows),expected=2*len(index),seconds=time.monotonic()-start))
            fill()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    summary=dict(identity=ident,scenes_per_distribution=len(index),scored=len(allrows),failed=sum(r['status']!='ok' for r in allrows),
        wall_seconds=time.monotonic()-start,CPU_waited_children_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
    for distribution in ('NATURAL','RECOVERY'):
        rows=[r for r in allrows if r['variant']==distribution]
        if rows:
            with (a.output/distribution/'scenes.csv').open('w') as f:
                w=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
        summary[distribution]=dict(scenes=len(rows),zero=sum(r['score']==0 for r in rows),PDMS=sum(r['score'] for r in rows)/max(1,len(rows)))
    complete=len(allrows)==2*len(index) and summary['failed']==0
    atomic_json(a.output/('COMPLETE.json' if complete else 'INCOMPLETE.json'),summary);print(json.dumps(summary))
    if not complete:raise SystemExit(1)


if __name__=='__main__':main()
