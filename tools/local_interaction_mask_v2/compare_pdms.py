"""Full-denominator paired v1 PDMS and log-cluster uncertainty; never selects test models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

METRICS=('score','no_at_fault_collisions','drivable_area_compliance','ego_progress',
         'time_to_collision_within_bound','comfort','driving_direction_compliance')


def read(path):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    result={r['token']:r for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate scored tokens')
    if not rows:raise ValueError('Empty scoring population')
    for row in rows:
        if not row.get('log') or row.get('status') not in ('ok','failed'):raise ValueError('Invalid scored scene identity/status')
        if row['status']=='ok' and any(not np.isfinite(float(row[k])) for k in METRICS):raise ValueError('Missing/nonfinite PDM factor')
    return result


def compare(first,second,bootstrap=10000,seed=20260926):
    if not first:raise ValueError('Empty paired population')
    if set(first)!=set(second):raise ValueError('Paired scene populations differ')
    tokens=sorted(first);logs=sorted({first[t]['log'] for t in tokens});which={log:i for i,log in enumerate(logs)}
    if any(first[t]['log']!=second[t]['log'] for t in tokens):raise ValueError('Log identity differs')
    failed_a=sum(first[t]['status']!='ok' for t in tokens);failed_b=sum(second[t]['status']!='ok' for t in tokens)
    result={'scenes':len(tokens),'logs':len(logs),'failed_first':failed_a,'failed_baseline':failed_b,'valid':not(failed_a or failed_b),
        'interval_scope':'paired log-cluster bootstrap for one fixed training seed; not cross-seed uncertainty','metrics':{}}
    # Failed rows stay at zero and mark the result invalid; never drop their tokens.
    weights=np.random.default_rng(seed).multinomial(len(logs),[1/len(logs)]*len(logs),size=bootstrap)
    counts=np.zeros(len(logs));
    for token in tokens:counts[which[first[token]['log']]]+=1
    rows=[]
    for metric in METRICS:
        a=np.array([float(first[t].get(metric) or 0) if first[t]['status']=='ok' else 0. for t in tokens])
        b=np.array([float(second[t].get(metric) or 0) if second[t]['status']=='ok' else 0. for t in tokens])
        delta=a-b;sums=np.zeros(len(logs))
        for token,d in zip(tokens,delta):sums[which[first[token]['log']]]+=d
        samples=(weights@sums)/(weights@counts)
        result['metrics'][metric]={'first':float(a.mean()),'baseline':float(b.mean()),'paired_delta':float(delta.mean()),
            'log_cluster_95_interval':np.quantile(samples,[.025,.975]).tolist() if len(logs)>=2 else None}
        if metric=='score':
            result.update(wins=int((delta>1e-12).sum()),losses=int((delta< -1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()),
                first_zero_fraction=float((a==0).mean()),baseline_zero_fraction=float((b==0).mean()),
                PDMS_first_percent=float(a.mean()*100),PDMS_baseline_percent=float(b.mean()*100),delta_percentage_points=float(delta.mean()*100))
            rows=[{'token':t,'log':first[t]['log'],'first_score':float(x),'baseline_score':float(y),'delta':float(d),
                   'first_status':first[t]['status'],'baseline_status':second[t]['status']} for t,x,y,d in zip(tokens,a,b,delta)]
    return result,rows


def verified_pair(first_path,second_path):
    summaries=[json.loads(Path(path).with_name('summary.json').read_text()) for path in (first_path,second_path)]
    if summaries[0]['evaluator_identity']!=summaries[1]['evaluator_identity']:
        raise ValueError('Paired official evaluator/protocol/index differs')
    exports=[summary['export_identity'] for summary in summaries]
    common=[{key:value for key,value in export.items() if key not in ('variant','bridge_sha256')} for export in exports]
    if common[0]!=common[1]:raise ValueError('Paired foundation/current cache/source/sampling protocol differs')
    return exports[0]


def compare_subsets(first,second,registry):
    groups={row['token']:row for row in registry['rows']}
    if len(groups)!=len(registry['rows']) or set(groups)!=set(first):raise ValueError('Subset full population differs')
    if any(groups[token]['log']!=row['log'] for token,row in first.items()):raise ValueError('Subset log identities differ')
    result={}
    for name in registry['rules']:
        tokens=[token for token,row in groups.items() if row[name]]
        result[name]=(compare({token:first[token] for token in tokens},{token:second[token] for token in tokens})[0]
                      if tokens else {'scenes':0,'status':'EMPTY','metrics':{}})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('first','baseline','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--navtest',action='store_true');p.add_argument('--subsets');a=p.parse_args()
    export=verified_pair(a.first,a.baseline)
    first,second=read(a.first),read(a.baseline);result,rows=compare(first,second)
    if a.navtest and (result['scenes']!=12146 or result['logs']!=136):raise ValueError('Incomplete Navtest comparison')
    result['input_csv_sha256']={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in [('first',a.first),('baseline',a.baseline)]}
    if a.subsets:
        registry=json.loads(Path(a.subsets).read_text())
        if registry['current_identity']!=export['current_identity'] or registry['current_manifest_sha256']!=export['current_manifest_sha256']:
            raise ValueError('Subset registry uses a different current observation cache')
        result['analysis_groups']=compare_subsets(first,second,registry)
        result['subset_rules']=registry['rules'];result['subset_registry_sha256']=hashlib.sha256(Path(a.subsets).read_bytes()).hexdigest()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'scenes.csv').open('w') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
