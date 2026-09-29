"""Matched five-run v1 planning summaries; log clusters, not repeated samples, form intervals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.local_interaction_mask_v2.compare_pdms import METRICS,read
from tools.foresight.score_pdms import atomic_json

PAIRS=(('B','A'),('C','A'),('D','B'),('D','C'),('A','R'))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sampling_contract(export):
    # W capacity differs intentionally for R. Auxiliary heads are absent in EVERY export.
    return {'current':export['current_identity'],
            'protocol':{k:v for k,v in export['protocol'].items() if k not in ('sampling_seed','W_retained')}}


def collapse_sampling_runs(runs,seeds):
    if set(runs)!=set(seeds):raise ValueError('Missing or unexpected inference seed')
    first=runs[seeds[0]];tokens=sorted(first)
    if not tokens:raise ValueError('Empty sampling population')
    for seed in seeds:
        if set(runs[seed])!=set(tokens) or any(runs[seed][t]['log']!=first[t]['log'] for t in tokens):raise ValueError('Sampling populations/logs differ')
    rows=[];values=[];failures=0;zero_samples=0
    for token in tokens:
        samples=[];failed=0
        for seed in seeds:
            row=runs[seed][token];bad=row['status']!='ok';failed+=bad
            samples.append([0. if bad else float(row[k]) for k in METRICS])
        x=np.asarray(samples,dtype=np.float64)
        if not np.isfinite(x).all() or (x< -1e-7).any() or (x>1+1e-7).any():raise ValueError('Invalid PDM factor')
        average=x.mean(0);failures+=failed;zero_samples+=int((x[:,0]==0).sum());values.append(x)
        rows.append({'token':token,'log':first[token]['log'],'failed_sampling_runs':failed,
                     **dict(zip(METRICS,average.tolist()))})
    cube=np.stack(values);means=cube.mean(0)[:,0]
    summary={'scenes':len(rows),'logs':len({r['log'] for r in rows}),'sampling_seeds':list(seeds),
             'failed_samples':failures,'failed_scenes':sum(r['failed_sampling_runs']>0 for r in rows),'valid':failures==0,
             'metrics':dict(zip(METRICS,cube.mean((0,1)).tolist())),
             'PDMS_per_sampling_seed':dict(zip(map(str,seeds),means.tolist())),
             'sampling_seed_population_PDMS_std':float(means.std(ddof=1)) if len(seeds)>1 else None,
             'zero_sample_fraction':zero_samples/(len(rows)*len(seeds)),
             'all_sampling_runs_zero_scene_fraction':float((cube[:,:,0]==0).all(1).mean())}
    return rows,summary


def paired_difference(first,baseline,bootstrap=10000,seed=20260928):
    if bootstrap<100:raise ValueError('At least100log bootstrap draws required')
    a={r['token']:r for r in first};b={r['token']:r for r in baseline}
    if not a or len(a)!=len(first) or len(b)!=len(baseline) or set(a)!=set(b):raise ValueError('Paired population mismatch')
    tokens=sorted(a)
    if any(a[t]['log']!=b[t]['log'] for t in tokens):raise ValueError('Paired log mismatch')
    logs=sorted({a[t]['log'] for t in tokens});lookup={log:i for i,log in enumerate(logs)}
    groups=np.asarray([lookup[a[t]['log']] for t in tokens]);counts=np.bincount(groups,minlength=len(logs))
    av=np.asarray([[a[t][k] for k in METRICS] for t in tokens]);bv=np.asarray([[b[t][k] for k in METRICS] for t in tokens]);delta=av-bv
    if not np.isfinite(delta).all():raise ValueError('Nonfinite paired result')
    sums=np.zeros((len(logs),len(METRICS)));np.add.at(sums,groups,delta)
    weights=np.random.default_rng(seed).multinomial(len(logs),[1/len(logs)]*len(logs),size=bootstrap)
    draws=(weights@sums)/(weights@counts)[:,None]
    report={'scenes':len(tokens),'logs':len(logs),'bootstrap_draws':bootstrap,'bootstrap_seed':seed,
            'interval_scope':'paired log clusters after averaging fixed inference seeds within each scene; one training seed',
            'valid':not any(a[t]['failed_sampling_runs'] or b[t]['failed_sampling_runs'] for t in tokens),
            'metrics':{k:{'first':float(av[:,i].mean()),'baseline':float(bv[:,i].mean()),'delta':float(delta[:,i].mean()),
                          'log_cluster_95_interval':np.quantile(draws[:,i],[.025,.975]).tolist() if len(logs)>1 else None} for i,k in enumerate(METRICS)},
            'wins':int((delta[:,0]>1e-12).sum()),'losses':int((delta[:,0]< -1e-12).sum()),'ties':int((abs(delta[:,0])<=1e-12).sum())}
    scene_rows=[{'token':t,'log':a[t]['log'],'first_failed_runs':a[t]['failed_sampling_runs'],'baseline_failed_runs':b[t]['failed_sampling_runs'],
                 **{k+'_delta':float(delta[j,i]) for i,k in enumerate(METRICS)}} for j,t in enumerate(tokens)]
    log_rows=[{'log':log,'scenes':int(counts[j]),**{k+'_delta':float(sums[j,i]/counts[j]) for i,k in enumerate(METRICS)}} for j,log in enumerate(logs)]
    return report,scene_rows,log_rows


def write_csv(path,rows):
    if not rows:raise ValueError('Cannot silently write an empty comparison')
    with Path(path).open('w') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)


def main(full_method=False):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--registry',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=('dev','navtest'),required=True);p.add_argument('--sampling-seeds',default='42,43,44,45,46')
    p.add_argument('--diagnostic',action='store_true')
    if full_method:
        p.add_argument('--screen',action='store_true',help='Preregistered single-inference-seed development screen')
        p.add_argument('--expected-arms',required=True,help='Explicit registered population of candidates/finalists/ablations')
    a=p.parse_args()
    seeds=tuple(map(int,a.sampling_seeds.split(',')))
    if len(set(seeds))!=len(seeds) or not seeds:raise ValueError('Invalid sampling protocol')
    screen=full_method and a.screen
    if screen and (a.split!='dev' or seeds!=(42,)):raise ValueError('Screen is development only, fixed inference seed42')
    if not a.diagnostic and not screen and seeds!=(42,43,44,45,46):raise ValueError('Formal protocol is five fixed inference seeds42–46')
    if a.diagnostic and a.split=='navtest':raise ValueError('No diagnostic Navtest model selection')
    registry=json.loads(Path(a.registry).read_text());groups={};identities={};evaluators=[];contracts=[];sources={}
    for entry in registry:
        arm,train_seed,infer_seed=entry['arm'],int(entry['training_seed']),int(entry['sampling_seed']);key=(arm,train_seed)
        allowed=tuple(f'C{i}' for i in range(6))+('NO_INTERACTION','NO_FUTURE','NO_CURRENT','W_ACTION_ONLY','NATIVE_ACTION_ONLY') if full_method else ('R','A','B','C','D')
        if arm not in allowed:raise ValueError('Unknown arm')
        folder=Path(entry['score_dir']);summary=json.loads((folder/'summary.json').read_text());export=summary['export_identity'];checkpoint=export['checkpoint']
        if summary['schema']!='foresight_official_pdms_v1' or export['current_identity']['split']!=a.split:raise ValueError('Wrong scoring protocol/split')
        if checkpoint['arm']!=arm or export['protocol']['sampling_seed']!=infer_seed:raise ValueError('Registry does not match scored artifact')
        if not a.diagnostic and (summary['diagnostic'] or checkpoint.get('training_seed')!=train_seed):raise ValueError('Formal comparison requires verified training seed and complete formal checkpoint')
        if a.split=='navtest' and not summary['full_navtest']:raise ValueError('Partial Navtest result')
        if key in identities and identities[key]['checkpoint']['sha256']!=checkpoint['sha256']:raise ValueError('Different model checkpoints mixed across inference seeds')
        identities[key]=export;evaluators.append(summary['evaluator_identity']);contracts.append(sampling_contract(export))
        if infer_seed in groups.setdefault(key,{}):raise ValueError('Duplicate arm/training/inference seed')
        rows=read(folder/'scenes.csv')
        if len(rows)!=summary['scenes'] or len({r['log'] for r in rows.values()})!=summary['logs']:
            raise ValueError('Scoring CSV population differs from its completed summary')
        if sum(r['status']!='ok' for r in rows.values())!=summary['failed']:
            raise ValueError('Scoring failure count changed')
        groups[key][infer_seed]=rows;sources[str(folder/'scenes.csv')]=digest(folder/'scenes.csv')
    if not groups:raise ValueError('Empty experiment registry')
    if any(x!=evaluators[0] for x in evaluators) or any(x!=contracts[0] for x in contracts):raise ValueError('Mixed evaluator/current input/sampling protocols')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);collapsed={};report={'schema':'foresight_paired_planning_v1',
        'split':a.split,'diagnostic':a.diagnostic,'registry_sha256':digest(a.registry),'source_csv_sha256':sources,
        'groups':{},'comparisons':{},'missing_comparisons':{},'training_seed_interpretation':'reported separately; inference seeds are not training repeats'}
    pairs=PAIRS
    if full_method:
        import itertools
        expected=a.expected_arms.split(',')
        if not expected or len(set(expected))!=len(expected) or any(x not in allowed for x in expected):raise ValueError('Invalid expected full-method matrix')
        pairs=tuple((first,base) for base,first in itertools.combinations(expected,2))
        report.update(schema='ddp_full_foresight_planning_v1',screen=screen,expected_arms=expected)
    for key,runs in groups.items():
        rows,summary=collapse_sampling_runs(runs,seeds);collapsed[key]=rows;name=f'{key[0]}_train{key[1]}'
        summary['checkpoint']=identities[key]['checkpoint'];report['groups'][name]=summary;write_csv(output/(name+'_scenes.csv'),rows)
    for train_seed in sorted({seed for _,seed in groups}):
        for first,base in pairs:
            name=f'{first}-{base}_train{train_seed}'
            if (first,train_seed) not in collapsed or (base,train_seed) not in collapsed:
                report['missing_comparisons'][name]='NOT_RUN';continue
            result,scenes,logs=paired_difference(collapsed[first,train_seed],collapsed[base,train_seed]);report['comparisons'][name]=result
            write_csv(output/(name+'_scenes.csv'),scenes);write_csv(output/(name+'_logs.csv'),logs)
    report['valid']=all(r['valid'] for r in report['groups'].values());report['complete_primary_matrix']=all((arm,42) in groups for arm in ('R','A','B','C','D'))
    if full_method:report['complete_primary_matrix']=all((arm,42) in groups for arm in expected)
    atomic_json(output/'summary.json',report)
    if not report['valid']:raise RuntimeError('Failed samples retained; paired report is not a valid complete experiment')


if __name__=='__main__':main()
