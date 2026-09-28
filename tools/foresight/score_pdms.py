"""Identity-checked adapter to the existing official single-trajectory CPU evaluator.

No metric formula is implemented here. GPU exports may continue concurrently;
CPU workers consume only atomic completed scene files. Navtest requires the
export's final lock; failed rows remain zero and invalidate the complete result.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tools.local_interaction_mask_v2.score_async import digest,python_tree_digest
from tools.local_interaction_mask_v2.compare_pdms import METRICS,read
from tools.local_interaction_mask_v2.merge_scores import merge_population
from tools.ddpolicy_vehicle.run_meter import metered_run


def identity_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def requested_population(current,metric,export):
    if not current or len({r['token'] for r in current})!=len(current):raise ValueError('Invalid current population')
    if identity_hash(current)!=export['current_identity']['index_sha256']:raise ValueError('Current index differs from GPU export')
    lookup={r['token']:r for r in metric}
    if len(lookup)!=len(metric):raise ValueError('Duplicate official cache rows')
    expected=current[:export['limit']] if export['limit'] else current
    rows=[]
    for scene in expected:
        item=lookup.get(scene['token'])
        if item is None or item['log']!=scene['log']:raise ValueError('Missing official scene/log cache identity')
        rows.append({'token':scene['token'],'log':scene['log'],'cache_path':item['cache_path']})
    return rows


def merge_parts(out,bank,index,identity,parts):
    directories=[out/'parts'/f'part_{i}' for i in range(parts)]
    if not all((p/'summary.json').exists() for p in directories):return None
    groups=[];evaluators=[];csv_hashes={}
    for i,part in enumerate(directories):
        ident=json.loads((part/'identity.json').read_text());args=ident['arguments']
        if ident['index_sha256']!=digest(out/'requested_index.json') or Path(args['predictions']).resolve()!=bank.resolve():
            raise ValueError('Official CPU scores changed index/export')
        if args['log_shard']!=i or args['log_shards']!=parts:raise ValueError('CPU log partitions changed')
        evaluators.append({k:v for k,v in ident.items() if k!='arguments'})
        group=read(part/'scenes.csv');logs=sorted({r['log'] for r in index});assigned=set(logs[i::parts])
        if set(group)!={r['token'] for r in index if r['log'] in assigned}:raise ValueError('CPU partition population mismatch')
        groups.append(group);csv_hashes[str(i)]=digest(part/'scenes.csv')
    if any(e!=evaluators[0] for e in evaluators):raise ValueError('Mixed official evaluator versions')
    export=identity['export'];export_hash=identity_hash(export)
    for shard in range(export['world_size']):
        state=json.loads((bank/f'shard_{shard}.json').read_text())
        if state['status']!='complete' or state['completed']!=len(index[shard::export['world_size']]) or state['identity_sha256']!=export_hash:
            raise ValueError('GPU export shard incomplete or changed')
    rows=merge_population(index,groups)
    for row in rows:
        meta=json.loads((bank/'predictions'/(row['token']+'.json')).read_text())
        if meta['identity_sha256']!=export_hash:raise ValueError('Scene inference identity changed')
        if row['status']=='ok' and (meta['status']!='ok' or row['proposal_sha256']!=meta['proposal_sha256']):raise ValueError('Scored trajectory differs from prediction')
    keys=sorted(set().union(*(r.keys() for r in rows)));temporary=out/f'scenes.{os.getpid()}.tmp'
    with temporary.open('w') as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(rows)
    temporary.replace(out/'scenes.csv')
    failed=sum(r['status']!='ok' for r in rows)
    summary={'schema':'foresight_official_pdms_v1','scenes':len(rows),'logs':len({r['log'] for r in rows}),
             'failed':failed,'valid':failed==0,'PDMS':sum(r['score'] for r in rows)/len(rows),
             'zero_fraction':sum(r['score']==0 for r in rows)/len(rows),
             'metrics':{k:sum(r[k] for r in rows)/len(rows) for k in METRICS},
             'export_identity':export,'evaluator_identity':evaluators[0],'parts_csv_sha256':csv_hashes,
             'full_navtest':export['current_identity']['split']=='navtest' and not export['limit'],
             'diagnostic':export['checkpoint']['scope']!='formal' or bool(export['limit']),
             'failure_policy':'retain all requested rows, zero on failure, any failure invalidates benchmark',
             'environment':'official full traffic/route environment; no vehicle-only scoring filter',
             'evaluator_protocol':'official NAVSIM v1.1, one executed ego trajectory, no learned scorer/oracle'}
    atomic_json(out/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('devkit','metric-index','current-index','predictions','output','campaign-root','run-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1)
    p.add_argument('--timeout-seconds',type=int,default=21600);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not 0<=a.part<a.parts or not 1<=a.workers<=32 or a.timeout_seconds<=0:raise ValueError('Invalid bounded CPU allocation')
    with metered_run(a.campaign_root,a.run_id,0,{'kind':'official_CPU_PDMS','real_optimizer_updates':0}) as (meter,_,save):
        bank=Path(a.predictions);export=json.loads((bank/'identity.json').read_text())
        protocol=export['protocol']
        if protocol['precision']!='FP32' or protocol['scorer'] is not None or protocol['candidates_per_scene']!=1 or protocol['future_conditioning']:
            raise ValueError('Only current-camera FP32 single ego exports can be scored')
        if export['checkpoint']['scope'] not in ('formal','startup','small_fit'):raise ValueError('Foreign checkpoint campaign')
        if export['current_identity']['split']=='navtest' and (export['checkpoint']['scope']!='formal' or export['limit']):raise ValueError('Navtest diagnostic model forbidden')
        current=json.loads(Path(a.current_index).read_text());metric=json.loads(Path(a.metric_index).read_text())
        index=requested_population(current,metric,export)
        if a.parts>len({r['log'] for r in index}):raise ValueError('More CPU parts than logs creates empty scoring shards')
        identity={'schema':'foresight_official_scoring_v1','export':export,'export_file_sha256':digest(bank/'identity.json'),
                  'current_index_sha256':digest(a.current_index),'metric_index_sha256':digest(a.metric_index),
                  'devkit':str(Path(a.devkit).resolve()),'devkit_python_sha256':python_tree_digest(Path(a.devkit)/'navsim'),
                  'adapter_sha256':digest(__file__),'validated_cpu_worker_sha256':digest(Path(__file__).parents[1]/'local_interaction_mask_v2/score_async.py'),
                  'cpu_parts':a.parts}
        out=Path(a.output)
        if out.exists():
            if not a.resume or json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Scoring identity changed or output already exists')
        else:out.mkdir(parents=True);atomic_json(out/'identity.json',identity);atomic_json(out/'requested_index.json',index)
        part=out/'parts'/f'part_{a.part}';part.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,'-m','tools.local_interaction_mask_v2.score_async','--devkit',a.devkit,'--index',str(out/'requested_index.json'),
             '--predictions',str(bank),'--output',str(part),'--workers',str(a.workers),'--chunk','8',
             '--export-shards',str(export['world_size']),'--log-shard',str(a.part),'--log-shards',str(a.parts),'--timeout-seconds',str(a.timeout_seconds)]
        if export['current_identity']['split']=='navtest':cmd+=['--benchmark-navtest']
        env=dict(os.environ)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):env[key]='1'
        with (out/f'part_{a.part}.log').open('a') as log:
            code=subprocess.call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        summary=merge_parts(out,bank,index,identity,a.parts)
        if summary:meter.update(inference_scenes=summary['scenes'],failed=summary['failed']);save()
        if code or summary and not summary['valid']:raise RuntimeError('Official scoring failed; partial rows and diagnostics retained')


if __name__=='__main__':main()
