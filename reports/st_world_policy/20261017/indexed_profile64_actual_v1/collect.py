"""Read final small PROFILE evidence, never model/cache tensors or new GPU work."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path('/mnt/project/st-world-policy-20261016/profile/world_indexed_provider_profile32_updates64_v1')
RUN=ROOT/'runs/indexed_provider_profile32_updates64_v1'
SOURCE='2d5e9483a8c8bc80b71348211756f3cab6aa21b1'


def stats(values):
    if not values:return {'n':0,'mean':None,'p50':None,'p95':None,'minimum':None,'maximum':None}
    if not all(type(x) in (int,float) and math.isfinite(x) for x in values):
        raise ValueError('Invalid measurement')
    ordered=sorted(values)
    def q(p):
        at=(len(ordered)-1)*p;lo=int(at);hi=min(lo+1,len(ordered)-1)
        return ordered[lo]+(at-lo)*(ordered[hi]-ordered[lo])
    return {'n':len(values),'mean':statistics.mean(values),'p50':q(.5),'p95':q(.95),
        'minimum':ordered[0],'maximum':ordered[-1]}


def timing(rows,launch_unix):
    # The producer records existing status.updated_unix, never iteration start.
    # Observed polling wall cannot substitute for the trainer's own timestamp.
    unique={}
    for row in rows:
        if type(row.get('completed')) is not int or row['completed']<0:
            raise ValueError('Invalid observed completed step')
        if type(row.get('updated_unix')) not in (int,float) or not math.isfinite(row['updated_unix']):
            raise ValueError('Invalid original status timestamp')
        key=(row['completed'],row['updated_unix'])
        if key in unique and unique[key]!=row:
            raise ValueError('Conflicting same original status observation')
        unique[key]=row
    ordered=sorted(unique.values(),key=lambda r:r['updated_unix'])
    first={}
    for row in ordered:first.setdefault(row['completed'],row)
    intervals=[];checkpoint_events=[];gaps=[]
    previous=None
    for n,row in sorted(first.items()):
        if previous is not None:
            pn,pr=previous
            if n!=pn+1:gaps.append({'from':pn,'to':n,'missing':list(range(pn+1,n))})
            else:
                seconds=row['updated_unix']-pr['updated_unix']
                if seconds<=0:raise ValueError('Original status time is not increasing')
                intervals.append({'from_step':pn,'to_step':n,'seconds':seconds,
                    'checkpoint_tag_changed':row.get('tag')!=pr.get('tag')})
        previous=(n,row)
    for old,new in zip(ordered,ordered[1:]):
        if old['completed']==new['completed'] and old.get('tag')!=new.get('tag'):
            checkpoint_events.append({'step':old['completed'],'seconds':new['updated_unix']-old['updated_unix'],
                'before_tag':old.get('tag'),'after_tag':new.get('tag')})
    normal=[r for r in intervals if not r['checkpoint_tag_changed']]
    steady=[r for r in normal if r['to_step']>8]
    rss=[(n,r['rank_rss_bytes']) for n,r in sorted(first.items()) if r.get('rank_rss_bytes') is not None]
    if any(type(v) is not int or v<0 for _,v in rss):raise ValueError('Invalid rank RSS')
    rss_first=[v for n,v in rss if 1<=n<=8];rss_last=[v for n,v in rss if n>=57]
    return {'source':'trainer status.updated_unix; read-only observer, not polling observed_unix',
        'observed_step_ids':sorted(first),'missing_consecutive_intervals':gaps,
        'first_step_observed':min(first) if first else None,
        'cold_launch_to_status0_seconds':first[0]['updated_unix']-launch_unix if 0 in first else None,
        'consecutive_intervals':intervals,'normal_complete_post_update_intervals':stats([r['seconds'] for r in normal]),
        'warmup_excluded_first_updates':8,'post_warmup_complete_intervals':stats([r['seconds'] for r in steady]),
        'same_step_checkpoint_events':checkpoint_events,
        'rank_rss':{'scope':'actual single training rank VmRSS, not process-tree sum',
            'first_status_per_step_bytes':rss,'all_observed_status_bytes':stats([r['rank_rss_bytes'] for r in ordered if r.get('rank_rss_bytes') is not None]),
            'first8_update_median_bytes':statistics.median(rss_first) if rss_first else None,
            'last8_update_median_bytes':statistics.median(rss_last) if rss_last else None,
            'last8_minus_first8_median_bytes':statistics.median(rss_last)-statistics.median(rss_first) if rss_first and rss_last else None}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('RESULT.json'))
    args=p.parse_args()
    if not (ROOT/'EXIT_initial.json').is_file():
        raise SystemExit('EXIT_RECEIPT_PENDING: no final report; do not dispatch or retry')
    total=0;pins={}
    def read(path,lines=False):
        nonlocal total
        path=Path(path)
        if path.suffix not in ('.json','.jsonl') or path.stat().st_size>32<<20:
            raise ValueError('Small JSON evidence only')
        if total+path.stat().st_size>128<<20:raise ValueError('Aggregate metadata read budget exceeded')
        body=path.read_bytes();total+=len(body)
        pins[str(path)]=hashlib.sha256(body).hexdigest()
        def invalid(x):raise ValueError('Nonfinite JSON '+x)
        return [json.loads(s,parse_constant=invalid) for s in body.decode().splitlines() if s] if lines else json.loads(body,parse_constant=invalid)
    def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    exited=read(ROOT/'EXIT_initial.json');launch=read(ROOT/'LAUNCH_initial.json')
    cfg=read(ROOT/'CONFIG.json');reg=read(ROOT/'REGISTRATION.json')
    if reg['source_sha']!=SOURCE or launch['source_sha']!=SOURCE:raise ValueError('Wrong runtime evidence')
    if reg['execution']!=cfg['execution'] or cfg['execution']['scope']!='PROFILE':raise ValueError('Wrong execution')
    common={'schema':'st_indexed_profile64_actual_v1','source_sha':SOURCE,
        'formal_approved':False,'scientific_gain':'UNTESTED','exact_resume_tested':False,
        'fullpopulation_throughput_claim':False,'runtime_or_model_changed_by_observer':False}
    if exited['returncode']!=0:
        output={**common,'status':'ACTUAL_RUN_FAILED','exit':exited,
            'failure':read(RUN/'FAILURE.json') if (RUN/'FAILURE.json').exists() else None,'automatic_retry':False}
    else:
        result=read(RUN/'RESULT.json');steps=read(RUN/'steps.jsonl',True)
        costs=read(RUN/'checkpoint_costs.jsonl',True);provider=read(RUN/'teacher_provider_rank0.json')
        selected=read(reg['artifacts']['data_manifest']['path']);descriptor=read(reg['artifacts']['teacher_manifest']['path'])
        observed=read(ROOT/'SUPPLY_STATUS_SAMPLES.jsonl',True);observer=read(ROOT/'SUPPLY_STATUS_OBSERVER.json')
        if observer['launch_sha256']!=pins[str(ROOT/'LAUNCH_initial.json')]:
            raise ValueError('Observer belongs to another launch')
        engine=read(RUN/'official_runtime/updated_engine_rank0.json')
        for name in ('config','data_manifest','teacher_manifest'):
            a=reg['artifacts'][name]
            if pins[a['path']]!=a['sha256']:raise ValueError('Changed registered artifact '+name)
        if (result['completed']!=64 or result['new_optimizer_updates']!=64 or result['exposure']!=128
                or result['status']!='PAUSED' or result.get('pause_reason')!='BUDGET_PAUSED'
                or [s['step'] for s in steps]!=list(range(1,65))):
            raise ValueError('Run did not complete exactly64 updates/128 exposures')
        totals={};ids=[]
        for step in steps:
            if (not step['fp32_master_update_verified'] or step['sampled_changed_elements']<=0
                    or not math.isfinite(step['maximum_sampled_change']) or step['maximum_sampled_change']<=0
                    or not math.isfinite(step['gradient_norm']) or step['gradient_norm']<=0
                    or not all(math.isfinite(v) for v in step['losses'].values())):
                raise ValueError('Invalid loss/gradient/master evidence')
            if len(step['sample_ids_by_rank'])!=1 or len(step['sample_ids_by_rank'][0])!=2:raise ValueError('Wrong sample batch')
            ids.extend(step['sample_ids_by_rank'][0])
            for k,v in step['counts'].items():
                if type(v) is not int or v<0:raise ValueError('Invalid denominator')
                totals[k]=totals.get(k,0)+v
        exposure=Counter(ids);available={r['token'] for r in selected['scenes']}
        if totals!=result['counts'] or set(exposure)!=available or len(exposure)!=32 or set(exposure.values())!={4}:
            raise ValueError('Exact32/four exposures/count join mismatch')
        fixed=provider['fixed_compute']
        if (provider!=result['teacher_provider_runtime'] or provider['provider_identity']!=descriptor['provider']['identity']
                or provider['failures']!=0 or provider['tensor_disk_bytes']!=0
                or provider['peak_resident_bytes']>provider['resident_limit_bytes']
                or fixed['compute_image_slots']!=8*fixed['calls']
                or fixed['compute_image_slots']!=fixed['requested_images']+fixed['padding_image_slots']):
            raise ValueError('Invalid fixed8 provider evidence')
        if engine['master_dtype']!='float32' or engine['execution_dtype']!='bfloat16' or engine['moment_elements']<=0:
            raise ValueError('Wrong actual execution/master state')
        native=[]
        for path in sorted((RUN/'checkpoints').glob('*/COMPLETE.json')):
            receipt=read(path)
            if digest({k:v for k,v in receipt.items() if k!='receipt_identity'})!=receipt['receipt_identity']:
                raise ValueError('Wrong receipt identity')
            if receipt['state']['identity']!=result['identity'] or receipt['world_size']!=1:raise ValueError('Wrong native lineage')
            for name,info in receipt['files'].items():
                if Path(name).name!=name or (path.parent/name).stat().st_size!=info['bytes']:raise ValueError('Native size mismatch')
            native.append({'step':receipt['state']['completed'],'tag':path.parent.name,
                'native_tensor_bytes':sum(v['bytes'] for v in receipt['files'].values()),
                'directory_logical_bytes':sum(f.stat().st_size for f in path.parent.iterdir() if f.is_file()),
                'files':receipt['files'],'payload_rehashed_by_observer':False})
        if [n['step'] for n in native]!=[64] or [n['step'] for n in costs]!=[64]:raise ValueError('Expected one native final save')
        tt=timing(observed,launch['started_unix'])
        steady=tt['post_warmup_complete_intervals'];timed=tt['normal_complete_post_update_intervals']
        tt['post_warmup_samples_per_second']=2/steady['mean'] if steady['n'] else None
        tt['checkpoint_inclusive_observed_mean_seconds']=(sum(v['seconds'] for v in tt['consecutive_intervals'])+sum(x['seconds'] for x in costs))/64 if len(tt['consecutive_intervals'])==64 and all(not r['checkpoint_tag_changed'] for r in tt['consecutive_intervals']) else None
        tt['checkpoint_inclusive_note']='64 observed normal intervals plus actual one native-save wall; coldload excluded; null if incomplete/contaminated'
        output={**common,'status':'BOUNDED_EXACT32_64_UPDATE_COMPLETE','execution':cfg['execution'],
            'result':result,'provider':provider,'updated_engine':engine,'checkpoint_costs':costs,'native_checkpoints':native,
            'timing':tt,'observer':observer,'sample_exposure_counts':dict(sorted(exposure.items())),
            'loss_summary':{k:stats([s['losses'][k] for s in steps]) for k in steps[0]['losses']},
            'gradient_norm':stats([s['gradient_norm'] for s in steps]),
            'master_changes':{'all64_verified':True,'sampled_changed_elements':stats([s['sampled_changed_elements'] for s in steps]),
                'maximum_sampled_change':stats([s['maximum_sampled_change'] for s in steps])},
            'internal_timer_diagnostic_not_complete_throughput':{'slowest_rank_seconds':stats([s['slowest_rank_seconds'] for s in steps]),
                'data_seconds':stats([s['per_rank_profile'][0][1] for s in steps])},
            'cuda_step_peaks':{'allocated_bytes':stats([s['per_rank_profile'][0][2] for s in steps]),
                'reserved_bytes':stats([s['per_rank_profile'][0][3] for s in steps])},
            'all_steps':steps,'coldload_compared_to_2update_speed':False,'new_teacher_disk_bytes':0}
    output.update(metadata_read_bytes=total,observed_unix=time.time(),pins=pins)
    if args.output.exists():raise FileExistsError('Preserve old observation')
    args.output.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':output['status'],'metadata_read_bytes':total,'output':str(args.output)}))


if __name__=='__main__':main()
