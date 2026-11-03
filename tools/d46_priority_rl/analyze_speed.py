"""CPU-only, complete-collection throughput from actual eight-rank journals."""
import argparse
from datetime import datetime,timezone
import hashlib,json,math,statistics
from pathlib import Path

ROOT=Path('/mnt/project/d46-priority-rl-20261103')


def fp(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def rows(path):
    data=[];lines=path.read_text().splitlines()
    for index,line in enumerate(lines):
        if not line.strip():continue
        try:data.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise
            # A single writer may be appending the last row during this read.
    return data


def analyze(run,collections=100):
    root=Path(run['output']);identity=json.loads((root/'identity.json').read_text());cfg=identity['recipe']
    ranks=[rows(root/f'metrics_rank{i}.jsonl') for i in range(8)]
    m=cfg['inner_updates'];batch=cfg['global_scenes'];group=cfg['group']
    common=min(len(r) for r in ranks);end=(common//m)*m;start=end-collections*m
    if start<1:raise ValueError('Insufficient complete steady collections')
    windows=[r[start-1:end] for r in ranks]
    for rank,window in enumerate(windows):
        if [r['updates']for r in window]!=list(range(start,end+1)):raise ValueError('Journal prefix gap or duplicate')
        if any(r['run']!=identity['sha256'] for r in window):raise ValueError('Foreign scientific run')
        if len({r['invocation'] for r in window})!=1:raise ValueError('Window crosses a resume; do not count paused time')
        if any(r['active_batch']['scenes']!=batch or r['active_batch']['group']!=group for r in window):raise ValueError('Window changed B/G')
    if len({w[0]['invocation'] for w in windows})!=1:raise ValueError('Rank invocation mismatch')
    count=end-start
    end_to_end=max(w[-1]['at']-w[0]['at'] for w in windows)/count
    update_critical=[max(w[i]['update_seconds'] for w in windows) for i in range(1,len(windows[0]))]
    exposures=[w[-1]['scene_exposures']-w[0]['scene_exposures'] for w in windows]
    candidates=[w[-1]['PG_candidates']-w[0]['PG_candidates'] for w in windows]
    if len(set(exposures))!=1 or len(set(candidates))!=1:raise ValueError('Rank global accounting disagrees')
    if exposures[0]!=collections*batch or candidates[0]!=collections*batch*group:
        raise ValueError('Window is padded or does not contain declared fresh collections')
    # These collection fields repeat at every inner update. Count each once.
    fresh_indices=[i for i in range(1,len(windows[0])) if windows[0][i]['inner_completed']==1]
    generation=sum(max(w[i]['collection_seconds_generation'] for w in windows) for i in fresh_indices)/count
    scoring=sum(max(w[i]['collection_seconds_scoring'] for w in windows) for i in fresh_indices)/count
    current=min(r[-1]['updates'] for r in ranks)
    current_rows=[r[current-1] for r in ranks]
    if len({r['scene_exposures'] for r in current_rows})!=1:raise ValueError('Latest common rank exposure disagrees')
    exposure=current_rows[0]['scene_exposures'];first_gate=.5
    gate=m*math.ceil(first_gate*103288/batch)
    return dict(arm=run['arm'],host=run['jobs'][-1]['host'],parent_IL_updates=46000,
        B=batch,G=group,M=m,world=8,latest_common_updates=current,
        valid_scene_exposures=exposure,exposure_epoch=exposure/103288,
        window=dict(anchor_update=start,last_update=end,Adam_intervals=count,complete_fresh_collections=collections,
                    first_unix=min(w[0]['at'] for w in windows),last_unix=max(w[-1]['at'] for w in windows),
                    same_invocation=True,rank_prefix_sha256=[fp(w) for w in windows]),
        optimizer_seconds_mean=statistics.mean(update_critical),optimizer_seconds_median=statistics.median(update_critical),
        end_to_end_seconds_per_Adam=end_to_end,
        recorded_generation_and_transition_recompute_seconds_per_Adam=generation,
        recorded_main_scoring_wait_seconds_per_Adam=scoring,
        unassigned_seconds_per_Adam=end_to_end-statistics.mean(update_critical)-generation-scoring,
        fresh_scenes_per_Adam=exposures[0]/count,fresh_main_candidates_per_Adam=candidates[0]/count,
        fresh_scenes_per_second=exposures[0]/(count*end_to_end),fresh_main_candidates_per_second=candidates[0]/(count*end_to_end),
        first_NAVTEST=dict(epoch=first_gate,trigger_updates=gate,requests_already_written=[str(f) for f in (root/'evaluation_requests').glob('*.json')],
                          hours_to_trigger_if_recent_rate_persists=max(0,gate-current)*end_to_end/3600,
                          additional_evaluation_time_not_included=True),
        mechanism_stage=dict(D1_endpoint_credit_active=exposure>=2*103288 if run['arm']=='D1' else None,
                             D2_nominal_rescue_active=exposure>=.5*103288 if run['arm']=='D2' else None),
        journal_paths=[str(root/f'metrics_rank{i}.jsonl') for i in range(8)])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--collections',type=int,default=100);p.add_argument('--output',type=Path)
    a=p.parse_args();campaign=json.loads((a.root/'CAMPAIGN.json').read_text())
    result=[analyze(run,a.collections) for run in campaign['runs'].values()]
    evaluations=[]
    for path in (a.root/'evaluations').glob('*/SUMMARY.json'):
        value=json.loads(path.read_text())
        if value.get('status')=='COMPLETE' and value.get('scenes')==12146 and value.get('invalid_scenes')==0:
            evaluations.append(dict(path=str(path),PDMS=value.get('PDMS')))
    value=dict(schema='D46_measured_eight_rank_throughput_review_v1',status='COMPLETE_CURRENT_SNAPSHOT',
        generated_UTC=datetime.now(timezone.utc).isoformat(),population=103288,runs=result,
        complete_D46_NAVTEST_results=evaluations,IL46_NAVTEST_PDMS=89.4955209758546,
        definitions=dict(optimizer='Mean across Adam steps of the slowest rank forward/backward/clip/Adam/scheduler CUDA-synchronized update timer.',
            end_to_end='Slowest rank elapsed wall time /200 consecutive Adam intervals over100 complete collections in one invocation; includes collection, scoring, preparations, updates and saves within the window.',
            generation='Main SDE generation plus current/old/reference transition recompute. This timer excludes later FPO/AWM MC material preparation and later D2 auxiliary probes.',
            residual='Uninstrumented preparation/material storage, normalization/barriers, saves and auxiliary work; do not assign it entirely to one subsystem.'),
        limitations=['Current actual cohosted workload, not an isolated hardware or algorithm speedup benchmark.',
            'B/G/M differ from the earliest D version; one Adam update is not a fixed amount of data or compute.',
            'D1 stageII and D2 rescue have not activated in this snapshot; later throughput must be measured again.',
            'Forecasts assume the current workload persists and exclude subsequent full evaluation time.'])
    path=a.output or a.root/'analysis/D46_THROUGHPUT_REVIEW_20261103.json';path.parent.mkdir(parents=True,exist_ok=True)
    value['sha256']=fp(value);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)
    print(json.dumps(dict(output=str(path),rows=[dict(arm=r['arm'],updates=r['latest_common_updates'],epoch=round(r['exposure_epoch'],4),
        update=round(r['optimizer_seconds_mean'],3),end_to_end=round(r['end_to_end_seconds_per_Adam'],3))for r in result],complete_D46_NAVTEST=len(evaluations))))


if __name__=='__main__':main()
