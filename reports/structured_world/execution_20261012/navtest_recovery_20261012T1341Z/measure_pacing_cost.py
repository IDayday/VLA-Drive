"""Bounded live scene-pacing comparison; always restore only our own override."""
from pathlib import Path
import argparse
import datetime
import fcntl
import json
import os
import statistics
import time
import uuid

ROOT = Path(__file__).resolve().parent
QUEUE = ROOT.parent/'navsim_navtest_dispatch_20261011T0225'
POLICY = QUEUE/'COHOST_COST_POLICY.json'
RUNS = {'G1': ROOT.parent/'formal_navsim_zt2_G1_G0_v3_io/G1_FULL_UNIFORM',
        'G3': ROOT.parent/'formal_navsim_local_G3_G2_v3_io/G3_EVENT_LOCAL'}
HOSTS = ['training-vla-zt', 'training-vlawm-zt2']


def write(path, data):
    tmp = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)


def snapshot():
    state = json.loads((QUEUE/'STATE.json').read_text()); key = state['request']['key']
    rows = {}
    for name, run in RUNS.items():
        p = run/'metrics.jsonl'
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-400000)); tail=f.read().splitlines()[1:]
        values=[json.loads(line) for line in tail][-32:]
        rows[name]={'update':values[-1]['updates'],
                    'mean_seconds':statistics.mean(v['seconds'] for v in values),
                    'median_seconds':statistics.median(v['seconds'] for v in values)}
    progress=json.loads((QUEUE/'jobs'/key/'scores/PROGRESS.json').read_text())
    return dict(unix=time.time(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                key=key, status=state['status'], training=rows,
                predictions=sum(1 for _ in (QUEUE/'jobs'/key/'records').glob('*.json')),
                score_progress=progress)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--duty',type=float,required=True)
    parser.add_argument('--seconds',type=int,default=240);args=parser.parse_args()
    assert .05 <= args.duty <= 1. and 60 <= args.seconds <= 300
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    record=ROOT/('PACING_PROBE_'+stamp+'.json');identity=str(uuid.uuid4())
    with (ROOT/'pacing_policy.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);old=json.loads(POLICY.read_text())
        assert 'pacing_cost_probe' not in old
        start=snapshot();assert start['status']=='INFERENCE_RUNNING'
        new=dict(old);new['owned_training_duty_by_host']=dict(old['owned_training_duty_by_host'])
        for host in HOSTS:new['owned_training_duty_by_host'][host]=args.duty
        new['pacing_cost_probe']={'id':identity,'seconds':args.seconds,'start_unix':time.time(),
                                 'purpose':'measure lost training updates per completed inference scene'}
        write(record,dict(status='RUNNING',start=start,duty=args.duty,prior_policy=old))
        write(POLICY,new)
    try:
        time.sleep(args.seconds)
        end=snapshot();elapsed=end['unix']-start['unix'];scenes=end['predictions']-start['predictions']
        baseline=json.loads((ROOT/'TRAINING_BASELINE.json').read_text())
        result=dict(status='MEASURED',start=start,end=end,duty=args.duty,seconds=elapsed,
                    completed_scenes=scenes,scenes_per_second=scenes/elapsed,
                    training_updates={k:end['training'][k]['update']-start['training'][k]['update'] for k in RUNS})
        result['lost_updates_vs_initial_baseline']={k:elapsed/baseline[k]['mean_seconds']-v
                                                   for k,v in result['training_updates'].items()}
        result['lost_updates_per_completed_scene']={k:v/scenes if scenes else None
                                      for k,v in result['lost_updates_vs_initial_baseline'].items()}
        write(record,result);print(json.dumps(result),flush=True)
    finally:
        with (ROOT/'pacing_policy.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);current=json.loads(POLICY.read_text())
            if current.get('pacing_cost_probe',{}).get('id') != identity:
                raise RuntimeError('Foreign policy edit; cannot overwrite its pacing')
            for host in HOSTS:
                current['owned_training_duty_by_host'][host]=old['owned_training_duty_by_host'][host]
            current.pop('pacing_cost_probe');write(POLICY,current)
            write(record.with_suffix('.RESTORED.json'),dict(restored=True,utc=snapshot()['utc']))


if __name__=='__main__':main()
