"""Read-only rolling throughput monitoring; cumulative averages are never step timings."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from ..contracts import require
from ..io import atomic_json


def training_efficiency(directory, warmup=3, maximum_seconds=None, window=200):
    path = Path(directory) / 'steps.jsonl'
    raw = path.read_bytes()
    # A writer can be appending the final line. Preserve all complete records;
    # malformed committed lines are errors, not silently discarded observations.
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines = lines[:-1]
    rows = [json.loads(line) for line in lines if line.strip()]
    require(window > 0 and warmup >= 0, 'invalid timing window/warmup')
    require(len(rows)>warmup, 'insufficient optimizer updates for steady-state timing')
    # A new launch or an explicit fork starts a new measurement segment.
    start = 0
    for i in range(1,len(rows)):
        if (rows[i]['optimizer_step'] <= rows[i-1]['optimizer_step'] or
            rows[i].get('launch_id') != rows[i-1].get('launch_id') or
            rows[i].get('runtime_identity_hash') != rows[i-1].get('runtime_identity_hash')):
            start = i
    measured = rows[start+warmup:][-window:]
    require(measured, 'insufficient updates after latest launch warmup')
    require(all('compute_update_seconds' in r for r in measured),
            'per-update timing missing: cumulative seconds_per_update is not a step duration')
    samples = [r['compute_update_seconds'] for r in measured]
    require(all(np.isfinite(samples)) and min(samples)>0, 'invalid timing evidence')
    p50,p95 = map(float,np.quantile(samples,[.5,.95]))
    def summary(key):
        values=[r[key] for r in measured if key in r]
        return {'count':len(values),'p50':float(np.median(values)), 'p95':float(np.quantile(values,.95)),
                'mean':float(np.mean(values))} if values else None
    finite = all(np.isfinite(r.get('gradient_norm',0)) and all(np.isfinite(v) for v in r.get('losses',{}).values()) for r in measured)
    global_batch = [r.get('actual_global_batch', r.get('global_batch_actual')) for r in measured]
    return {'status':'PROFILED' if finite else 'NONFINITE_TRAINING', 'updates':len(rows),
        'latest_optimizer_step':rows[-1]['optimizer_step'], 'window_updates':len(measured), 'window_limit':window,
        'window_start_step':measured[0]['optimizer_step'], 'warmup_excluded_per_launch':warmup,
        'p50_update_seconds':p50, 'p95_update_seconds':p95, 'finite_loss_and_gradient':finite,
        'max_allowed_seconds':maximum_seconds,
        'throughput_gate_passed':maximum_seconds is not None and p95<=maximum_seconds and finite,
        'compute':summary('compute_update_seconds'), 'wall_including_previous_save':summary('step_wall_seconds'),
        'loader_wait':summary('data_wait_seconds'), 'data_load':summary('data_load_seconds'),
        'checkpoint_seconds_in_window':sum(r.get('previous_checkpoint_seconds',0) for r in measured),
        'samples_per_compute_second':sum(global_batch)/sum(samples) if all(global_batch) else None,
        'raw_log':str(path), 'read_snapshot_hash':hashlib.sha256(raw).hexdigest(),
        'log_age_seconds':max(0., time.time()-path.stat().st_mtime),
        'automatic_semantic_changes':False, 'scientific_gain':'UNTESTED'}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--directory',required=True); p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--max-seconds-per-update',type=float,required=True)
    p.add_argument('--watch-seconds',type=int,default=0); p.add_argument('--window',type=int,default=200)
    a=p.parse_args(); require(a.watch_seconds>=0 and a.max_seconds_per_update>0,'monitor budgets')
    from ..resources import qualify
    qualify('cpu')
    end=time.monotonic()+a.watch_seconds
    while True:
        try:
            result=training_efficiency(a.directory,a.warmup,a.max_seconds_per_update,a.window)
            atomic_json(Path(a.directory)/'EFFICIENCY.json',result)
            print(json.dumps(result),flush=True)
        except (FileNotFoundError,ValueError) as e:
            result={'status':'WAITING_FOR_OWN_TRAINING_EVIDENCE','error':str(e)}
            print(json.dumps(result),flush=True)
        if time.monotonic()>=end: break
        time.sleep(min(30,end-time.monotonic()))
    return 0 if result.get('throughput_gate_passed') else 2


if __name__=='__main__': raise SystemExit(main())
