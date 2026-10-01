"""Release only our verified pressure reserves while registered training is live.

Independent CPU observer; never modifies or signals training, never starts GPU
workloads, and leaves pressure policy to the existing allocator when training is
paused or complete. Addresses container host-PID visibility in the old allocator.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

from .navtest_milestones import release_owned_pressure
from .navtest_schedule import atomic, lease, load_registration, read, source_identity


def eligible(state, model, now):
    return (state.get('status') == 'RUNNING' and
            state.get('identity') == model['run_identity'] and
            state.get('source_sha') == model['training_source_sha'] and
            0 <= now-state.get('updated_unix', 0) < 30 and
            state.get('host') == model['hostname'])


def check(config, model, root):
    if socket.gethostname() != model['hostname']:
        raise ValueError('Wrong authorized host')
    state = read(Path(model['training_run'])/'status.json')
    job = Path(root)/model['arm'];before = read(job/'pressure_releases.json') if (job/'pressure_releases.json').exists() else []
    releases = release_owned_pressure(config, model, job,
        reason='User-authorized pressure release for active formal training; no trainer signal') if eligible(state, model, time.time()) else before
    result = {'host': socket.gethostname(), 'arm': model['arm'], 'unix': time.time(),
              'training_status': state['status'], 'update': state['completed'],
              'eligible': eligible(state, model, time.time()), 'new_releases': len(releases)-len(before),
              'total_releases': len(releases), 'training_signals': 0, 'unrelated_signals': 0}
    atomic(job/'status.json', result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--registration', required=True);p.add_argument('--output', required=True)
    p.add_argument('--guard-source-sha', required=True);p.add_argument('--arm', choices=('C0','C1','C4'))
    p.add_argument('--interval', type=int, default=30);p.add_argument('--max-seconds', type=int, default=172800)
    a = p.parse_args()
    source = Path(__file__).resolve().parents[2]
    if source_identity(source) != a.guard_source_sha or a.interval < 5 or a.max_seconds <= 0:
        raise ValueError('Guard source/configuration changed')
    reg = load_registration(a.registration);cfg = reg['config'];root = Path(a.output)
    if a.arm:
        model = next(m for m in cfg['models'] if m['arm'] == a.arm)
        print(json.dumps(check(cfg,model,root)), flush=True);return
    with lease(root/'observer.lock'):
        started = time.time();results = []
        def query(model):
            cmd = ['/usr/bin/python3','-m','tools.full_foresight.training_pressure_guard',
                   '--registration',str(Path(a.registration).resolve()),'--output',str(root.resolve()),
                   '--guard-source-sha',a.guard_source_sha,'--arm',model['arm']]
            if model['host'] != 'local':
                cmd = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',model['host'],
                       'cd '+shlex.quote(str(source))+' && exec '+shlex.join(cmd)]
            try:
                return json.loads(subprocess.check_output(cmd,cwd=source,text=True,timeout=45))
            except Exception as error:
                return {'arm':model['arm'],'error':repr(error),'unix':time.time()}
        with ThreadPoolExecutor(max_workers=3) as pool:
            while time.time()-started < a.max_seconds and not (root/'STOP').exists():
                results = list(pool.map(query,cfg['models']))
                atomic(root/'status.json',{'status':'RUNNING','pid':os.getpid(),'source':a.guard_source_sha,
                    'registration':reg['identity'],'unix':time.time(),'hosts':results,'optimizer_updates':0})
                if all(r.get('training_status') in ('COMPLETE','FAILED') for r in results):
                    break
                time.sleep(a.interval)
        atomic(root/'status.json',{'status':'FINISHED','pid':os.getpid(),'source':a.guard_source_sha,
               'registration':reg['identity'],'unix':time.time(),'hosts':results,'optimizer_updates':0})


if __name__ == '__main__':main()
