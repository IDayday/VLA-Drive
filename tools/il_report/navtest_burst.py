"""High-concurrency scene sharding around qualified, unchanged IL evaluators.

Worker count changes only token partitioning. Model batch=1, token-bound root
noise, single K10 shift5 trajectory and the canonical scorer remain unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def prechild_refusal(result):
    """No model execution occurred; this is capacity waiting, not bad science."""
    return (result.get('status') == 'REFUSED_RESOURCE'
            and result.get('child_not_started') is True and result.get('returncode') == 75)


def _sealed(path):
    value = json.loads(Path(path).read_text())
    body = dict(value); saved = body.pop('sha256', None)
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if not saved or saved != actual:
        raise ValueError('Changed sealed evaluation metadata: '+str(path))
    return value


def rank_completion(request, name):
    """EXIT0 alone is not a completed fixed-population evaluator shard."""
    if not name.startswith('rank') or not name[4:].isdigit():
        raise ValueError('Full rank completion is only defined for rankN jobs')
    rank = int(name[4:]); path = request.output/(name+'.json')
    if not path.exists():return False
    identity = _sealed(request.output/'identity.json')
    row = json.loads(path.read_text())
    expected = len(identity['tokens'][rank::request.world])
    return (0 <= rank < request.world and row.get('status') == 'COMPLETE'
            and row.get('rank') == rank and row.get('world') == request.world
            and row.get('identity') == identity['sha256']
            and row.get('new', 0)+row.get('reused', 0) == expected)


def review_resource_boundary(request, name):
    """Requeue a proven native deadline return, retaining its exact scene cache.

    Arbitrary user stops, forced exits, failed inference and malformed identities
    remain explicit review cases. The new worker still validates every reused
    scene and resumes the SAME rank/world/source/export/seed.
    """
    if not name.startswith('rank') or not name[4:].isdigit():return False
    spec = request.jobs[name]; receipt = Path(spec['receipt_dir'])
    result_path = receipt/'RESULT.json'
    if not result_path.exists():return False
    result = json.loads(result_path.read_text())
    if (result.get('status') != 'EXITED' or result.get('returncode') != 0
            or result.get('forced_expiry')
            or result.get('migration_required') is not True or result.get('temporary') is not True
            or not (receipt/'STOP_REQUESTED').exists()):return False
    row_path = request.output/(name+'.json')
    if not row_path.exists():return False
    row = json.loads(row_path.read_text());rank = int(name[4:])
    identity = _sealed(request.output/'identity.json')
    registration = _sealed(request.reg)
    if (identity.get('cohort') != 'NAVTEST' or identity.get('qualification') is not False
            or identity.get('seed') != 2101 or identity.get('count') != 12146
            or len(identity.get('tokens', [])) != 12146
            or identity.get('inference') != 'single_candidate_cleanx_K10_shift5'
            or registration.get('model_sha256') != identity.get('model_sha256')
            or registration.get('protocol_sha256') != identity.get('protocol', {}).get('sha256')
            or row.get('status') != 'STOPPED_AT_SCENE_BOUNDARY'
            or row.get('rank') != rank or row.get('world') != request.world
            or row.get('identity') != identity['sha256'] or row.get('host') != result.get('host')
            or spec.get('source') != str(request.source) or result.get('cwd') != str(request.source)
            or result.get('command') != spec.get('command')):return False
    command = spec['command']
    def argument(flag):
        return command[command.index(flag)+1] if command.count(flag) == 1 and command.index(flag)+1 < len(command) else None
    expected = {'-m':'tools.qwen_world_compare.evaluate', '--output':str(request.output),
                '--export':str(request.export), '--rank':str(rank), '--world':str(request.world),
                '--seed':'2101', '--cohort':'NAVTEST'}
    if any(argument(k) != v for k,v in expected.items()):return False
    total = len(identity['tokens'][rank::request.world]);done = row.get('new',0)+row.get('reused',0)
    # The evaluator polls the immutable guard itself. It may publish its
    # boundary exit BEFORE the worker's next timer writes STOP_REQUESTED.
    guard = result.get('guard_unix'); deadline = result.get('deadline_unix')
    stopped = result.get('stop_requested_unix')
    if (not isinstance(guard, (int,float)) or not isinstance(deadline, (int,float))
            or not guard < deadline or not 0 <= rank < request.world or not 0 <= done < total
            or not guard <= row.get('end_unix',0) <= result.get('end_unix',0) <= deadline
            or (stopped is not None and not guard <= stopped <= result.get('end_unix',0))):
        return False
    write(receipt/'OPERATOR_REQUEUE.json', dict(status='REVIEWED_TERMINAL_REQUEUE',
        at=time.time(), spec_sha256=hashlib.sha256((receipt/'spec.json').read_bytes()).hexdigest(),
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        rank_receipt_sha256=hashlib.sha256(row_path.read_bytes()).hexdigest(),
        identity_sha256=identity['sha256'], reason='NATIVE_DEADLINE_SCENE_BOUNDARY_RETURN',
        rank=rank, world=request.world, completed_scenes_to_reuse=done, remaining_scenes=total-done,
        source_export_world_rank_seed_unchanged=True, old_attempt_preserved=True,
        cached_scores_not_modified=True, no_fake_COMPLETE_or_RESULT=True))
    return True


def summary_log(directory):
    """A failed summary log is evidence, not an obstacle to a legal resume."""
    path = directory/'summary.log'
    if path.exists():path = directory/f'summary_resume_{time.time_ns()}_{os.getpid()}.log'
    return path.open('x')


def notify_completed_report(label, root, summary):
    report = Path('/mnt/project/qwen-world-il-research-20261030/reporting')
    write(report/'refresh_request.json', dict(id='complete:'+label+':'+summary['sha256'],
        reason='full_IL_Navtest_completed', scientific_result=True, administrative=False,
        root=str(root), label=label, scenes=12146, at=time.time()))


def resource_worker_overrides(source):
    # Text I/O must stay UTF-8 even if a native scorer changes LC_CTYPE to C.
    # No numerical/backend/precision variables are changed.
    return dict(PYTHONPATH=str(source), PYTHONUTF8='1')


def review_unacknowledged_startup(spec, remote):
    """Review a proven pre-child crash; never infer success from a dead PID."""
    receipt = Path(spec['receipt_dir'])
    if any((receipt/n).exists() for n in ('RUNNING.json','RESULT.json','stdout.log')):
        return False
    pending = receipt/'STARTUP_PENDING.json'
    operator = receipt/'operator.json'
    if not pending.exists() or not operator.exists():return False
    if time.time()-json.loads(pending.read_text())['at'] < 60:return False
    evidence = json.loads(operator.read_text())
    program = '''from pathlib import Path
import json,fcntl
s=__DATA__;d=Path(s['receipt']);o=s['operator'];p=Path('/proc')/str(o['pid'])
live=p.exists() and int((p/'stat').read_text().split()[21])==o['start_ticks'] and (p/'stat').read_text().split()[2]not in ['Z','X']
matched=[]
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  a=[x for x in (q/'cmdline').read_text().split(chr(0)) if x];st=(q/'stat').read_text().split()
  if st[2]in ['Z','X']or '-c'in a:continue
  owned=('--receipt-root'in a and a[a.index('--receipt-root')+1]==str(d))or a==s['command']
  inherited=(b'WORLD_POSTCOT_STOP_FILE='+str(d/'STOP_REQUESTED').encode()+bytes([0]))in(q/'environ').read_bytes()
  if owned or inherited:matched.append(dict(pid=int(q.name),start_ticks=int(st[21])))
 except(OSError,ValueError):pass
f=(d/'worker.lock').open('a');free=True
try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:free=False
print(json.dumps(dict(worker_alive=live,matching_processes=matched,worker_lock_free=free,
 receipt_children=[n for n in ['RUNNING.json','RESULT.json','stdout.log']if(d/n).exists()])))
'''.replace('__DATA__',repr(dict(receipt=str(receipt),operator=evidence,command=spec['command'])))
    proof = json.loads(remote(spec['host'], ['/usr/bin/python3','-c',program]))
    if (proof.get('worker_alive') is not False or proof.get('matching_processes') != []
            or proof.get('worker_lock_free') is not True or proof.get('receipt_children') != []):
        return False
    transport = (receipt/'transport.log').read_text()
    # This legacy exception was raised inside _gpu_admission before Popen.
    # Other missing receipts/unknown errors demand explicit review.
    if ('in _gpu_admission' not in transport or
            not transport.rstrip().endswith('ValueError: Sharing requires an explicitly authorized host and measured valid headroom')):
        raise RuntimeError('Dead startup without a child receipt; review required: '+str(receipt))
    write(receipt/'OPERATOR_REQUEUE.json',dict(status='REVIEWED_TERMINAL_REQUEUE',
        at=time.time(),spec_sha256=hashlib.sha256((receipt/'spec.json').read_bytes()).hexdigest(),
        transport_sha256=hashlib.sha256((receipt/'transport.log').read_bytes()).hexdigest(),
        host=spec['host'],process_exit_proof=proof,original_attempt_preserved=True,
        child_not_started=True,model_forward_calls=0,scorer_calls=0,
        reason='Legacy finite-host sharing rejection before Popen; retry with current qualified worker'))
    return True


def protected_hosts():
    policy=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    return set(policy.get('placement_preferences',{}).get('qwen_world_IL_B_D',{}).get('hosts',[]))


def cpu_host(delegate):
    hosts=[h for h in delegate.allowed_hosts()if h not in protected_hosts()]
    preferred=['training-vlawm-tsinghua-zt2','training-vlawm-zt2','training-vlawm-zt']
    return next((h for h in preferred if h in hosts),hosts[0])


def vacant_slot(uuid, *, family, root=Path('/mnt/project')):
    slots=[('shared','runtime_gpu_shared_workers')]
    if family=='four':slots.append(('primary','runtime_gpu_leases'))
    for slot,folder in slots:
        path=root/folder/(uuid+'.lock');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as stream:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            return slot
    return None


def capacity_probe(delegate,host):
    result=delegate.probe(host)
    if (result.get('worker_python_exists') is not None and result.get('MemAvailable_bytes') is not None
            and all('total_MiB'in g for g in result['gpus'])
            and all('used_MiB'in o for o in result['owners'])):
        return result
    # Legacy World probe has a different RAM field and no used/total memory.
    # Obtain real accounting through THAT SAME task-authorized remote entry.
    code='''import subprocess,json,os
from pathlib import Path
g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus=[dict(index=int(i),uuid=u.strip(),total_MiB=int(t),free_MiB=int(f),utilization=int(v))for i,u,t,f,v in(x.split(',')for x in g.splitlines())]
o=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True);owners=[]
for line in o.splitlines():
 p,u,m=line.split(',');path=Path('/proc')/p.strip()/'cmdline';args=path.read_bytes().decode().split(chr(0))if path.is_file()else[]
 prefixes=['/mnt/project/qwen-','/mnt/project/VLA-Drive-qwen','/mnt/project/VLA-AD/experiments/recogdrive','/mnt/project/recogdrive']
 owners.append(dict(pid=int(p),uuid=u.strip(),used_MiB=int(m)if m.strip().isdigit()else None,visible=bool(args),known_project=any(a.startswith(prefix)for a in args for prefix in prefixes)))
free=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))
limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
if limit.exists()and current.exists()and limit.read_text().strip()!='max':free=min(free,max(0,int(limit.read_text())-int(current.read_text())))
print(json.dumps(dict(gpus=gpus,owners=owners,MemAvailable_bytes=free,CPU_affinity=len(os.sched_getaffinity(0)),load1=os.getloadavg()[0],worker_python_exists=Path('/mnt/project/flow-safe-improvement-20261013/venv/bin/python').is_file())))'''
    return json.loads(delegate.remote(host,['/usr/bin/python3','-c',code]))


def adopt_four_jobs(request):
    """Adopt exact existing commands, including still-running shards, once."""
    for path in sorted((request.root/'jobs').glob('*_attempt*/spec.json')):
        spec = json.loads(path.read_text())
        if spec['source'] != str(request.source) or spec['source_sha'] != request.source_sha:
            raise ValueError('Cross-source shard adoption is not allowed')
        logical, number = path.parent.name.rsplit('_attempt', 1)
        command = spec['command']
        if (logical not in ['export', 'parity', *('rank'+str(i) for i in range(request.world))]
                or '--output' not in command
                or command[command.index('--output')+1] != str(request.export if logical == 'export' else request.output)
                or spec['receipt_dir'] != str(path.parent)):
            raise ValueError('Existing shard command/identity differs from this evaluation')
        reviewed = path.parent/'OPERATOR_REQUEUE.json'
        if reviewed.exists():
            decision = json.loads(reviewed.read_text())
            import hashlib
            if (decision.get('status') != 'REVIEWED_TERMINAL_REQUEUE'
                    or decision.get('spec_sha256') != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError('Unbound operator requeue decision')
            if int(number) >= request.attempts.get(logical, 0):
                request.attempts[logical] = int(number)
                request.jobs.pop(logical, None)
            continue
        if int(number) > request.attempts.get(logical, 0):
            request.attempts[logical] = int(number)
            request.jobs[logical] = spec


def write(path, value):
    path = Path(path); tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


class Pool:
    """Cached administrative probes; each worker still repeats its admission."""
    def __init__(self, delegate, admission_source=None):
        self.delegate = delegate; self.sample = None; self.at = 0
        self.admission_source = admission_source

    def __getattr__(self, key):
        return getattr(self.delegate, key)

    def available(self, min_free_MiB=32768):
        if self.sample is not None and time.monotonic()-self.at < 15:
            return self.sample
        from starVLA.rl.qwen_world_postcot.resources import gpu_activity_issue
        hosts = [h for h in self.delegate.allowed_hosts() if h not in protected_hosts()]
        def get(host):
            try:
                observed = capacity_probe(self.delegate,host)
                if self.admission_source:
                    # The old qualified worker has a narrower owner allowlist
                    # than the generic scheduler. Use its EXACT read-only guard;
                    # do not loosen sharing or touch the immutable worker.
                    program = '''import sys,json
sys.path.insert(0,__SOURCE__)
from tools.qwen_world_compare.job_worker import _admission_sample,_known_owners,shared_training_host
from starVLA.rl.qwen_world_compare.contracts import authorization,require_dispatch
host=require_dispatch(admit_long_job=True)
sharing_allowed=shared_training_host(host,authorization()[0])
devices,owners=_admission_sample();accepted=[];rejected={}
for index,device in devices.items():
 try:_known_owners(owners,{index:device['uuid']},[index]);accepted.append(device['uuid'])
 except Exception as error:rejected[device['uuid']]=str(error)
print(json.dumps(dict(accepted=accepted,rejected=rejected,sharing_allowed=sharing_allowed)))'''.replace('__SOURCE__',repr(str(self.admission_source)))
                    admission = json.loads(self.delegate.remote(host, ['/usr/bin/python3', '-c', program]))
                    observed['qualified_worker_admission'] = admission
                return host, observed, None
            except Exception as error:
                return host, None, str(error)
        samples, errors = [], {}
        for _ in range(2):
            with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as executor:
                rows = list(executor.map(get, hosts))
            samples.append({h: d for h, d, e in rows if e is None})
            errors.update({h: e for h, d, e in rows if e is not None})
        routes = []; seen = set()
        for host, after in samples[1].items():
            before = samples[0].get(host)
            if not before or [g['uuid'] for g in before['gpus']] != [g['uuid'] for g in after['gpus']]:
                continue
            after['available_memory_bytes'] = after['MemAvailable_bytes']
            for a, b in zip(before['gpus'], after['gpus']):
                if (b['uuid'] in seen or min(a['free_MiB'], b['free_MiB']) < min_free_MiB
                        or any(self.admission_source and s['qualified_worker_admission'].get('sharing_allowed') is False for s in(before,after))
                        or any(self.admission_source and b['uuid'] not in s['qualified_worker_admission']['accepted'] for s in (before,after))
                        or not after['worker_python_exists']
                        or any(not o['visible'] for s in (before, after) for o in s['owners'] if o['uuid'] == b['uuid'])
                        or any(not self.admission_source and o.get('known_project')is not True for s in(before,after)for o in s['owners']if o['uuid']==b['uuid'])
                        or gpu_activity_issue(a, before['owners']) or gpu_activity_issue(b, after['owners'])):
                    continue
                seen.add(b['uuid'])
                routes.append(dict(host=host, gpu=b['index'], uuid=b['uuid'],
                                   free_MiB=min(a['free_MiB'], b['free_MiB'])))
        self.sample = routes, dict(first=samples[0], second=samples[1], errors=errors)
        self.at = time.monotonic()
        return self.sample


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--family', required=True, choices=['four', 'world'])
    parser.add_argument('--workers', type=int, choices=[16, 32], default=32)
    parser.add_argument('--host-load-limit',type=int,choices=(2,4,8),default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker-source', type=Path,
                        help='Qualified resource-only worker; predictor/scorer still run from --source')
    args = parser.parse_args(); root = args.root.resolve()
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'owner.json').exists() and not args.resume:
            raise FileExistsError('Existing burst invocation must be reviewed; never duplicate')
        if args.resume:
            previous = json.loads((root/'owner.json').read_text())
            attempt = root/'operator_attempts'/str(previous['pid']); attempt.mkdir(parents=True, exist_ok=False)
            (attempt/'owner.json').write_bytes((root/'owner.json').read_bytes())
            if (root/'failure.json').exists():
                (root/'failure.json').rename(attempt/'failure.json')
        sys.path.insert(0, str(args.source.resolve()))
        if args.family == 'four':
            from tools.qwen_world_compare.interim_owner import InterimRequest as Qualified
        else:
            from tools.qwen_world_postcot.latest_snapshot_owner import SnapshotRequest as Qualified
        from tools.qwen_world_postcot.user_interim_owner import Request as Base
        dispatch_lock = Path('/mnt/project/runtime_periodic_navtest_dispatch.lock')
        worker_source = (args.worker_source or args.source).resolve()
        if not (worker_source/'tools/qwen_world_compare/job_worker.py').is_file():
            raise ValueError('Missing explicitly qualified admission worker')

        class Burst(Qualified):
            def done(self, name):
                if args.family == 'four':
                    terminal = super().done(name)
                    if terminal is True and name.startswith('rank'):
                        return rank_completion(self, name)
                    return terminal
                if name not in self.jobs: return None
                path = Path(self.jobs[name]['receipt_dir'])/'result.json'
                if not path.exists(): return None
                result = json.loads(path.read_text())
                return result.get('status') == 'PROCESS_COMPLETE' and result.get('exit_code') == 0

            def route(self, routes, inventory, *, parity=False):
                guard = dispatch_lock.open('a'); fcntl.flock(guard, fcntl.LOCK_EX)
                try:
                    used={u for j in self.active()for u in j['gpu_uuids']}
                    result=None
                    for route in sorted(routes,key=lambda r:-r['free_MiB']):
                        if route['host']in protected_hosts() or route['uuid']in used:continue
                        if self.pin.get('evaluation_hosts')and route['host']not in self.pin['evaluation_hosts']:continue
                        if route['free_MiB']<(49152 if parity else 32768):continue
                        host=inventory['second'][route['host']]
                        if host['available_memory_bytes']<128*1024**3:continue
                        loading=sum(j['host']==route['host'] and self.done(name)is None
                            and (not(self.output/(name+'.json')).exists() or json.loads((self.output/(name+'.json')).read_text()).get('new',0)<1)
                            for name,j in self.jobs.items())
                        limit=min(args.host_load_limit,max(1,int((host['available_memory_bytes']-128*1024**3)//(32*1024**3))))
                        if loading>=limit:continue
                        slot=vacant_slot(route['uuid'],family=args.family)
                        if slot is None:continue
                        # Qualified worker repeats cohost/UUID/headroom checks
                        # under this exact slot. This is a new free slot, never
                        # a stolen lock or an actor/scorer change.
                        result=dict(route,lease_slot=slot);break
                    if result:
                        self.dispatch_guard = guard
                        return result
                except BaseException:
                    guard.close(); raise
                guard.close()
                return None

            def submit(self, name, module, argv, route=None):
                try:
                    if args.family == 'four':
                        number=self.attempts.get(name,0)+1;self.attempts[name]=number
                        receipt=self.root/'jobs'/f'{name}_attempt{number:03d}';receipt.mkdir(parents=True,exist_ok=False)
                        module='tools.qwen_world_compare.export'if name=='export'else'tools.qwen_world_compare.evaluate'
                        host=route['host']if route else cpu_host(self.resources)
                        command=[self.resources.MODEL_PYTHON,'-u','-m',module,*argv]
                        spec=dict(host=host,command=command,source=str(self.source),source_sha=self.source_sha,
                            gpu_uuids=[route['uuid']]if route else[],receipt_dir=str(receipt),
                            execution_only=dict(lease_slot=route.get('lease_slot')if route else None,
                                host_load_limit=args.host_load_limit, worker_source=str(worker_source)))
                        self.jobs[name]=spec;write(receipt/'spec.json',spec)
                        invocation=['/usr/bin/python3','-u','-m','tools.qwen_world_compare.job_worker',
                            '--receipt-root',str(receipt),'--command-json',json.dumps(command),'--cwd',str(self.source)]
                        if route:
                            minimum=49152 if name=='parity'else 32768
                            invocation+=['--gpus',str(route['gpu']),'--expected-gpu-uuids-json',json.dumps([route['uuid']]),
                                '--share-min-free-mib',str(minimum),'--lease-slot',route['lease_slot']]
                        code='''import subprocess,os,json
from pathlib import Path
s=__SPEC__;d=Path(s['directory']);f=open(d/'transport.log','ab',buffering=0)
p=subprocess.Popen(s['invocation'],cwd=s['source'],env=dict(os.environ,**s['environment_overrides']),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,start_ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21]))))'''.replace('__SPEC__',repr(dict(invocation=invocation,directory=str(receipt),source=str(worker_source),environment_overrides=resource_worker_overrides(worker_source))))
                        write(receipt/'operator.json',json.loads(self.resources.remote(host,['/usr/bin/python3','-c',code])))
                    else:
                        # Platform-only adaptation: unique retry receipt dirs.
                        # Commands, worker, scopes, memory guards and register()
                        # are the qualified World implementation unchanged.
                        number = self.burst_attempts.get(name, 0)+1
                        self.burst_attempts[name] = number
                        receipt = self.root/'jobs'/f'{name}_attempt{number:03d}'
                        receipt.mkdir(parents=True, exist_ok=False)
                        argv = ['--user-snapshot-registration' if a == '--user-interim-registration' else a for a in argv]
                        spec = dict(source=str(self.source), source_sha=self.source_sha,
                            host=route['host'] if route else cpu_host(self.resources),
                            gpus=[route['gpu']] if route else [], gpu_uuids=[route['uuid']] if route else [],
                            min_free_MiB=(49152 if name == 'rank0' else 32768) if route else 0,
                            world_shared_lease=bool(route), module=module,
                            command=[self.resources.MODEL_PYTHON, '-u', '-m', module, *argv], receipt_dir=str(receipt))
                        write(receipt/'spec.json', spec); self.jobs[name] = spec
                        write(receipt/'operator.json', self.resources.launch(receipt/'spec.json', str(self.source)))
                    if route:
                        receipt = Path(self.jobs[name]['receipt_dir'])
                        deadline = time.monotonic()+180
                        while not any((receipt/n).exists() for n in ('RUNNING.json', 'process.json', 'RESULT.json', 'result.json')):
                            if time.monotonic() > deadline:
                                write(receipt/'STARTUP_PENDING.json', dict(at=time.time(),
                                    state='UNCONFIRMED_STARTUP_WAIT', no_redispatch=True,
                                    reason='Existing durable intent retained; process metadata may be delayed'))
                                break
                            time.sleep(.1)
                finally:
                    if getattr(self, 'dispatch_guard', None):
                        self.dispatch_guard.close(); self.dispatch_guard = None

        pins = json.loads((root/'PINNED.json').read_text())
        label, pin = next(iter(pins['models'].items()))
        if len(pins['models']) != 1:
            raise ValueError('One fixed checkpoint per burst')
        directory = root/label; directory.mkdir(exist_ok=args.resume)
        request = Burst(directory, pin); request.world = args.workers
        request.burst_attempts = {}; retry_after = {}
        if pin.get('reuse_export') and not request.export.exists():
            export = Path(pin['reuse_export'])
            if not (export/'COMPLETE').exists(): raise ValueError('Missing qualified reused export')
            request.export.symlink_to(export, target_is_directory=True)
        if args.resume:
            if args.family == 'four':
                adopt_four_jobs(request)
            else:
                for path in sorted((directory/'jobs').glob('*_attempt*/spec.json')):
                    spec = json.loads(path.read_text())
                    if spec['source'] != str(request.source) or spec['source_sha'] != request.source_sha:
                        raise ValueError('Cross-source adoption requires a separate receipt')
                    logical, number = path.parent.name.rsplit('_attempt', 1)
                    if int(number) > request.burst_attempts.get(logical, 0):
                        request.burst_attempts[logical] = int(number)
                        request.jobs[logical] = spec
            write(root/'OPERATOR_ADOPTION.json', dict(at=time.time(), previous_owner=previous,
                adopted_jobs=list(request.jobs), completed_shards_not_reexecuted=True,
                existing_registration_and_scene_identity_preserved=True))
        # The façade changes administrative probing only, no worker imports.
        if args.family == 'world':
            request.resources = Pool(request.resources)
        else:
            # Match this task's grant rather than the older SourceFlow grant.
            # The probe and worker guards still independently reject sharing.
            pool = Pool(request.resources, admission_source=worker_source)
        owner = dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            status='RUNNING', at=time.time(), source=str(args.source),
            workers=args.workers, model_batch=1, family=args.family, full_population=12146,
            noise_seed='seed_for(native_qwen_sourceflow_ODE,2101,scene_token)',
            model_and_scoring_code_unchanged=True)
        owner['qualified_resource_worker_source'] = str(worker_source)
        write(root/'owner.json', owner)
        try:
            if args.family == 'world':
                if not (request.export/'COMPLETE').exists():
                    request.submit('export', 'tools.qwen_world_postcot.export',
                                   ['--checkpoint', pin['checkpoint'], '--output', str(request.export)])
                while not all(request.done('rank'+str(i)) is True for i in range(args.workers)):
                    for name in list(request.jobs):
                        if request.done(name) is not False: continue
                        receipt = Path(request.jobs[name]['receipt_dir'])
                        result = json.loads((receipt/'result.json').read_text())
                        prechild = (not (receipt/'process.json').exists() and not (receipt/'worker.log').exists()
                            and ('BlockingIOError' in result.get('error', '') or 'Insufficient measured headroom' in result.get('error', '')))
                        if prechild:
                            delay=min(300,30*request.burst_attempts[name])
                            write(receipt/'PRECHILD_RESOURCE_CLASSIFICATION.json', dict(
                                child_not_started=True, scorer_calls=0, physical_inferences=0,
                                retained_failed_attempt=True, retry_after_seconds=delay))
                            retry_after[name] = time.monotonic()+delay
                            request.jobs.pop(name); continue
                        while request.active(): time.sleep(15)
                        raise RuntimeError('World worker failed; retain all receipts and completed scene records: '+str(receipt))
                    exported = (request.export/'COMPLETE').exists() and ('export' not in request.jobs or request.done('export') is True)
                    if exported:
                        if not request.reg.exists(): request.register()
                        parity = request.proof.exists() and json.loads(request.proof.read_text()).get('status') == 'PASS'
                        pending = [rank for rank in (range(args.workers) if parity else [0])
                                   if 'rank'+str(rank) not in request.jobs
                                   and time.monotonic() >= retry_after.get('rank'+str(rank), 0)]
                        routes, inventory = request.resources.available() if pending else ([], {})
                        for rank in pending:
                            name = 'rank'+str(rank)
                            if name in request.jobs or time.monotonic() < retry_after.get(name, 0): continue
                            route = request.route(routes, inventory, parity=rank == 0)
                            if route is None: break
                            argv = ['--export', str(request.export), '--output', str(request.output), '--cohort', 'NAVTEST',
                                '--data', '/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST', '--seed', '2101',
                                '--rank', str(rank), '--world', str(args.workers), '--user-interim-registration', str(request.reg)]
                            if rank == 0: argv += ['--native-parity-output', str(request.proof)]
                            request.submit(name, 'tools.qwen_world_postcot.evaluate', argv, route)
                        write(root/'burst_progress.json', dict(at=time.time(), submitted=list(request.jobs), workers=args.workers))
                    time.sleep(15)
                with summary_log(directory) as out:
                    subprocess.run([request.resources.MODEL_PYTHON, '-m', 'tools.qwen_world_postcot.summarize',
                        '--evaluation', str(request.output), '--output', str(directory/'SUMMARY.json')],
                        cwd=request.source, stdout=out, stderr=subprocess.STDOUT, check=True)
                from tools.qwen_world_postcot.user_interim_owner import validate_complete_summary
                summary = json.loads((directory/'SUMMARY.json').read_text())
                counts = validate_complete_summary(summary, json.loads((request.output/'identity.json').read_text()),
                    json.loads(Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json').read_text()))
                write(directory/'COMPLETE.json', dict(status='COMPLETE', PDMS=summary['PDMS'], scenes=12146,
                    logs=counts['manifest_logs'], physical_logs=counts['physical_logs'], all_owned_workers_exited=True))
                request.publish(summary)
            else:
                if 'export' not in request.jobs:
                    request.submit('export', '', ['--checkpoint', pin['checkpoint'], '--output', str(request.export)])
                while not all(request.done('rank'+str(i)) is True for i in range(args.workers)):
                    for name in list(request.jobs):
                        if request.done(name) is None and review_unacknowledged_startup(request.jobs[name],request.resources.remote):
                            retry_after[name]=time.monotonic()+30
                            request.jobs.pop(name);continue
                        if request.done(name) is False:
                            result = json.loads((Path(request.jobs[name]['receipt_dir'])/'RESULT.json').read_text())
                            if prechild_refusal(result):
                                delay = min(300, 30*request.attempts[name])
                                retry_after[name] = time.monotonic()+delay
                                write(Path(request.jobs[name]['receipt_dir'])/'RESOURCE_WAIT_CLASSIFICATION.json', dict(
                                    child_not_started=True,model_forward_calls=0,scorer_calls=0,
                                    retry_after_seconds=delay,reason=result.get('reason'),failed_attempt_preserved=True))
                                request.jobs.pop(name); continue
                            if review_resource_boundary(request, name):
                                retry_after[name] = time.monotonic()+30
                                request.jobs.pop(name); continue
                            raise RuntimeError('Qualified worker failed; evidence retained: '+str(request.jobs[name]['receipt_dir']))
                    if request.done('export') is True:
                        if not request.reg.exists(): request.register()
                        parity = (request.proof.exists() and json.loads(request.proof.read_text()).get('status') == 'PASS'
                                  and request.done('parity') is True)
                        pending = [rank for rank in (range(args.workers) if parity else [0])
                                   if ('rank'+str(rank) if parity else 'parity') not in request.jobs
                                   and time.monotonic() >= retry_after.get('rank'+str(rank) if parity else 'parity',0)]
                        routes, inventory = pool.available() if pending else ([], {})
                        for rank in pending:
                            name = 'rank'+str(rank) if parity else 'parity'
                            if name in request.jobs: continue
                            route = request.route(routes, inventory, parity=not parity)
                            if route is None: break
                            argv = ['--export', str(request.export), '--output', str(request.output), '--cohort', 'NAVTEST',
                                '--data', '/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST', '--rank', str(rank),
                                '--world', str(args.workers), '--seed', '2101', '--user-interim-registration', str(request.reg)]
                            if rank == 0: argv += ['--native-parity-output', str(request.proof)]
                            if not parity: argv += ['--parity-only']
                            request.submit(name, '', argv, route)
                        write(root/'burst_progress.json', dict(at=time.time(), submitted=list(request.jobs), workers=args.workers))
                    time.sleep(15)
                with summary_log(directory) as out:
                    subprocess.run([request.resources.MODEL_PYTHON, '-m', 'tools.qwen_world_compare.evaluate', '--summarize',
                        '--evaluation', str(request.output), '--output', str(directory/'SUMMARY.json')], cwd=request.source,
                        stdout=out, stderr=subprocess.STDOUT, check=True)
                summary = json.loads((directory/'SUMMARY.json').read_text())
                if summary['scenes'] != 12146 or summary['invalid_scenes'] or summary['qualification']:
                    raise ValueError('Incomplete full Navtest')
                write(directory/'COMPLETE.json', dict(status='COMPLETE', PDMS=summary['PDMS'], scenes=12146,
                    all_owned_workers_exited=True))
            owner.update(status='COMPLETE', ended_unix=time.time())
            write(root/'COMPLETE.json', dict(status='COMPLETE', models=[label], all_owned_workers_exited=True))
            notify_completed_report(label, root, summary)
        except BaseException:
            owner.update(status='FAILED', error=traceback.format_exc(), ended_unix=time.time())
            write(root/'failure.json', owner)
            raise
        finally:
            write(root/'owner.json', owner)


if __name__ == '__main__':
    main()
