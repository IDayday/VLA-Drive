"""Qualify cross-host flock, atomic publication and generation fencing.

Only a fresh shared_lock_validation_* directory is touched. All workers are
bounded stdlib CPU processes; production locks, GPUs and signals are unused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ART = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
BASE = ART/'navtest_dispatch_20261011'
POLICY = ART/'NAVSIM_EVALUATION_POLICY_20261011.json'
LOCAL = socket.gethostname().removesuffix('-worker-0')


def require(ok, message):
    if not ok: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorize(host):
    task = read('/mnt/project/server_dispatch_policy.json')['task_authorizations']['structured_world_fgtr_round1']
    policy = read(POLICY)
    require(host in task['allowed_hosts'] and host in policy['inference_allowed_hosts'], 'Host unauthorized: '+host)


def scope(output):
    output = output.resolve()
    require(output.parent == BASE.resolve() and output.name.startswith('shared_lock_validation_'), 'Use the isolated qualification directory')
    return output


def atomic(path, value):
    path = Path(path); tmp = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('x') as stream:
        json.dump(value, stream, sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


@contextmanager
def locked(output, nonblocking=False):
    with (output/'advisory.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def publish(output, generation, revision):
    with locked(output):
        current = read(output/'LEASE.json')['generation']
        if current != generation:
            return {'accepted': False, 'fenced': True, 'proposed_generation': generation, 'current_generation': current}
        payload = hashlib.sha256(str(revision).encode()).hexdigest()*512
        atomic(output/'PUBLICATION.json', {'generation': generation, 'revision': revision, 'payload': payload})
        return {'accepted': True, 'generation': generation, 'revision': revision, 'sha256': sha(output/'PUBLICATION.json')}


def publication(output):
    value = read(output/'PUBLICATION.json')
    require(value['payload'] == hashlib.sha256(str(value['revision']).encode()).hexdigest()*512, 'Partial or inconsistent atomic publication')
    return value


def identity(output):
    st = output.stat(); lock = (output/'advisory.lock').stat()
    mount = subprocess.check_output(['findmnt', '--json', '-T', str(output), '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS'], text=True)
    return {'host': LOCAL, 'mount': json.loads(mount), 'directory_stat': {'device': st.st_dev, 'inode': st.st_ino},
            'lock_stat': {'device': lock.st_dev, 'inode': lock.st_ino}, 'worker_source': str(Path(__file__).resolve()),
            'worker_source_sha256': sha(__file__), 'python': sys.executable}


def worker(output, packet):
    output = scope(output); authorize(LOCAL)
    require(packet['host'] == LOCAL, 'Worker is on a different host')
    mode = packet['mode']
    if mode == 'blocked':
        try:
            with locked(output, nonblocking=True): pass
        except BlockingIOError: return {'blocked': True, **identity(output)}
        return {'blocked': False, **identity(output)}
    if mode == 'takeover':
        with locked(output, nonblocking=True):
            old = read(output/'LEASE.json')['generation']; new = old+1
            atomic(output/'LEASE.json', {'generation': new, 'owner': LOCAL})
        return {'acquired': True, 'generation': new, 'identity': identity(output)}
    if mode == 'publish': return publish(output, packet['generation'], packet['revision'])
    if mode == 'read': return {'publication': publication(output), 'lease': read(output/'LEASE.json'), 'identity': identity(output)}
    if mode == 'observe':
        atomic(output/('OBSERVER_READY_'+LOCAL+'.json'), {'pid': os.getpid(), 'host': LOCAL})
        deadline = time.monotonic()+10
        while not (output/'ATOMIC_START.json').exists():
            require(time.monotonic() < deadline, 'Atomic observer barrier timed out'); time.sleep(.01)
        revisions = set()
        for _ in range(200):
            value = publication(output); require(value['generation'] == packet['generation'], 'Atomic generation changed')
            revisions.add(value['revision']); time.sleep(.015)
        return {'samples': 200, 'revisions_seen': sorted(revisions), 'all_reads_valid': True, 'host': LOCAL}
    if mode == 'append':
        for i in range(packet['entries']):
            with locked(output):
                require(read(output/'LEASE.json')['generation'] == packet['generation'], 'Append writer lost its generation')
                critical = output/'CRITICAL_OWNER.json'
                with critical.open('x') as marker: json.dump({'host': LOCAL, 'slot': packet['slot']}, marker)
                try:
                    time.sleep(.002)
                    with (output/'APPEND.jsonl').open('a') as stream:
                        stream.write(json.dumps({'id': LOCAL+'_'+str(packet['slot'])+'_'+str(i), 'host': LOCAL,
                                                 'slot': packet['slot'], 'entry': i, 'generation': packet['generation']})+'\n')
                        stream.flush(); os.fsync(stream.fileno())
                finally: critical.unlink()
        return {'host': LOCAL, 'slot': packet['slot'], 'entries': packet['entries'], 'exclusive_critical_section': True}
    raise RuntimeError('Unknown owned CPU worker mode')


def invoke(source, output, host, mode, **values):
    authorize(host)
    command = ['/usr/bin/python3', '-B', str(source), '--worker', '--output', str(output)]
    if host != LOCAL:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(command)]
    result = subprocess.run(command, input=json.dumps({'host': host, 'mode': mode, **values}),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def qualify(args):
    output = scope(args.output); output.mkdir(parents=True, exist_ok=False)
    source = output/'VALIDATE_SHARED_LEASES.py'; source.write_bytes(Path(__file__).read_bytes())
    atomic(output/'SOURCE.json', {'origin': str(Path(__file__).resolve()), 'sha256': sha(source),
        'policy_file': str(POLICY), 'policy_sha256': sha(POLICY), 'operator_python': sys.executable,
        'started_unix': time.time(), 'worker_or_GPU_signals': [], 'GPU_commands': [], 'production_locks': []})
    checks = {}
    try:
        with locked(output):
            atomic(output/'LEASE.json', {'generation': 1, 'owner': LOCAL})
            with ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
                blocked = list(pool.map(lambda h: invoke(source, output, h, 'blocked'), args.hosts))
            require(all(v['blocked'] for v in blocked), 'Remote acquired while local held flock')
            require(all(v['worker_source_sha256'] == sha(source) for v in blocked), 'Remote source differs')
            checks['remote_nonblocking_rejects_local_held_lock'] = blocked
        takeovers = []
        for host in args.hosts:
            result = invoke(source, output, host, 'takeover')
            require(read(output/'LEASE.json') == {'generation': result['generation'], 'owner': host}, 'Remote generation publication not visible locally')
            takeovers.append(result)
        checks['remote_acquisition_generation_visible_locally'] = takeovers
        generation = read(output/'LEASE.json')['generation']
        require(generation == 1+len(args.hosts), 'Generation did not advance')
        require(publish(output, generation, 0)['accepted'], 'Current generation was rejected')
        with ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
            observers = [pool.submit(invoke, source, output, host, 'observe', generation=generation) for host in args.hosts]
            deadline = time.monotonic()+10
            while not all((output/('OBSERVER_READY_'+h+'.json')).exists() for h in args.hosts):
                require(time.monotonic() < deadline, 'Remote observer startup timed out'); time.sleep(.01)
            atomic(output/'ATOMIC_START.json', {'generation': generation})
            for revision in range(1, 25):
                require(publish(output, generation, revision)['accepted'], 'Atomic writer was fenced'); time.sleep(.02)
            observations = [v.result() for v in observers]
        require(all(v['all_reads_valid'] and 24 in v['revisions_seen'] and len(v['revisions_seen']) >= 2 for v in observations),
                'Remote atomic publication visibility/consistency failed')
        checks['atomic_rename_observations'] = observations
        with ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
            reads = list(pool.map(lambda h: invoke(source, output, h, 'read'), args.hosts))
        require(all(v['publication'] == publication(output) and v['lease'] == read(output/'LEASE.json') for v in reads),
                'Final shared publication differs across hosts')
        checks['final_atomic_publication_shared'] = [{'host': v['identity']['host'], 'generation': v['publication']['generation'],
            'revision': v['publication']['revision'], 'mount_identity': v['identity']} for v in reads]
        before = sha(output/'PUBLICATION.json')
        fenced = [invoke(source, output, args.hosts[0], 'publish', generation=generation-1, revision=999),
                  publish(output, 1, 999)]
        require(all(v['fenced'] and not v['accepted'] for v in fenced) and sha(output/'PUBLICATION.json') == before,
                'Stale generation published over current owner')
        checks['stale_generations_fenced_without_publication'] = fenced
        jobs = [(host, slot) for host in [LOCAL]+args.hosts for slot in range(2)]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            results = list(pool.map(lambda j: invoke(source, output, j[0], 'append', generation=generation, slot=j[1], entries=25), jobs))
        rows = [json.loads(v) for v in (output/'APPEND.jsonl').read_text().splitlines()]
        expected = {h+'_'+str(slot)+'_'+str(i) for h, slot in jobs for i in range(25)}
        require(len(rows) == len(expected) and {v['id'] for v in rows} == expected and all(v['generation'] == generation for v in rows),
                'Concurrent append was lost, duplicated or foreign')
        checks['concurrent_locked_append'] = {'workers': results, 'rows': len(rows), 'unique_rows': len(expected),
            'journal_sha256': sha(output/'APPEND.jsonl'), 'exclusive_critical_sections_passed': True}
        atomic(output/'AGGREGATE.json', {'passed': True, 'checks': checks, 'local_mount_identity': identity(output),
            'hosts': [LOCAL]+args.hosts, 'source_sha256': sha(source), 'finished_unix': time.time(),
            'limits': 'Qualification of advisory cooperative flock and atomic replace on this shared mount; stale generation fencing requires the same locked compare-and-publish protocol. No crash/durability or arbitrary uncooperative-writer protection claimed.'})
        print(json.dumps({'passed': True, 'evidence': str(output/'AGGREGATE.json'), 'source': str(source), 'append_rows': len(rows)}))
    except BaseException as error:
        atomic(output/'FAILED.json', {'passed': False, 'error': repr(error), 'stderr': getattr(error, 'stderr', None), 'checks': checks})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--hosts', nargs='+', default=['training-vlawm-zt3', 'training-vlawm-zt4'])
    args = parser.parse_args(); authorize(LOCAL)
    if args.worker: print(json.dumps(worker(args.output, json.load(sys.stdin)))); return
    require(LOCAL == 'training-vla-zt' and len(set(args.hosts)) == len(args.hosts) and LOCAL not in args.hosts, 'Invalid qualification placement')
    for host in args.hosts: authorize(host)
    qualify(args)


if __name__ == '__main__': main()
