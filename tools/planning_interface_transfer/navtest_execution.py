"""Allocation-only overlay for immutable A/V Navtest registrations.

The native exporter, checkpoint, four scene partitions and metric contract are
unchanged. A live trainer may coexist only when its exact registered run is
visible and sufficient device/host memory remains. No process is evicted.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from starVLA.model.modules.vehicle_joint.initialization import identity_hash
from tools.full_foresight.navtest_schedule import atomic, read, sha, source_identity


def load(path):
    r = read(path)
    if (r['schema'] != 'planning_interface_navtest_execution_v1'
            or r['identity'] != identity_hash({k:v for k,v in r.items() if k != 'identity'})
            or source_identity(r['source_worktree']) != r['source_sha']
            or r['world_size'] != 4 or r['max_parallel_future_tasks'] != 3):
        raise ValueError('Execution allocation/source changed')
    for p, expected in r['asset_hashes'].items():
        if sha(p) != expected:
            raise ValueError('Allocation provenance changed: '+p)
    for update, arms in r['allocations'].items():
        if int(update) not in (75000,80000,90000,100000):
            raise ValueError('Unrequested checkpoint')
        used = set()
        for arm, a in arms.items():
            if arm not in ('A_ACTION','A_NO_MAE','V_QUERY') or len(a['gpus']) != 4:
                raise ValueError('Invalid arm/partition count')
            if a['host'] not in r['authorized_hosts']:
                raise ValueError('Unregistered host')
            for card in a['gpus']:
                if card not in r['authorized_hosts'][a['host']]['allowed_gpus']:
                    raise ValueError('Unauthorized GPU')
                key = (a['host'],card)
                if key in used:
                    raise ValueError('Overlapping simultaneous allocations')
                used.add(key)
    return r


def allocation(r, update, arm):
    return r['allocations'][str(update)][arm]


def close_dead_export_meter(path, end_unix, reason):
    """Record confirmed termination, keeping the original failed-cost evidence.

    A killed native child cannot execute the meter's finally block. Its last
    saved RUNNING record otherwise charges GPU time forever. Observation time
    is an upper bound, not a fabricated exact time of death.
    """
    path=Path(path)
    if not path.exists(): return
    original=read(path)
    if original['status'] != 'RUNNING': return
    if (original['kind'] != 'foresight_current_camera_export' or original['gpu_count'] != 1
            or original['real_optimizer_updates'] != 0 or end_unix < original['start_unix']):
        raise ValueError('Only confirmed owned inference meters may be closed')
    receipt=path.parent/f'termination_receipt_{time.time_ns()}.json'
    atomic(receipt,dict(original=original,original_sha256=sha(path),reason=reason,
                       observed_end_unix=end_unix,cost_boundary='conservative observation upper bound'))
    original.update(status='FAILED',end_unix=end_unix,wall_seconds=end_unix-original['start_unix'],
        gpu_hours=(end_unix-original['start_unix'])/3600,error=reason,
        termination_receipt=str(receipt),cost_boundary='conservative observation upper bound')
    atomic(path,original)


def lease_main(args):
    r = load(args.allocation); a = allocation(r,args.update,args.arm)
    card = a['gpus'][args.rank]; host = r['authorized_hosts'][a['host']]
    if socket.gethostname() != host['hostname']:
        raise ValueError('Wrong allocation host')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.rank not in range(4):
        raise ValueError('Missing native command/invalid rank')
    handles, child = [], None
    try:
        for name in ('navtest_pdms','recogdrive_mtopd_research'):
            p = Path(f'/var/tmp/{name}_gpu{card}.lock'); f = p.open('a')
            handles.append(f)
            try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('Evaluation GPU lease busy',flush=True); return 75
        raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
                                      '--format=csv,noheader,nounits'],text=True)
        devices = {int(x.split(',')[0]):x.strip().split(', ') for x in raw.splitlines()}
        gpu = devices[card]; uuid = gpu[1]
        if uuid != a['gpu_uuids'][str(card)]:
            raise ValueError('Physical GPU identity changed')
        if int(gpu[2]) < r['minimum_free_mib']:
            print('Insufficient live GPU memory',flush=True); return 75
        mem = {k:int(v.split()[0]) for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < r['minimum_host_available_kib']:
            print('Insufficient host memory for FP32 master loading',flush=True); return 75
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                       '--format=csv,noheader,nounits'],text=True)
        # Only the specifically registered trainer may already use this card.
        # Invisible or foreign processes are rejected, even with free memory.
        for line in apps.splitlines():
            parts = [x.strip() for x in line.split(',')]
            if parts[0] != uuid: continue
            try: argv = Path('/proc')/parts[1]/'cmdline'; words = argv.read_bytes().split(b'\0')
            except OSError:
                print('GPU process ownership unavailable',flush=True); return 75
            words = [w.decode() for w in words if w]
            allowed = ('tools.foresight.train_student' in words and '--run-id' in words
                       and words[words.index('--run-id')+1] == host['co_resident_training_run_id'])
            if not allowed:
                print('GPU has a different live task; preserve it',flush=True); return 75
        env = dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=uuid)
        child = subprocess.Popen(command,cwd=args.cwd,env=env,start_new_session=True)
        print(json.dumps(dict(host=socket.gethostname(),physical_gpu=card,uuid=uuid,
            child_pid=child.pid,allocation=r['identity'],co_resident_training_run=host['co_resident_training_run_id'])),flush=True)
        interrupted = []
        for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            signal.signal(s,lambda received,frame:interrupted.append(received))
        while child.poll() is None:
            if interrupted:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL); child.wait()
                return 128+interrupted[0]
            time.sleep(.2)
        return child.returncode
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        for f in reversed(handles): f.close()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--allocation',required=True);p.add_argument('--arm',required=True)
    p.add_argument('--update',required=True,type=int);p.add_argument('--rank',required=True,type=int)
    p.add_argument('--cwd',required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    sys.exit(lease_main(p.parse_args()))


if __name__ == '__main__': main()
