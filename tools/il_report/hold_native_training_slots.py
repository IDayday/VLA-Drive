"""Protect one verified native invocation; existing cohosts finish naturally.

CPU only. Reuses the established D guardian's four-namespace lease mechanism.
No GPU contexts, process signals, ownership broadening or trainer changes.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time


STOP = False


def halt(*_):
    global STOP
    STOP = True


def write(path, value):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--invocation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    a = p.parse_args();a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output/'guardian.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(a.source.resolve()))
    from starVLA.rl.qwen_world_compare.contracts import allowed_hosts
    from starVLA.rl.qwen_world_compare.resources import remote
    live = json.loads((a.invocation/'RUNNING.json').read_text())
    host, uuids = live['host'], live['gpu_uuids']
    if len(uuids) != 8 or len(set(uuids)) != 8 or live['lease_slot'] not in ('primary','shared'):
        raise ValueError('Expected exact eight-card native invocation')
    def verified():
        code = "import pathlib,json;p=pathlib.Path('/proc')/str(%d);s=(p/'stat').read_text().split()if p.exists()else[];print(json.dumps(bool(s and s[2]not in('Z','X')and int(s[21])==%d)))" % (live['worker_pid'],live['worker_start_ticks'])
        return host in allowed_hosts() and json.loads(remote(host,['/usr/bin/python3','-c',code],timeout=30))
    if not verified() or (a.invocation/'RESULT.json').exists():
        raise ValueError('Native invocation is no longer active')
    slots = dict(primary='runtime_gpu_leases',shared='runtime_gpu_shared_workers',
                 research='runtime_gpu_research_slots',evaluation='runtime_gpu_research_evaluation_slots')
    wanted = [Path('/mnt/project')/folder/(u+'.lock') for slot,folder in slots.items()
              if slot != live['lease_slot'] for u in uuids]
    remaining, held = list(wanted), []
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,halt)
    def publish(status):
        write(a.output/'STATE.json',dict(status=status,pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),at=time.time(),
            invocation=str(a.invocation),host=host,held=len(held),required=len(wanted),
            native_slot=live['lease_slot'],CUDA_contexts=0,model_signals=0))
    try:
        last_check = 0
        publish('WAITING_FOR_EXISTING_COHOSTS')
        while not STOP and not (a.invocation/'RESULT.json').exists() and not (a.output/'RELEASE').exists():
            if time.monotonic()-last_check >= 30:
                if not verified():break
                last_check=time.monotonic()
            changed=False
            for path in list(remaining):
                path.parent.mkdir(exist_ok=True);stream=path.open('a')
                try:fcntl.flock(stream,fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:stream.close();continue
                held.append(stream);remaining.remove(path);changed=True
            if changed:publish('HELD' if not remaining else 'WAITING_FOR_EXISTING_COHOSTS')
            time.sleep(5 if remaining else 30)
    finally:
        for stream in held:stream.close()
        publish('RELEASED')


if __name__ == '__main__':main()
