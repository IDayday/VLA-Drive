"""One finite snapshot request after a prior owner releases its workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json


def predecessor_ready(root):
    root=Path(root);complete=root/'COMPLETE.json';publication=root/'PUBLICATION.json'
    if complete.exists():
        c=json.loads(complete.read_text())
        if c.get('status')!='COMPLETE' or c.get('all_owned_workers_exited')is not True:
            raise ValueError('Previous evaluation has not released all owned workers')
        if not publication.exists():raise ValueError('Previous completion lacks publication receipt')
        p=json.loads(publication.read_text())
        if p.get('status')!='PUBLISHED' or not p.get('commit') or p.get('commit')!=p.get('remote_SHA'):
            raise ValueError('Previous publication is not verified')
        return True
    if (root/'failure.json').exists():raise RuntimeError('Previous evaluation failed; retain evidence and review before queue takeover')
    return False


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);p.add_argument('--report-dir',required=True);a=p.parse_args()
    root=Path(a.root);source=Path(__file__).resolve().parents[2];pins=json.loads((root/'PINNED.json').read_text())
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'QUEUE_OWNER.json').exists() or (root/'owner.json').exists():raise FileExistsError('Do not duplicate the registered queue/evaluation owner')
    atomic_json(root/'QUEUE_OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        source=str(source),at=time.time(),predecessor=pins['predecessor_root']))
    try:
        while not predecessor_ready(pins['predecessor_root']):
            atomic_json(root/'QUEUE_STATUS.json',dict(stage='WAITING_PREDECESSOR_FULL_TEST',at=time.time()))
            time.sleep(30)
        atomic_json(root/'QUEUE_STATUS.json',dict(stage='EVALUATING_REGISTERED_MIDPOINTS',at=time.time(),models=list(pins['models'])))
        command=[os.environ.get('WORLD_MODEL_PYTHON','/mnt/project/flow-safe-improvement-20261013/venv/bin/python'),
            '-u','-m','tools.qwen_world_postcot.latest_snapshot_owner','--root',str(root),'--report-dir',str(Path(a.report_dir).resolve())]
        with (root/'evaluation_owner.log').open('xb')as log:
            result=subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=False)
        if result.returncode or not (root/'COMPLETE.json').exists():raise RuntimeError('Midpoint evaluator did not publish a complete result')
        atomic_json(root/'QUEUE_STATUS.json',dict(stage='COMPLETE',at=time.time()))
    except BaseException:
        atomic_json(root/'QUEUE_FAILURE.json',dict(at=time.time(),error=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':main()
