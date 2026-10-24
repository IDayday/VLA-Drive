"""Operator-only extension of the existing worker for exact NAVTEST-only GPUs."""
import argparse
import fcntl
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--spec',required=True);a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text());source=Path(spec['source']);sys.path.insert(0,str(source))
    from scripts.analysis.qwen_sourceflow import job_worker
    from starVLA.rl.qwen_flow_exploration.evaluation_admission import require_navtest_dispatch,ROOT
    from starVLA.rl.qwen_sourceflow.contracts import file_sha
    entry='scripts.analysis.qwen_flow_exploration.evaluate_sharded';cmd=spec['command']
    if (spec['module']!=entry or len(spec['gpus'])!=1 or
        cmd[:4]!=['/mnt/project/flow-safe-improvement-20261013/venv/bin/python','-u','-m',entry] or
        '--split' not in cmd or cmd[cmd.index('--split')+1]!='NAVTEST' or
        any(flag in cmd for flag in ['--qualification','--limit'])):
        raise PermissionError('Spare-GPU capability only permits the locked full NAVTEST evaluator')
    entry_sha=file_sha(source/'scripts/analysis/qwen_flow_exploration/evaluate_sharded.py')
    def admitted():return require_navtest_dispatch(spec['gpus'],entry_sha)
    admitted()
    directory=ROOT/'device_locks';directory.mkdir(exist_ok=True)
    lock=(directory/(spec['gpu_uuids'][0]+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Only process-local resource admission changes. No model or scoring code
    # is patched; the qualified child checks its frozen scope independently.
    job_worker.MODULES=job_worker.MODULES|{entry};job_worker.require_dispatch=admitted
    job_worker.main()


if __name__=='__main__':main()
