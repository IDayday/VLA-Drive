"""One owned process invocation, durable exit/cost receipt, no automatic retry."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from starVLA.rl.qwen_sourceflow.contracts import require_dispatch,file_sha,fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

MODULES={'scripts.analysis.qwen_sourceflow.train_il','scripts.analysis.qwen_sourceflow.train_rl',
    'scripts.analysis.qwen_sourceflow.evaluate_policy','scripts.analysis.qwen_sourceflow.validate_deployment',
    'scripts.analysis.qwen_sourceflow.export_policy','scripts.analysis.qwen_sourceflow.validate_rl_deployment',
    'scripts.analysis.qwen_sourceflow.profile_efficiency','scripts.analysis.qwen_sourceflow.benchmark_collectives'}


def request_native_stop(child, module):
    """Signal only this invocation's native training workers, never torchrun."""
    signaled=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        pid=int(path.name)
        try:
            if os.getpgid(pid)!=child.pid:continue
            argv=(path/'cmdline').read_bytes().decode().split('\0')
            if '-m' not in argv or argv[argv.index('-m')+1]!=module:continue
            # pidfd pins this exact process. The host operator uses system Python.
            fd=os.pidfd_open(pid)
            try:signal.pidfd_send_signal(fd,signal.SIGUSR1)
            finally:os.close(fd)
            signaled.append(pid)
        except (FileNotFoundError,ProcessLookupError):pass
    return signaled


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--spec',required=True);a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text());root=Path(spec['receipt_dir']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'process.json').exists():raise FileExistsError('Never duplicate or silently retry an invocation')
    host=require_dispatch();source=Path(spec['source']);started=time.time();record=dict(status='PREPARING',start_unix=started,host=host,spec_sha256=fingerprint(spec))
    try:
        if host!=spec['host']:raise ValueError('Wrong worker host')
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=spec['source_sha']:
            raise ValueError('Changed immutable source')
        if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():raise ValueError('Dirty worker source')
        cmd=spec['command'];module=spec['module']
        if module not in MODULES or module not in cmd:raise ValueError('Unregistered native job entry')
        if spec['gpus']:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
            gpus=[(u.strip(),int(f)) for u,f in (line.split(',') for line in rows.splitlines())]
            if [gpus[i][0] for i in spec['gpus']]!=spec['gpu_uuids']:raise ValueError('Physical GPU identity changed')
            if min(gpus[i][1] for i in spec['gpus'])<spec['min_free_MiB']:raise ValueError('Insufficient measured headroom')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,spec['gpus'])),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
        with (root/'worker.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(status='RUNNING',pid=child.pid,start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]),command=cmd,source=str(source),gpus=spec['gpus'],gpu_uuids=spec['gpu_uuids'])
            atomic_json(root/'process.json',record);stopping=False
            while True:
                try:exit_code=child.wait(timeout=60);break
                except subprocess.TimeoutExpired:pass
                try:require_dispatch();withdrawn=False
                except PermissionError:withdrawn=True
                if not stopping and (withdrawn or (root/'STOP_AFTER_BOUNDARY').exists()):
                    if module.endswith(('train_il','train_rl')):
                        pids=request_native_stop(child,module)
                        atomic_json(root/'stop_request.json',dict(native_boundary=True,pids=pids,withdrawn=withdrawn,time_unix=time.time()))
                    else:
                        # Evaluation scene records are atomic; an interrupted
                        # invocation stays incomplete, never a zero-score row.
                        child.terminate()
                    stopping=True
        record.update(status='PROCESS_COMPLETE' if exit_code==0 else 'FAILED',exit_code=exit_code,end_unix=time.time(),wall_seconds=time.time()-started,
            GPU_time_semantics='worker wall occupancy; possibly shared, not exclusive compute time')
    except BaseException:
        record.update(status='FAILED',error=traceback.format_exc(),end_unix=time.time(),wall_seconds=time.time()-started)
        raise
    finally:atomic_json(root/'result.json',record)


if __name__=='__main__':main()
