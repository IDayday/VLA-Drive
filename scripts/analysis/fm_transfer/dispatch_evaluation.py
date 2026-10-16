"""Admit eight raw-observation milestone shards on a currently authorized host."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',required=True)
    p.add_argument('--export',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cohort',choices=['DEV','TRAIN_DIAGNOSTIC'],required=True)
    args=p.parse_args();host=require_dispatch();source=Path(args.source).resolve();export=Path(args.export)
    if not json.loads((source/'SOURCE_COMPLETE.json').read_text())['git_archive_extraction_complete']:
        raise ValueError('Unpublished evaluation source')
    # A frozen public evaluation protocol is required; this entry never opens
    # Navtest or invents another seed to fill available devices.
    info=json.loads((export/'export.json').read_text())
    if info['qualification']:raise ValueError('No full evaluation of qualification weights')
    sha=info['weight_sha256']
    if file_sha(export/'pytorch_model.pt')!=sha:raise ValueError('Changed exported weights')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if list(out.glob('dispatches/*.json')):raise ValueError('Existing dispatch: inspect completion before explicit shard recovery')
    free=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()))
    if len(free)!=8 or min(free)<24000:raise RuntimeError('No eight-GPU evaluation headroom')
    invocation=uuid.uuid4().hex;receipts=[];dest=out/'dispatches';dest.mkdir(exist_ok=True)
    for rank in range(8):
        require_dispatch()
        current=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()))
        if current[rank]<24000:raise RuntimeError('Resource changed; preserve already launched workers')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false',
            PYTHONPATH=str(source),VLM_ATTN_IMPLEMENTATION='flash_attention_2')
        cmd=['/mnt/project/flow-safe-improvement-20261013/venv/bin/python','-m','scripts.analysis.fm_transfer.evaluate_policy',
            '--export',str(export),'--export-sha256',sha,'--output',str(out),'--cohort',args.cohort,'--shard',str(rank),'--shards','8']
        logfile=dest/(invocation+f'_rank{rank}.log')
        with logfile.open('xb') as log:
            child=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        record=dict(host=host,rank=rank,pid=child.pid,start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]),
            source=str(source),start_unix=time.time(),command=cmd,initial_free_MiB=free[rank],log=str(logfile),
            export_sha256=sha,dispatcher_sha256=file_sha(__file__))
        atomic_json(dest/(invocation+f'_rank{rank}.json'),record);receipts.append(record)
    print(json.dumps(dict(status='EIGHT_SHARDS_DISPATCHED',output=str(out),workers=receipts)))


if __name__=='__main__':main()
