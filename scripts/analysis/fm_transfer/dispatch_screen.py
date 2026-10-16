"""Dispatch the registered full TRAIN single-candidate screen of a complete SFT.

Shard count changes only execution routing. Each scene keeps its bound seed and
batch-one forward. Failed workers need explicit same-source shard continuation;
this entry never duplicates a partially launched screen.
"""
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


def completed_sft(checkpoint):
    checkpoint=Path(checkpoint)
    seal=json.loads((checkpoint/'COMPLETE').read_text())
    files=json.loads((checkpoint/'files.json').read_text())
    identity=json.loads((checkpoint/'identity.json').read_text())
    if ('.incomplete' in checkpoint.name or seal['phase']!='READY' or
            file_sha(checkpoint/'files.json')!=seal['files_sha256'] or
            file_sha(checkpoint/'identity.json')!=files['identity.json'] or
            identity['qualification'] or seal['updates']!=identity['config']['max_updates']):
        raise ValueError('Requires the sealed complete formal supervised endpoint')
    return identity,files


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--sft-checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--source',required=True)
    p.add_argument('--shards',type=int,choices=(8,16,24),default=24);a=p.parse_args()
    host=require_dispatch();source=Path(a.source).resolve();out=Path(a.output)
    if not json.loads((source/'SOURCE_COMPLETE.json').read_text())['git_archive_extraction_complete']:
        raise ValueError('Unpublished screen source')
    identity,files=completed_sft(a.sft_checkpoint)
    if file_sha(Path(a.sft_checkpoint)/'model.pt')!=files['model.pt']:raise ValueError('Changed SFT model')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if list(out.glob('dispatches/*.json')):raise ValueError('Existing dispatch; inspect and explicitly resume only missing shards')
    free=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()))
    # Reserve all pending workers before any asynchronous model loading begins.
    # This is headroom admission, not an assertion of measured exclusive usage.
    required=16000+10000*(a.shards//8)
    if len(free)!=8 or min(free)<required:raise RuntimeError('Insufficient current capacity for the entire shard plan')
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    attempt=uuid.uuid4().hex;dest=out/'dispatches';dest.mkdir(exist_ok=True);records=[]
    for rank in range(a.shards):
        require_dispatch();gpu=rank%8
        current=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()))
        remaining=sum(r%8==gpu for r in range(rank,a.shards))
        if current[gpu]<16000+10000*remaining:
            raise RuntimeError('Capacity changed during admission; preserve and inspect already launched shards')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(source))
        cmd=['/mnt/project/flow-safe-improvement-20261013/venv/bin/python','-m','scripts.analysis.fm_transfer.screen_train',
            '--sft-checkpoint',a.sft_checkpoint,'--output',str(out),'--shard',str(rank),'--shards',str(a.shards)]
        logfile=dest/(attempt+f'_rank{rank}.log')
        with logfile.open('xb') as log:
            child=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        record=dict(host=host,rank=rank,gpu=gpu,pid=child.pid,
            start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]),
            start_unix=time.time(),command=cmd,source=str(source),log=str(logfile),
            SFT_identity=identity['sha256'],SFT_model_sha256=files['model.pt'],
            initial_free_MiB=free,worker_start_free_MiB=current[gpu],initial_GPU_owners=owners,admission_reserved_MiB_per_GPU=required,
            dispatcher_sha256=file_sha(__file__))
        atomic_json(dest/(attempt+f'_rank{rank}.json'),record);records.append(record)
    print(json.dumps(dict(status='FULL_TRAIN_SCREEN_DISPATCHED',workers=records)))


if __name__=='__main__':main()
