"""Dispatch the registered full TRAIN single-candidate screen of a complete SFT.

Shard count changes only execution routing. Each scene keeps its bound seed and
batch-one forward. Failed workers need explicit same-source shard continuation;
this entry never duplicates a partially launched screen.
"""
import argparse
from collections import Counter
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


def partition_plan(shards, ranks=None):
    """Route immutable global shards without changing seeds or batch layout."""
    if shards not in (8,16,24):raise ValueError('Unregistered global shard count')
    selected=list(range(shards)) if ranks is None else list(ranks)
    if not selected or len(set(selected))!=len(selected) or any(r<0 or r>=shards for r in selected):
        raise ValueError('Invalid, empty or duplicate shard partition')
    selected.sort()
    counts=Counter(r%8 for r in selected)
    return selected,{gpu:16000+10000*count for gpu,count in counts.items()}


def admit_partition(out, contract, ranks, explicit):
    """Must run under the shared dispatch lock; never overwrite a prior shard."""
    prior=[json.loads(p.read_text()) for p in sorted(out.glob('dispatches/*.json'))]
    recorded=out/'dispatch_contract.json'
    if prior and not explicit:raise ValueError('Existing dispatch; inspect and explicitly select only unlaunched shards')
    if prior and not recorded.exists():raise ValueError('Prior dispatcher has no partition contract; inspect explicitly')
    if recorded.exists() and json.loads(recorded.read_text())!=contract:
        raise ValueError('Different source/SFT/shard plan at the same screen destination')
    if not prior and (out/'manifest.json').exists():raise ValueError('Untracked existing screen; no duplicate admission')
    seen=[r['rank'] for r in prior]
    if len(set(seen))!=len(seen) or set(seen)&set(ranks):
        raise ValueError('A global screen shard already has a dispatch; no automatic retry')
    for r in prior:
        if (r['SFT_identity']!=contract['SFT_identity'] or r['SFT_model_sha256']!=contract['SFT_model_sha256'] or
                r['source']!=contract['worker_source'] or r['global_shards']!=contract['global_shards']):
            raise ValueError('Existing shard has a different screen identity')
    atomic_json(recorded,contract)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--sft-checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--source',required=True)
    p.add_argument('--shards',type=int,choices=(8,16,24),default=24)
    p.add_argument('--ranks',help='Explicit comma-separated subset of global shards for this host; no repeat/retry')
    a=p.parse_args()
    selected,reserved=partition_plan(a.shards,None if a.ranks is None else [int(x) for x in a.ranks.split(',')])
    host=require_dispatch();source=Path(a.source).resolve();out=Path(a.output)
    if not json.loads((source/'SOURCE_COMPLETE.json').read_text())['git_archive_extraction_complete']:
        raise ValueError('Unpublished screen source')
    identity,files=completed_sft(a.sft_checkpoint)
    if file_sha(Path(a.sft_checkpoint)/'model.pt')!=files['model.pt']:raise ValueError('Changed SFT model')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    contract=dict(schema='fm_transfer_multihost_screen_partition_v1',SFT_identity=identity['sha256'],
        SFT_model_sha256=files['model.pt'],worker_source=str(source),global_shards=a.shards,
        output=str(out.resolve()),action_batch=1,seed=2101)
    admit_partition(out,contract,selected,a.ranks is not None)
    free=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()))
    # Reserve all pending workers before any asynchronous model loading begins.
    # This is headroom admission, not an assertion of measured exclusive usage.
    if len(free)!=8 or any(free[gpu]<required for gpu,required in reserved.items()):
        raise RuntimeError('Insufficient current capacity for the entire host partition')
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    attempt=uuid.uuid4().hex;dest=out/'dispatches';dest.mkdir(exist_ok=True);records=[]
    for index,rank in enumerate(selected):
        require_dispatch();gpu=rank%8
        current=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()))
        remaining=sum(r%8==gpu for r in selected[index:])
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
            initial_free_MiB=free,worker_start_free_MiB=current[gpu],initial_GPU_owners=owners,
            admission_reserved_MiB_per_GPU=reserved[gpu],global_shards=a.shards,partition_ranks=selected,
            dispatcher_sha256=file_sha(__file__))
        atomic_json(dest/(attempt+f'_rank{rank}.json'),record);records.append(record)
    print(json.dumps(dict(status='FULL_TRAIN_SCREEN_PARTITION_DISPATCHED',global_shards=a.shards,
        partition_ranks=selected,workers=records)))


if __name__=='__main__':main()
