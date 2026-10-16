"""Single-host admission and launch receipt; never evict or retry jobs implicitly."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from starVLA.rl.fm_transfer.contracts import require_dispatch, ROOT, config
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--variant',required=True,choices=['A_DDP_TIME','B_SOURCE_ACTION_DIT'])
    p.add_argument('--output',required=True);p.add_argument('--resume')
    p.add_argument('--qualification',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();host=require_dispatch();source=Path.cwd().resolve();out=Path(a.output)
    seal=json.loads((source/'SOURCE_COMPLETE.json').read_text())
    if not seal['git_archive_extraction_complete']:raise ValueError('Source snapshot is not published')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'identity.json').exists() and not a.resume:raise ValueError('Existing run: no duplicate launch')
    if not a.qualification:
        if a.stop_after:raise ValueError('Formal budget is fixed before training')
        reg=json.loads((ROOT/'experiment_registration.json').read_text())
        if (reg['SFT_source_commit']!=seal['commit'] or
                reg['SFT'][a.variant]!=config(a.variant)):
            raise ValueError('Dispatch differs from the fixed registration')
        for gate in reg['SFT_acceptance'][a.variant]:
            if file_sha(gate['path'])!=gate['sha256'] or json.loads(Path(gate['path']).read_text())['status']!='PASS':
                raise ValueError('SFT acceptance gate missing/changed')
    for record in out.glob('dispatches/*.json'):
        prior=json.loads(record.read_text())
        if prior['host']==host:
            proc=Path('/proc')/str(prior['pid'])/'stat'
            if proc.exists() and int(proc.read_text().split()[21])==prior['start_ticks']:
                raise ValueError('The recorded trainer is still running')
    resources=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    rows=[r.split(',') for r in resources.splitlines()]
    if len(rows)!=8 or any(int(r[1])<36000 for r in rows):raise RuntimeError('Insufficient current eight-GPU headroom')
    process_list=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'],text=True)
    attempt=uuid.uuid4().hex;dest=out/'dispatches';dest.mkdir(exist_ok=True)
    log=(dest/(attempt+'.log')).open('xb')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',
        TOKENIZERS_PARALLELISM='false',VLM_ATTN_IMPLEMENTATION='flash_attention_2',PYTHONPATH=str(source))
    cmd=['/mnt/project/flow-safe-improvement-20261013/venv/bin/python','-m','torch.distributed.run',
        '--standalone','--nproc_per_node=8','--module','scripts.analysis.fm_transfer.train_sft',
        '--variant',a.variant,'--output',str(out)]
    if a.resume:cmd+=['--resume',a.resume]
    if a.qualification:cmd+=['--qualification']
    if a.stop_after:cmd+=['--stop-after',str(a.stop_after)]
    require_dispatch()  # Recheck just before creation; no pressure-job heuristic.
    child=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,start_new_session=True)
    receipt=dict(schema='fm_transfer_SFT_dispatch_v1',host=host,pid=child.pid,
        start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]),
        attempt=attempt,start_unix=time.time(),source=str(source),source_commit=seal['commit'],
        variant=a.variant,qualification=a.qualification,resume=a.resume,command=cmd,
        live_GPU_resources=resources,live_GPU_processes=process_list,
        dispatch_policy_sha256=file_sha('/mnt/project/server_dispatch_policy.json'),
        log=str(dest/(attempt+'.log')),output=str(out),config_sha256=fingerprint(config(a.variant)))
    atomic_json(dest/(attempt+'.json'),receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
