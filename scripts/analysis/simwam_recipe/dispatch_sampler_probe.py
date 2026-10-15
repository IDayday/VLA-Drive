"""Dispatch only pre-registered, finite TRAIN sampler diagnostics."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from starVLA.rl.simwam_recipe.contracts import file_sha
from scripts.analysis.simwam_recipe.controller_runtime import require_worker_host


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--host',required=True);p.add_argument('--model',required=True,choices=['SFT','R_CANONICAL_20000'])
    p.add_argument('--gpus',nargs='+',type=int,required=True)
    a=p.parse_args();require_worker_host(a.host)
    if len(set(a.gpus))!=len(a.gpus) or any(g not in range(8) for g in a.gpus):raise ValueError('Invalid GPU/shard mapping')
    reg=json.loads((Path(a.root)/'registration.json').read_text())
    source=Path(a.source).resolve()
    if reg['source_script_sha256']!=file_sha(source/'scripts/analysis/simwam_recipe/probe_sampler_transfer.py'):
        raise ValueError('Source differs from pre-registered sampler probe')
    payload=dict(vars(a),source=str(source),python='/mnt/project/flow-safe-improvement-20261013/venv/bin/python',
                 registration_sha256=reg['sha256'])
    code='''import json,os,pathlib,subprocess,time
p=json.loads(PAYLOAD)
out=pathlib.Path(p['root'])/'dispatch'/p['model'];out.mkdir(parents=True,exist_ok=True)
for gpu in p['gpus']:
 receipt=out/('rank'+str(gpu)+'.json')
 if receipt.exists():raise RuntimeError('Existing receipt requires explicit inspection: '+str(receipt))
 free=[int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)) for _ in range(2)]
 if min(free)<14000:raise RuntimeError('Insufficient live free GPU memory; no eviction')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),LOCAL_RANK='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',PYTHONPATH=p['source'])
 cmd=[p['python'],'-m','scripts.analysis.simwam_recipe.probe_sampler_transfer','worker','--root',p['root'],'--model',p['model'],'--shard',str(gpu),'--shards','8']
 log=(out/('rank'+str(gpu)+'.log')).open('ab',buffering=0)
 child=subprocess.Popen(cmd,cwd=p['source'],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 value=dict(p,pid=child.pid,start_ticks=pathlib.Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],gpu=gpu,command=cmd,started=time.time(),free_MiB=free)
 receipt.write_text(json.dumps(value,indent=2)+'\\n');print(p['model'],gpu,child.pid)
'''.replace('PAYLOAD',repr(json.dumps(payload)))
    subprocess.run(['ssh','-o','BatchMode=yes',a.host,shlex.join([payload['python'],'-c',code])],check=True)


if __name__=='__main__':main()
