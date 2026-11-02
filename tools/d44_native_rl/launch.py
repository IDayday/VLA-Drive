"""Exact command dispatch through existing UUID/lease/expiry admission."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def launch(host,source,job,command,gpus,*,share_min_free_mib=None):
    source=Path(source).resolve();job=Path(job).resolve();job.mkdir(parents=True,exist_ok=True)
    if(job/'LAUNCH.json').exists():raise FileExistsError('Invocation already dispatched; inspect its receipt')
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():raise ValueError('Freeze scientific source before dispatch')
    code="import subprocess,json;print(json.dumps(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)))"
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3','-'],input=code,text=True,capture_output=True,timeout=20)
    if r.returncode:raise RuntimeError(r.stderr)
    mapping={int(x.split(',')[0]):x.split(',')[1].strip()for x in json.loads(r.stdout).splitlines()}
    uuids=[mapping[i]for i in gpus]
    args=['/usr/bin/python3','-u','-m','tools.qwen_world_compare.job_worker','--receipt-root',str(job),
        '--gpus',','.join(map(str,gpus)),'--expected-gpu-uuids-json',json.dumps(uuids),
        '--command-json',json.dumps(command),'--cwd',str(source),'--minimum-free-mib','65000']
    if share_min_free_mib is not None:
        args=args[:-2]+['--share-min-free-mib',str(share_min_free_mib),'--lease-slot','shared']
    program='import subprocess,os;f=open('+repr(str(job/'transport.log'))+',"ab",buffering=0);p=subprocess.Popen('+repr(args)+',cwd='+repr(str(source))+',env=dict(os.environ,PYTHONPATH='+repr(str(source))+',PYTHONUTF8="1",WORLD_COMPARE_ROOT="/mnt/project/d44-native-rl-20261103",WORLD_COMPARE_TASK="d44_native_rl_20261103"),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)'
    r=subprocess.run(['ssh','-o','BatchMode=yes',host,'/usr/bin/python3','-'],input=program,text=True,capture_output=True,timeout=20)
    if r.returncode:raise RuntimeError(r.stderr)
    receipt=dict(host=host,pid=int(r.stdout),source=str(source),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),command=command,job=str(job),at=time.time())
    tmp=job/'LAUNCH.json.tmp';tmp.write_text(json.dumps(receipt,ensure_ascii=True,indent=2)+'\n');os.replace(tmp,job/'LAUNCH.json');return receipt


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--host',required=True);p.add_argument('--source',required=True);p.add_argument('--job',required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--command-json',required=True)
    p.add_argument('--share-min-free-mib',type=int);a=p.parse_args()
    print(json.dumps(launch(a.host,a.source,a.job,json.loads(a.command_json),[int(x)for x in a.gpus.split(',')],share_min_free_mib=a.share_min_free_mib),ensure_ascii=True))


if __name__=='__main__':main()
