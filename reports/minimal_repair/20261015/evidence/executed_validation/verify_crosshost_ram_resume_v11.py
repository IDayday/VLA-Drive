import json, os, shlex, subprocess, sys, time
from pathlib import Path
import torch
w=Path('/mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015')
sys.path.insert(0,str(w))
from tools.decision_local.common import atomic_json,check_host,digest,read
from tools.decision_local.verify_resume import equal
r=Path('/mnt/project/minimal-repair-artifacts/20261015');out=r/'resume_RAM_v11'
check_host();host='training-vlawm-tsinghua-zt4'
policy=read('/mnt/project/server_dispatch_policy.json')
assert host in policy['allowed_hosts'] and host not in policy['global_dispatch_prohibitions']
original=read(out/'JOBS.json');jobs=[]
for mode,index in [('stop7',2),('resume_ram',3)]:
    argv=list(original[index]['argv']);argv[argv.index('--output')+1]=str(out/'crosshost')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    if mode=='resume_ram':
        os.link(out/'crosshost/latest.pt',out/'crosshost/origin7.pt')
        argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
              shlex.join(['env','CUDA_VISIBLE_DEVICES=1','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1']+argv)]
    started=time.time()
    with (out/('crosshost_'+mode+'.log')).open('x') as log:
        p=subprocess.run(argv,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=330)
    jobs.append(dict(argv=argv,mode=mode,returncode=p.returncode,started_unix=started,ended_unix=time.time()))
    atomic_json(out/'CROSSHOST_JOBS.json',jobs);p.check_returncode()
a=torch.load(out/'continuous_A/latest.pt',map_location='cpu',weights_only=False)
b=torch.load(out/'crosshost/latest.pt',map_location='cpu',weights_only=False)
keys=['identity','model','optimizer','scheduler','rng','completed','epoch','offset','exposure','order','effective_optimizer_updates','exposure_counts']
for k in keys:equal(a[k],b[k],k)
atomic_json(out/'CROSSHOST_ACCEPTANCE.json',dict(exact=True,keys=keys,completed=b['completed'],
    source_sha=b['identity']['source_sha'],resume_host=host,initial_host='training-vla-zt',
    script_sha256=digest(__file__),scope='shared-storage update7 on local A800, exact RAM-backed native resume through48 on tsinghua-zt4 A800'))
print('cross-host native state exact through48',flush=True)
