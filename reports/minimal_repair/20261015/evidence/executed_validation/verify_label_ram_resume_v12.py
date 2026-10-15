import json,os,subprocess,sys,time
from pathlib import Path
import torch
w=Path('/mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015');sys.path.insert(0,str(w))
from tools.decision_local.common import atomic_json,check_host,digest,read
from tools.decision_local.verify_resume import equal
r=Path('/mnt/project/minimal-repair-artifacts/20261015');out=r/'resume_RAM_v11';check_host()
assert read(r/'RAM_LABEL_SMOKE_v12.json')['all_data_and_record_bytes_exact']
jobs=[]
for mode,index in [('stop7',2),('resume_labels',3)]:
    argv=list(read(out/'JOBS.json')[index]['argv'])
    argv[argv.index('--output')+1]=str(out/'labels_interrupted')
    argv[argv.index('--features')+1]='/dev/shm/decision-local-minimal-repair-v5-20261015/anchor_train_full_v1'
    if mode=='resume_labels':
        os.link(out/'labels_interrupted/latest.pt',out/'labels_interrupted/origin7.pt')
        argv[argv.index('--labels')+1]='/dev/shm/decision-local-repair-labels-smoke-v12-20261015/targets_train64_v5'
    started=time.time()
    with (out/('labels_'+mode+'.log')).open('x') as log:
        p=subprocess.run(argv,cwd=w,stdout=log,stderr=subprocess.STDOUT,timeout=330)
    jobs.append(dict(argv=argv,mode=mode,returncode=p.returncode,started_unix=started,ended_unix=time.time()))
    atomic_json(out/'LABEL_RAM_JOBS.json',jobs);p.check_returncode()
a=torch.load(out/'continuous_A/latest.pt',map_location='cpu',weights_only=False)
b=torch.load(out/'labels_interrupted/latest.pt',map_location='cpu',weights_only=False)
keys=['identity','model','optimizer','scheduler','rng','completed','epoch','offset','exposure','order','effective_optimizer_updates','exposure_counts']
for k in keys:equal(a[k],b[k],k)
atomic_json(out/'LABEL_RAM_ACCEPTANCE.json',dict(exact=True,keys=keys,completed=b['completed'],source_sha=b['identity']['source_sha'],
    script_sha256=digest(__file__),scope='Same native trainer; temporary byte-identical labels and feature storage; optimizer7 interruption through48'))
print('label and feature RAM native state exact through48',flush=True)
