"""Exactly 4 vs 2+2 SYNTHETIC updates per device under a shared capped ledger."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from tools.joint_local_scene_v3.synthetic import write_synthetic
from tools.joint_local_scene_v3.budget import atomic_json


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--ledger',required=True);p.add_argument('--config',required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new verification output')
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(a.config).read_text());cfg['model'].update(dim=32,heads=4,layers=2,steps=3,condition_dim=12)
    cfg['graph'].update(max_neighbors=2,primary_neighbors=1,max_context=3);cfg.update(warmup_steps=1,sampling_steps=2,role_loss_weight=2.,condition_dim_source='Synthetic width12, no VLM in this fixture')
    atomic_json(out/'config.json',cfg);write_synthetic(out/'data',cfg)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    base=[sys.executable,'-m','tools.joint_local_scene_v3.train_mechanism','--train',str(out/'data'),'--holdout',str(out/'data'),'--config',str(out/'config.json'),
        '--mode','mask','--updates','4','--schedule-updates','4','--batch','2','--eval-every','2','--ledger',a.ledger,'--device',a.device,'--deterministic']
    commands=[]
    def invoke(name,extra=(),expect=0):
        cmd=base+['--output',str(out/name),'--run-id',f'v3_review_resume_{a.device}_{name}']+list(extra);commands.append(cmd)
        with (out/(name+'_process.log')).open('a') as log:
            proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        if expect==0 and proc.returncode!=0:raise RuntimeError(f'Run failed ({proc.returncode}); see {name}_process.log')
        if expect!=0 and proc.returncode==0:raise RuntimeError('Expected a pre-update rejection')
    invoke('continuous')
    invoke('resumed',['--stop-after','2'])
    paused=torch.load(out/'resumed'/'checkpoint.pt',map_location='cpu',weights_only=False);assert paused['step']==2
    before=(out/'resumed'/'checkpoint.pt').read_bytes()
    invoke('resumed',['--resume'],expect=1)
    assert before==(out/'resumed'/'checkpoint.pt').read_bytes()
    invoke('resumed',['--resume','--acknowledge-stop'])
    first=torch.load(out/'continuous'/'checkpoint.pt',map_location='cpu',weights_only=False);second=torch.load(out/'resumed'/'checkpoint.pt',map_location='cpu',weights_only=False)
    checks={k:equal(first[k],second[k]) for k in first}
    # Wall time/allocated bytes are not claimed deterministic. All learning/log quantities are.
    def logs(name):
        return [{k:v for k,v in json.loads(line).items() if k not in ('seconds','peak_gpu_bytes')} for line in (out/name/'train.jsonl').read_text().splitlines()]
    checks['learning_log_sequence']=logs('continuous')==logs('resumed')
    checks['explicit_stop_archive']=(out/'resumed'/'STOP_REQUESTED.acknowledged_step2').exists() and not (out/'resumed'/'STOP_REQUESTED').exists()
    report={'status':'PASS' if all(checks.values()) else 'FAIL','device':a.device,'synthetic_optimizer_updates':8,'real_optimizer_updates':0,
        'comparison':'4 continuous vs2+2 with same device/source/data/config and deterministic algorithms; not cross-device equivalence','checks':checks,'commands':commands}
    atomic_json(out/'summary.json',report);print(json.dumps(report,indent=2))
    if not all(checks.values()):raise AssertionError('Synthetic continuation differs')


if __name__=='__main__':main()
