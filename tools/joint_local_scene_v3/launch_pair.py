"""Launch a matched pair in a pinned checkout, recording exact commands."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for key in ('campaign','source','train','holdout'):p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,choices=[42,43],default=42);p.add_argument('--gpus',type=int,nargs=2,required=True)
    p.add_argument('--resume-to64',action='store_true');a=p.parse_args();root=Path(a.campaign);source=Path(a.source)
    if a.gpus[0]==a.gpus[1]:raise ValueError('Independent runs require distinct GPUs')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    subprocess.run(['git','diff','--exit-code','--quiet'],cwd=source,check=True)
    registration=json.loads((root/'registration.json').read_text());assert registration['updates_per_epoch']==228 and registration['batch']==32
    config='configs/joint_local_scene_v3/'+('mechanism.json' if a.seed==42 else 'mechanism_seed43.json')
    commands=[]
    for mode,gpu in zip(('all','mask'),a.gpus):
        name=f'formal{a.seed}_{mode}';out=root/name
        if out.exists() and not a.resume_to64:raise FileExistsError('Existing run: '+str(out))
        cmd=[sys.executable,'-m','tools.joint_local_scene_v3.train_mechanism','--train',a.train,'--holdout',a.holdout,'--config',config,'--mode',mode,'--updates','14592','--schedule-updates','14592','--batch','32','--eval-milestones','0,228,456,912,1824,3648,5472,7296,10944,14592','--save-every','100','--eval-train','--diagnostic-subset',str(root/'train64.json'),'--ledger',str(root/'budget_ledger.json'),'--run-id',name,'--output',str(out),'--device','cuda','--deterministic']
        cmd+=['--resume','--acknowledge-stop'] if a.resume_to64 else ['--stop-after','7296']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        record={'run_id':name,'cwd':str(source),'source_sha':sha,'argv':cmd,'CUDA_VISIBLE_DEVICES':str(gpu),'resume_to64':a.resume_to64,'started_utc_seconds':time.time()}
        commands.append((record,env,out))
    running=[]
    for record,env,out in commands:
        with (root/'commands.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        log=(root/(record['run_id']+'_process.log')).open('a');proc=subprocess.Popen(record['argv'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT);running.append((proc,log,out));print(json.dumps({'run_id':record['run_id'],'pid':proc.pid}),flush=True)
    while any(proc.poll() is None for proc,_,_ in running):
        if any(proc.poll() not in (None,0) for proc,_,_ in running):
            for proc,_,out in running:
                if proc.poll() is None and out.exists() and not (out/'STOP_REQUESTED').exists():(out/'STOP_REQUESTED').write_text('Paired run failed; preserve counterpart at safe boundary.\n')
        time.sleep(1)
    for _,log,_ in running:log.close()
    if any(proc.returncode for proc,_,_ in running):raise RuntimeError('A paired process failed; checkpoints retained')
    print('Pair processes exited; inspect status for planned32-epoch pause or64-epoch completion.',flush=True)


if __name__=='__main__':main()
