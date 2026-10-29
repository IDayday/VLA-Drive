"""Owned finite full-model IL/restore qualifications, then explicit acceptance.

Qualification weights are never formal initialization. Successful CPU arithmetic
tests alone cannot publish ENGINEERING_ACCEPTANCE.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
if __package__ in(None,''):
    script_dir=Path(__file__).resolve().parent
    sys.path[:]=[x for x in sys.path if Path(x or os.curdir).resolve()!=script_dir]
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from starVLA.rl.qwen_world_compare.contracts import ROOT,file_sha
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from tools.qwen_world_compare.dispatch import choose_host,launch,job_state

PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'

def tree_compare(a,b,path=''):
    if torch.is_tensor(a):
        if not torch.is_tensor(b)or a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):raise ValueError('Native tensor resume mismatch: '+path)
    elif isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Native recovery keys differ: '+path)
        for k in a:tree_compare(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise ValueError('Native recovery list differs: '+path)
        for i,(x,y)in enumerate(zip(a,b)):tree_compare(x,y,path+f'.{i}')
    else:
        import numpy as np
        if isinstance(a,np.ndarray):
            if not np.array_equal(a,b):raise ValueError('Native numpy RNG differs: '+path)
        elif a!=b:raise ValueError('Native recovery scalar differs: '+path)

def compare_native(left,right,output):
    left,right=Path(left),Path(right)
    ls=json.loads((left/'COMPLETE').read_text());rs=json.loads((right/'COMPLETE').read_text())
    if ls['updates']!=4 or rs['updates']!=4:raise ValueError('Qualification compare requires two complete fourth updates')
    a=torch.load(left/ls['model_file'],map_location='cpu',weights_only=False)
    b=torch.load(right/rs['model_file'],map_location='cpu',weights_only=False)
    tree_compare(a['module'],b['module'],'model')
    if 'lr_scheduler'in a:tree_compare(a['lr_scheduler'],b['lr_scheduler'],'scheduler')
    del a,b
    for p in sorted((left/'native').glob('*_optim_states.pt')):
        a=torch.load(p,map_location='cpu',weights_only=False);b=torch.load(right/'native'/p.name,map_location='cpu',weights_only=False)
        tree_compare(a,b,'optimizer.'+p.name);del a,b
    for rank in range(8):
        a=torch.load(left/f'rank{rank:02d}_state.pt',map_location='cpu',weights_only=False);b=torch.load(right/f'rank{rank:02d}_state.pt',map_location='cpu',weights_only=False)
        for key in('state','builder','rng','identity_sha256'):tree_compare(a[key],b[key],f'rank{rank}.{key}')
    atomic_json(output,dict(status='PASS',model_bitwise_equal=True,optimizer_scheduler_equal=True,all_rank_rng_builder_cursor_equal=True,left=str(left),right=str(right),comparison='continuous4 vs native2+resume2; all FP32 tensors and full native state'))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source');p.add_argument('--architectures',nargs='+',default=['A','B','C','D']);p.add_argument('--compare',nargs=2);p.add_argument('--output')
    a=p.parse_args()
    if a.compare:compare_native(*a.compare,a.output);return
    source=Path(a.source).resolve();q=ROOT/'qualification';q.mkdir(exist_ok=True);lk=(q/'owner.lock').open('a');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_tag=subprocess.check_output(['git','rev-parse','--short=8','HEAD'],cwd=source,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():raise ValueError('Qualification runs from an immutable clean worktree')
    state_path=q/'STATE.json';state=json.loads(state_path.read_text())if state_path.exists()else dict(source=str(source),arms={x:dict(stage='BASELINE_PENDING')for x in a.architectures},jobs={})
    if state['source']!=str(source):raise ValueError('Another qualification source owns this root')
    atomic_json(q/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source=str(source),at=time.time()))
    while True:
        excluded=set();changed=False
        for name,j in state['jobs'].items():
            try:r=job_state(j['receipt'])
            except ValueError:
                # A launcher may have crossed the SSH/Popen boundary without
                # returning its PID. Never guess that it is safe to duplicate.
                if time.time()-j.get('intent_unix',time.time())<120:
                    excluded.update(j['uuids']);continue
                state['arms'][j['architecture']].update(stage='FAILED',error='Uncertain launch intent; inspect owned remote worker before any retry')
                j['handled']=True;changed=True;continue
            status=r['status']
            if status in('RUNNING','DISPATCHED')or('returncode'not in r and status not in('REFUSED_RESOURCE',)):
                excluded.update(j['uuids']);continue
            if j.get('handled'):continue
            arm=state['arms'][j['architecture']];stage=j['stage']
            if r.get('returncode')!=0:
                attempts=sum(x['architecture']==j['architecture']and x['stage']==stage for x in state['jobs'].values())
                if r.get('status')=='REFUSED_RESOURCE'and r.get('child_not_started')is True and attempts<3:
                    arm['stage']=stage+'_PENDING';j['handled']=True;changed=True;continue
                arm.update(stage='FAILED',error=dict(job=name,result=r));j['handled']=True;changed=True;continue
            arm['stage']={'BASELINE':'RESUME_PENDING','RESUME':'COMPARE_PENDING','COMPARE':'ACCEPTANCE_PENDING'}[stage]
            j['handled']=True;changed=True
        for arch,arm in state['arms'].items():
            stage=arm['stage'];base=q/arch;cfg=source/f'configs/qwen_world_compare/arch_{arch.lower()}.yaml'
            if stage=='ACCEPTANCE_PENDING':
                check=json.loads((base/'NATIVE_RESUME_CHECK.json').read_text())
                if check['status']!='PASS':raise ValueError('Native continuation did not pass')
                audits=[json.loads((base/'IL_joint'/f'GRADIENT_AUDIT_start000000_rank{rank}.json').read_text())for rank in range(8)]
                for role in('pretrained_language_merger','pretrained_vision'):
                    if not all(any(r.get('nonzero',0)>0 for k,r in row['roles'].items()if k==role)for row in audits):raise ValueError('No real full-model gradient for '+role)
                loss_totals={k:0. for k in('x','dino','wa_pre','object','future_track','wa_post','evaluation')}
                import math
                for rank in range(8):
                    rows=[json.loads(line)for line in(base/'IL_joint'/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
                    if [r['updates']for r in rows]!=[1,2,3,4]or any(r['optimizer_skipped_steps']!=0 for r in rows):raise ValueError('Not four real, finite, unskipped updates')
                    for row in rows:
                        if set(row['losses'])!=set(loss_totals):raise ValueError('Qualification omitted a common loss')
                        for k,v in row['losses'].items():
                            if not math.isfinite(v)or v<0:raise ValueError('Invalid common qualification loss')
                            loss_totals[k]+=v
                if any(v<=0 for v in loss_totals.values()):raise ValueError('No actual supervised evidence for one of seven losses')
                from tools.qwen_world_compare.campaign import expected_source_files
                native=json.loads((base/'IL_joint/IDENTITY.json').read_text())
                receipt=dict(status='PASS',architecture_id=arch,source_files=expected_source_files(source),config_sha256=file_sha(cfg),qualification_identity=native,loss_totals=loss_totals,checks=dict(real_all_seven_losses=True,real_full_model_gradients=True,native_resume=True),micro_batch=native['execution']['micro'],backend=native['execution']['backend'],formal_initialized_from_qualification=False)
                atomic_json(base/'ENGINEERING_ACCEPTANCE.json',receipt);arm['stage']='COMPLETE';changed=True
            elif stage in('BASELINE_PENDING','RESUME_PENDING'):
                selected=choose_host(excluded_uuids=excluded)
                if selected is None:continue
                host,inventory=selected;uuids=[g['uuid']for g in inventory['gpus']];kind='BASELINE'if stage=='BASELINE_PENDING'else'RESUME'
                out=base/('IL_joint'if kind=='BASELINE'else'IL_resume_from2')
                command=[PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8','-m','tools.qwen_world_compare.train_il','--config',str(cfg),'--output',str(out),'--qualification','--joint','--stop-after','4','--evidence-override',str(q/'evidence_first32')]
                if kind=='BASELINE':command+=['--qualification-save-at','2']
                else:command+=['--resume',str(base/'IL_joint/checkpoints/update000002')]
                attempt=1+sum(x['architecture']==arch and x['stage']==kind for x in state['jobs'].values())
                name=f'qualify_{source_tag}_{arch}_{kind.lower()}_attempt{attempt:03d}'
                state['jobs'][name]=dict(receipt=str(ROOT/'jobs'/name),architecture=arch,stage=kind,host=host,uuids=uuids,intent_unix=time.time())
                arm['stage']=kind+'_RUNNING';atomic_json(state_path,state)
                receipt=launch(host,str(source),command,name,gpus='0,1,2,3,4,5,6,7',expected_uuids=uuids,minimum=65000)
                state['jobs'][name]=dict(receipt=str(receipt),architecture=arch,stage=kind,host=host,uuids=uuids);arm['stage']=kind+'_RUNNING';excluded.update(uuids);changed=True
            elif stage=='COMPARE_PENDING':
                # Large CPU native-state comparison uses the just-tested host,
                # not the lower-memory front-end container.
                prior=[j for j in state['jobs'].values()if j['architecture']==arch and j['stage']=='RESUME'][-1]
                command=[PYTHON,'-u','-m','tools.qwen_world_compare.qualify','--compare',str(base/'IL_joint/checkpoints/update000004'),str(base/'IL_resume_from2/checkpoints/update000004'),'--output',str(base/'NATIVE_RESUME_CHECK.json')]
                name=f'qualify_{source_tag}_{arch}_native_compare_attempt001'
                state['jobs'][name]=dict(receipt=str(ROOT/'jobs'/name),architecture=arch,stage='COMPARE',host=prior['host'],uuids=[],intent_unix=time.time())
                arm['stage']='COMPARE_RUNNING';atomic_json(state_path,state)
                receipt=launch(prior['host'],str(source),command,name)
                state['jobs'][name]=dict(receipt=str(receipt),architecture=arch,stage='COMPARE',host=prior['host'],uuids=[]);arm['stage']='COMPARE_RUNNING';changed=True
        if changed:atomic_json(state_path,state)
        if all(arm['stage']in('COMPLETE','FAILED')for arm in state['arms'].values()):
            atomic_json(q/'OWNER_COMPLETE.json',dict(status='COMPLETE'if all(x['stage']=='COMPLETE'for x in state['arms'].values())else'FAILED_REVIEW_REQUIRED',arms=state['arms']));return
        time.sleep(20)

if __name__=='__main__':main()
