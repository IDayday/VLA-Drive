"""Finite coordinate-only study: qualification, paired native IL, full Test.

One owner, disjoint eight-GPU jobs, no original campaign restart. Qualification
weights are discarded. A failure is evidence and stops this owner for review.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint
from starVLA.rl.qwen_world_postcot.resources import remote, probe, MODEL_PYTHON
from starVLA.rl.qwen_world_postcot.coordinate_objective import tolerance_weights

SOURCE=Path(__file__).resolve().parents[2]
TASK='qwen_world_coordinate_loss_20261028'


def read(path):return json.loads(Path(path).read_text())


def launch(root,name,host,gpus,command):
    directory=root/'jobs'/name
    if directory.exists():raise FileExistsError('Invocation already exists: '+name)
    sample=probe(host)
    selected=[g for g in sample['gpus'] if g['index'] in gpus]
    if (len(selected)!=len(gpus) or min(g['free_MiB'] for g in selected)<60000
            or any(o['uuid'] in {g['uuid'] for g in selected} for o in sample['owners'])):
        raise RuntimeError('Registered idle resources unavailable; no process evicted: '+host)
    directory.mkdir(parents=True)
    args=['/usr/bin/python3','-u','-m','tools.qwen_world_postcot.job_worker',
          '--receipt-root',str(directory),'--gpus',','.join(map(str,gpus)),
          '--expected-gpu-uuids-json',json.dumps([g['uuid'] for g in selected]),
          '--minimum-free-mib','60000','--command-json',json.dumps(command),'--cwd',str(SOURCE)]
    payload=dict(args=args,cwd=str(SOURCE),directory=str(directory),task=TASK)
    code=('import json,os,pathlib,subprocess,time\ns='+repr(payload)+'\n'
          'env=dict(os.environ,PYTHONPATH=s["cwd"],WORLD_POSTCOT_TASK=s["task"])\n'
          'f=open(pathlib.Path(s["directory"])/"outer.log","ab",buffering=0)\n'
          'p=subprocess.Popen(s["args"],cwd=s["cwd"],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)\n'
          'print(json.dumps(dict(pid=p.pid,actual_host=os.uname().nodename,at=time.time())))')
    row=json.loads(remote(host,['/usr/bin/python3','-c',code]))
    atomic_json(directory/'LAUNCH.json',dict(host_alias=host,command=command,**row))
    return directory


def wait_all(directories):
    pending=set(map(Path,directories))
    while pending:
        for d in list(pending):
            result=d/'RESULT.json'
            if result.exists():
                row=read(result)
                if row.get('returncode')!=0:raise RuntimeError('Worker failed: '+str(d)+': '+repr(row))
                pending.remove(d)
            elif (d/'LAUNCH.json').exists() and time.time()-read(d/'LAUNCH.json')['at']>180:
                # A worker can fail before Popen/RESULT publication. Check its
                # exact PID instead of waiting forever or blindly relaunching.
                launch_row=read(d/'LAUNCH.json');pid=launch_row['pid']
                code=f'import os,json; print(json.dumps(dict(alive=os.path.exists("/proc/{pid}"))))'
                if not json.loads(remote(launch_row['host_alias'],['/usr/bin/python3','-c',code]))['alive']:
                    raise RuntimeError('Worker disappeared before a result; inspect '+str(d))
        if pending:time.sleep(30)


def train_command(root,arm,output,*,qualification=0,pause=0,resume=None):
    result=[MODEL_PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            '-m','tools.qwen_world_postcot.train_il','--output',str(output),
            '--coordinate-registration',str(root/'REGISTRATION.json'),'--coordinate-arm',arm]
    if qualification:result+=['--coordinate-qualification-updates',str(qualification)]
    if pause:result+=['--coordinate-pause-after',str(pause)]
    if resume:result+=['--resume',str(resume)]
    return result


def assert_tensor_tree(a,b,path=''):
    import torch
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or not torch.equal(a,b):raise ValueError('Resume tensor mismatch: '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Resume key mismatch: '+path)
        for k in a:assert_tensor_tree(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise ValueError('Resume sequence mismatch: '+path)
        for i,(x,y) in enumerate(zip(a,b)):assert_tensor_tree(x,y,path+'.'+str(i))
    elif type(a).__module__=='numpy':
        import numpy as np
        if not np.array_equal(a,b):raise ValueError('Resume numpy mismatch: '+path)
    elif hasattr(a,'__dict__') and type(a) is type(b):
        assert_tensor_tree(vars(a),vars(b),path+'.__dict__')
    elif a!=b:raise ValueError('Resume scalar mismatch: '+path)


def qualify_resume(root):
    import torch
    a=root/'qualification/continuous/checkpoints/update063003'
    b=root/'qualification/resumed/checkpoints/update063003'
    # Native checkpoint metadata may contain elapsed timing and paths; model,
    # each rank RNG/builder/cursor, and full Adam/scheduler values are compared.
    ma=torch.load(a/'native/mp_rank_00_model_states.pt',map_location='cpu',weights_only=False)
    mb=torch.load(b/'native/mp_rank_00_model_states.pt',map_location='cpu',weights_only=False)
    assert_tensor_tree(ma['module'],mb['module'],'module')
    assert_tensor_tree(ma['lr_scheduler'],mb['lr_scheduler'],'scheduler')
    del ma,mb
    for rank in range(8):
        ra=torch.load(a/f'rank{rank:02d}_state.pt',map_location='cpu',weights_only=False)
        rb=torch.load(b/f'rank{rank:02d}_state.pt',map_location='cpu',weights_only=False)
        for name in ('state','builder','rng'):assert_tensor_tree(ra[name],rb[name],f'rank{rank}.{name}')
    # Compare all tensor/scalar optimizer content, preserving each shard's
    # partition layout; ignore no scientific fields.
    for pa in sorted((a/'native').glob('*_optim_states.pt')):
        pb=b/'native'/pa.name
        va=torch.load(pa,map_location='cpu',weights_only=False);vb=torch.load(pb,map_location='cpu',weights_only=False)
        assert_tensor_tree(va,vb,'optimizer.'+pa.name)
        del va,vb
    atomic_json(root/'RESUME_QUALIFICATION.json',dict(status='PASS',world=8,
        continuous_updates=3,interrupted_after=1,resumed_remaining=2,
        model_optimizer_scheduler_RNG_builder_exact=True,qualification_weights_used_in_formal=False))


def register(root):
    import torch
    pre=read(root/'PREREGISTRATION.json');cal=read(root/'calibration/CALIBRATION.json')
    body=dict(cal);seal=body.pop('sha256')
    if fingerprint(body)!=seal or cal['status']!='PASS' or cal['split']!='TRAIN' or cal['scenes']!=32:
        raise ValueError('Invalid one-time TRAIN calibration')
    if cal['parent_model_sha256']!=pre['parent_model_sha256']:
        raise ValueError('Calibration belongs to another parent')
    client=torch.load(Path(pre['parent_checkpoint'])/'rank00_state.pt',map_location='cpu',weights_only=False)
    row=dict(schema='world_coordinate_loss_study_v1',parent_checkpoint=pre['parent_checkpoint'],
        parent_model_sha256=pre['parent_model_sha256'],parent_identity_sha256=pre['parent_identity_sha256'],
        parent_updates=63000,added_updates=2000,seed=42,evaluation_seed=2101,
        physical_tolerances=pre['physical_tolerances'],horizon_weights=[1.]*8,
        freeze_parent_candidate_bank=True,candidate_bank_hash=client['builder']['bank_hash'],
        arms=dict(CONTROL=dict(weights=[1.,1.,1.],scale=1.),
            PHYSICAL=dict(weights=tolerance_weights(pre['physical_tolerances']),scale=cal['fixed_scale'])),
        calibration_sha256=seal,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip())
    row['sha256']=fingerprint(row);atomic_json(root/'REGISTRATION.json',row)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    lock=(root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'OWNER.json').exists():raise FileExistsError('Do not duplicate study owner')
    atomic_json(root/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),source=str(SOURCE),at=time.time()))
    try:
        wait_all([root/'jobs/calibration_attempt003']);register(root)
        first='training-vlawm-tsinghua-zt';second='training-vlawm-tsinghua-zt3'
        q=root/'qualification';jobs=[launch(root,'qual_continuous',first,list(range(8)),train_command(root,'CONTROL',q/'continuous',qualification=3)),
            launch(root,'qual_interrupt',second,list(range(8)),train_command(root,'CONTROL',q/'resumed',qualification=3,pause=1))]
        wait_all(jobs)
        jobs=[launch(root,'qual_physical',first,list(range(8)),train_command(root,'PHYSICAL',q/'physical',qualification=3)),
            launch(root,'qual_resume',second,list(range(8)),train_command(root,'CONTROL',q/'resumed',qualification=3,resume=q/'resumed/checkpoints/update063001'))]
        wait_all(jobs);qualify_resume(root)
        atomic_json(root/'STATUS.json',dict(stage='FORMAL_TRAINING',at=time.time(),endpoint=65000))
        jobs=[launch(root,'train_'+arm,host,list(range(8)),train_command(root,arm,root/'runs'/arm))
              for arm,host in [('CONTROL',first),('PHYSICAL',second)]]
        wait_all(jobs)
        models={}
        for arm in ('CONTROL','PHYSICAL'):
            run=root/'runs'/arm;progress=read(run/'progress.json')
            if progress['status']!='STUDY_COMPLETE' or progress['updates']!=65000:raise ValueError('Missing registered endpoint')
            checkpoint=Path(progress['checkpoint']);seal=read(checkpoint/'COMPLETE');identity=read(checkpoint/'identity.json')
            label=arm+'_065000'
            models[label]=dict(label=label,kind='IL',updates=65000,checkpoint=str(checkpoint),parent_checkpoint=str(checkpoint),
                files_sha256=seal['files_sha256'],native_identity_sha256=fingerprint(identity),source=str(SOURCE),
                user_instruction='按建议进行坐标损失匹配实验，以完整Navtest评价固定终点')
        evaluation=root/'NAVTEST';evaluation.mkdir()
        atomic_json(evaluation/'PINNED.json',dict(schema='world_checkpoint_sweep_NAVTEST_v1',models=models,
            max_concurrent_models=2,reference_evaluation='/mnt/project/qwen-world-postcot-20261025/IL_midlate_loss_NAVTEST_20261028T102909Z/WORLD_IL_063000/evaluation',
            seed=2101,full_population=12146,coordinate_loss_study=dict(root=str(root))))
        atomic_json(root/'STATUS.json',dict(stage='FULL_NAVTEST',at=time.time()))
        with (evaluation/'owner.log').open('xb') as log:
            subprocess.run([MODEL_PYTHON,'-u','-m','tools.qwen_world_postcot.latest_snapshot_owner',
                '--root',str(evaluation),'--report-dir',str(SOURCE/'reports/qwen_world_coordinate_loss_20261028')],
                cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (evaluation/'COMPLETE.json').exists():raise RuntimeError('Test owner did not complete')
        atomic_json(root/'COMPLETE.json',dict(status='COMPLETE',at=time.time(),publication=read(evaluation/'PUBLICATION.json')))
    except BaseException:
        atomic_json(root/'FAILURE.json',dict(at=time.time(),error=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':main()
