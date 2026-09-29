"""Finite full-four-loss profile queue: single4, true concurrent2x4, or single8."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.campaign import charged_gpu_hours


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('campaign-root','local-root','index','qwen','sources','host-tag','pressure-script','pressure-python'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--mode',choices=('startup4','single4','paired4','single8'),required=True)
    p.add_argument('--candidates',default='C0,C5');p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--base-port',type=int,default=29641);p.add_argument('--gpu-hours-cap',type=float,default=192)
    a=p.parse_args();root=Path(a.campaign_root);local=Path(a.local_root);work=Path.cwd()
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Lock profile source')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    candidates=a.candidates.split(',');gpus=list(map(int,a.gpus.split(',')))
    if len(set(candidates))!=len(candidates) or not candidates or any(c not in [f'C{i}' for i in range(6)] for c in candidates):
        raise ValueError('Unique registered candidates required')
    if len(gpus)!=8 or len(set(gpus))!=8 or min(gpus)<0 or max(gpus)>7:raise ValueError('Explicit host-local eight-GPU inventory required')
    if a.mode=='paired4' and len(candidates)%2:raise ValueError('Concurrent profile requires complete pairs')
    cal=root/'four_loss_calibration_v1.json';teacher=root/'teacher_reuse_verification.json'
    if not json.loads(cal.read_text())['passed'] or not json.loads((root/'full_real_resume_comparison.json').read_text())['passed']:
        raise ValueError('Real full-method gradients and resume must pass first')
    tag=f'{a.host_tag}_{a.mode}_{sha[:7]}';queue=root/'queues'/f'{tag}.json';queue.parent.mkdir(exist_ok=True)
    lock=queue.with_suffix('.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if queue.exists():raise FileExistsError('Existing queue must be inspected, never blindly relaunched')
    stage=json.loads((local/'student_train_v1/local_stage.json').read_text())
    if stage['scenes']<3840:raise ValueError('Profile current camera population not staged')
    rows=json.loads((Path(a.index)/'train_scenes.json').read_text())[:3840]
    needed=1+max(x for row in rows for h in row['images'] for x in h)
    for c in candidates:
        replica=json.loads((local/'targets'/c/'local_replica.json').read_text())
        identity=json.loads((local/'targets'/c/'identity.json').read_text())
        if replica['identity']!=identity['identity'] or replica['images']<needed:
            raise ValueError('Full current/future profile targets not staged: '+c)
    ii=json.loads((local/'interaction_train_v1/COMPLETE.json').read_text())
    if ii['scenes']!=len(json.loads((local/'student_train_v1/index.json').read_text())):
        raise ValueError('Local frozen MAE export population differs')
    queue_state={'source_sha':sha,'mode':a.mode,'status':'RUNNING','candidates':candidates,
                 'scene_prefix':3840,'global_batch':32,'all_four_losses':True,'jobs':[],'started_unix':time.time()}
    atomic_json(queue,queue_state)
    width=2 if a.mode=='paired4' else 1
    for first in range(0,len(candidates),width):
        if (root/'STOP_REQUESTED').exists() or charged_gpu_hours(root)>=a.gpu_hours_cap:
            queue_state['status']='PAUSED';atomic_json(queue,queue_state);return
        processes=[]
        for slot,c in enumerate(candidates[first:first+width]):
            cards=gpus if a.mode=='single8' else gpus[slot*4:slot*4+4]
            run_id=f'p0_{tag}_{c}';updates=4 if a.mode=='startup4' else 120
            record=root/'allocations'/(run_id+'.json');record.parent.mkdir(exist_ok=True)
            cmd=[sys.executable,'-m','tools.foresight.run_allocated','--gpus',','.join(map(str,cards)),
                '--worktree',str(work),'--record',str(record),'--pressure-script',a.pressure_script,
                '--pressure-python',a.pressure_python,'--',sys.executable,'-m','tools.full_foresight.run_student',
                '--candidate',c,'--campaign-root',str(root),'--data',str(local/'student_train_v1'),
                '--image-root',str(local/'images'),'--targets',str(local/'targets'),'--index',a.index,
                '--interaction-root',str(local/'interaction_train_v1'),'--calibration',str(cal),
                '--teacher-verification',str(teacher),'--qwen',a.qwen,'--sources',a.sources,'--run-id',run_id,
                '--gpus',str(len(cards)),'--micro-batch',str(32//len(cards)),
                '--master-port',str(a.base_port+first+slot),'--scope','startup' if updates==4 else 'profile',
                '--updates',str(updates),'--schedule-updates','100000','--warmup','5000',
                '--max-seconds','1800' if updates==4 else '14400','--campaign-gpu-hours',str(a.gpu_hours_cap),
                '--limit','3840','--save-every','100','--milestones','0','--deterministic']
            log=root/'logs'/(run_id+'.log');log.parent.mkdir(exist_ok=True)
            with log.open('xb') as stream:proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
            entry={'run_id':run_id,'gpu_ids':cards,'pid':proc.pid,'command':cmd,'started_unix':time.time()}
            processes.append((proc,entry));queue_state['jobs'].append(entry)
        atomic_json(queue,queue_state)
        while any(proc.poll() is None for proc,_ in processes):
            if (root/'STOP_REQUESTED').exists() or charged_gpu_hours(root)>=a.gpu_hours_cap:
                for proc,entry in processes:
                    folder=root/'students'/entry['run_id']
                    if proc.poll() is None and folder.exists():
                        (folder/'STOP_REQUESTED').write_text('Full-method queue user stop/budget; save at optimizer boundary\n')
            time.sleep(5)
        for proc,entry in processes:
            entry.update(exit_code=proc.returncode,ended_unix=time.time())
            path=root/'students'/entry['run_id']/'status.json'
            entry['student_status']=json.loads(path.read_text())['status'] if path.exists() else 'NOT_STARTED'
        atomic_json(queue,queue_state)
        if any(entry['exit_code'] or entry['student_status']!='COMPLETE' for _,entry in processes):
            queue_state['status']='PAUSED' if (root/'STOP_REQUESTED').exists() else 'FAILED';atomic_json(queue,queue_state);return
    queue_state.update(status='COMPLETE',ended_unix=time.time());atomic_json(queue,queue_state)

if __name__=='__main__':main()
