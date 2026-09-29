"""Finite registered candidate queue on one authorized GPU slot; milestone export and CPU scoring."""
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
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('registration','campaign-root','local-root','qwen','sources','candidates','gpus','slot-id',
              'pressure-script','pressure-python','dev-data','devkit','metric-index','scoring-python'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--master-port',type=int,required=True);p.add_argument('--resume-slot',action='store_true')
    a=p.parse_args();root=Path(a.campaign_root);local=Path(a.local_root);reg=read(a.registration);work=Path.cwd()
    if subprocess.check_output(['git','status','--porcelain']).strip() or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=reg['training_source_sha']:raise ValueError('Registered clean source required')
    cards=a.gpus.split(',');candidates=a.candidates.split(',')
    if len(set(cards))!=len(cards) or len(set(candidates))!=len(candidates):raise ValueError('Unique slot/candidate assignment required')
    out=root/'formal_queues'/(a.slot_id+'.json');out.parent.mkdir(exist_ok=True)
    lock=out.with_suffix('.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    contract={'registration_sha256':file_sha256(a.registration),'gpus':cards,'candidates':candidates,'source':reg['training_source_sha']}
    if out.exists():
        state=read(out)
        if not a.resume_slot or state['contract']!=contract:raise ValueError('Existing slot requires explicit unchanged resume')
    else:state={'contract':contract,'status':'RUNNING','events':[]}
    atomic_json(out,state)
    def publish(event):state['events'].append(event);atomic_json(out,state)
    def allocated(label,cmd):
        record=root/'allocations'/(label+'.json')
        if record.exists():raise FileExistsError('Allocation ID already used')
        wrapped=[sys.executable,'-m','tools.foresight.run_allocated','--gpus',a.gpus,'--worktree',str(work),'--record',str(record),
                 '--pressure-script',a.pressure_script,'--pressure-python',a.pressure_python,'--',*cmd]
        with (root/'logs'/(label+'.log')).open('xb') as stream:
            proc=subprocess.Popen(wrapped,stdout=stream,stderr=subprocess.STDOUT)
        publish({'event':'START','label':label,'pid':proc.pid,'command':wrapped,'time':time.time()})
        while proc.poll() is None:
            if (root/'STOP_REQUESTED').exists() or charged_gpu_hours(root)>=reg['gpu_hours_cap']:
                folder=root/'students'/run_id
                if folder.exists():(folder/'STOP_REQUESTED').write_text('Registered slot stop/budget: save at optimizer boundary\n')
            time.sleep(5)
        publish({'event':'EXIT','label':label,'exit_code':proc.returncode,'time':time.time()})
        if proc.returncode:raise RuntimeError('Registered workload failed: '+label)
    try:
        for candidate in candidates:
            run_id=f'formal_{candidate}_seed42_v1';spec=reg['runs'][run_id]
            if len(cards)!=spec['gpus']:raise ValueError('Wrong GPU count for registered slot')
            if (root/'STOP_REQUESTED').exists() or charged_gpu_hours(root)>=reg['gpu_hours_cap']:
                state['status']='PAUSED';atomic_json(out,state);return
            replica=local/'targets'/candidate/'local_replica.json'
            if not replica.exists() or read(replica)['images']!=read(root/'dino_targets_v1'/candidate/'identity.json')['image_count']:
                cmd=[sys.executable,'-m','tools.dino_tradeoff.stage_targets','--source',str(root/'dino_targets_v1'),'--output',str(local/'targets'),'--candidate',candidate]
                with (root/'logs'/(a.slot_id+'_'+candidate+'_stage.log')).open('a') as stream:
                    subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
            run=root/'students'/run_id
            for endpoint in [x for x in reg['development_updates'] if x<=reg['screen_updates']]:
                status=read(run/'status.json') if (run/'status.json').exists() else None
                if not status or status['completed']<endpoint:
                    if status and not a.resume_slot and status['status'] not in ('PAUSED',):raise ValueError('Do not duplicate an existing active/failed run')
                    attempt=time.time_ns();cmd=[sys.executable,'-m','tools.full_foresight.run_student',
                        '--candidate',candidate,'--campaign-root',str(root),'--data',str(local/'student_train_v1'),
                        '--image-root',str(local/'images'),'--targets',str(local/'targets'),'--index',str(root/'dino_index_v1'),
                        '--interaction-root',str(local/'interaction_train_v1'),'--calibration',str(root/'four_loss_calibration_v1.json'),
                        '--teacher-verification',str(root/'teacher_reuse_verification.json'),'--qwen',a.qwen,'--sources',a.sources,
                        '--run-id',run_id,'--gpus',str(spec['gpus']),'--micro-batch',str(spec['micro_batch']),'--master-port',str(a.master_port),
                        '--scope','formal','--updates',str(spec['updates']),'--schedule-updates',str(spec['schedule_updates']),
                        '--warmup',str(reg['warmup']),'--max-seconds','216000','--campaign-gpu-hours',str(reg['gpu_hours_cap']),
                        '--save-every',str(reg['save_every']),'--milestones',','.join(map(str,reg['milestones'])),'--stop-after',str(endpoint),
                        '--registration',a.registration,'--deterministic']
                    if status:cmd+=['--resume','--acknowledge-stop']
                    allocated(f'{run_id}_to{endpoint}_{attempt}',cmd)
                    status=read(run/'status.json')
                    if status['completed']!=endpoint or status['status']!='PAUSED':
                        state['status']='PAUSED';atomic_json(out,state);return
                label=f'{run_id}_dev{endpoint}_seed42';export=root/'evaluations'/label
                done=all((export/f'shard_{i}.json').exists() and read(export/f'shard_{i}.json')['status']=='complete' for i in range(spec['gpus']))
                if not done:
                    allocated(label+'_'+str(time.time_ns()),[sys.executable,'-m','tools.full_foresight.export_development',
                        '--training-run',str(run),'--checkpoint-tag',f'milestone_{endpoint:06d}','--current-root',a.dev_data,
                        '--output',str(export),'--campaign-root',str(root),'--run-id',label+'_'+str(time.time_ns()),
                        '--gpus',str(spec['gpus']),'--sampling-seed','42','--campaign-gpu-hours',str(reg['gpu_hours_cap'])])
                scores=root/'scores'/label;scores.parent.mkdir(exist_ok=True)
                scoring_gate=root/'scoring_protocol_v1.json'
                approved=scoring_gate.exists() and read(scoring_gate).get('passed') is True
                if not approved:publish({'event':'SCORING_PENDING_AUDIT','export':str(export),'time':time.time()})
                if approved and not (scores/'summary.json').exists():
                    cmd=[a.scoring_python,'-m','tools.foresight.score_pdms','--devkit',a.devkit,'--metric-index',a.metric_index,
                        '--current-index',str(Path(a.dev_data)/'index.json'),'--predictions',str(export),'--output',str(scores),
                        '--campaign-root',str(root),'--run-id',label+'_score_'+str(time.time_ns()),'--workers','16']
                    if scores.exists():cmd+=['--resume']
                    with (root/'logs'/(label+'_score.log')).open('a') as stream:
                        proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    publish({'event':'CPU_SCORE','pid':proc.pid,'command':cmd,'time':time.time()})
                ego=root/'evaluations'/(label+'_ego.json')
                if not ego.exists():
                    subprocess.run([sys.executable,'-m','tools.foresight.evaluate_ego','--predictions',str(export),'--current-root',a.dev_data,'--output',str(ego)],check=True)
            publish({'event':'SCREEN_PAUSED','run_id':run_id,'updates':reg['screen_updates'],'time':time.time()})
        state['status']='SCREEN_TRAINING_COMPLETE_EVALUATION_TRACKED_SEPARATELY';atomic_json(out,state)
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));atomic_json(out,state);raise


if __name__=='__main__':main()
