"""Bounded existing-resource campaign runner; never launches historic controllers.

The immutable JSON plan declares allocated host/device lists and clean source.
Each arm starts from generic/random initialization. Only this run's planned
allocation pauses may resume automatically. A STOP_REQUESTED file always wins.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
from .prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def occupied_devices(host):
    command=['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits']
    if host: command=['ssh','-o','BatchMode=yes',host,shlex.join(command)]
    result=subprocess.check_output(command,text=True,timeout=30)
    return {int(line.split(',')[0]):int(line.split(',')[1]) for line in result.splitlines() if line.strip()}


def charged_gpu_hours(root):
    paths=list((root/'runs').glob('*/status.json'))+list((root/'training').glob('*/attempt_*.json'))
    total=0.
    for path in paths:
        value=json.loads(path.read_text())
        end=time.time() if value['status']=='RUNNING' else value.get('end_unix',value['start_unix'])
        total+=(end-value['start_unix'])*value['gpu_count']/3600
    return total


def validate_allocations(plan):
    for phase in (42,43):
        slots=[m['training_slot'] for m in plan['models'] if m['seed']==phase]
        resources=[(s.get('host'),g) for s in slots for g in s['gpus']]
        if len(set(resources))!=len(resources):raise ValueError('Overlapping parallel training device assignments')
    resources=[(s.get('host'),g) for s in plan['evaluation_slots'] for g in s['gpus']]
    if len(set(resources))!=len(resources):raise ValueError('Overlapping evaluation device assignments')
    for slot in [m['training_slot'] for m in plan['models']]+plan['evaluation_slots']:
        if slot.get('host') not in (None,'training-vla-zt2') or not slot['gpus'] or any(g not in range(8) for g in slot['gpus']):
            raise ValueError('Only explicitly authorized existing local/vla-zt2 resources are supported')
    for model in plan['models']:
        if not re.fullmatch(r'formal_[ABC]_seed(?:42|43)_\d+',model['run_id']):
            raise ValueError('Use a new named formal run inside this campaign')


class Campaign:
    def __init__(self, plan, plan_path, directory):
        self.plan,self.directory=plan,Path(directory)
        self.plan_path=str(Path(plan_path).resolve())
        self.root=Path(plan['campaign_root']);self.abort=threading.Event()
        self.active_train_runs=set();self.guard=threading.Lock()
        self.status={'status':'RUNNING','pid':os.getpid(),'host':socket.gethostname(),
            'plan_sha256':file_sha256(plan_path),'source_sha':plan['source_sha'],
            'phase':'starting','started_unix':time.time(),'completed_jobs':[]}

    def stopped(self):
        return self.abort.is_set() or (self.directory/'STOP_REQUESTED').exists()

    def budget_available(self):
        # Preserve a buffer for checkpoint writes and in-flight export scenes.
        return charged_gpu_hours(self.root) < self.plan['gpu_hour_cap']-10

    def save(self):
        with self.guard:
            self.status['gpu_hours']=charged_gpu_hours(self.root)
            self.status['updated_unix']=time.time()
            atomic_json(self.directory/'status.json',self.status)

    def device_wait(self, slot):
        while not self.stopped() and self.budget_available():
            memory=occupied_devices(slot.get('host'))
            if all(memory.get(g,100000)<512 for g in slot['gpus']):return
            time.sleep(15)
        raise InterruptedError('Campaign stopped or budget reserve reached before GPU allocation')

    def command(self, args, slot, label):
        uses_gpu=bool(slot['gpus'])
        if self.stopped() or (uses_gpu and not self.budget_available()):raise InterruptedError('Campaign stopped/budget')
        worktree=self.plan['worktree']
        threads='4' if uses_gpu else '1'
        env={'CUDA_VISIBLE_DEVICES':','.join(map(str,slot['gpus'])), 'OMP_NUM_THREADS':threads,
             'MKL_NUM_THREADS':threads,'OPENBLAS_NUM_THREADS':threads,'NO_ALBUMENTATIONS_UPDATE':'1',
             'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1','MAX_JOBS':'4'}
        argv=['env',*[f'{k}={v}' for k,v in env.items()],*map(str,args)]
        if slot.get('host'):
            argv=['ssh','-o','BatchMode=yes',slot['host'],'cd '+shlex.quote(worktree)+' && '+shlex.join(argv)]
            cwd=None
        else:cwd=worktree
        log=self.directory/(label+f'_{time.time_ns()}.log')
        with log.open('w') as stream:
            process=subprocess.Popen(argv,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if self.stopped() or (uses_gpu and not self.budget_available()):
                    self.abort.set()
                    # Only campaign-owned trainers receive a cooperative stop.
                    # Prediction exporters have an explicit short time bound.
                    with self.guard:
                        for run in self.active_train_runs:
                            folder=self.root/'training'/run
                            if folder.exists():(folder/'STOP_REQUESTED').touch(exist_ok=True)
                time.sleep(5)
            if process.returncode:raise RuntimeError(f'{label} failed ({process.returncode}); log={log}')

    def train(self, model):
        slot=model['training_slot'];run=model['run_id'];directory=self.root/'training'/run
        common=self.plan['data'];completed=False
        training_cap=self.plan['gpu_hour_cap']-self.plan['final_evaluation_reserve_gpu_hours']
        while not completed:
            if file_sha256(model['config'])!=model['config_sha256']:raise ValueError('Registered model configuration changed')
            if self.stopped():raise InterruptedError('Explicit campaign stop')
            if charged_gpu_hours(self.root)>=training_cap-2:return 'BUDGET_PAUSED'
            previous=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else None
            if previous and previous['status']=='COMPLETE':return
            if previous and previous['status']!='PAUSED':
                raise RuntimeError(f'Refuse duplicate or failed run {run}: {previous["status"]}')
            if (directory/'STOP_REQUESTED').exists():raise InterruptedError(f'Explicit trainer stop: {run}')
            self.device_wait(slot)
            args=[self.plan['python'],'-m','torch.distributed.run','--standalone',
                f'--nproc_per_node={len(slot["gpus"])}','-m','tools.ddpolicy_vehicle.train',
                '--config',model['config'],'--processed-root',common['processed_root'],
                '--vehicle-root',common['vehicle_root'],'--depth-root',common['depth_root'],
                '--tokens',common['train_tokens'],'--campaign-root',str(self.root),'--run-id',run,
                '--global-batch','32','--micro-batch','4','--updates','100000','--save-every','1000',
                '--milestones','0,1000,5000,10000,25000,50000,75000,90000,100000',
                '--campaign-gpu-hours',str(training_cap-2),
                '--max-seconds',str(self.plan.get('allocation_seconds',86400))]
            if previous:args+=['--resume','--acknowledge-stop']
            with self.guard:self.active_train_runs.add(run)
            try:self.command(args,slot,run)
            finally:
                with self.guard:self.active_train_runs.discard(run)
            state=json.loads((directory/'status.json').read_text())
            completed=state['status']=='COMPLETE'
            if not completed and state['status']!='PAUSED':raise RuntimeError(f'Unexpected training state: {state["status"]}')
        return 'COMPLETE'

    def prediction_score(self, job, slot):
        model,tag,split=job
        root=self.root/'formal_evaluation'/model['run_id']/tag/split
        for seed in self.plan['sampling_seeds']:
            prediction=root/f'predictions_seed{seed}';score=root/f'scores_seed{seed}'
            marker=prediction/'shard_0.json'
            while not marker.exists() or json.loads(marker.read_text())['status']!='complete':
                self.device_wait(slot)
                run_id=f'export_{model["run_id"]}_{tag}_{split}_{seed}_{time.time_ns()}'
                args=[self.plan['python'],'-m','tools.ddpolicy_vehicle.export_predictions',
                    '--training-run',str(self.root/'training'/model['run_id']),'--checkpoint-tag',tag,
                    '--current-root',self.plan['data'][f'current_{split}'],'--output',str(prediction),
                    '--sampling-seed',str(seed),'--campaign-root',str(self.root),'--run-id',run_id,
                    '--max-seconds','3600','--local-checkpoint-cache',self.plan['local_checkpoint_cache']]
                if split=='navtest':args+=['--final-lock',str(self.directory/'final_lock.json')]
                self.command(args,slot,run_id)
            if json.loads(marker.read_text())['failed']:raise RuntimeError('Failed predictions retained')
            if not (score/'summary.json').exists():
                args=[self.plan['scoring_python'],'-m','tools.local_interaction_mask_v2.score_async',
                    '--devkit',self.plan['devkit'],'--index',self.plan['data'][f'metric_{split}'],
                    '--predictions',str(prediction),'--output',str(score),'--workers','8','--chunk','8','--export-shards','1']
                if split=='navtest':args+=['--benchmark-navtest']
                self.command(args,{'gpus':[]},f'score_{model["run_id"]}_{tag}_{split}_{seed}')
            summary=json.loads((score/'summary.json').read_text())
            expected=1696 if split=='dev' else 12146
            if not summary['valid'] or summary['scenes']!=expected or summary['failed']:
                raise RuntimeError('Incomplete/invalid full evaluation')

    def parallel(self, jobs, function, phase):
        self.status['phase']=phase;self.save()
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures={pool.submit(function,job):str(job) for job in jobs}
            while futures:
                finished,_=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
                for future in finished:
                    label=futures.pop(future)
                    try:future.result()
                    except BaseException:
                        self.abort.set();raise
                    self.status['completed_jobs'].append({'phase':phase,'job':label})
                self.save()

    def evaluate(self, models, split, selections=None):
        jobs=[(m,tag,split) for m in models for tag in (
            self.plan['development_tags'] if selections is None else [selections[m['run_id']]])]
        slots=self.plan['evaluation_slots']
        partitions=[jobs[i::len(slots)] for i in range(len(slots))]
        def worker(item):
            tasks,slot=item
            for job in tasks:self.prediction_score(job,slot)
        self.parallel(list(zip(partitions,slots)),worker,split+'_evaluation')

    def run(self):
        primary=[m for m in self.plan['models'] if m['seed']==42]
        second=[m for m in self.plan['models'] if m['seed']==43]
        self.parallel(primary,self.train,'primary_training')
        complete=lambda m:(self.root/'training'/m['run_id']/'status.json').exists() and json.loads((self.root/'training'/m['run_id']/'status.json').read_text())['status']=='COMPLETE'
        if not all(complete(m) for m in primary):
            self.status.update(status='BUDGET_PAUSED',uncompleted_phase='primary_training');self.save();return
        self.evaluate(primary,'dev')
        evaluated=primary
        if second:
            self.parallel(second,self.train,'second_seed_training')
            if all(complete(m) for m in second):
                self.evaluate(second,'dev');evaluated+=second
            else:self.status['second_seed']='NOT_RUN_TO_COMPLETION; paired checkpoints preserved, primary final evaluation prioritized'
        selection=[];tags={}
        for model in evaluated:
            candidates=[]
            for tag in self.plan['development_tags']:
                folders=[self.root/'formal_evaluation'/model['run_id']/tag/'dev'/f'scores_seed{s}' for s in self.plan['sampling_seeds']]
                value=sum(json.loads((f/'summary.json').read_text())['PDMS'] for f in folders)/len(folders)
                candidates.append((value,tag,folders))
            _,tag,folders=max(candidates,key=lambda v:(v[0],v[1]))
            tags[model['run_id']]=tag
            selection.append({'run':str(self.root/'training'/model['run_id']),'tag':tag,'dev_scores':list(map(str,folders))})
        selection_path=self.directory/'selection.json'
        if selection_path.exists() and json.loads(selection_path.read_text())!=selection:
            raise ValueError('Previously frozen development selection changed')
        if not selection_path.exists():atomic_json(selection_path,selection)
        if not (self.directory/'final_lock.json').exists():
            self.command([self.plan['python'],'-m','tools.ddpolicy_vehicle.lock_final','--selection',
                str(self.directory/'selection.json'),'--output',str(self.directory/'final_lock.json')],{'gpus':[]},'final_lock')
        self.evaluate(evaluated,'navtest',tags)
        self.status['phase']='paired_analysis';self.save()
        self.command([self.plan['python'],'-m','tools.ddpolicy_vehicle.analyse_campaign',
            '--plan',self.plan_path,'--campaign-directory',str(self.directory)],{'gpus':[]},'paired_analysis')
        self.status['status']='COMPLETE';self.save()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True);p.add_argument('--directory',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--acknowledge-stop',action='store_true')
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());directory=Path(a.directory)
    if not 0<plan['final_evaluation_reserve_gpu_hours']<plan['gpu_hour_cap'] or plan['sampling_seeds']!=[42,43,44,45,46]:raise ValueError('Invalid registered campaign protocol')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=plan['worktree'],text=True).strip()
    if actual!=plan['source_sha'] or subprocess.check_output(['git','status','--porcelain'],cwd=plan['worktree']):
        raise ValueError('Use the immutable reviewed training/evaluation source')
    keys={(m['arm'],m['seed']) for m in plan['models']}
    if len(plan['models'])!=5 or keys!={('A',42),('B',42),('C',42),('B',43),('C',43)}:raise ValueError('Expected one primary trio and one complete B/C second seed')
    if plan['development_tags']!=['milestone_025000','milestone_050000','milestone_075000','milestone_100000']:
        raise ValueError('Use the fixed common development checkpoint grid')
    validate_allocations(plan)
    from omegaconf import OmegaConf
    for model in plan['models']:
        if file_sha256(model['config'])!=model['config_sha256']:raise ValueError('Registered configuration hash mismatch')
        cfg=OmegaConf.load(model['config'])
        if cfg.from_scratch.arm!=model['arm'] or int(cfg.seed)!=model['seed'] or cfg.trainer.max_train_steps!=100000:
            raise ValueError('Formal configuration differs from declared arm/seed/schedule')
    if directory.exists() and not a.resume:raise FileExistsError('Campaign already exists')
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'RUN.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.resume:
        previous=json.loads((directory/'status.json').read_text())
        if previous['plan_sha256']!=file_sha256(a.plan):raise ValueError('Campaign plan changed')
        if not a.acknowledge_stop:raise ValueError('Explicit resume acknowledgement required')
        if (directory/'STOP_REQUESTED').exists():
            os.replace(directory/'STOP_REQUESTED',directory/f'STOP_ACKNOWLEDGED_{time.time_ns()}')
    runner=Campaign(plan,a.plan,directory)
    def stop(*_):runner.abort.set()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:runner.run()
    except BaseException as error:
        runner.status.update(status='PAUSED' if isinstance(error,InterruptedError) else 'FAILED',error=repr(error));runner.save();raise
    finally:lock.close()


if __name__=='__main__':main()
