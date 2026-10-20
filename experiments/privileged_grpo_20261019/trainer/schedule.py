"""Bounded independent-replica queue; no cross-model DDP gradient sharing."""
import argparse,collections,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/grpo_smoke_v1'))
from input_protocol import validate_protocol
STOP=False
def stop(signum,frame):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def jobs(cfg,phase):
    if phase=='initial':return [{'arm':a,'seed':0} for a in ['A','B','C','F','E']]
    # Keep A/C/E within a scope together; fixed queue, identical budgets.
    ordered=['A','C','E','A_VLM','C_VLM','E_VLM','A_DiT','C_DiT','E_DiT','B','F']
    return [{'arm':a,'seed':s} for s in cfg['training_seeds'] for a in ordered]
def recover_queue(root, pending, cfg, phase_identity):
    root=Path(root)
    if read(root/'TRAINING.json')!=cfg or read(root/'PHASE_IDENTITY.json')!=phase_identity:raise ValueError('Recovery phase identity changed')
    if read(root/'QUEUE.json')['phase']!='train':raise ValueError('Only training queues support checkpoint recovery')
    for process in root.glob('*.process.json'):
        record=read(process);cmdline=Path('/proc')/str(record['pid'])/'cmdline'
        if cmdline.exists():
            args=cmdline.read_bytes().split(b'\0')
            if str(ROOT/'run.py').encode() in args and any(str(root).encode() in arg for arg in args):raise ValueError('Old phase still has a live task process')
    old_results=read(root/'RESULT.json').get('jobs',[]) if (root/'RESULT.json').exists() else []
    by_job={(r['job']['arm'],r['job']['seed']):r for r in old_results}
    queue=[];reused=[]
    for job in pending:
        name=f"{job['arm']}_seed{job['seed']}";prior=by_job.get((job['arm'],job['seed']));folder=Path(prior['output']) if prior else root/name
        result=read(folder/'RESULT.json') if (folder/'RESULT.json').exists() else {}
        if result.get('status')=='COMPLETE' and result.get('completed_updates')==cfg['epochs']*316:
            reused.append({'job':job,'exit_code':0,'status':'COMPLETE','output':str(folder),'reused_from':str(root)});continue
        checkpoints=[]
        for sidecar in (folder/'checkpoints').glob('*.json'):
            checkpoint=sidecar.with_suffix('.pt')
            if checkpoint.exists():checkpoints.append((read(sidecar)['completed_updates'],str(checkpoint)))
        recovered=dict(job)
        if checkpoints:recovered['resume']=max(checkpoints)[1]
        queue.append(recovered)
    return collections.deque(queue),reused

def main(a):
    global STOP
    cfg=read(a.training)
    validate_protocol(cfg)
    if not cfg.get('formal_protocol_locked') or not cfg.get('epochs') or not cfg.get('training_seeds') or not cfg.get('wall_seconds'):raise ValueError('Formal configuration still unlocked')
    prep=read(Path(a.prepared)/'PREPARATION.json')
    if prep['status']!='PASS':raise ValueError('Full F coverage gate failed')
    phase_identity={'prepared_sha256':hashlib.sha256((Path(a.prepared)/'PREPARATION.json').read_bytes()).hexdigest(),'runtime_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(list(ROOT.glob('*.py'))+list((ROOT/'vendor').rglob('*.py'))) if '__pycache__' not in p.parts}}
    devices=a.devices.split(',')
    if len(devices)!=len(set(devices)) or any(not x.isdigit() for x in devices):raise ValueError('Device list must have unique integer indices')
    phase_root=Path(a.out).resolve();phase_root.mkdir(parents=True,exist_ok=False);queue=collections.deque(jobs(cfg,a.phase));save(phase_root/'QUEUE.json',{'phase':a.phase,'devices':devices,'jobs':list(queue),'cross_model_gradient_sync':False})
    if a.plan_only:print(json.dumps({'jobs':len(queue),'max_parallel':len(devices),'launch':False}));return
    import torch
    if not torch.cuda.is_available() or max(map(int,devices))>=torch.cuda.device_count():raise ValueError('Requested physical devices not visible')
    if a.phase!='initial':
        if not a.previous:raise ValueError('Prior phase acceptance required')
        previous=read(Path(a.previous)/'RESULT.json')
        if previous['status']!='COMPLETE' or previous['phase']!=('initial' if a.phase=='train' else 'train'):raise ValueError('Prior phase not complete')
        if read(Path(a.previous)/'TRAINING.json')!=cfg:raise ValueError('Prior phase configuration changed')
        if read(Path(a.previous)/'PHASE_IDENTITY.json')!=phase_identity:raise ValueError('Prior phase prompts/runtime changed')
    save(phase_root/'TRAINING.json',cfg);save(phase_root/'PHASE_IDENTITY.json',phase_identity);active={};results=[];failed=False;start=time.monotonic();last_report=0
    if a.recover_from:
        if a.phase!='train':raise ValueError('Queue recovery currently requires phase=train')
        queue,results=recover_queue(a.recover_from,list(queue),cfg,phase_identity)
    while queue or active:
        if time.monotonic()-start>=a.scheduler_wall_seconds:STOP=True
        if STOP or failed:
            queue.clear()
            for item in active.values():
                if not item['stop_sent']:item['process'].send_signal(signal.SIGTERM);item['stop_sent']=True
        for device in devices:
            if device in active or not queue or STOP or failed:continue
            job=queue.popleft();name=f"{job['arm']}_seed{job['seed']}";out=phase_root/name
            command=[sys.executable,'-u',str(ROOT/'run.py'),'--assets',str(Path(a.assets).resolve()),'--training',str(Path(a.training).resolve()),'--prepared',str(Path(a.prepared).resolve()),'--out',str(out),'--arm',job['arm'],'--seed',str(job['seed']),'--mode',a.phase]
            if a.phase=='final':
                prior=next(r for r in previous['jobs'] if r['job']['arm']==job['arm'] and r['job']['seed']==job['seed'])
                trained=read(Path(prior['output'])/'RESULT.json');command+=['--resume',trained['checkpoint']]
            elif job.get('resume'):command+=['--resume',job['resume']]
            log=(phase_root/(name+'.log')).open('x');env={**os.environ,'CUDA_VISIBLE_DEVICES':device,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            active[device]={'process':proc,'log':log,'out':out,'job':job,'stop_sent':False};save(phase_root/(name+'.process.json'),{'pid':proc.pid,'device':device,'command':command,'job':job})
            save(phase_root/'STATUS.json',{'phase':a.phase,'stage':'launching','active':{d:{'pid':x['process'].pid,**x['job']} for d,x in active.items()},'queued':len(queue),'completed':len(results),'elapsed_s':time.monotonic()-start})
            # Limit simultaneous large checkpoint loads; do not change compute budget.
            time.sleep(a.launch_stagger_seconds)
        for device,item in list(active.items()):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();result=read(item['out']/'RESULT.json') if (item['out']/'RESULT.json').exists() else {'status':'MISSING_RESULT'}
            ok=code==0 and result['status']=='COMPLETE';results.append({'job':item['job'],'exit_code':code,'status':result['status'],'output':str(item['out'])});failed|=not ok;del active[device]
        if time.monotonic()-last_report>=60:
            save(phase_root/'STATUS.json',{'phase':a.phase,'active':{d:{'pid':x['process'].pid,**x['job']} for d,x in active.items()},'queued':len(queue),'completed':len(results),'failed':failed,'stopping':STOP,'elapsed_s':time.monotonic()-start});last_report=time.monotonic()
        if active:time.sleep(1)
    expected=jobs(cfg,a.phase);complete=not failed and not STOP and len(results)==len(expected)
    save(phase_root/'RESULT.json',{'status':'COMPLETE' if complete else 'INCOMPLETE','phase':a.phase,'jobs':results,'expected_jobs':len(expected),'elapsed_s':time.monotonic()-start});return 0 if complete else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--training',required=True);p.add_argument('--prepared',required=True);p.add_argument('--phase',choices=['initial','train','final'],required=True);p.add_argument('--devices',required=True);p.add_argument('--out',required=True);p.add_argument('--previous');p.add_argument('--recover-from');p.add_argument('--plan-only',action='store_true');p.add_argument('--launch-stagger-seconds',type=float,default=20);p.add_argument('--scheduler-wall-seconds',type=float,required=True);a=p.parse_args()
    if a.scheduler_wall_seconds<=0 or not 0<=a.launch_stagger_seconds<=60:p.error('Invalid scheduler bounds')
    raise SystemExit(main(a))
