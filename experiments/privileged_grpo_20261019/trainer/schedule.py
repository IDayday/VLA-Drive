"""Bounded independent-replica queue; no cross-model DDP gradient sharing."""
import argparse,collections,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/grpo_smoke_v1'))
from input_protocol import validate_protocol
from contracts import validate_training,validate_prepared,device_ids
STOP=False
def stop(signum,frame):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def accepted_job(folder,job,phase,cfg):
    folder=Path(folder);result=read(folder/'RESULT.json')
    if result.get('status')!='COMPLETE' or result.get('mode')!=phase or result.get('arm')!=job['arm'] or result.get('seed')!=job['seed']:
        raise ValueError('Job result identity/status mismatch: '+str(folder))
    if phase=='train':
        cp=Path(result['checkpoint']);side=read(cp.with_suffix('.json'))
        if result.get('completed_updates')!=cfg['epochs']*316 or side['completed_updates']!=cfg['epochs']*316 or not cp.is_file():raise ValueError('Training endpoint/checkpoint incomplete')
        if side['arm']!=job['arm'] or side['seed']!=job['seed']:raise ValueError('Checkpoint identity mismatch')
    else:
        receipt=read(folder/'evaluation'/phase/'ACCEPTANCE.json')
        expected=141*3
        if receipt.get('status')!='PASS' or receipt.get('groups')!=expected or receipt.get('candidates')!=expected*8 or receipt.get('token_count')!=141 or len(set(receipt.get('files',[])))!=expected:raise ValueError('Evaluation coverage incomplete')
    return result
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
        if result.get('status')=='COMPLETE':
            accepted_job(folder,job,'train',cfg)
            reused.append({'job':job,'exit_code':0,'status':'COMPLETE','output':str(folder),'reused_from':str(root)});continue
        checkpoints=[]
        for sidecar in (folder/'checkpoints').glob('*.json'):
            checkpoint=sidecar.with_suffix('.pt')
            if checkpoint.exists():
                side=read(sidecar)
                if side.get('arm')!=job['arm'] or side.get('seed')!=job['seed']:raise ValueError('Recovery checkpoint belongs to another job')
                checkpoints.append((side['completed_updates'],str(checkpoint)))
        recovered=dict(job)
        if checkpoints:recovered['resume']=max(checkpoints)[1]
        queue.append(recovered)
    return collections.deque(queue),reused

def main(a):
    global STOP
    STOP=False
    cfg=read(a.training)
    validate_training(cfg,formal=not a.plan_only)
    prepared_error=None
    try:validate_prepared(a.prepared,cfg,complete=True)
    except (ValueError,KeyError,OSError) as exc:
        if not a.plan_only:raise
        prepared_error=str(exc)
    devices=device_ids(a.devices)
    if a.plan_only:
        pending=jobs(cfg,a.phase) if a.phase=='initial' or cfg.get('training_seeds') else []
        phase_root=Path(a.out).resolve();phase_root.mkdir(parents=True,exist_ok=False)
        value={'phase':a.phase,'devices':devices,'jobs':pending,'cross_model_gradient_sync':False,'launch':False,'formal_locked':cfg.get('formal_protocol_locked',False),'prepared_error':prepared_error}
        save(phase_root/'QUEUE.json',value);print(json.dumps(value));return 0
    phase_identity={'assets_sha256':hashlib.sha256(Path(a.assets).read_bytes()).hexdigest(),'prepared_sha256':hashlib.sha256((Path(a.prepared)/'PREPARATION.json').read_bytes()).hexdigest(),'runtime_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(list(ROOT.glob('*.py'))+list((ROOT/'vendor').rglob('*.py'))) if '__pycache__' not in p.parts}}
    import torch
    if not torch.cuda.is_available() or max(map(int,devices))>=torch.cuda.device_count():raise ValueError('Requested physical devices not visible')
    if a.phase!='initial':
        if not a.previous:raise ValueError('Prior phase acceptance required')
        previous=read(Path(a.previous)/'RESULT.json')
        if previous['status']!='COMPLETE' or previous['phase']!=('initial' if a.phase=='train' else 'train'):raise ValueError('Prior phase not complete')
        if read(Path(a.previous)/'TRAINING.json')!=cfg:raise ValueError('Prior phase configuration changed')
        if read(Path(a.previous)/'PHASE_IDENTITY.json')!=phase_identity:raise ValueError('Prior phase prompts/runtime changed')
        expected_previous=jobs(cfg,'initial' if a.phase=='train' else 'train')
        if {(r['job']['arm'],r['job']['seed']) for r in previous['jobs']}!={(r['arm'],r['seed']) for r in expected_previous} or len(previous['jobs'])!=len(expected_previous):raise ValueError('Prior phase job coverage incomplete')
        for item in previous['jobs']:accepted_job(item['output'],item['job'],'initial' if a.phase=='train' else 'train',cfg)
    phase_root=Path(a.out).resolve();phase_root.mkdir(parents=True,exist_ok=False);queue=collections.deque(jobs(cfg,a.phase));save(phase_root/'QUEUE.json',{'phase':a.phase,'devices':devices,'jobs':list(queue),'cross_model_gradient_sync':False})
    save(phase_root/'TRAINING.json',cfg);save(phase_root/'PHASE_IDENTITY.json',phase_identity);active={};results=[];failed=False;start=time.monotonic();last_report=0
    if a.recover_from:
        if a.phase!='train':raise ValueError('Queue recovery currently requires phase=train')
        queue,results=recover_queue(a.recover_from,list(queue),cfg,phase_identity)
    try:
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
                try:proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
                except BaseException:log.close();raise
                active[device]={'process':proc,'log':log,'out':out,'job':job,'stop_sent':False};save(phase_root/(name+'.process.json'),{'pid':proc.pid,'device':device,'command':command,'job':job})
                save(phase_root/'STATUS.json',{'phase':a.phase,'stage':'launching','active':{d:{'pid':x['process'].pid,**x['job']} for d,x in active.items()},'queued':len(queue),'completed':len(results),'elapsed_s':time.monotonic()-start})
                # Limit simultaneous large checkpoint loads; do not change compute budget.
                time.sleep(a.launch_stagger_seconds)
            for device,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();result=read(item['out']/'RESULT.json') if (item['out']/'RESULT.json').exists() else {'status':'MISSING_RESULT'}
                try:accepted_job(item['out'],item['job'],a.phase,cfg);ok=code==0
                except (ValueError,KeyError,OSError) as exc:ok=False;result={**result,'status':'REJECTED','reason':str(exc)}
                results.append({'job':item['job'],'exit_code':code,'status':result['status'],'output':str(item['out'])});failed|=not ok;del active[device]
            if time.monotonic()-last_report>=60:
                save(phase_root/'STATUS.json',{'phase':a.phase,'active':{d:{'pid':x['process'].pid,**x['job']} for d,x in active.items()},'queued':len(queue),'completed':len(results),'failed':failed,'stopping':STOP,'elapsed_s':time.monotonic()-start});last_report=time.monotonic()
            if active:time.sleep(1)
    except BaseException as exc:
        save(phase_root/'ERROR.json',{'type':type(exc).__name__,'message':str(exc)})
        save(phase_root/'RESULT.json',{'status':'INCOMPLETE','phase':a.phase,'jobs':results,'error':str(exc),'elapsed_s':time.monotonic()-start})
        raise
    finally:
        # Never orphan our own children if loading a receipt or launching a job fails.
        for item in active.values():
            if item['process'].poll() is None:
                item['process'].send_signal(signal.SIGTERM)
        for item in active.values():
            item['process'].wait()
            item['log'].close()
    expected=jobs(cfg,a.phase);complete=not failed and not STOP and len(results)==len(expected)
    save(phase_root/'RESULT.json',{'status':'COMPLETE' if complete else 'INCOMPLETE','phase':a.phase,'jobs':results,'expected_jobs':len(expected),'elapsed_s':time.monotonic()-start});return 0 if complete else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--training',required=True);p.add_argument('--prepared',required=True);p.add_argument('--phase',choices=['initial','train','final'],required=True);p.add_argument('--devices',required=True);p.add_argument('--out',required=True);p.add_argument('--previous');p.add_argument('--recover-from');p.add_argument('--plan-only',action='store_true');p.add_argument('--launch-stagger-seconds',type=float,default=20);p.add_argument('--scheduler-wall-seconds',type=float,required=True);a=p.parse_args()
    if a.scheduler_wall_seconds<=0 or not 0<=a.launch_stagger_seconds<=60:p.error('Invalid scheduler bounds')
    raise SystemExit(main(a))
