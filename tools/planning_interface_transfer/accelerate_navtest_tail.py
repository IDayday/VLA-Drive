"""Temporarily delegate untouched tails of native four-rank Navtest banks.

Native writers are paused, never replaced. Helpers use the exact native model,
noise, preprocessing and FP32 path; native writers resume to publish original
completion receipts. All teacher/label inputs remain absent. Existing rows are
immutable. This script imports model code only after changing to its pinned
native worktree, so its orchestration source cannot change the model class.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)


def partition_missing(index,existing,world_size,guard_per_rank,helpers):
    if world_size!=4 or guard_per_rank<2 or helpers<1:raise ValueError('Invalid delegation')
    guards,tail=[],[]
    for rank in range(world_size):
        missing=[i for i in range(rank,len(index),world_size) if index[i]['token'] not in existing]
        guards.extend(missing[:guard_per_rank]);tail.extend(missing[guard_per_rank:])
    tail.sort()
    return sorted(guards),[tail[i::helpers] for i in range(helpers)]


def verify_owned(pid,bank):
    p=Path('/proc')/str(pid)/'cmdline'
    if not p.exists():return False
    words=[x.decode() for x in p.read_bytes().split(b'\0') if x]
    return ('-m' in words and words[words.index('-m')+1]=='tools.foresight.export_predictions'
            and '--output' in words and words[words.index('--output')+1]==bank)


def lease_worker(a,c):
    if socket.gethostname()!=c['hostname']:raise ValueError('Wrong GPU host')
    handles=[]
    for name in ['navtest_pdms_shared','navtest_pdms','recogdrive_mtopd_research']:
        f=Path(f'/var/tmp/{name}_gpu{c["gpu"]}.lock').open('a');handles.append(f)
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            stat=os.fstat(f.fileno());device=(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino)
            owners={int(row.split()[4]) for row in Path('/proc/locks').read_text().splitlines()
                    if len(row.split())>5 and row.split()[1]=='FLOCK' and tuple(int(x,16 if k<2 else 10) for k,x in enumerate(row.split()[5].split(':')))==device}
            approved=c.get('approved_training_lease_owner')
            if name!='navtest_pdms_shared' and approved and owners=={approved['pid']}:
                p=Path('/proc')/str(approved['pid'])/'cmdline'
                if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==approved['cmdline_sha256']:
                    f.close();handles.remove(f);continue
            for h in handles:h.close()
            return None
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    devices={int(x.split(',')[0]):x.split(', ') for x in raw.strip().splitlines()};gpu=devices[c['gpu']]
    if gpu[1]!=c['uuid']:raise ValueError('GPU UUID changed')
    if int(gpu[2])<30000:
        for f in handles:f.close()
        return None
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    for row in raw.strip().splitlines():
        if not row:continue
        uuid,pid=[x.strip() for x in row.split(',')]
        if uuid!=c['uuid']:continue
        try:words=[x.decode() for x in (Path('/proc')/pid/'cmdline').read_bytes().split(b'\0') if x]
        except OSError:
            for f in handles:f.close()
            return None
        if not ('-m' in words and words[words.index('-m')+1]=='tools.foresight.train_student'
                and '--run-id' in words and words[words.index('--run-id')+1]==c['trainer_run_id']):
            for f in handles:f.close()
            return None
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=c['uuid']
    return handles


def worker(a):
    c=read(a.request)
    if sha(__file__)!=c['helper_file_sha256']:raise ValueError('Helper source changed')
    handles=lease_worker(a,c)
    if handles is None:return 75
    try:
        atomic(Path(c['request_root'])/'process.json',dict(pid=os.getpid(),request=str(Path(a.request).resolve()),host=socket.gethostname()))
        os.chdir(c['native_worktree']);sys.path.insert(0,c['native_worktree'])
        if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=c['native_source'] or subprocess.check_output(['git','status','--porcelain']).strip():
            raise ValueError('Native model source changed')
        import numpy as np
        import torch
        from tools.foresight.checkpoints import checkpoint_identity,load_student,scene_noise
        from tools.ddpolicy_vehicle.run_meter import metered_run
        from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
        from starVLA.model.modules.vehicle_joint.initialization import identity_hash
        bank=Path(c['bank']);export=read(bank/'identity.json')
        if export!=c['bank_identity'] or export['world_size']!=4:raise ValueError('Bank identity/partition changed')
        data=ForesightCurrentDataset(c['current_root'])
        if data.identity!=export['current_identity'] or len(data)!=12146:raise ValueError('Current population changed')
        training,ckpt=checkpoint_identity(c['training_run'],c['tag'])
        if ckpt!=export['checkpoint']:raise ValueError('Checkpoint changed')
        expected=dict(precision='FP32',tf32=False,candidates_per_scene=1,sampling_seed=42,steps=10,future_conditioning=False)
        if any(export['protocol'][k]!=v for k,v in expected.items()):raise ValueError('Inference protocol changed')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(.25)
        with metered_run(c['campaign_root'],c['run_id'],1,dict(kind='foresight_supplemental_export',real_optimizer_updates=0)) as (record,folder,save):
            record.update(bank=str(bank),helper_source=c['helper_source'],native_source=c['native_source']);save()
            model=load_student(c['training_run'],c['tag'],training)
            if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('FP32 required')
            # Each helper independently checks the unchanged native path against
            # one retained completed scene before writing any new prediction.
            probe=c['probe_index'];scene=data.index[probe]
            with torch.inference_mode():pose=model.predict_action([data[probe]],initial_noise=scene_noise(scene['token'],42,'cuda'))[0]
            original=np.load(bank/'predictions'/(scene['token']+'.npz'))['trajectory']
            error=float(np.max(np.abs(pose.cpu().numpy()-original)))
            if error>1e-6:raise ValueError(f'Native inference parity failed: {error}')
            atomic(Path(c['request_root'])/'parity.json',dict(passed=True,max_pose_error=error,native_source=c['native_source'],helper_source=c['helper_source']))
            work=Path(c['request_root'])/'work.json'
            while not work.exists():
                if (Path(c['request_root'])/'STOP_DRAIN').exists() or time.time()-record['start_unix']>600:return 0
                time.sleep(.5)
            c['indices']=read(work)['indices'];record['delegated_indices']=c['indices'];save()
            signature=identity_hash(export);done=0
            for idx in c['indices']:
                if (Path(c['request_root'])/'STOP_DRAIN').exists() or time.time()-record['start_unix']>c.get('helper_max_seconds',900):
                    record['status']='PAUSED';break
                scene=data.index[idx];token=scene['token'];dest=bank/'predictions'/(token+'.npz');meta=dest.with_suffix('.json')
                if meta.exists():
                    row=read(meta)
                    if row['status']!='ok' or row['identity_sha256']!=signature or sha(dest)!=row['proposal_sha256']:
                        raise ValueError('Existing scene changed')
                    continue
                with torch.inference_mode():pose=model.predict_action([data[idx]],initial_noise=scene_noise(token,42,'cuda'))[0]
                if pose.shape!=(8,3) or not torch.isfinite(pose).all():raise FloatingPointError('Invalid ego prediction')
                tmp=dest.with_suffix(f'.{os.getpid()}.tail.tmp')
                with tmp.open('wb') as stream:np.savez_compressed(stream,trajectory=pose.cpu().numpy())
                tmp.replace(dest)
                atomic(meta,dict(token=token,log=scene['log'],identity_sha256=signature,status='ok',proposal_sha256=sha(dest),
                    supplemental_source=c['helper_source'],native_inference_source=c['native_source']))
                done+=1;record.update(inference_scenes=done,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save()
            atomic(Path(c['request_root'])/'result.json',dict(status='COMPLETE' if done==len(c['indices']) else 'PAUSED',generated=done,requested=len(c['indices']),max_pose_error=error,optimizer_updates=0))
    finally:
        for f in handles:f.close()
    return 0


def signal_owned(host,records,signum):
    code='import os,signal,pathlib,json;records='+repr(records)+'\n'
    code+='''for r in records:
 p=pathlib.Path('/proc')/str(r['pid'])/'cmdline'
 if not p.exists():continue
 words=[x.decode() for x in p.read_bytes().split(b'\\0') if x]
 if not ('-m' in words and words[words.index('-m')+1]=='tools.foresight.export_predictions' and words[words.index('--output')+1]==r['bank']):raise RuntimeError('Owned native PID mismatch')
 os.kill(r['pid'],'''+str(int(signum))+')\n'
    import shlex
    subprocess.run(['ssh',host,'python3 -c '+shlex.quote(code)],check=True,timeout=20)


def run(a):
    import shlex
    c=read(a.plan);root=Path(c['root'])
    if sha(__file__)!=c['helper_file_sha256']:raise ValueError('Locked helper source changed')
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children=[];paused={};interrupted=[]
        for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            signal.signal(s,lambda received,frame:interrupted.append(received))
        try:
            index=read(Path(c['current_root'])/'index.json')
            for job in c['jobs']:
                bank=Path(job['bank']);rows=read(bank/'identity.json')
                # No helper can overlap a live native writer. Pause only exact
                # registered export PIDs, never a trainer or CPU scorer.
                existing={p.stem for p in (bank/'predictions').glob('*.json')}
                probes=[i for i,x in enumerate(index) if x['token'] in existing]
                seen=set();distinct=[]
                for i in probes:
                    if index[i]['log'] not in seen:seen.add(index[i]['log']);distinct.append(i)
                    if len(distinct)>=len(job['workers']):break
                for k,allocation in enumerate(job['workers']):
                    folder=root/job['name']/f'helper{k}';folder.mkdir(parents=True,exist_ok=True)
                    request=dict(allocation,bank=str(bank),bank_identity=rows,current_root=c['current_root'],
                        training_run=job['training_run'],tag=job['tag'],native_worktree=job['native_worktree'],native_source=job['native_source'],
                        helper_source=c['helper_source'],helper_file_sha256=c['helper_file_sha256'],campaign_root=str(root),
                        request_root=str(folder),run_id=job['name']+f'_helper{k}',probe_index=distinct[k%len(distinct)],helper_max_seconds=c.get('helper_max_seconds',900))
                    atomic(folder/'request.json',request)
                    command=[c['python'],'-u',str(Path(__file__).resolve()),'worker','--request',str(folder/'request.json')]
                    remote=['env','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','TOKENIZERS_PARALLELISM=false','CUBLAS_WORKSPACE_CONFIG=:4096:8',*command]
                    with (folder/'worker.log').open('x') as stream:
                        child=subprocess.Popen(['ssh',allocation['host'],'exec '+shlex.join(remote)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    children.append((child,folder))
            # Original exporters keep working during expensive model loading.
            # Pause only after helpers are ready, then issue disjoint tails.
            loading_deadline=time.time()+360
            while time.time()<loading_deadline and not interrupted:
                if all(p.poll() is not None or (f/'parity.json').exists() for p,f in children):break
                time.sleep(2)
            for job in c['jobs']:
                ready=[f for p,f in children if f.parent.name==job['name'] and p.poll() is None and (f/'parity.json').exists()]
                if not ready:continue
                paused.setdefault(job['native_host'],[]).extend(job['native_processes'])
                atomic(root/'paused.json',dict(parent_pid=os.getpid(),paused=paused,helper_file=str(Path(__file__).resolve()),plan=str(Path(a.plan).resolve())))
                signal_owned(job['native_host'],job['native_processes'],signal.SIGSTOP)
                bank=Path(job['bank']);existing={p.stem for p in (bank/'predictions').glob('*.json')}
                guards,parts=partition_missing(index,existing,4,32,len(ready))
                atomic(root/job['name']/'partition.json',dict(guards=guards,parts=parts,bank_identity=read(bank/'identity.json')))
                for folder,indices in zip(ready,parts):atomic(folder/'work.json',dict(indices=indices))
            for p,f in children:
                if p.poll() is None and not (f/'work.json').exists():atomic(f/'STOP_DRAIN',dict(reason='Helper not ready within finite loading window'))
            atomic(root/'status.json',dict(status='RUNNING',pid=os.getpid(),helpers=len(children),paused_natives=paused,started_unix=time.time(),optimizer_updates=0))
            deadline=time.time()+c.get('helper_max_seconds',900)+100
            while any(p.poll() is None for p,_ in children) and time.time()<deadline and not interrupted:time.sleep(3)
            if any(p.poll() is None for p,_ in children):raise TimeoutError('Bounded supplemental window exhausted')
            atomic(root/'helper_exits.json',[dict(folder=str(f),exit_code=p.returncode) for p,f in children])
        finally:
            for p,f in children:
                if p.poll() is None:atomic(f/'STOP_DRAIN',dict(reason='Controller draining before native resume'))
            for p,f in children:
                if p.poll() is None:
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:stop_owned_helper(f,c)
            # Native originals finish the32 guarded scenes/rank, verify all new
            # atomic cache rows and publish the original four shard receipts.
            for host,records in paused.items():signal_owned(host,records,signal.SIGCONT)
            atomic(root/'status.json',dict(status='HELPERS_FINISHED_NATIVE_RESUMED',pid=os.getpid(),optimizer_updates=0,updated_unix=time.time()))


def stop_owned_helper(folder,c):
    """SIGTERM only the manifest-bound helper, never a trainer or native writer."""
    request=read(folder/'request.json')
    if not (folder/'process.json').exists():return
    pid=read(folder/'process.json')['pid']
    import shlex
    code='import os,signal,pathlib;pid='+repr(pid)+';request='+repr(str(folder/'request.json'))+'\n'
    code+='''p=pathlib.Path('/proc')/str(pid)/'cmdline'
if p.exists():
 w=[x.decode() for x in p.read_bytes().split(b'\\0') if x]
 if not ('--request' in w and w[w.index('--request')+1]==request and 'worker' in w):raise RuntimeError('Helper PID ownership changed')
 os.kill(pid,signal.SIGTERM)
 import time
 deadline=time.time()+10
 while p.exists() and time.time()<deadline:time.sleep(.1)
 if p.exists():raise RuntimeError('Helper has not stopped; refuse concurrent native writing')
'''
    subprocess.run(['ssh',request['host'],'python3 -c '+shlex.quote(code)],check=True,timeout=15)


def rescue(a):
    """Independent bounded fail-safe resumes owned native writers after a crash."""
    c=read(a.plan);root=Path(c['root']);deadline=time.time()+c.get('helper_max_seconds',900)+500
    while time.time()<deadline:
        if (root/'status.json').exists() and read(root/'status.json')['status']=='HELPERS_FINISHED_NATIVE_RESUMED':return
        if (root/'paused.json').exists():
            pid=read(root/'paused.json')['parent_pid']
            if not (Path('/proc')/str(pid)).exists():break
        time.sleep(5)
    for folder in root.glob('*/helper*'):
        atomic(folder/'STOP_DRAIN',dict(reason='Independent finite rescue'))
    time.sleep(5)
    for folder in root.glob('*/helper*'):stop_owned_helper(folder,c)
    if (root/'paused.json').exists():
        for host,records in read(root/'paused.json')['paused'].items():signal_owned(host,records,signal.SIGCONT)
    atomic(root/'rescue_receipt.json',dict(resumed_unix=time.time(),optimizer_updates=0))


def watch(a):
    """Feed only existing registered 90k/100k banks; never start another evaluator."""
    import shlex
    c=read(a.plan);root=Path(c['root']);root.mkdir(parents=True,exist_ok=True)
    if sha(__file__)!=c['helper_file_sha256']:raise ValueError('Companion source changed')
    if sha(c['queue_registration'])!=c['queue_registration_sha256']:raise ValueError('Registered queue changed')
    requested={f'{arm}_{step:06d}' for arm in ('A_ACTION','A_NO_MAE','V_QUERY') for step in (90000,100000)}
    if set(c['jobs'])!=requested:raise ValueError('Companion must cover only the six requested future banks')
    with (root/'companion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children={};completed={};deadline=time.time()+86400
        if (root/'companion_status.json').exists():completed=read(root/'companion_status.json')['completed']
        while time.time()<deadline and len(completed)<len(requested):
            for name,p in list(children.items()):
                if p.poll() is not None:completed[name]=dict(exit_code=p.returncode);del children[name]
            for name,spec in c['jobs'].items():
                if name in completed or name in children:continue
                bank=Path(spec['bank'])
                if not (bank/'identity.json').exists() or len(list((bank/'predictions').glob('*.json')))<128:continue
                if all((bank/f'shard_{i}.json').exists() and read(bank/f'shard_{i}.json')['status']=='complete' for i in range(4)):
                    completed[name]=dict(status='NATIVE_ALREADY_COMPLETE');continue
                code="""import pathlib,json
rows=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:w=[x.decode() for x in p.read_bytes().split(b'\\0') if x]
 except (OSError,UnicodeError):continue
 if '-m' in w and w[w.index('-m')+1]=='tools.foresight.export_predictions' and '--output' in w and w[w.index('--output')+1]==BANK:rows.append(dict(pid=int(p.parent.name),bank=BANK,words=w))
print(json.dumps(rows))
""".replace('BANK',repr(str(bank)))
                try:
                    records=json.loads(subprocess.check_output(['ssh',spec['native_host'],'python3 -c '+shlex.quote(code)],text=True,timeout=20))
                except (subprocess.SubprocessError,json.JSONDecodeError):continue
                if not records:continue
                words=records[0]['words'];identity=read(bank/'identity.json')
                job=dict(spec,name=name,native_processes=[dict(pid=r['pid'],bank=str(bank)) for r in records],
                    training_run=words[words.index('--training-run')+1],tag=words[words.index('--checkpoint-tag')+1],native_source=identity['evaluation_source'])
                folder=root/name;folder.mkdir();plan={k:v for k,v in c.items() if k not in ('jobs','queue_registration','queue_registration_sha256')}
                plan.update(root=str(folder),jobs=[job]);atomic(folder/'plan.json',plan)
                handles={}
                for mode in ('run','rescue'):
                    with (folder/(mode+'.log')).open('x') as log:
                        p=subprocess.Popen([c['python'],'-u',str(Path(__file__).resolve()),mode,'--plan',str(folder/'plan.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    handles[mode]=p.pid
                    if mode=='run':children[name]=p
                atomic(folder/'launch.json',handles)
            atomic(root/'companion_status.json',dict(status='RUNNING',pid=os.getpid(),completed=completed,active=list(children),requested=sorted(requested),optimizer_updates=0,updated_unix=time.time()))
            time.sleep(10)
        atomic(root/'companion_status.json',dict(status='COMPLETE' if len(completed)==len(requested) else 'PAUSED',completed=completed,requested=sorted(requested),optimizer_updates=0,updated_unix=time.time()))


def main():
    p=argparse.ArgumentParser(__doc__);s=p.add_subparsers(dest='mode',required=True)
    x=s.add_parser('worker');x.add_argument('--request',required=True)
    x=s.add_parser('run');x.add_argument('--plan',required=True)
    x=s.add_parser('rescue');x.add_argument('--plan',required=True)
    x=s.add_parser('watch');x.add_argument('--plan',required=True)
    a=p.parse_args()
    if a.mode=='worker':sys.exit(worker(a))
    elif a.mode=='run':run(a)
    elif a.mode=='rescue':rescue(a)
    else:watch(a)


if __name__=='__main__':main()
