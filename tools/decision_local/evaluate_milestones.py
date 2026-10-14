"""Execute a finite registered list of development exports and canonical scores.

No checkpoint selection, new training or retry is performed. Final Navtest
requires a separate locked endpoint and completed acceptance receipt. Existing
prediction and scoring tools retain all model/metric validation. CPU scoring may
overlap the next small-head export. Only this task's own GPU processes
may share an inference device; unknown/other owners are skipped.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha


def eligible_gpus(allowed):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    owners={}
    for line in processes.splitlines():
        uuid,pid=[x.strip() for x in line.split(',')]
        try:cmd=Path('/proc/'+pid+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except OSError:cmd='unknown'
        owners.setdefault(uuid,[]).append(cmd)
    free=[];shared=[]
    for line in rows.splitlines():
        idx,uuid,memory,util=[x.strip() for x in line.split(',')];idx=int(idx)
        if idx not in allowed:continue
        if int(memory)<512 and int(util)<10 and not owners.get(uuid):free.append(idx)
        elif int(memory)<20000 and owners.get(uuid) and all(
                '/mnt/project/decision-local-consequence-artifacts/20261014/' in x and
                any('/decision_local/'+name in x for name in ('train_small_model.py','build_fixed_anchor_cache.py'))
                for x in owners[uuid]):shared.append(idx)
    return free+shared


def reuse_q0(baseline,predictions,destination,index):
    """Use the scorer's existing resume mechanism after exact trajectory checks."""
    bm=read(baseline/'identity.json');bd=read(baseline/'COMPLETE.json')
    pm=read(predictions/'identity.json')
    if not bd['q0']['valid'] or bm['index_sha256']!=digest(index):raise ValueError('Invalid common canonical q0 bank')
    source=read(Path(read(baseline/'RESEARCH_INPUT.json')['predictions'])/'identity.json')
    for key in ('base_identity','features','sampling_seed'):
        if pm[key]!=source[key]:raise ValueError('Different shared S0 input/proposal')
    rows=read(index);bank=destination/'q0/records';bank.mkdir(parents=True,exist_ok=True)
    for row in rows:
        token=row['token'];path=predictions/'predictions'/(token+'.npz')
        if digest(path)!=read(predictions/'records'/(token+'.json'))['sha256']:raise ValueError('Changed exported trajectory')
        with np.load(path) as new,np.load(baseline/'q0/proposals'/(token+'.npz')) as old:
            if not np.array_equal(new['q0'],old['trajectory']):raise ValueError('Different q0 cannot reuse scores')
        previous=baseline/'q0/records'/(token+'.json');target=bank/previous.name
        if target.exists():raise FileExistsError('Score reuse must be prepared once')
        # Both result roots remain immutable after scoring. No copy of a
        # trajectory, label or model is substituted in the execution path.
        os.link(previous,target)
    atomic_json(destination/'Q0_REUSE.json',{'baseline':str(baseline),'scenes':len(rows),
        'baseline_identity':bm['identity'],'exact_initial_trajectories':True,
        'canonical_will_recheck_each_proposal_and_metric_cache_hash':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--gpus',nargs='+',type=int,required=True)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--cpu-parallel',type=int,default=2)
    a=p.parse_args();resource=check_host();registration=read(a.registry);registered_hash=digest(a.registry)
    if registration['population'] not in ('debug','dev','navtest'):raise ValueError('Unsupported evaluation population')
    final=registration['population']=='navtest'
    if final:
        lock=Path(registration['method_lock'])
        if not lock.is_file() or digest(lock)!=registration['method_lock_sha256']:
            raise ValueError('Final Navtest requires its immutable method/endpoint lock')
        locked=read(lock)
        if locked['main_endpoint_epochs']!=24 or locked['proposal_sampling_seed']!=42:
            raise ValueError('Only the registered common endpoint and proposal seed are supported')
    score_timeout=7200 if final else 3600
    admission_seconds=score_timeout+700
    if not 1<=a.cpu_parallel<=2 or a.max_seconds<=0:raise ValueError('Bounded concurrency required')
    state=Path(registration['runtime']);state.mkdir(parents=True,exist_ok=True)
    (state/'claims').mkdir(exist_ok=True);(state/'results').mkdir(exist_ok=True)
    start=time.monotonic();active={};baseline=None
    atomic_json(state/'WORKER.json',{'pid':os.getpid(),'resources':resource,'registry_sha256':registered_hash,
        'started_unix':time.time(),'source_sha':source_sha(ROOT),'max_seconds':a.max_seconds})
    while time.monotonic()-start<a.max_seconds or active:
        check_host()
        if digest(a.registry)!=registered_hash:raise ValueError('Registered evaluation list changed')
        for name,(child,job,stream,began) in list(active.items()):
            code=child.poll()
            if code is None and time.monotonic()-began>score_timeout:
                # Only this observer's stateless canonical CPU process group.
                os.killpg(child.pid,signal.SIGTERM);code=child.wait(timeout=30)
            if code is None:continue
            stream.close();score=Path(job['scores']);complete=score/'COMPLETE.json'
            success=code==0 and complete.exists()
            atomic_json(state/'results'/(name+'.json'),{'returncode':code,'complete':success,'finished_unix':time.time()})
            if success and baseline is None:baseline=score
            del active[name]
        ready=True
        if final:
            receipt=Path(registration['final_readiness'])
            ready=receipt.exists()
            if ready:
                acceptance=read(receipt)
                if not acceptance.get('accepted') or acceptance['method_lock_sha256']!=registration['method_lock_sha256']:
                    raise ValueError('Unaccepted or mismatched final endpoint')
                if digest(lock)!=registration['method_lock_sha256']:raise ValueError('Locked final method changed')
        if ready and len(active)<a.cpu_parallel and time.monotonic()-start<a.max_seconds-admission_seconds:
            for job in registration['jobs']:
                name=job['id'];claim=state/'claims'/name
                if claim.exists() or not all(Path(x).exists() for x in job['requires']):continue
                devices=eligible_gpus(a.gpus)
                if not devices:break
                try:claim.mkdir()
                except FileExistsError:continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(devices[0]),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
                atomic_json(claim/'COMMANDS.json',{'commands':job['commands'],'gpu':devices[0],
                    'started_unix':time.time(),'source_sha':source_sha(ROOT)})
                score=Path(job['scores']);score.mkdir(exist_ok=True,parents=True)
                atomic_json(score/'RESEARCH_INPUT.json',{'predictions':job['predictions'],'checkpoint':job['checkpoint']})
                streaming=bool(registration.get('streaming',False))
                if streaming:
                    # The mature streaming scorer verifies every new q0 against
                    # the proposal SHA and metric cache SHA of these prior rows.
                    # No unverified score is aggregated, even before GPU export.
                    if baseline is not None:
                        bm=read(baseline/'identity.json');bd=read(baseline/'COMPLETE.json')
                        if not bd['q0']['valid'] or bm['index_sha256']!=digest(Path(registration['metric_index'])):
                            raise ValueError('Invalid initial-plan scoring bank')
                        target=score/'q0/records';target.mkdir(parents=True,exist_ok=True)
                        for path in (baseline/'q0/records').glob('*.json'):os.link(path,target/path.name)
                        atomic_json(score/'Q0_REUSE.json',{'baseline':str(baseline),'validation':'Streaming scorer must exactly match each newly exported q0 proposal and cache hash'})
                    stream=(state/(name+'_score.log')).open('w')
                    child=subprocess.Popen(job['commands']['score'],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                # Device count is measured now; placement never changes the
                # model's semantic identity, batch1 arithmetic or proposal RNG.
                chosen=devices if registration.get('shard_exports',False) else devices[:1]
                exports=[]
                for rank,gpu in enumerate(chosen):
                    command=list(job['commands']['export'])
                    if registration.get('shard_exports',False):command+=['--rank',str(rank),'--shards',str(len(chosen))]
                    logfile=(state/(name+f'_export_{rank:02d}.log')).open('w')
                    worker=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                        stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
                    exports.append((worker,logfile))
                    atomic_json(claim/f'EXPORT_{rank:02d}.json',{'pid':worker.pid,'gpu':gpu,'command':command,'started_unix':time.time()})
                codes=[];deadline=time.monotonic()+660
                for worker,logfile in exports:
                    try:code=worker.wait(timeout=max(.1,deadline-time.monotonic()))
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid,signal.SIGTERM);code=worker.wait(timeout=30)
                    logfile.close();codes.append(code)
                if any(codes):
                    atomic_json(Path(job['predictions'])/'INFERENCE_FAILED.json',{'returncodes':codes,'owned_shards_only':True})
                    if streaming:
                        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);stream.close()
                    atomic_json(state/'results'/(name+'.json'),{'complete':False,'export_returncodes':codes});continue
                if not streaming:
                    if baseline is not None:reuse_q0(baseline,Path(job['predictions']),score,Path(registration['metric_index']))
                    stream=(state/(name+'_score.log')).open('w')
                    child=subprocess.Popen(job['commands']['score'],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[name]=(child,job,stream,time.monotonic())
                if len(active)>=a.cpu_parallel:break
        if not active and all((state/'claims'/j['id']).exists() for j in registration['jobs']):break
        if not active and time.monotonic()-start>=a.max_seconds:break
        time.sleep(10.)
    atomic_json(state/'FINISHED.json',{'seconds':time.monotonic()-start,
        'pending':[j['id'] for j in registration['jobs'] if not (state/'claims'/j['id']).exists()],
        'scope':'Only registered observations; final Navtest additionally gated on endpoint readiness; failures preserved, no automatic retry or model selection'})


if __name__=='__main__':main()
