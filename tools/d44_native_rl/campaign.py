"""Single-owner queue for independent D44 runs; never adopts other experiments."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
from tools.d44_native_rl.launch import launch
from starVLA.rl.d44_native_rl.config import defaults
from starVLA.rl.qwen_world_compare.contracts import allowed_hosts,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json

ROOT=Path('/mnt/project/d44-native-rl-20261103')
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
PROTECTED={'training-vla-zt','training-vlawm-tsinghua-zt'}
PREFERRED={'D0':['training-vlawm-zt4'],'D2':['training-vlawm-tsinghua-zt2'],'D5':['training-vlawm-tsinghua-zt3']}
FALLBACK=['training-vlawm-tsinghua-zt5','training-vlawm-zt5','training-vla-zt','training-vlawm-tsinghua-zt']


def read(path):return json.loads(Path(path).read_text())


def wait_for_completion(campaign,seconds):
    """Job receipt events trigger dispatch; regular status remains half-hourly."""
    import ctypes,os,select
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
    if fd<0:time.sleep(seconds);return
    try:
        for run in campaign['runs'].values():
            paths=[Path(run['qualification'])]
            paths += [Path(j['job'])for j in run['jobs'][-1:]]
            for path in paths:
                if path.is_dir():libc.inotify_add_watch(fd,os.fsencode(path),0x8|0x80|0x100)
        select.select([fd],[],[],seconds)
    finally:os.close(fd)


def register(source):
    import yaml
    source=Path(source).resolve()
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():raise ValueError('Frozen source required')
    configroot=ROOT/'registered_configs';configroot.mkdir(exist_ok=True)
    runs={}
    for arm in ('D0','D2','D5'):
        cfg=defaults(arm);cfg['checkpoint_policy']='all_off';cfg['probability_layout']='sequential'
        path=configroot/(arm+'.yaml');text=yaml.safe_dump(cfg,sort_keys=False)
        if path.exists()and path.read_text()!=text:raise ValueError('Registered formal recipe cannot be overwritten')
        path.write_text(text)
        runs[arm]=dict(arm=arm,source=str(source),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
            config=str(path),config_sha256=file_sha(path),output=str(ROOT/'runs'/f'{arm}_B64G16_M2_v1'),
            qualification=str(ROOT/'qualification'/f'{arm}_B64G16_v7'),status='WAITING_FOR_QUALIFICATION',jobs=[],
            initial_target_epochs=cfg['target_epochs'],requires_fresh_Adam=True,source_checkpoint='D_IL44K',automatic_budget_extension=False)
    registry=ROOT/'CAMPAIGN.json'
    if registry.exists():raise FileExistsError('Campaign registration already exists')
    atomic_json(registry,dict(schema='d44_campaign_v1',at=time.time(),runs=runs,closed=False,
        completion_claim=False,resource_policy='current dispatch policy, bounded expiry, preserve B/D IL'))
    return registry


def qualified(run):
    root=Path(run['qualification'])
    for rank in range(8):
        path=root/f'result_rank{rank}.json'
        if not path.exists():return False
        result=read(path)
        if (result['updates']<10 or not result['qualification']or result['status']!='QUALIFICATION_STOPPED'
                or result.get('frozen_before')!=result.get('frozen_after')or not result.get('frozen_before')):return False
    identity=read(root/'identity.json');c=identity['recipe']
    if c['global_scenes']!=64 or c['group']!=16 or c['inner_updates']!=2 or identity['parent_model_sha256']!=read(ROOT/'parent_audit/AUDIT.json')['parent']['model_sha256']:
        raise ValueError('Wrong D44 B64/G16 qualification identity')
    # Actual saved material, not a batch_size label, proves distinct scenes.
    evidence=root/'B64_QUALIFICATION.json'
    if not evidence.exists():
        import torch
        files=list((root/'behavior').glob('*/collection000001/rank*.pt'));tokens=[]
        if len(files)!=8:raise ValueError('Missing real eight-rank collection')
        for path in files:
            batch=torch.load(path,map_location='cpu',weights_only=False)
            if len(batch['scenes'])!=8:raise ValueError('Not eight local scenes')
            for r in batch['scenes']:
                if r['chain'].shape!=(16,11,8,3):raise ValueError('Wrong real G16/K10 chain')
                tokens.append(r['token'])
        if len(tokens)!=64 or len(set(tokens))!=64:raise ValueError('Global batch is not 64 distinct scenes')
        atomic_json(evidence,dict(status='PASS',distinct_scenes=64,candidates_per_scene=16,complete_chain_states=11,
            scope='actual saved eight-rank first fresh collection',parent=identity['parent_model_sha256']))
    return True


def tick(registry):
    campaign=read(registry)
    if campaign['closed']:return campaign
    occupied=set()
    for run in campaign['runs'].values():
        for job in run['jobs'][-1:]:
            if not(Path(job['job'])/'RESULT.json').exists():occupied.add(job['host'])
    for arm,run in campaign['runs'].items():
        if run['status']in('COMPLETE','FAILED_REQUIRES_REVIEW'):continue
        if not qualified(run):run['status']='WAITING_FOR_QUALIFICATION';continue
        if file_sha(run['config'])!=run['config_sha256']:raise ValueError('Formal config changed')
        if run['jobs']:
            last=run['jobs'][-1];result=Path(last['job'])/'RESULT.json'
            if not result.exists():run['status']='RUNNING_OR_ADMITTING';continue
            result=read(result)
            if result.get('returncode',result.get('exit_code',0))not in(0,75):
                run.update(status='FAILED_REQUIRES_REVIEW',last_result=result);continue
            # A resource refusal starts no actor and carries no training state.
            if result.get('status')not in('EXITED','REFUSED_RESOURCE'):
                run.update(status='FAILED_REQUIRES_REVIEW',last_result=result);continue
        root=Path(run['output']);resume=None;done=0
        if(root/'latest.json').exists():
            latest=read(root/'latest.json');resume=latest['checkpoint'];seal=read(Path(resume)/'COMPLETE')
            if seal['phase']not in('READY','ROLLOUT_COMMITTED','INNER_COMMITTED','COMPLETE'):raise ValueError('Illegal latest phase')
            if seal['files_sha256']!=file_sha(Path(resume)/'files.json'):raise ValueError('Changed recovery inventory')
            done=seal['updates']
            import torch
            state=torch.load(Path(resume)/'rank0.pt',map_location='cpu',weights_only=False)['state']
            if state['valid_scene_exposures']>=run['initial_target_epochs']*103288 and state['next_inner']==0:
                run.update(status='COMPLETE',checkpoint=resume);continue
        hosts=[h for h in PREFERRED.get(arm,[])+FALLBACK if h in allowed_hosts(admit_long_job=True)and h not in occupied]
        if not hosts:run['status']='WAITING_FOR_AUTHORIZED_CAPACITY';continue
        # The worker rechecks UUIDs, explicit sharing, known owner ancestry,
        # locks and measured free memory. Failed admission never starts CUDA.
        host=hosts[len(run['jobs'])%len(hosts)];index=len(run['jobs'])+1
        job=ROOT/'jobs'/arm/f'attempt{index:03}'
        cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','tools.d44_native_rl.train',
            '--config',run['config'],'--output',run['output']]
        if resume:cmd+=['--resume',resume]
        if done<100:cmd+=['--stop-after-update','100']
        try:
            # Original B/D IL hosts are eligible only when ALL GPUs are free;
            # their workers never share or evict those protected trainers.
            receipt=launch(host,run['source'],job,cmd,list(range(8)),share_min_free_mib=None if host in PROTECTED else 41000)
            run['jobs'].append(receipt);run['status']='RUNNING_OR_ADMITTING';occupied.add(host)
        except (subprocess.SubprocessError,OSError,RuntimeError)as error:
            run.update(status='DISPATCH_RETRY_REQUIRED',last_dispatch_error=repr(error))
    campaign['at']=time.time();campaign['completion_claim']=False;atomic_json(registry,campaign)
    return campaign


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['register','run','status','tick']);p.add_argument('--source');p.add_argument('--registry',default=str(ROOT/'CAMPAIGN.json'));p.add_argument('--interval',type=int,default=1800);a=p.parse_args()
    if a.command=='register':print(register(a.source));return
    if a.command=='status':print(json.dumps(read(a.registry),indent=2));return
    with(Path(a.registry).with_suffix('.owner.lock')).open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(ROOT/'CAMPAIGN_OWNER.json',dict(pid=__import__('os').getpid(),started=time.time(),registry=a.registry))
        while True:
            campaign=tick(a.registry)
            if a.command=='tick' or campaign['closed']:break
            # Metadata admission only; no training journal or GPU log polling.
            wait_for_completion(campaign,min(a.interval,60)if any(r['status']=='WAITING_FOR_QUALIFICATION'for r in campaign['runs'].values())else a.interval)


if __name__=='__main__':main()
