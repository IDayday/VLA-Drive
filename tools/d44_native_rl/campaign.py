"""Single-owner queue for independent D44 runs; never adopts other experiments."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
import math
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


def epoch_update(arm,epoch,batch):
    # Every completed data epoch includes its own explicit padded tail.
    def collections(e,b):
        whole=math.floor(e);return whole*math.ceil(103288/b)+math.ceil((e-whole)*103288/b)
    if arm=='D6':return 2*(collections(min(epoch,1),32)+collections(max(0,epoch-1),64))
    return 2*collections(epoch,batch)


def next_stop(run,done):
    cfg=read_config(run['config']);gates=[.25,.5,1,2,3,4,6,8,12,20]
    nodes=[100]+[epoch_update(run['arm'],e,cfg['global_scenes'])for e in gates if e<=cfg['target_epochs']]
    return min((n for n in nodes if n>done),default=None)


def read_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def pending_evaluation(campaign,run):
    if not run.get('pause_for_evaluation',True):return False
    for request in Path(run['output']).glob('evaluation_requests/epoch*.json'):
        case=campaign.get('evaluations',{}).get(run['arm']+'_'+request.stem)
        if case is None or case['status']!='COMPLETE':return True
    return False


def wait_for_completion(campaign,seconds):
    """Job receipt events trigger dispatch; regular status remains half-hourly."""
    import ctypes,os,select,struct
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
    if fd<0:time.sleep(seconds);return
    try:
        paths=[]
        for run in campaign['runs'].values():
            paths += [Path(run['qualification']),Path(run['output']),Path(run['output'])/'evaluation_requests']
            paths += [Path(j['job'])for j in run['jobs'][-1:]]
        for case in campaign.get('evaluations',{}).values():
            paths += [Path(case['output'])]
            paths += [Path(j['job'])for jobs in case['jobs'].values()for j in jobs[-1:]]
        for path in paths:
            if path.is_dir():libc.inotify_add_watch(fd,os.fsencode(path),0x8|0x80|0x100)
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            if not select.select([fd],[],[],max(0,until-time.monotonic()))[0]:break
            data=os.read(fd,65536);offset=0
            while offset<len(data):
                wd,mask,cookie,length=struct.unpack_from('iIII',data,offset);offset+=16
                name=data[offset:offset+length].split(b'\0')[0].decode();offset+=length
                if name=='RESULT.json'or name=='evaluation_requests'or name.startswith(('result_rank','epoch','rank'))and name.endswith('.json'):return
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
    for name in ('D0_REAL_MID_INNER_EXACT_RESUME.json','NATIVE_DEPLOYMENT_EXPORT_PARITY_16.json','CANONICAL_SCORING_CONSISTENCY.json'):
        proof=ROOT/'qualification'/name
        if not proof.exists()or read(proof).get('status')!='PASS':return False
    root=Path(run['qualification'])
    if run['arm']=='D6':
        path=ROOT/'qualification/D6_REAL_STAGE_INNER_RESUME.json'
        if not path.exists()or read(path).get('status')!='PASS':return False
        expected={p:file_sha(Path(run['source'])/p)for p in ('tools/d44_native_rl/train.py','starVLA/rl/d44_native_rl/stages.py','starVLA/rl/d44_native_rl/history.py','starVLA/rl/d44_native_rl/recovery.py')}
        return read(path).get('source_files')==expected
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
            if result.get('returncode')==75 and time.time()<result['end_unix']+300:
                run['status']='WAITING_FOR_AUTHORIZED_CAPACITY';continue
            if result.get('returncode',result.get('exit_code',0))not in(0,75):
                run.update(status='FAILED_REQUIRES_REVIEW',last_result=result);continue
            # A resource refusal starts no actor and carries no training state.
            if result.get('status')not in('EXITED','REFUSED_RESOURCE'):
                run.update(status='FAILED_REQUIRES_REVIEW',last_result=result);continue
        if pending_evaluation(campaign,run):
            run['status']='SAVED_AND_PAUSED_FOR_REGISTERED_FULL_NAVTEST';continue
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
        # Operational qualification stops preserve the identical scientific
        # run/Adam/RNG. The quarter-epoch boundary is 404 B64 collections.
        limit=next_stop(run,done)
        if limit is not None:cmd+=['--stop-after-update',str(limit)]
        try:
            # Original B/D IL hosts are eligible only when ALL GPUs are free;
            # their workers never share or evict those protected trainers.
            receipt=launch(host,run['source'],job,cmd,list(range(8)),share_min_free_mib=None if host in PROTECTED else 41000,worker_source=Path(__file__).resolve().parents[2])
            run['jobs'].append(receipt);run['status']='RUNNING_OR_ADMITTING';occupied.add(host)
        except (subprocess.SubprocessError,OSError,RuntimeError)as error:
            run.update(status='DISPATCH_RETRY_REQUIRED',last_dispatch_error=repr(error))
    evaluation_tick(campaign)
    campaign['at']=time.time();campaign['completion_claim']=False;atomic_json(registry,campaign)
    return campaign


def free_evaluation_gpus(host,*,shared=False):
    code="import subprocess,json;print(json.dumps(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)))"
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,'/usr/bin/python3','-'],input=code,text=True,capture_output=True,timeout=12)
    if r.returncode:return []
    result=[]
    for line in json.loads(r.stdout).splitlines():
        index,free,util=map(int,line.split(','))
        if host=='training-vlawm-zt2'and index==6:continue
        if free>=(24000 if shared else 65000)and util<=(80 if shared else 10):result.append(index)
    return result


def evaluation_tick(campaign):
    """Registered complete Navtest only; free-GPU admission never evicts IL."""
    cases=campaign.setdefault('evaluations',{});allowed=allowed_hosts(admit_long_job=True)
    for arm,run in campaign['runs'].items():
        for request in Path(run['output']).glob('evaluation_requests/epoch*.json'):
            req=read(request);key=arm+'_'+request.stem
            cases.setdefault(key,dict(request=req,source=run['source'],status='PENDING',world=8,jobs={},
                output=str(ROOT/'evaluations'/key/'evaluation')))
    pending=[v for v in cases.values()if v['status']not in('COMPLETE','FAILED_REQUIRES_REVIEW','FAILED_SUMMARY_REQUIRES_REVIEW')]
    if not pending:return
    reserved=set()
    alljobs=[j for run in campaign['runs'].values()for j in run['jobs'][-1:]]
    alljobs += [j for case in cases.values()for jobs in case['jobs'].values()for j in jobs[-1:]]
    for j in alljobs:
        if not(Path(j['job'])/'RESULT.json').exists():reserved.update((j['host'],gpu)for gpu in j.get('gpus',range(8)))
    pool=[]
    released=set()
    for run in campaign['runs'].values():
        if run['status']=='SAVED_AND_PAUSED_FOR_REGISTERED_FULL_NAVTEST'and run['jobs']:
            released.add(run['jobs'][-1]['host'])
    for host in dict.fromkeys(list(sorted(released))+['training-vlawm-zt2','training-vlawm-zt5','training-vlawm-tsinghua-zt5','training-vla-zt','training-vlawm-tsinghua-zt']):
        if host not in allowed:continue
        shared=host in released and host not in PROTECTED
        try:pool += [(host,gpu,shared)for gpu in free_evaluation_gpus(host,shared=shared)if(host,gpu)not in reserved]
        except (subprocess.SubprocessError,OSError,ValueError):continue
    for case in pending:
        output=Path(case['output']);completed=[]
        for rank in range(case['world']):
            jobs=case['jobs'].setdefault(str(rank),[])
            status=output/f'rank{rank}.json'
            finished=jobs and(Path(jobs[-1]['job'])/'RESULT.json').exists()
            if status.exists()and read(status).get('status')=='COMPLETE'and finished and read(Path(jobs[-1]['job'])/'RESULT.json').get('returncode')==0:
                completed.append(rank);continue
            if jobs and not(Path(jobs[-1]['job'])/'RESULT.json').exists():continue
            if jobs:
                result=read(Path(jobs[-1]['job'])/'RESULT.json')
                if result.get('returncode')==75 and time.time()<result['end_unix']+300:continue
                if result.get('returncode',0)not in(0,75):case['status']='FAILED_REQUIRES_REVIEW';continue
            if not pool:continue
            host,gpu,shared=pool.pop(0);job=ROOT/'jobs/EVALUATE'/Path(case['output']).parent.name/f'rank{rank}_attempt{len(jobs)+1:03}'
            cmd=[PYTHON,'-m','tools.d44_native_rl.evaluate','--checkpoint',case['request']['checkpoint'],
                '--output',case['output'],'--rank',str(rank),'--world',str(case['world'])]
            try:jobs.append(launch(host,case['source'],job,cmd,[gpu],share_min_free_mib=24000 if shared else None,worker_source=Path(__file__).resolve().parents[2]));case['status']='RUNNING'
            except (subprocess.SubprocessError,OSError,RuntimeError)as error:case['last_dispatch_error']=repr(error)
        if len(completed)==case['world']:
            from tools.qwen_world_compare.evaluate import summarize
            baseline='/mnt/project/qwen-world-four-frameworks-20261029/periodic_NAVTEST_ARCH_D_044000_20261102T142147Z/D_IL_044000/evaluation'
            try:
                value=summarize(output,baseline);atomic_json(output.parent/'SUMMARY.json',value)
                case['status']='COMPLETE';case['summary']=str(output.parent/'SUMMARY.json')
            except Exception as error:case.update(status='FAILED_SUMMARY_REQUIRES_REVIEW',summary_error=repr(error))


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
            if any(r['status']=='SAVED_AND_PAUSED_FOR_REGISTERED_FULL_NAVTEST'and not pending_evaluation(campaign,r)for r in campaign['runs'].values()):continue
            # Metadata admission only; no training journal or GPU log polling.
            wait_for_completion(campaign,min(a.interval,60)if any(r['status']=='WAITING_FOR_QUALIFICATION'for r in campaign['runs'].values())else a.interval)


if __name__=='__main__':main()
