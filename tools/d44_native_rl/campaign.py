"""Single-owner queue for independent D44 runs; never adopts other experiments."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
import math
from tools.d44_native_rl.launch import launch,DispatchPreflightFailed
from starVLA.rl.d44_native_rl.config import defaults
from starVLA.rl.qwen_world_compare.contracts import allowed_hosts,file_sha,authorization
from starVLA.rl.flow_grpo.transactions import atomic_json

ROOT=Path('/mnt/project/d44-native-rl-20261103')
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
PROTECTED={'training-vla-zt','training-vlawm-tsinghua-zt'}
PREFERRED={'D0':['training-vlawm-zt4'],'D2':['training-vlawm-tsinghua-zt2'],'D5':['training-vlawm-tsinghua-zt3'],
    'D1':['training-vlawm-zt'],'D3_FPO':['training-vlawm-zt'],'D3_AWM':['training-vlawm-zt'],'D4':['training-vlawm-zt'],'D6':['training-vlawm-zt']}
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


def add_run(registry,arm,source,qualification,*,qualification_world=8,qualification_scenes=64):
    import yaml
    registry=Path(registry);campaign=read(registry);source=Path(source).resolve()
    if not campaign['closed']:raise ValueError('Quiesce the CPU owner before changing campaign membership')
    if arm in campaign['runs']:raise FileExistsError('Registered algorithm ID exists')
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():raise ValueError('Frozen source required')
    cfg=defaults(arm);cfg['checkpoint_policy']='all_off';config=ROOT/'registered_configs'/(arm+'.yaml')
    if config.exists():raise FileExistsError('Registered config exists')
    config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    campaign['runs'][arm]=dict(arm=arm,source=str(source),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        config=str(config),config_sha256=file_sha(config),output=str(ROOT/'runs'/(arm+'_STAGED_v1'if arm=='D6'else arm+f'_B{cfg["global_scenes"]}G16_M2_v1')),
        qualification=str(Path(qualification).resolve()),qualification_world=qualification_world,qualification_scenes=qualification_scenes,
        status='WAITING_FOR_QUALIFICATION',jobs=[],initial_target_epochs=cfg['target_epochs'],requires_fresh_Adam=True,
        source_checkpoint='D_IL44K',automatic_budget_extension=False,priority=2 if arm in('D4','D6')else 1)
    atomic_json(registry,campaign);return campaign


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
    required_world=run.get('qualification_world',8)
    for rank in range(required_world):
        path=root/f'result_rank{rank}.json'
        if not path.exists():return False
        result=read(path)
        if (result['updates']<10 or not result['qualification']or result['status']!='QUALIFICATION_STOPPED'
                or result.get('frozen_before')!=result.get('frozen_after')or not result.get('frozen_before')):return False
    identity=read(root/'identity.json');c=identity['recipe']
    if c['arm']!=run['arm']or identity['architecture_id']!='D':raise ValueError('Qualification used another algorithm/model')
    formal=read_config(run['config'])
    for key in ('text_rank','film_last_layers','train_action_io','post_feedback'):
        if c[key]!=formal[key]:raise ValueError('Qualification scope differs: '+key)
    # The declared pre-formal cost/endpoint/teacher-gate delta is documented;
    # the actual native forward/probability/adapter/codec kernels must match.
    for name in ('starVLA/rl/d44_native_rl/actor.py','starVLA/rl/d44_native_rl/gaussian_objective.py',
                 'starVLA/rl/d44_native_rl/expanded_scope.py','starVLA/rl/d44_native_rl/native_cleanx_aux.py',
                 'starVLA/rl/qwen_world_compare/policy_adapter.py','starVLA/model/modules/qwen_world_compare/qwen_flow_adapter.py',
                 'starVLA/model/modules/qwen_sourceflow/action_codec.py'):
        if identity['source_files'].get(name)!=file_sha(Path(run['source'])/name):raise ValueError('Qualification native kernel source differs: '+name)
    expected_scenes=run.get('qualification_scenes',64)
    if c['global_scenes']!=expected_scenes or c['group']!=16 or c['inner_updates']!=2 or identity['world']!=required_world or identity['parent_model_sha256']!=read(ROOT/'parent_audit/AUDIT.json')['parent']['model_sha256']:
        raise ValueError('Wrong D44 B64/G16 qualification identity')
    # Actual saved material, not a batch_size label, proves distinct scenes.
    evidence=root/'MULTISCENE_QUALIFICATION.json'
    if not evidence.exists():
        import torch
        files=list((root/'behavior').glob('*/collection000001/rank*.pt'));tokens=[]
        if len(files)!=required_world:raise ValueError('Missing real multi-rank collection')
        for path in files:
            batch=torch.load(path,map_location='cpu',weights_only=False)
            if len(batch['scenes'])!=expected_scenes//required_world:raise ValueError('Wrong local scene accumulation')
            for r in batch['scenes']:
                if r['chain'].shape!=(16,11,8,3):raise ValueError('Wrong real G16/K10 chain')
                tokens.append(r['token'])
        if len(tokens)!=expected_scenes or len(set(tokens))!=expected_scenes:raise ValueError('Global batch scenes are not distinct')
        atomic_json(evidence,dict(status='PASS',distinct_scenes=expected_scenes,candidates_per_scene=16,complete_chain_states=11,
            scope='actual saved multi-rank first fresh collection',actual_world=required_world,parent=identity['parent_model_sha256']))
    return True


def tick(registry):
    campaign=read(registry)
    if campaign['closed']:return campaign
    occupied=set()
    for run in campaign['runs'].values():
        for job in run['jobs'][-1:]:
            if not(Path(job['job'])/'RESULT.json').exists():occupied.add(job['host'])
    for arm,run in sorted(campaign['runs'].items(),key=lambda item:item[1].get('priority',0)):
        if run['status']in('COMPLETE','FAILED_REQUIRES_REVIEW'):continue
        if not qualified(run):run['status']='WAITING_FOR_QUALIFICATION';continue
        if file_sha(run['config'])!=run['config_sha256']:raise ValueError('Formal config changed')
        if run['jobs']:
            last=run['jobs'][-1];result=Path(last['job'])/'RESULT.json'
            if not result.exists():run['status']='RUNNING_OR_ADMITTING';continue
            result=read(result)
            if result.get('returncode')==75:run.setdefault('host_retry_after',{})[last['host']]=result['end_unix']+300
            elif result.get('returncode')==0:run.setdefault('host_retry_after',{}).pop(last['host'],None)
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
        hosts=[h for h in PREFERRED.get(arm,[])+FALLBACK if h in allowed_hosts(admit_long_job=True)and h not in occupied and run.get('host_retry_after',{}).get(h,0)<=time.time()]
        if not hosts:run['status']='WAITING_FOR_AUTHORIZED_CAPACITY';continue
        # The worker rechecks UUIDs, explicit sharing, known owner ancestry,
        # locks and measured free memory. Failed admission never starts CUDA.
        prior=run['jobs'][-1]if run['jobs']else None
        healthy=prior is not None and read(Path(prior['job'])/'RESULT.json').get('returncode')==0
        if healthy and prior['host']in hosts:hosts=[prior['host']]+[h for h in hosts if h!=prior['host']]
        elif prior is not None and prior['host']in hosts:hosts=[h for h in hosts if h!=prior['host']]+[prior['host']]
        host=hosts[0];index=len(run['jobs'])+1
        job=ROOT/'jobs'/arm/f'attempt{index:03}'
        cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','tools.d44_native_rl.train',
            '--config',run['config'],'--output',run['output']]
        if resume:cmd+=['--resume',resume]
        # Operational qualification stops preserve the identical scientific
        # run/Adam/RNG. The quarter-epoch boundary is 404 B64 collections.
        limit=next_stop(run,done)
        if limit is not None:cmd+=['--stop-after-update',str(limit)]
        for host in hosts:
            try:
                # Original B/D IL hosts are eligible only when ALL GPUs are free.
                profiled=arm=='D0'and resume is not None and host not in PROTECTED
                receipt=launch(host,run['source'],job,cmd,list(range(8)),share_min_free_mib=None if host in PROTECTED else 40960 if profiled else 41000,
                    worker_source=Path(__file__).resolve().parents[2],profiled_d0_third_slot=profiled)
                run['jobs'].append(receipt);run['status']='RUNNING_OR_ADMITTING';occupied.add(host);break
            except DispatchPreflightFailed as error:
                run.setdefault('prechild_dispatch_failures',[]).append(dict(host=host,error=repr(error),at=time.time(),child_started=False));continue
            except (subprocess.SubprocessError,OSError,RuntimeError)as error:
                run.update(status='DISPATCH_RETRY_REQUIRED',last_dispatch_error=repr(error));break
        else:run['status']='WAITING_FOR_REACHABLE_AUTHORIZED_HOST'
    evaluation_tick(campaign)
    yield_owned_lower_priority(campaign)
    campaign['at']=time.time();campaign['completion_claim']=False;atomic_json(registry,campaign)
    return campaign


def free_evaluation_gpus(host,*,shared=False):
    code="import subprocess,json;print(json.dumps(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)))"
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,'/usr/bin/python3','-'],input=code,text=True,capture_output=True,timeout=12)
    if r.returncode:return []
    rows=[tuple(map(int,line.split(',')))for line in json.loads(r.stdout).splitlines()]
    if host in PROTECTED and any(free<79000 or util>10 for _,free,util in rows):return []
    result=[]
    for index,free,util in rows:
        if host=='training-vlawm-zt2'and index==6:continue
        if free>=(24000 if shared else 65000)and(shared or util<=10):result.append(index)
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
        shared=host not in PROTECTED
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
                try:
                    from tools.d44_native_rl.analyze import analyze
                    analysis=analyze(output,baseline);atomic_json(output.parent/'PAIRED_ANALYSIS.json',analysis)
                    case['paired_analysis']=str(output.parent/'PAIRED_ANALYSIS.json')
                except Exception as error:case['analysis_error']=repr(error)
            except Exception as error:case.update(status='FAILED_SUMMARY_REQUIRES_REVIEW',summary_error=repr(error))


def yield_owned_lower_priority(campaign):
    """Only our own lower-priority job yields a native boundary for waiting P0.

    This never signals a process, changes optimizer state, or preempts B/D IL
    or another project. A STOP file invokes the existing exact native save.
    """
    grant,_=authorization();stable=set(grant['permanent_hosts']+grant.get('unbounded_temporary_hosts',[]))&set(allowed_hosts(admit_long_job=True))
    pending=campaign.setdefault('priority_yields',[])
    for arm,run in campaign['runs'].items():
        if run.get('priority',0)!=0 or run['status']not in('WAITING_FOR_AUTHORIZED_CAPACITY','WAITING_FOR_REACHABLE_AUTHORIZED_HOST')or not run['jobs']:continue
        result=Path(run['jobs'][-1]['job'])/'RESULT.json'
        if not result.exists()or read(result).get('returncode')not in(0,75):continue
        if any(r['beneficiary']==arm and not(Path(r['yielded_job'])/'RESULT.json').exists()for r in pending):continue
        candidates=[(other,v)for other,v in campaign['runs'].items()if v.get('priority',0)>0 and v['jobs']and v['jobs'][-1]['host']in stable]
        for other,lower in sorted(candidates,key=lambda pair:-pair[1].get('priority',0)):
            job=lower['jobs'][-1];root=Path(job['job']);running=root/'RUNNING.json'
            if not root.resolve().is_relative_to((ROOT/'jobs').resolve()):raise ValueError('Priority yield is outside our own job root')
            if(root/'RESULT.json').exists()or(root/'STOP_REQUESTED').exists()or not running.exists():continue
            state=read(running)
            if state['command']!=job['command']or state['cwd']!=lower['source']:raise ValueError('Own-job priority yield identity differs')
            (root/'STOP_REQUESTED').touch()
            pending.append(dict(beneficiary=arm,yielded_algorithm=other,yielded_job=str(root),host=job['host'],at=time.time(),
                scientific_source=lower['source_sha'],native_boundary_requested=True,processes_signalled=0,reset_optimizer=False))
            break


def selection_index(campaign):
    """Test-selected deployment indexes; never alter training or TRAIN sampling."""
    rows={}
    for arm,run in campaign['runs'].items():
        complete=[]
        for case in campaign.get('evaluations',{}).values():
            if case.get('status')!='COMPLETE':continue
            cp=Path(case['request']['checkpoint']).resolve()
            if not cp.is_relative_to(Path(run['output']).resolve()):continue
            summary=read(case['summary'])
            if summary['qualification']or summary['scenes']!=12146 or summary['invalid_scenes']!=0:raise ValueError('Selection requires complete B1 Navtest')
            complete.append(dict(checkpoint=str(cp),PDMS=summary['PDMS'],new_zero=summary['paired']['new_zero'],
                epoch=case['request']['epoch'],summary=case['summary']))
        item=dict(last_checkpoint=read(Path(run['output'])/'latest.json')['checkpoint']if(Path(run['output'])/'latest.json').exists()else None,
                  evaluated_checkpoints=len(complete),test_selected=True)
        if complete:
            best=max(complete,key=lambda row:(row['PDMS'],-row['epoch']))
            stable=min((row for row in complete if row['PDMS']>=best['PDMS']-.2),key=lambda row:(row['new_zero'],-row['PDMS'],row['epoch']))
            item.update(best_score_checkpoint=best,best_stable_proxy_checkpoint=stable)
        rows[arm]=item
    return dict(schema='D44_test_selected_checkpoint_index_v1',runs=rows,
        stable_proxy_rule='Lowest new-zero count among checkpoints within0.2 PDMS points of the run maximum; then higher PDMS, then earlier epoch',
        limits='Descriptive test-selected index; not unseen-test validation or a stability/safety guarantee. No test data feed training.')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['register','add','run','status','tick']);p.add_argument('--source');p.add_argument('--registry',default=str(ROOT/'CAMPAIGN.json'));p.add_argument('--interval',type=int,default=1800)
    p.add_argument('--arm');p.add_argument('--qualification');p.add_argument('--qualification-world',type=int,default=8);p.add_argument('--qualification-scenes',type=int,default=64);a=p.parse_args()
    if a.command=='register':print(register(a.source));return
    if a.command=='add':
        with(Path(a.registry).with_suffix('.owner.lock')).open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);add_run(a.registry,a.arm,a.source,a.qualification,qualification_world=a.qualification_world,qualification_scenes=a.qualification_scenes)
        return
    if a.command=='status':print(json.dumps(read(a.registry),indent=2));return
    with(Path(a.registry).with_suffix('.owner.lock')).open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(ROOT/'CAMPAIGN_OWNER.json',dict(pid=__import__('os').getpid(),started=time.time(),registry=a.registry))
        while True:
            campaign=tick(a.registry)
            atomic_json(ROOT/'CHECKPOINT_SELECTION.json',selection_index(campaign))
            if a.command=='tick' or campaign['closed']:break
            if any(r['status']=='SAVED_AND_PAUSED_FOR_REGISTERED_FULL_NAVTEST'and not pending_evaluation(campaign,r)for r in campaign['runs'].values()):continue
            # Metadata admission only; no training journal or GPU log polling.
            wait_for_completion(campaign,min(a.interval,60)if any(r['status']=='WAITING_FOR_QUALIFICATION'for r in campaign['runs'].values())else a.interval)


if __name__=='__main__':main()
