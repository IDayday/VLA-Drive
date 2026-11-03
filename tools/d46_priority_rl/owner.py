"""Fresh IL46 D-native queue with native recovery and canonical evaluations.

No checkpoint from the IL70 or D44 RL campaigns is accepted. Qualification
weights are never a formal parent. Existing D-native train/evaluate kernels,
scoring, source-chain conventions and atomic recovery are reused unchanged.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import yaml

from starVLA.rl.qwen_world_compare.contracts import allowed_hosts, file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from tools.d44_native_rl.launch import launch, DispatchPreflightFailed
from tools.d44_native_rl.qualification_queue import capacity

ROOT=Path('/mnt/project/d46-priority-rl-20261103')
TASK='d46_priority_rl_20261103'
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
MODEL_SHA='386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
ORDER=('D3_FPO','D3_AWM','D5','D1','D2','D0')
HOSTS=('training-vlawm-zt4','training-rl-zt2','training-vlawm-tsinghua-zt2',
       'training-vlawm-zt','training-vlawm-tsinghua-zt3','training-vlawm-zt2',
       'training-vlawm-tsinghua-zt5','training-vlawm-tsinghua-zt')
PREFERRED=dict(zip(ORDER,HOSTS))
SOURCE=Path(__file__).resolve().parents[2]


def read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def registration():
    from starVLA.rl.d44_native_rl.config import defaults46
    if (ROOT/'CAMPAIGN.json').exists():raise FileExistsError('Never overwrite a registered campaign')
    if subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip():raise ValueError('Frozen source required')
    runs={}
    for priority,arm in enumerate(ORDER):
        cfg=defaults46(arm);cfg.update(target_epochs=6.,maximum_epochs=20.,checkpoint_policy='all_off')
        config=ROOT/'registered_configs'/(arm+'.yaml');config.parent.mkdir(exist_ok=True)
        config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        qcfg=dict(cfg,rescue_start_epoch=0. if arm=='D2' else cfg['rescue_start_epoch'])
        qconfig=config.with_name(arm+'_qualification.yaml');qconfig.write_text(yaml.safe_dump(qcfg,sort_keys=False))
        runs[arm]=dict(arm=arm,priority=priority,source=str(SOURCE),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
            config=str(config),config_sha256=file_sha(config),qualification_config=str(qconfig),qualification_config_sha256=file_sha(qconfig),
            output=str(ROOT/'runs'/('D46_'+arm)),qualification=str(ROOT/'qualification'/('D46_'+arm)),
            source_checkpoint='D_IL46K',requires_fresh_Adam=True,initial_target_epochs=6.,automatic_budget_extension=False,
            status='WAITING_FOR_QUALIFICATION',jobs=[],qualification_jobs=[],pause_for_evaluation=True)
    value=dict(schema='d46_priority_native_campaign_v1',created=time.time(),runs=runs,closed=False,evaluations={},evaluation_world=32,completion_claim=False,
               parent_contract=str(ROOT/'PARENT_CONTRACT.json'),parent_model_sha256=MODEL_SHA,
               resource_policy='D46 priority on current authorized capacity; B/D IL protected; expired hosts excluded')
    atomic_json(ROOT/'CAMPAIGN.json',value);return value


def qualification_pass(run):
    root=Path(run['qualification']);identity=read(root/'identity.json',{})
    if identity.get('parent_model_sha256')!=MODEL_SHA or identity.get('world')!=8:return False
    cfg=identity['recipe']
    if cfg['arm']!=run['arm'] or cfg['parent']!='D_IL46K' or cfg['global_scenes']!=64 or cfg['group']!=16 or cfg['inner_updates']!=2:
        raise ValueError('Qualification used another policy/batch/group')
    rows=[read(root/f'result_rank{i}.json',{}) for i in range(8)]
    if any(r.get('updates')!=10 or r.get('status')!='QUALIFICATION_STOPPED' or not r.get('frozen_before') or
           r.get('frozen_before')!=r.get('frozen_after') for r in rows):return False
    for i in range(8):
        parity=read(root/f'initial_native_parity_rank{i}.json',{})
        if parity.get('status')!='PASS' or parity.get('parent')!=MODEL_SHA or parity.get('ODE_chain_max_abs')!=0 or parity.get('original_cleanx_deployment_max_abs')!=0:
            raise ValueError('IL46 native wrapper changed actual deployment')
    formal=yaml.safe_load(Path(run['config']).read_text())
    if any(cfg[k]!=formal[k] for k in ('text_rank','film_last_layers','train_action_io','post_feedback')):
        raise ValueError('Qualification trainable scope differs')
    for name,digest in identity['source_files'].items():
        if file_sha(Path(run['source'])/name)!=digest:raise ValueError('Qualified source changed: '+name)
    return True


def inspect_latest(run):
    root=Path(run['output']).resolve();latest=read(root/'latest.json')
    if not latest:return None,None
    cp=Path(latest['checkpoint']).resolve()
    if not cp.is_relative_to(root/'checkpoints'):raise ValueError('Foreign native checkpoint')
    seal=read(cp/'COMPLETE',{});identity=read(cp/'identity.json',{})
    if seal.get('world')!=8 or seal.get('files_sha256')!=file_sha(cp/'files.json') or identity.get('parent_model_sha256')!=MODEL_SHA or identity.get('qualification') is not False:
        raise ValueError('Resume is not sealed formal IL46 world8')
    if identity['recipe']!=yaml.safe_load(Path(run['config']).read_text()):raise ValueError('Resume scientific recipe changed')
    return str(cp),seal


def startup_hosts(campaign):
    """Reserve a launching world until all8 ranks prove a current update.

    GPU allocation can lag Popen by minutes. Free memory observed during that
    interval is not an invitation to start another world on the same devices.
    Once ready, the worker's existing max2/UUID/identity/memory checks decide.
    """
    blocked=set(campaign.get('world16_reserved_hosts',[]))
    blocked.update(j['host']for j in campaign.get('world16_stage_jobs',[])if not(Path(j['job'])/'RESULT.json').exists())
    for run in campaign['runs'].values():
        for key,output in [('jobs',run['output']),('qualification_jobs',run['qualification'])]:
            if not run[key]:continue
            job=run[key][-1]
            if (Path(job['job'])/'RESULT.json').exists():continue
            ready=True
            for i in range(8):
                f=Path(output)/f'metrics_rank{i}.jsonl'
                try:
                    with f.open('rb') as stream:
                        stream.seek(max(0,f.stat().st_size-65536));rows=stream.read().splitlines()
                    row=json.loads(rows[-1])
                    if not job['at']<=row['at']<=time.time() or time.time()-row['at']>300:ready=False
                except (OSError,IndexError,ValueError,KeyError):ready=False
            if not ready:blocked.add(job['host'])
    return blocked



def capacity_resume(campaign,control,run,hold,cache):
    """Explicit resource hold release, never a failure or world16 override.

    Cache at most the two authorized host probes for this tick. A reservation
    is taken before dispatch, including ambiguous/prechild launch outcomes.
    Matching receipts preserve already-running releases; changed hold text
    restores the ordinary cooperative hold on the next tick.
    """
    cfg=control.get('capacity_resume',{})
    arms=('D3_AWM','D5','D1');hosts=('training-vlawm-zt','training-vlawm-tsinghua-zt2')
    if (cfg.get('enabled')is not True or run['arm']not in arms or run['arm']not in cfg.get('arms',[])
        or not isinstance(hold,str)or not hold or cfg.get('minimum_free_MiB')!=79000
        or not set(cfg.get('arms',[]))<=set(arms)or not set(cfg.get('hosts',[]))<=set(hosts)
        or run.get('world16_stage')):return None
    hold_sha=hashlib.sha256(hold.encode()).hexdigest()
    previous=run.get('capacity_resume_release',{})
    # A new operator hold is a new decision, not permission to auto-release it.
    if previous and previous.get('hold_sha256')!=hold_sha:return None
    live=any(not(Path(j['job'])/'RESULT.json').is_file()for key in('jobs','qualification_jobs')for j in run[key])
    if live:
        if (previous.get('schema')=='d46_explicit_capacity_hold_release_v1'and previous.get('hold_sha256')==hold_sha
            and previous.get('host')in cfg.get('hosts',[])and previous.get('arm')==run['arm']):return previous['host']
        return None
    blocked=startup_hosts(campaign)|cache.setdefault('reserved',set())
    if 'allowed'not in cache:cache['allowed']=set(allowed_hosts(admit_long_job=True))
    for host in dict.fromkeys(cfg.get('hosts',[])):
        if host not in cache['allowed']or host in blocked:continue
        if host not in cache:
            try:cache[host]=capacity(host,79000)
            except(DispatchPreflightFailed,subprocess.SubprocessError,OSError,ValueError):cache[host]=[]
        if sorted(cache[host])!=list(range(8)):continue
        value=dict(schema='d46_explicit_capacity_hold_release_v1',arm=run['arm'],hold_sha256=hold_sha,
            hold_text=hold,host=host,at=time.time(),minimum_free_MiB=79000,GPUs=list(range(8)),
            authority='explicit EXECUTION_POLICY.capacity_resume; independent worker lease/UUID/owner checks unchanged')
        run.setdefault('capacity_resume_receipts',[]).append(value);run['capacity_resume_release']=value
        cache['reserved'].add(host)
        return host
    return None


def startup_control(run,config,resume,qualification=False):
    """Explicit administrative startup only; original scientific run unchanged."""
    binding=run.get('startup_control')or read(ROOT/'operations/EXECUTION_POLICY.json',{}).get('startup_controls',{}).get(run['arm'])
    if not binding or qualification:return None
    if not resume:raise ValueError('Startup control requires exact original native resume')
    source=Path(binding['science_source']).resolve();profile=Path(binding['profile']).resolve()
    if file_sha(profile)!=binding['profile_sha256']:raise ValueError('Startup control profile changed')
    value=read(profile);body=dict(value);digest=body.pop('sha256',None)
    identity=read(Path(resume)/'identity.json',{})
    if (value.get('schema')!='d46_world8_control_startup_execution_profile_v1'or fingerprint(body)!=digest
        or value.get('protocol',{}).get('world')!=8 or value['protocol'].get('timeout_seconds')!=600
        or value.get('output')!=run['output']or value.get('arm')!=run['arm']or value.get('config')!=str(Path(config).resolve())
        or value.get('config_sha256')!=file_sha(config)or value.get('science_source')!=str(source)
        or value.get('identity_sha256')!=identity.get('sha256')or identity.get('world')!=8
        or value.get('parent_model_sha256')!=MODEL_SHA or value.get('source_files_sha256')!=fingerprint(identity['source_files'])
        or file_sha(source/'tools/d46_priority_rl/train_control.py')!=value.get('wrapper_sha256')):
        raise ValueError('Startup wrapper is not this own unchanged world8 algorithm')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=binding['source_sha']:
        raise ValueError('Startup control source changed')
    for name,digest in identity['source_files'].items():
        if file_sha(source/name)!=digest:raise ValueError('Startup science differs: '+name)
    return dict(source=str(source),entry='tools.d46_priority_rl.train_control',args=['--control-profile',str(profile),'--control-profile-sha256',binding['profile_sha256']])

def dispatch(run,qualification=False,blocked_hosts=(),capacity_host=None):
    # The native trainer commits its checkpoint before emitting milestone
    # requests. Provision this administrative directory before any GPU child.
    (Path(run['qualification']if qualification else run['output'])/'evaluation_requests').mkdir(parents=True,exist_ok=True)
    jobs=run['qualification_jobs'] if qualification else run['jobs']
    if jobs:
        result=read(Path(jobs[-1]['job'])/'RESULT.json')
        if result is None:return 'RUNNING_QUALIFICATION' if qualification else 'RUNNING_OR_ADMITTING'
        if result.get('returncode') not in (0,75):
            review=run.get('qualification_resource_retry_review',{})
            permitted=(qualification and review.get('job')==jobs[-1]['job']
                and review.get('returncode')==result.get('returncode')
                and Path(review.get('receipt','/nonexistent')).is_file()
                and file_sha(review['receipt'])==review.get('receipt_sha256'))
            native_review=run.get('formal_boundary_recovery_review',{})
            permitted=permitted or(not qualification and native_review.get('job')==jobs[-1]['job']
                and native_review.get('returncode')==result.get('returncode')
                and Path(native_review.get('receipt','/nonexistent')).is_file()
                and file_sha(native_review['receipt'])==native_review.get('receipt_sha256'))
            if not permitted:raise RuntimeError('Actual child failure retained: '+jobs[-1]['job'])
        if result.get('returncode')==75 and time.time()<result['end_unix']+60:return 'WAITING_FOR_AUTHORIZED_CAPACITY'
    output=run['qualification'] if qualification else run['output']
    cfg=run['qualification_config'] if qualification else run['config']
    if file_sha(cfg)!=run['qualification_config_sha256' if qualification else 'config_sha256']:raise ValueError('Registered config changed')
    resume=None
    if qualification:
        latest=read(Path(output)/'latest.json')
        if latest:resume=latest['checkpoint']
    else:resume,_=inspect_latest(run)
    adapter=startup_control(run,cfg,resume,qualification)
    source=adapter['source']if adapter else run['source']
    entry=adapter['entry']if adapter else'tools.d44_native_rl.train'
    command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m',entry,'--config',cfg,'--output',output]
    if adapter:command+=adapter['args']
    if qualification:command+=['--qualification','--stop-after-update','10']
    if resume:command+=['--resume',resume]
    # One own full-world task per host initially; worker still independently
    # checks current owner identities, physical UUIDs, both leases and memory.
    preferred=read(ROOT/'operations/EXECUTION_POLICY.json',{}).get('preferred_hosts',{}).get(run['arm'],[])
    routes=(capacity_host,)if capacity_host else tuple(preferred)+(PREFERRED[run['arm']],)+HOSTS
    for host in dict.fromkeys(routes):
        if host in blocked_hosts or host not in allowed_hosts(admit_long_job=True):continue
        try:
            minimum=79000 if capacity_host else 49152
            if not capacity_host and len(capacity(host,49152))!=8:continue
            job=ROOT/'jobs'/run['arm']/('qualification' if qualification else 'formal')/f'attempt{len(jobs)+1:03}'
            receipt=launch(host,source,job,command,list(range(8)),share_min_free_mib=minimum,worker_source=SOURCE,task_root=ROOT,task=TASK)
            jobs.append(receipt);return 'RUNNING_QUALIFICATION' if qualification else 'RUNNING_OR_ADMITTING'
        except DispatchPreflightFailed as error:
            run.setdefault('prechild_failures',[]).append(dict(host=host,at=time.time(),error=str(error),child_started=False))
        except (OSError,subprocess.SubprocessError) as error:
            raise RuntimeError('Ambiguous transport retained for review; no duplicate launch') from error
    return 'WAITING_FOR_AUTHORIZED_CAPACITY'


def status(campaign):
    rows=[];active=0
    for arm,run in campaign['runs'].items():
        row=dict(id='D46_'+arm,status=run['status'],host=None,updates=0,job=None,scene_exposures=None)
        stage=run.get('world16_stage')
        if stage:
            point=stage.get('latest_native')or{}
            row.update(status=stage['status'],world=16,host=', '.join(stage['hosts']),updates=point.get('updates',0),scene_exposures=point.get('valid_scene_exposures'),execution_stage=stage['stage_id'])
            current=[]
            for i in range(16):
                f=Path(stage['output'])/f'metrics_rank{i}.jsonl'
                try:
                    with f.open('rb')as stream:
                        stream.seek(max(0,f.stat().st_size-65536));lines=stream.read().splitlines()
                    item=json.loads(lines[-1])
                    if item['run']==stage['identity_sha256']and stage['last_attempt_started']<=item['at']<=time.time()and time.time()-item['at']<300:current.append(item)
                except(OSError,IndexError,ValueError,KeyError):pass
            if len(current)==16 and len(stage.get('active_jobs',[]))==2 and all(not(Path(j['job'])/'RESULT.json').exists()for j in stage['active_jobs']):
                row.update(status='TRAINING_WORLD16',updates=min(r['updates']for r in current),scene_exposures=min(r['scene_exposures']for r in current));active+=1
            rows.append(row);continue
        if run['jobs']:
            job=run['jobs'][-1];row.update(host=job['host'],job=job['job'])
            last=[]
            for i in range(8):
                f=Path(run['output'])/f'metrics_rank{i}.jsonl'
                if f.is_file():
                    with f.open('rb') as stream:
                        stream.seek(max(0,f.stat().st_size-65536));lines=stream.read().splitlines()
                    if lines:last.append(json.loads(lines[-1]))
            live=not(Path(job['job'])/'RESULT.json').exists()
            if len(last)==8:
                row.update(updates=min(r['updates'] for r in last),scene_exposures=min(r['scene_exposures'] for r in last))
                if live and all(job['at']<=r['at']<=time.time() and time.time()-r['at']<300 for r in last):row['status']='TRAINING';active+=1
        rows.append(row)
    previous=read(ROOT/'PRIORITY_STATUS.json',{})
    previous.update(schema='D46_priority_status_v1',updated_at=time.time(),status='D46_FRESH_IL46_PRIORITY_CAMPAIGN',runs=rows,actual_training=active,
        operations_source=str(SOURCE),scientific_sources=sorted({r['source']for r in campaign['runs'].values()}|{r['world16_stage']['source']for r in campaign['runs'].values()if r.get('world16_stage')}),scientific_sources_frozen=True,qualification_weights_never_formal_initialization=True)
    atomic_json(ROOT/'PRIORITY_STATUS.json',previous)


def tick(campaign):
    from tools.d44_native_rl.campaign import pending_evaluation,evaluation_tick,selection_index
    # Operational scheduling only: no recipe, optimizer or training-clock edit.
    control=read(ROOT/'operations/EXECUTION_POLICY.json',{})
    early=read(ROOT/'operations/EARLY_EVAL_20261103/STATE.json',{})
    early_owner=read(ROOT/'operations/EARLY_EVAL_20261103/CONTROLLER_LAUNCH.json',{})
    early_live=False
    try:
        early_live=(int(Path(f"/proc/{early_owner['pid']}/stat").read_text().split()[21])==early_owner['start_ticks'])
    except (OSError,KeyError,ValueError):pass
    defer_evaluation=(control.get('defer_registered_evaluations',False)
        or(control.get('defer_registered_evaluations_during_early_batch',False)
            and early_live and not early.get('complete',False)))
    from tools.d46_priority_rl.world16_owner import ingest
    ingest(campaign,control)
    for arm,review in control.get('formal_native_recovery_reviews',{}).items():
        run=campaign['runs'][arm]
        if run['status']=='FAILED_REQUIRES_REVIEW'and run['jobs']and review.get('job')==run['jobs'][-1]['job']:
            if file_sha(review['receipt'])!=review['receipt_sha256']:raise ValueError('Native recovery review changed')
            run.setdefault('reviewed_errors',[]).append(dict(error=run.pop('error',''),review=review,at=time.time()))
            run.update(formal_boundary_recovery_review=review,status='REVIEWED_NATIVE_BOUNDARY_READY_TO_RESUME')
    capacity_cache={}
    for arm in ORDER:
        run=campaign['runs'][arm]
        if run.get('world16_stage'):
            if run['jobs']and not(Path(run['jobs'][-1]['job'])/'RESULT.json').exists():
                (Path(run['jobs'][-1]['job'])/'STOP_REQUESTED').touch()
            run['status']=run['world16_stage']['status'];continue
        if run['status'] in ('FAILED_REQUIRES_REVIEW','COMPLETE'):continue
        hold=control.get('execution_holds',{}).get(arm)
        capacity_host=capacity_resume(campaign,control,run,hold,capacity_cache)if hold else None
        if hold and not capacity_host:
            # A staged same-run continuation must never race an automatic
            # restart of its old world. STOP invokes the existing native save.
            if run['jobs']and not(Path(run['jobs'][-1]['job'])/'RESULT.json').exists():
                (Path(run['jobs'][-1]['job'])/'STOP_REQUESTED').touch()
            run.update(status='SAVED_AND_HELD_FOR_REGISTERED_STAGE',execution_hold=hold)
            continue
        try:
            if not qualification_pass(run):
                run['status']=dispatch(run,True,startup_hosts(campaign),capacity_host=capacity_host)
                if qualification_pass(run):run['status']='QUALIFIED_FROM_TRUE_IL46'
                continue
            resume,seal=inspect_latest(run)
            if seal:
                import torch
                state=torch.load(Path(resume)/'rank0.pt',map_location='cpu',weights_only=False)['state']
                if state['valid_scene_exposures']>=6*103288 and state['next_inner']==0:
                    run['status']='COMPLETE';continue
            if pending_evaluation(campaign,run)and not control.get('continue_training_during_evaluation',False):
                # Own learner continues only until a native stop is explicitly
                # requested; no inference result feeds back as training labels.
                if run['jobs'] and not(Path(run['jobs'][-1]['job'])/'RESULT.json').exists():
                    (Path(run['jobs'][-1]['job'])/'STOP_REQUESTED').touch()
                run['status']='SAVED_AND_PAUSED_FOR_REGISTERED_FULL_NAVTEST';continue
            run['status']=dispatch(run,False,startup_hosts(campaign),capacity_host=capacity_host)
        except Exception:
            run.update(status='FAILED_REQUIRES_REVIEW',error=traceback.format_exc())
    if not defer_evaluation:evaluation_tick(campaign)
    campaign['evaluation_operations']=dict(control=control,early_batch_live=early_live,
        registered_dispatch_deferred=defer_evaluation,at=time.time())
    atomic_json(ROOT/'CHECKPOINT_SELECTION.json',selection_index(campaign))
    campaign.update(at=time.time(),completion_claim=False);atomic_json(ROOT/'CAMPAIGN.json',campaign);status(campaign)
    return campaign


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('register','run','once'));p.add_argument('--interval',type=int,default=60);a=p.parse_args()
    if Path(os.environ.get('WORLD_COMPARE_ROOT','')).resolve()!=ROOT or os.environ.get('WORLD_COMPARE_TASK')!=TASK or os.environ.get('D_NATIVE_CAMPAIGN_ROOT')!=str(ROOT):raise ValueError('Explicit independent D46 root/task required')
    if a.action=='register':registration();return
    with (ROOT/'CAMPAIGN.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(ROOT/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source=str(SOURCE),started=time.time()))
        while not(ROOT/'STOP_OWNER').exists():
            campaign=read(ROOT/'CAMPAIGN.json')
            if campaign['closed']:break
            tick(campaign)
            if a.action=='once':break
            time.sleep(a.interval)


if __name__=='__main__':main()
