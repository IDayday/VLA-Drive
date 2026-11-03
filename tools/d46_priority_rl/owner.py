"""Fresh IL46 D-native queue with native recovery and canonical evaluations.

No checkpoint from the IL70 or D44 RL campaigns is accepted. Qualification
weights are never a formal parent. Existing D-native train/evaluate kernels,
scoring, source-chain conventions and atomic recovery are reused unchanged.
"""
import argparse
import fcntl
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
    blocked=set()
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


def dispatch(run,qualification=False,blocked_hosts=()):
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
    command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','tools.d44_native_rl.train','--config',cfg,'--output',output]
    if qualification:command+=['--qualification','--stop-after-update','10']
    if resume:command+=['--resume',resume]
    # One own full-world task per host initially; worker still independently
    # checks current owner identities, physical UUIDs, both leases and memory.
    preferred=read(ROOT/'operations/EXECUTION_POLICY.json',{}).get('preferred_hosts',{}).get(run['arm'],[])
    for host in dict.fromkeys(tuple(preferred)+(PREFERRED[run['arm']],)+HOSTS):
        if host in blocked_hosts or host not in allowed_hosts(admit_long_job=True):continue
        try:
            if len(capacity(host,49152))!=8:continue
            job=ROOT/'jobs'/run['arm']/('qualification' if qualification else 'formal')/f'attempt{len(jobs)+1:03}'
            receipt=launch(host,run['source'],job,command,list(range(8)),share_min_free_mib=49152,worker_source=SOURCE,task_root=ROOT,task=TASK)
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
        operations_source=str(SOURCE),scientific_sources=sorted({r['source']for r in campaign['runs'].values()}),scientific_sources_frozen=True,qualification_weights_never_formal_initialization=True)
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
    for arm in ORDER:
        run=campaign['runs'][arm]
        if run['status'] in ('FAILED_REQUIRES_REVIEW','COMPLETE'):continue
        hold=control.get('execution_holds',{}).get(arm)
        if hold:
            # A staged same-run continuation must never race an automatic
            # restart of its old world. STOP invokes the existing native save.
            if run['jobs']and not(Path(run['jobs'][-1]['job'])/'RESULT.json').exists():
                (Path(run['jobs'][-1]['job'])/'STOP_REQUESTED').touch()
            run.update(status='SAVED_AND_HELD_FOR_REGISTERED_STAGE',execution_hold=hold)
            continue
        try:
            if not qualification_pass(run):
                run['status']=dispatch(run,True,startup_hosts(campaign))
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
            run['status']=dispatch(run,False,startup_hosts(campaign))
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
