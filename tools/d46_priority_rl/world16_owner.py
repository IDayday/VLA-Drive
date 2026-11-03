"""Own only a sealed two-node formal16 stage; CAMPAIGN has one other writer.

No conversion, optimizer changes, resource policy, trainer copy or GPU signals.
Root must register an execution_hold and fixed world16_stages pointer first.
"""
import argparse
import copy
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import time
import traceback
import yaml
from starVLA.rl.qwen_world_compare.contracts import allowed_hosts,file_sha,fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from tools.d44_native_rl.launch import launch,DispatchPreflightFailed
from tools.d44_native_rl.qualification_queue import capacity

ROOT=Path('/mnt/project/d46-priority-rl-20261103')
SOURCE=Path(__file__).resolve().parents[2]
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
TASK='d46_priority_rl_20261103'
MODEL_SHA='386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
PLAN='d46_formal_world16_stage_plan_v1'
STATE='d46_formal_world16_stage_state_v1'
TERMINAL={'COMPLETE_AT_REGISTERED_6E','FAILED_REQUIRES_REVIEW','AMBIGUOUS_LAUNCH_REQUIRES_REVIEW','STOPPED_AT_NATIVE_BOUNDARY','RESOURCE75_RETRY_LIMIT'}


def read(path,default=None):return json.loads(Path(path).read_text())if Path(path).is_file()else default

def owner_root(plan):return Path(plan.get('owner_root',ROOT/'operations/WORLD16_STAGES'/plan['stage_id'])).resolve()

def load_plan(path,*,check_hold=True):
    path=Path(path).resolve();p=read(path);digest=file_sha(path)
    if p.get('schema')!=PLAN or p.get('arm')not in('D0','D5','D3_FPO')or not p.get('stage_id'):raise ValueError('Exact D0/D5/D3_FPO stage plan required')
    if p.get('target_epochs')!=6. or len(p.get('hosts',[]))!=2 or len(set(p['hosts']))!=2:raise ValueError('Two fixed hosts, original6E only')
    if not 1024<=p['master_port']<=65535 or not p.get('master_addr'):raise ValueError('Explicit rendezvous endpoint')
    if p.get('minimum_free_MiB',65000)<49152 or p.get('resource75_retry_limit',3)not in(1,2,3):raise ValueError('Unprofiled shared memory/retry budget')
    output=Path(p['output']).resolve();identity=read(output/'identity.json',{});transition=identity.get('native_world_transition',{})
    body=dict(identity);saved=body.pop('sha256',None)
    cfg=yaml.safe_load(Path(p['config']).read_text())
    if (fingerprint(body)!=saved or p['identity_sha256']!=saved or identity.get('world')!=16 or identity.get('qualification')is not False
        or identity.get('parent_model_sha256')!=MODEL_SHA or cfg!=identity.get('recipe')or cfg['arm']!=p['arm']
        or(cfg['global_scenes'],cfg['group'],cfg['inner_updates'],cfg['target_epochs'])!=(64,16,2,6.)):
        raise ValueError('Not exact formal IL46 B64G16M2 native16 stage')
    receipt=read(output/'WORLD_TRANSITION.json',{});r=dict(receipt);seal=r.pop('sha256',None)
    if (fingerprint(r)!=seal or file_sha(output/'WORLD_TRANSITION.json')!=p['transition_receipt_sha256']
        or receipt.get('target_identity_sha256')!=saved or receipt.get('qualification')is not False
        or transition.get('schema')!='d46_exact_native_READY_world8_to16_v1'or transition.get('old_owner_may_adopt')is not False
        or transition.get('source_qualification')is not False or transition.get('target_qualification')is not False):raise ValueError('Typed formal transition mismatch')
    if file_sha(p['config'])!=p['config_sha256']:raise ValueError('Sealed recipe changed')
    source=Path(p['science_source']).resolve()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=p['source_sha']:raise ValueError('Scientific commit changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():raise ValueError('Freeze scientific source')
    if not output.is_relative_to(ROOT)or not owner_root(p).is_relative_to(ROOT):raise ValueError('Stage outside D46 root')
    if check_hold:
        control=read(ROOT/'operations/EXECUTION_POLICY.json',{});ref=control.get('world16_stages',{}).get(p['arm'],{})
        if not control.get('execution_holds',{}).get(p['arm'])or ref.get('plan')!=str(path)or ref.get('plan_sha256')!=digest:
            raise ValueError('Main owner must hold oldworld and bind this exact stage')
        campaign=read(ROOT/'CAMPAIGN.json',{});run=campaign.get('runs',{}).get(p['arm'],{})
        if run.get('config_sha256')!=p['config_sha256']:raise ValueError('Not the same logical registered algorithm')
        parent=Path(transition.get('source_checkpoint','/nonexistent')).resolve()
        if not parent.is_relative_to(Path(run['output']).resolve()/'checkpoints'):raise ValueError('Transition is not this algorithm own native ancestor')
        if run.get('jobs')and not(Path(run['jobs'][-1]['job'])/'RESULT.json').is_file():raise ValueError('Oldworld8 must actually exit before stage launch')
    p=dict(p,_path=str(path),_sha256=digest);return p


def latest(plan,cache=None):
    output=Path(plan['output']).resolve();point=read(output/'latest.json',{});cp=Path(point.get('checkpoint','/nonexistent')).resolve()
    if not cp.is_relative_to(output/'checkpoints'):raise ValueError('Foreign stage native path')
    seal=read(cp/'COMPLETE',{});files=read(cp/'files.json',{});identity=read(cp/'identity.json',{})
    if (seal.get('world')!=16 or seal.get('files_sha256')!=file_sha(cp/'files.json')or identity.get('sha256')!=plan['identity_sha256']
        or identity.get('qualification')is not False or identity.get('recipe',{}).get('arm')!=plan['arm']
        or file_sha(cp/'identity.json')!=files.get('identity.json')):raise ValueError('Not a sealed own formal16 native')
    if cache and cache.get('checkpoint')==str(cp)and cache.get('files_sha256')==seal['files_sha256']:return cache
    if file_sha(cp/'rank0.pt')!=files.get('rank0.pt'):raise ValueError('Changed native state')
    import torch
    state=torch.load(cp/'rank0.pt',map_location='cpu',weights_only=False)['state']
    if state['updates']!=seal['updates']or state['next_inner']!=seal['next_inner']:raise ValueError('Native clock differs')
    return dict(checkpoint=str(cp),files_sha256=seal['files_sha256'],updates=seal['updates'],next_inner=seal['next_inner'],phase=seal['phase'],
        valid_scene_exposures=int(state['valid_scene_exposures']),scene_cursor=int(state['scene_cursor']),virtual_rng_in_native=True)


def initial(plan):return dict(schema=STATE,stage_id=plan['stage_id'],arm=plan['arm'],plan_sha256=plan['_sha256'],status='PREPARED',attempts=[],active_jobs=[],evaluation_requests=[],resource75_retries=0,updated=time.time())

def persist(plan,state):
    root=owner_root(plan);root.mkdir(parents=True,exist_ok=True);state['updated']=time.time()
    state['reserved_hosts']=plan['hosts']if state['status']not in TERMINAL else[]
    state['active_jobs']=[n['receipt']for a in state['attempts']for n in a['nodes']if n.get('receipt')and not(Path(n['receipt']['job'])/'RESULT.json').exists()]
    atomic_json(root/'STATE.json',state)
    atomic_json(root/'EVALUATION_MAILBOX.json',dict(schema='d46_world16_evaluation_mailbox_v1',stage_id=plan['stage_id'],arm=plan['arm'],plan_sha256=plan['_sha256'],
        output=plan['output'],science_source=plan['science_source'],requests=state['evaluation_requests'],active_jobs=state['active_jobs']))


def requests(plan):
    result=[]
    for path in sorted((Path(plan['output'])/'evaluation_requests').glob('epoch*.json')):
        value=read(path);cp=Path(value['checkpoint']).resolve();seal=read(cp/'COMPLETE',{});ident=read(cp/'identity.json',{});files=read(cp/'files.json',{})
        if (not cp.is_relative_to(Path(plan['output']).resolve()/'checkpoints')or seal.get('world')!=16
            or ident.get('sha256')!=plan['identity_sha256']or ident.get('qualification')is not False
            or seal.get('files_sha256')!=file_sha(cp/'files.json')or files.get('identity.json')!=file_sha(cp/'identity.json')
            or value.get('source_run')!=plan['identity_sha256'] or value.get('updates')!=seal.get('updates')
            or value.get('seed')!=2101 or value.get('protocol')!='single_B1_cleanx_ODE_K10_shift5_canonical_v1.1'):
            raise ValueError('Invalid stage evaluation request')
        result.append(dict(path=str(path),sha256=file_sha(path),request=value))
    return result


def command(plan,node,resume):
    return ['/usr/bin/env','NCCL_IB_DISABLE=1','NCCL_SOCKET_IFNAME=eth0',PYTHON,'-m','torch.distributed.run','--nnodes=2','--nproc_per_node=8',f'--node_rank={node}',f"--master_addr={plan['master_addr']}",f"--master_port={plan['master_port']}",
        '--rdzv_conf=timeout=120','--max_restarts=0','-m','tools.d46_world16.train','--config',plan['config'],'--output',plan['output'],'--resume',resume]


def stop_owned(state,reason):
    for a in state['attempts'][-1:]:
        for node in a['nodes']:
            if node.get('receipt')and not(Path(node['receipt']['job'])/'RESULT.json').exists():
                (Path(node['receipt']['job'])/'STOP_REQUESTED').touch();node['cooperative_stop_reason']=reason


def reconcile(node):
    if node.get('receipt')or node.get('phase')!='LAUNCH_INTENT':return
    job=Path(node['job']);actual=read(job/'LAUNCH.json')
    if actual:
        if actual.get('host')!=node['host']or actual.get('command')!=node['command']:raise ValueError('Foreign launch receipt')
        node.update(receipt=actual,phase='DISPATCHED');return
    actual=read(job/'RUNNING.json')or read(job/'RESULT.json')
    if actual:
        if actual.get('host')!=node['host']or actual.get('command')!=node['command']or actual.get('cwd')!=node['source']:raise ValueError('Ambiguous worker identity')
        node.update(receipt=dict(host=node['host'],job=str(job),gpus=list(range(8)),command=node['command'],at=actual.get('start_unix'),adopted_actual_worker=True),phase='DISPATCHED');return
    raise RuntimeError('Launch intent has no conclusive local receipt; review before retry')



def launch_remaining(plan,state,attempt):
    minimum=plan.get('minimum_free_MiB',65000)
    (Path(plan['output'])/'evaluation_requests').mkdir(parents=True,exist_ok=True)
    for i,node in enumerate(attempt['nodes']):
        if node.get('phase')!='NOT_LAUNCHED':continue
        node['phase']='LAUNCH_INTENT';persist(plan,state)
        try:
            value=launch(node['host'],plan['science_source'],node['job'],node['command'],list(range(8)),
                share_min_free_mib=minimum if plan.get('shared_admission',False)else None,worker_source=plan['worker_source'],task_root=ROOT,task=TASK)
            node.update(receipt=value,phase='DISPATCHED');persist(plan,state)
            result=read(Path(node['job'])/'RESULT.json')
            if result and result.get('returncode')==75:
                for rest in attempt['nodes'][i+1:]:rest['phase']='NOT_LAUNCHED_PEER_REFUSED'
                break
        except DispatchPreflightFailed as error:
            node.update(phase='PRECHILD_PREFLIGHT_FAILED',error=repr(error));attempt['preflight_failed']=True
            stop_owned(state,'peer_preflight_failure');break
        except Exception:
            node['launch_error']=traceback.format_exc()
            try:reconcile(node)
            except Exception:pass
            state['status']='AMBIGUOUS_LAUNCH_REQUIRES_REVIEW';stop_owned(state,'ambiguous_launch');persist(plan,state);return state
    attempt['launch_complete']=all(n.get('receipt')for n in attempt['nodes'])
    return state

def tick(plan,state):
    # Recheck the main-owner hold; only Root may transfer ownership.
    current=load_plan(plan['_path'])
    if current['_sha256']!=state['plan_sha256']:raise ValueError('Plan changed')
    state['latest_native']=latest(plan,state.get('latest_native'));state['evaluation_requests']=requests(plan)
    if state['status']in TERMINAL:persist(plan,state);return state
    if not state['attempts']and state['latest_native']['valid_scene_exposures']>=6*103288 and state['latest_native']['phase']=='READY':
        state['status']='COMPLETE_AT_REGISTERED_6E';persist(plan,state);return state
    if (owner_root(plan)/'STOP_REQUESTED').exists()or(plan.get('owner_deadline_unix')is not None and time.time()>=plan['owner_deadline_unix']):
        stop_owned(state,'operator_or_explicit_deadline');state['status']='STOP_REQUESTED_OPERATOR'
        if not state['attempts']:
            state['status']='STOPPED_AT_NATIVE_BOUNDARY';persist(plan,state);return state
    if state['attempts']:
        a=state['attempts'][-1]
        for node in a['nodes']:
            try:reconcile(node)
            except Exception:
                state.update(status='AMBIGUOUS_LAUNCH_REQUIRES_REVIEW',error=traceback.format_exc());stop_owned(state,'ambiguous_gang');persist(plan,state);return state
        if state['status']=='STOP_REQUESTED_OPERATOR':
            stop_owned(state,'operator_or_explicit_deadline')
        outputs=[read(Path(n['receipt']['job'])/'RESULT.json')if n.get('receipt')else None for n in a['nodes']]
        real_failure=[r for r in outputs if r and r.get('returncode')not in(0,75)]
        if real_failure:
            state.update(status='FAILED_REQUIRES_REVIEW',failure_results=real_failure);stop_owned(state,'peer_actual_failure');persist(plan,state);return state
        if state['status']!='STOP_REQUESTED_OPERATOR'and any(n.get('phase')=='NOT_LAUNCHED'for n in a['nodes'])and not any(r and r.get('returncode')==75 for r in outputs):
            launch_remaining(plan,state,a)
            if state['status']=='AMBIGUOUS_LAUNCH_REQUIRES_REVIEW':return state
            outputs=[read(Path(n['receipt']['job'])/'RESULT.json')if n.get('receipt')else None for n in a['nodes']]
        live=any(n.get('receipt')and r is None for n,r in zip(a['nodes'],outputs))
        if live:
            if any(r and r.get('returncode')==75 for r in outputs):stop_owned(state,'peer_prechild_resource75')
            if state['status']!='STOP_REQUESTED_OPERATOR':state['status']='RUNNING_OR_ADMITTING'
            persist(plan,state);return state
        if state['status']=='STOP_REQUESTED_OPERATOR':state['status']='STOPPED_AT_NATIVE_BOUNDARY';persist(plan,state);return state
        refusals=[r for r in outputs if r and r.get('returncode')==75]
        if refusals:
            if any(r.get('child_not_started')is not True for r in refusals):state.update(status='FAILED_REQUIRES_REVIEW',error='75 without actual prechild proof');persist(plan,state);return state
            if a.get('resource75_handled')is not True:
                state['resource75_retries']+=1;a['resource75_handled']=True
            if state['resource75_retries']>=plan.get('resource75_retry_limit',3):state['status']='RESOURCE75_RETRY_LIMIT';persist(plan,state);return state
            state['retry_not_before']=max(r['end_unix']for r in refusals)+60
        elif a.get('launch_complete')and len(outputs)==2 and all(r and r.get('returncode')==0 for r in outputs):
            if state['latest_native']['valid_scene_exposures']>=6*103288 and state['latest_native']['phase']=='READY':
                state['status']='COMPLETE_AT_REGISTERED_6E';persist(plan,state);return state
        elif not a.get('preflight_failed'):
            state.update(status='AMBIGUOUS_LAUNCH_REQUIRES_REVIEW',error='Incomplete gang terminal proof');persist(plan,state);return state
    if state.get('retry_not_before',0)>time.time():state['status']='RESOURCE_BACKOFF';persist(plan,state);return state
    allowed=allowed_hosts(admit_long_job=True);minimum=plan.get('minimum_free_MiB',65000)
    for host in plan['hosts']:
        if host not in allowed:state['status']='WAITING_AUTHORIZED_CAPACITY';persist(plan,state);return state
        try:
            if capacity(host,minimum)!=list(range(8)):state['status']='WAITING_AUTHORIZED_CAPACITY';persist(plan,state);return state
        except(DispatchPreflightFailed,subprocess.SubprocessError,OSError)as error:state.update(status='WAITING_REACHABLE_HOST',preflight_error=repr(error));persist(plan,state);return state
    # One durable intent per node before remote Popen. Never redo an ambiguous launch.
    attempt=dict(number=len(state['attempts'])+1,created=time.time(),resume=state['latest_native']['checkpoint'],nodes=[])
    for i,host in enumerate(plan['hosts']):
        job=ROOT/'jobs/WORLD16_FORMAL'/plan['stage_id']/f"attempt{attempt['number']:03}"/f'node{i}'
        attempt['nodes'].append(dict(host=host,job=str(job),source=plan['science_source'],command=command(plan,i,attempt['resume']),phase='NOT_LAUNCHED'))
    state['attempts'].append(attempt);persist(plan,state)
    launch_remaining(plan,state,attempt)
    if state['status']=='AMBIGUOUS_LAUNCH_REQUIRES_REVIEW':return state
    attempt['launch_complete']=all(n.get('receipt')for n in attempt['nodes'])
    state['status']='RUNNING_OR_ADMITTING'if any(n.get('receipt')for n in attempt['nodes'])else'WAITING_REACHABLE_HOST';persist(plan,state);return state


def ingest(campaign,control):
    """Called ONLY by main campaign owner while it holds CAMPAIGN.owner.lock."""
    stagejobs=[];reserved_hosts=set()
    for arm,reference in control.get('world16_stages',{}).items():
        if arm not in campaign['runs']or arm not in('D0','D5','D3_FPO'):raise ValueError('Unknown world16 logical arm')
        if not control.get('execution_holds',{}).get(arm):raise ValueError('World16 stage cannot coexist with automatic old8 restart')
        if file_sha(reference['plan'])!=reference['plan_sha256']:raise ValueError('Stage pointer changed')
        p=read(reference['plan']);s=read(owner_root(p)/'STATE.json',{})
        if p.get('schema')!=PLAN or p['arm']!=arm or(s and(s.get('schema')!=STATE or s.get('stage_id')!=p['stage_id']or s.get('arm')!=arm or s.get('plan_sha256')!=reference['plan_sha256'])):raise ValueError('Foreign stage state')
        run=campaign['runs'][arm];run['world16_stage']=dict(plan=reference['plan'],plan_sha256=reference['plan_sha256'],stage_id=p['stage_id'],output=p['output'],source=p['science_source'],
            status=s.get('status','WAITING_STAGE_OWNER'),latest_native=s.get('latest_native'),hosts=p['hosts'],active_jobs=s.get('active_jobs',[]),
            identity_sha256=p['identity_sha256'],optimizer='inherited_AdamW_no_reset',world=16,last_attempt_started=s.get('attempts',[{}])[-1].get('created',0)if s.get('attempts')else 0)
        stagejobs+=s.get('active_jobs',[])
        if s.get('status','PREPARED')not in TERMINAL:reserved_hosts.update(p['hosts'])
        mailbox=read(owner_root(p)/'EVALUATION_MAILBOX.json',{})
        if mailbox and(mailbox.get('plan_sha256')!=reference['plan_sha256']or mailbox.get('stage_id')!=p['stage_id']or mailbox.get('arm')!=arm):raise ValueError('Foreign stage evaluation mailbox')
        for record in mailbox.get('requests',[]):
            path=Path(record['path'])
            if not path.resolve().is_relative_to(Path(p['output']).resolve()/'evaluation_requests')or file_sha(path)!=record['sha256']:raise ValueError('Mailbox request changed')
            req=read(path);key=arm+'_'+path.stem;cases=campaign.setdefault('evaluations',{})
            if key in cases and cases[key]['request']!=req:raise ValueError('Never overwrite an old full evaluation')
            cases.setdefault(key,dict(request=req,source=p['science_source'],status='PENDING',world=campaign.get('evaluation_world',32),jobs={},output=str(ROOT/'evaluations'/key/'evaluation'),world16_stage=p['stage_id']))
    campaign['world16_stage_jobs']=stagejobs;campaign['world16_reserved_hosts']=sorted(reserved_hosts)


def wait_metadata(plan,state,seconds):
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
    if fd<0:time.sleep(seconds);return
    try:
        paths=[Path(plan['output']),Path(plan['output'])/'evaluation_requests',owner_root(plan),ROOT/'operations']
        paths += [Path(n['job'])for a in state['attempts'][-1:]for n in a['nodes']]
        for path in paths:
            if path.is_dir():libc.inotify_add_watch(fd,os.fsencode(path),0x8|0x80|0x100)
        select.select([fd],[],[],seconds)
    finally:os.close(fd)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('prepare','run','once'));p.add_argument('--plan',required=True);p.add_argument('--interval',type=int,default=60);a=p.parse_args()
    if os.environ.get('WORLD_COMPARE_ROOT')!=str(ROOT)or os.environ.get('WORLD_COMPARE_TASK')!=TASK:raise ValueError('Explicit D46 root/task required')
    plan=load_plan(a.plan);root=owner_root(plan);root.mkdir(parents=True,exist_ok=True)
    with(root/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(root/'STATE.json')or initial(plan)
        if state['plan_sha256']!=plan['_sha256']:raise ValueError('Existing stage uses another sealed plan')
        atomic_json(root/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source=str(SOURCE),plan_sha256=plan['_sha256'],started=time.time()))
        if a.action=='prepare':persist(plan,state);return
        while True:
            try:tick(plan,state)
            except Exception:state.update(status='FAILED_REQUIRES_REVIEW',error=traceback.format_exc());stop_owned(state,'stage_control_failure');persist(plan,state)
            if a.action=='once'or state['status']in TERMINAL:break
            wait_metadata(plan,state,min(300,max(10,a.interval)))


if __name__=='__main__':main()
