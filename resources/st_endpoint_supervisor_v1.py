"""Lead-only fixed4096 DEV endpoint supervisor; no automatic dispatch or retry.

External control only. Frozen runtime751 supplies every model, export, canonical
score and paired-analysis operation. A real endpoint and fresh independent
admission are mandatory; status/--help never launches anything.
"""
from __future__ import annotations
import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time

CAMPAIGN=Path('/mnt/project/st-world-policy-20261016')
RUNTIME=Path('/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018')
SOURCE='751fcfa2b41549ae16f7b3204aef8dd18b912016'
RUN_ID='C1_rollout_std_fulltrain_seed42_official_4096_v1'
JOB_ID='C1_rollout_std_step004096_dev1696_seed42_v1'
TRAIN_REG=CAMPAIGN/'formal/consequence/registrations'/RUN_ID/'REGISTRATION.json'
TRAIN_REG_SHA='928b9788404c4dcb9316b8457dc1a627010f892b85308f52aacf07b2cbb0783f'
OUTPUT=CAMPAIGN/'formal/evaluation/jobs'/JOB_ID
ANALYSIS=CAMPAIGN/'formal/evaluation/analyses/C1_rollout_std_minus_C1_COMMON_ENDPOINT_dev1696_v1'
PARENT=CAMPAIGN/'resources/launch_formal_dev_export_v5.py'
DISPATCH_POLICY=Path('/mnt/project/server_dispatch_policy.json')
AUTHORIZATION=CAMPAIGN/'resources/AUTHORIZATION_20261016T203442.json'
PARENT_SHA='ad0ee24987bdba9ee3fbc46a5efbdb1445b0a6a69d4d588fc32e960738b876bf'
SHARED_SOURCE='100.97.192.3@tcp:100.97.192.4@tcp:/hbTcp001/78ebbad2386e922193d44ac249bbb12e_kl4gv5vivlcf06o2[/mnt/project]'
STATIC={
    'commands':'213cb24ca991c970b9cbc39436ab315afeb88363ef35bd94fe21b17dc6be5c41',
    'decision':'d77a8a0686e08700b65cc4a6f56baedfbdb788510a682a34064b23c852e9ad5c',
    'comparison':'2b978dc4e1e3d1ed1b47a84294bba123dbcf7f85e03626990522d76efc394055',
    'fixed_inputs':'96cf23f32267064cabc7a180b8162c78135d696cf60236c75a9db26ba82d146c',
    'analysis_template':'45ab6bb27f0439868934be7bac53809b2bb7d8d285d4c0ac23a1806a310129cf'}
LIMITS={'export_world':4,'scenes':1696,'logs':16,'seed':42,'steps':10,
    'export_outer':7800,'export_grace':60,'export_rank_RSS':96*2**30,'export_all_RSS':384*2**30,
    'fresh_available_RAM':512*2**30,'GPU_driver_bytes':32*2**30,'loaders':2,
    'score_outer':4200,'score_grace':60,'score_RSS':64*2**30,'score_global_calls':1728,
    'analysis_outer':1800,'analysis_grace':60,'analysis_RSS':8*2**30,
    'output_bytes':2**30,'invocations_per_unit':2,'shared_free_floor':1024*2**30}
THREAD_ENV={'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
    'MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}
IMPLEMENTER='/root/road_failure_direction_runner'


class WaitingCheckpoint(ValueError):pass


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def stable_binding(value):
    # Model/data/output/controller/declared route+lease remain immutable. Fresh
    # writer observations and current authorization are per-invocation receipts.
    return {key:item for key,item in value.items()
            if key not in ('storage_reservations','dispatch_policy','authorization')}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*2**20),b''):h.update(chunk)
    return h.hexdigest()


def small(path):
    with Path(path).open('rb') as stream:raw=stream.read(2*2**20+1)
    if len(raw)>2*2**20:raise ValueError('Oversized endpoint control')
    return json.loads(raw)


def verify_pin(ref):
    if (set(ref) not in ({'path','sha256'},{'path','sha256','bytes'})
            or not Path(ref['path']).is_absolute() or sha(ref['path'])!=ref['sha256']
            or ('bytes' in ref and Path(ref['path']).stat().st_size!=ref['bytes'])):
        raise ValueError('Changed endpoint control pin')


def pinned(ref):
    verify_pin(ref)
    return small(ref['path'])


def immutable(path,value):
    # Same mature no-replace primitive as export/scoring; loaded only on use.
    if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))
    from tools.st_world_policy.base_binding import immutable_json
    immutable_json(Path(path),value)


def mature_helpers():
    """Reuse exact small v5 helpers without executing its top-level dispatcher."""
    if sha(PARENT)!=PARENT_SHA:raise ValueError('Mature v5 helper source changed')
    names={'sha','check_route_time_v3','fresh_environment_v3','shared_lock_mount_v3',
           'cpu_capacity_v3','fresh_gpu_v3'}
    tree=ast.parse(PARENT.read_text())
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in functions}!=names:raise ValueError('Missing mature helper')
    class BoundedCommands:
        @staticmethod
        def check_output(*args,**kwargs):
            kwargs['timeout']=min(kwargs.get('timeout',15),15)
            return subprocess.check_output(*args,**kwargs)
    namespace={**globals(),'ROOT':CAMPAIGN,'ROUTE_SHARED_SOURCE_V3':SHARED_SOURCE,'subprocess':BoundedCommands}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(PARENT),'exec'),namespace)
    return {name:namespace[name] for name in names}


def validate_admission(path,expected_sha):
    if sha(path)!=expected_sha:raise ValueError('Admission checksum mismatch')
    receipt=small(path);binding=receipt['binding']
    if (receipt.get('schema')!='st_rollout_endpoint_admission_v1' or receipt.get('status')!='APPROVED'
            or receipt.get('binding_sha256')!=digest(binding)
            or receipt.get('stable_binding_sha256')!=digest(stable_binding(binding))
            or binding.get('source_sha')!=SOURCE or binding.get('run_id')!=RUN_ID
            or binding.get('controller_sha256')!=sha(__file__) or binding.get('limits')!=LIMITS):
        raise ValueError('Foreign/stale endpoint admission')
    reviewers=[]
    for kind in ('evidence','correctness','resources'):
        review=pinned(receipt['reviews'][kind]);reviewer=review.get('reviewer')
        if (review.get('schema')!='st_endpoint_supervisor_review_v1' or review.get('status')!='PASS'
                or review.get('review_kind')!=kind or review.get('binding_sha256')!=digest(stable_binding(binding))
                or not reviewer or reviewer in (IMPLEMENTER,receipt['issuer'])
                or not review.get('evidence_artifacts')):
            raise ValueError('Independent endpoint execution review missing/stale')
        for ref in review['evidence_artifacts']:verify_pin(ref)
        reviewers.append(reviewer)
    if len(set(reviewers))!=3:raise ValueError('Three distinct endpoint reviewers required')
    documents={}
    if set(binding['sealed'])!=set(STATIC):raise ValueError('Sealed endpoint inputs incomplete')
    for name,expected in STATIC.items():
        ref=binding['sealed'][name]
        if ref['sha256']!=expected:raise ValueError('Scientific endpoint control changed: '+name)
        documents[name]=pinned(ref)
    return receipt,binding,documents


def validate_endpoint(binding,documents):
    """Bind exact native metadata; frozen exporter rehashes ALL native bytes."""
    if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))
    from tools.st_world_policy.check_formal_approval import check_formal_approval
    from tools.st_world_policy.training_state import require_native_inventory,TERMINATIONS,terminal
    training=binding['training'];reg=pinned(training['registration']);approval=pinned(training['approval'])
    identity=pinned(training['identity']);launch=pinned(training['launch'])
    config=pinned(reg['artifacts']['config'])
    if (training['registration']!={'path':str(TRAIN_REG),'sha256':TRAIN_REG_SHA}
            or binding['environment']!=reg['artifacts']['environment']
            or reg['source_sha']!=SOURCE or reg['run_id']!=RUN_ID or reg['execution']['scope']!='FORMAL'
            or reg['execution']['max_updates']!=4096 or reg['execution']['sample_limit']!=0
            or identity['source_sha']!=SOURCE or identity['registration_sha256']!=training['registration']['sha256']
            or digest({k:v for k,v in identity.items() if k!='identity'})!=identity['identity']
            or identity['config']!=config or not identity['weights_usable_for_formal_init']
            or launch['source_sha']!=SOURCE or launch['diagnostic_weights_used'] is not False
            or launch['approval_sha256']!=training['approval']['sha256']
            or launch['registration_sha256']!=training['registration']['sha256']):
        raise ValueError('Wrong formal training lineage')
    gate=check_formal_approval(training['registration']['path'],training['approval']['path'],repo=RUNTIME)
    if not gate['formal_launch_allowed']:raise ValueError('Formal lineage lacks full independent approval')
    endpoint=binding['checkpoint'];tag=endpoint['tag']
    if not re.fullmatch(r'step_004096_[0-9]+',tag):raise ValueError('Only the exact4096 native endpoint is allowed')
    folder=Path(config['runtime']['output_root'])/RUN_ID/'checkpoints'/tag
    if not (folder/'COMPLETE.json').is_file():raise WaitingCheckpoint('WAITING_CHECKPOINT')
    if endpoint['complete']['path']!=str(folder/'COMPLETE.json'):raise ValueError('Foreign endpoint receipt path')
    complete=pinned(endpoint['complete']);state=complete['state']
    if (complete['schema']!='st_native_checkpoint_v1' or complete['world_size']!=4
            or digest({k:v for k,v in complete.items() if k!='receipt_identity'})!=complete['receipt_identity']
            or endpoint['receipt_identity']!=complete['receipt_identity']
            or state['completed']!=4096 or state['policy_version']!=4096 or state['tag']!=tag
            or state['identity']!=identity['identity'] or state.get('pending_old_policy','missing') is not None
            or state.get('pending_feedback','missing') is not None
            or state['termination_reason'] not in TERMINATIONS or not terminal(state,4096)):
        raise ValueError('Incomplete/foreign optimizer endpoint')
    require_native_inventory(complete['files'],4)
    if {p.name for p in folder.glob('*.pt')}!=set(complete['files']):raise ValueError('Native inventory changed')
    for name,info in complete['files'].items():
        if Path(name).name!=name or (folder/name).stat().st_size!=info['bytes']:
            raise ValueError('Incomplete native shard size')
    inputs=documents['fixed_inputs']['pins'];manifest=pinned(inputs['DEV_manifest'])
    scenes=manifest['scenes'];index=pinned(inputs['DEV_metric_index'])
    if (manifest['split']!='dev' or len(scenes)!=1696 or len({r['token'] for r in scenes})!=1696
            or len({r['log'] for r in scenes})!=16 or len(index)!=1696
            or {(r['token'],r['log']) for r in index}!={(r['token'],r['log']) for r in scenes}):
        raise ValueError('Complete fixed DEV population/context join changed')
    old=pinned(inputs['old_score_COMPLETE'])
    if (old.get('status')!='COMPLETE' or old.get('requested_scenes')!=1696 or old.get('valid_scored')!=1696
            or old.get('invalid')!=0 or old.get('unresolved')!=0
            or old.get('scenes_csv_sha256')!=documents['fixed_inputs']['legacy_csv_sha256']):
        raise ValueError('Original C1 control is not complete and unchanged')
    plan=pinned(binding['analysis_plan']);draft=documents['analysis_template']
    allowed_changes={'status','formal_launch_allowed','formal_runs','evaluation_runtime','execution_amendment'}
    if any(plan.get(k)!=v for k,v in draft.items() if k not in allowed_changes):
        raise ValueError('Sealed scientific analysis definitions changed')
    if not plan.get('formal_launch_allowed') or plan['formal_runs']['C1']!=draft['formal_runs']['C1']:
        raise ValueError('Final analysis plan or original control changed')
    amendment={**draft['execution_amendment'],'status':'APPROVED_WORLD4_EXECUTION'}
    if plan.get('execution_amendment')!=amendment:raise ValueError('Unreviewed execution amendment')
    arm=plan['formal_runs']['C1_ROLLOUT_STD']
    for key,new in (('registration','registration'),('approval','approval'),('run_identity_file','identity'),('launch','launch')):
        if arm[key]!=training[new]:raise ValueError('Final analysis training binding differs')
    if plan['score_protocol_sha256']!='db383e17b887e08beb358b282fbeab223c80c558164bead81d52a45b1b798143':
        raise ValueError('Canonical score protocol changed')
    return {'receipt_identity':complete['receipt_identity'],'training_identity':identity['identity'],
            'manifest':inputs['DEV_manifest'],'scenes':scenes}


def route_checks(binding,phase,helpers):
    if (binding['dispatch_policy']['path']!=str(DISPATCH_POLICY)
            or binding['authorization']['path']!=str(AUTHORIZATION)):
        raise ValueError('Current campaign authorization/live dispatch policy paths required')
    route=binding['route'];policy=pinned(binding['dispatch_policy']);auth=pinned(binding['authorization'])
    host=socket.gethostname()
    canonical=[name for name,value in policy['allowed_hosts'].items() if host in value['aliases']]
    if (len(canonical)!=1 or canonical[0]!='training-vlawm-tsinghua-zt6' or host!=route['hostname']
            or route['host']!=canonical[0] or host not in policy['allowed_aliases']
            or canonical[0] not in auth['allowed_hosts']):raise ValueError('Host outside fresh allowed endpoint route')
    for name,item in policy['global_dispatch_prohibitions'].items():
        if host==name or canonical[0]==name or host in item.get('aliases',[]) or canonical[0] in item.get('aliases',[]):
            raise ValueError('Host prohibited')
    duration={'export':7860,'score':4260,'analysis':1860}[phase]
    helpers['check_route_time_v3'](route,time.time(),duration)
    if (route['gpus']!=list(range(4)) or len(set(route['gpu_uuids']))!=4
            or any(not re.fullmatch(r'GPU-[0-9a-f-]{36}',u) for u in route['gpu_uuids'])):
        raise ValueError('Four real GPU UUIDs0..3 required; placeholders refused')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True,timeout=15).strip()!=SOURCE or subprocess.check_output(['git','status','--porcelain'],cwd=RUNTIME,timeout=15).strip():
        raise ValueError('Frozen evaluation source changed')
    helpers['fresh_environment_v3'](binding['environment']);helpers['shared_lock_mount_v3']();capacity=helpers['cpu_capacity_v3']()
    if not set(range(12)).issubset(os.sched_getaffinity(0)) or (capacity['quota_us']>=0 and capacity['quota_us']/capacity['period_us']<8):
        raise ValueError('Insufficient fixed CPU affinity/quota')
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    if phase=='export' and int(mem['MemAvailable'].split()[0])*1024<LIMITS['fresh_available_RAM']:
        raise ValueError('Fresh512GiB available host RAM required')
    storage=pinned(binding['storage_reservations'])
    if (not 0<=time.time()-storage['observed_unix']<=300 or storage['future_reserved_bytes']<0
            or shutil.disk_usage(CAMPAIGN).free<LIMITS['shared_free_floor']+storage['future_reserved_bytes']+LIMITS['output_bytes']):
        raise ValueError('Refresh active writer/storage commitments and free floor')
    return route


def argv_for(phase,documents,binding,rank=None):
    commands=documents['commands']
    if phase=='export':argv=list(commands['exports'][rank]['argv_template'])
    elif phase=='score':argv=list(commands['score']['argv'])
    else:argv=list(commands['analysis']['argv_template'])
    replacements={'__ACTUAL_VERIFIED_COMPLETED_4096_NATIVE_TAG__':binding['checkpoint']['tag'],
                  '__SEALED_ANALYSIS_PLAN_WITH_REAL_APPROVAL_AND_RUNTIME_HASHES__':binding['analysis_plan']['path']}
    argv=[replacements.get(value,value) for value in argv]
    if any('__' in value for value in argv):raise ValueError('Unresolved command placeholder')
    if phase=='export':cpus=f'{2*rank},{2*rank+1}'
    elif phase=='score':cpus='8-11'
    else:cpus='8,9'
    return ['timeout','--signal=TERM',f"--kill-after={LIMITS[phase+'_grace']}s",f"{LIMITS[phase+'_outer']}s",'taskset','-c',cpus,*argv]


def claim_invocation(root,unit,*,resume,binding_sha,argv):
    target=Path(root)/'invocations'/unit;prior=[path for path in target.glob('*') if path.is_dir()]
    if len(prior)>=2:raise ValueError('Two durable invocation admissions exhausted: '+unit)
    if prior and not resume:raise ValueError('Explicit same-identity resume required; never automatically retry')
    for folder in prior:
        path=folder/'ADMISSION.json'
        if path.exists():
            if small(path)['binding_sha256']!=binding_sha:raise ValueError('Cannot resume changed endpoint identity')
        elif not (Path(root)/'BINDING.json').exists() or digest(small(Path(root)/'BINDING.json'))!=binding_sha:
            raise ValueError('Unbound partial intent remains charged; explicit lineage reconciliation required')
    attempt=target/str(time.time_ns());attempt.mkdir(parents=True,exist_ok=False)
    immutable(attempt/'ADMISSION.json',{'status':'ADMITTED_BEFORE_POPEN_OUTCOME_UNKNOWN','ordinal':len(prior)+1,
        'maximum_invocations':2,'binding_sha256':binding_sha,'unit':unit,'argv':argv,'admitted_unix':time.time()})
    return attempt


def process_tree(pid):
    rows={};pending=[pid]
    while pending:
        current=pending.pop()
        try:
            fields=Path(f'/proc/{current}/stat').read_text().rsplit(')',1)[1].split()
            rows[current]={'start_ticks':int(fields[19]),'rss':int(fields[21])*os.sysconf('SC_PAGE_SIZE'),
                'state':fields[0],'cpu_seconds':sum(int(fields[i]) for i in (11,12,13,14))/os.sysconf('SC_CLK_TCK')}
            for path in Path(f'/proc/{current}/task').glob('*/children'):
                try:pending.extend(int(p) for p in path.read_text().split() if int(p) not in rows)
                except FileNotFoundError:pass
        except (FileNotFoundError,ProcessLookupError):pass
    return rows


def session_members(session_id,root_start=None):
    """Linux session membership catches reparented and threaded descendants.

    A fresh setsid session can contain only descendants of our Popen child.
    A reused session leader PID with a different start stamp is foreign.
    Only filtered owned-session metadata is returned/persisted.
    """
    rows={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            if os.getsid(int(path.parent.name))!=session_id:continue
            fields=path.read_text().rsplit(')',1)[1].split()
            if int(fields[3])!=session_id:continue
            pid=int(path.parent.name)
            rows[pid]={'start_ticks':int(fields[19]),'rss':int(fields[21])*os.sysconf('SC_PAGE_SIZE'),
                'state':fields[0],'cpu_seconds':sum(int(fields[i]) for i in (11,12,13,14))/os.sysconf('SC_CLK_TCK'),
                'pgrp':int(fields[2])}
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    leader=rows.get(session_id)
    if leader and root_start is not None and leader['start_ticks']!=root_start:return {}
    return rows


def owned_output_bytes():
    total=0
    for root in (OUTPUT,ANALYSIS):
        for path in root.rglob('*'):
            try:
                if path.is_file():total+=path.stat().st_size
            except FileNotFoundError:pass
    return total


def require_export_identity(signature,expected):
    checkpoint=signature['checkpoint']
    if (digest({k:v for k,v in signature.items() if k!='identity'})!=signature['identity']
            or signature['source_sha']!=SOURCE or signature['world_size']!=4
            or signature['seed']!=42 or signature['precision']!='FP32_master_eval_TF32_off'
            or signature['solver']!='original_Euler_10' or signature['candidates_per_scene']!=1
            or signature['requested_scenes']!=1696 or signature['targets_read'] is not False
            or signature['teacher_cache_read'] is not False or signature['scorer'] is not None
            or signature['selected_scenes_sha256']!=expected['selected_scenes_sha256']
            or signature['observations']['manifest_sha256']!=expected['manifest_sha256']
            or checkpoint['completed']!=4096 or checkpoint['tag']!=expected['tag']
            or checkpoint['run_identity']!=expected['run_identity'] or checkpoint['scope']!='FORMAL'
            or checkpoint['weights_usable_for_formal_init'] is not True
            or checkpoint['receipt_identity']!=expected['receipt_identity']
            or checkpoint['native_receipt_sha256']!=expected['native_receipt_sha256']
            or checkpoint['native_checkpoint_sha256']!=expected['native_checkpoint_sha256']
            or digest({k:v for k,v in checkpoint.items() if k!='sha256'})!=checkpoint['sha256']):
        raise ValueError('Export differs from the exact fixed formal endpoint')
    return checkpoint


def load_finished(rank,owned,*,after=0.,expected):
    for path in (OUTPUT/'export/attempts').glob('*/LOAD_AUDIT.json'):
        invocation=path.with_name('INVOCATION.json')
        if not invocation.exists():continue
        record=small(invocation)
        if (record['arguments']['rank']==rank and record['pid'] in owned
                and record['started_unix']>=after):
            if (record['arguments']['world_size']!=4 or record['arguments']['output']!=str(OUTPUT/'export')):
                raise ValueError('Foreign exporter load audit')
            args=record['arguments'];signature=small(OUTPUT/'export/IDENTITY.json')
            checkpoint=require_export_identity(signature,expected)
            if (args['registration']!=expected['registration'] or args['checkpoint_tag']!=expected['tag']
                    or args['device']!='cuda:0' or args['seed']!=42 or args['max_scenes']!=1696
                    or record['identity']!=signature['identity']):
                raise ValueError('LOAD_AUDIT belongs to another export/checkpoint identity')
            audit=small(path)
            exact={'schema':'st_flow_fp32_restoration_audit_v1','checkpoint_record_sha256':checkpoint['sha256'],
                'parent_checkpoint_sha256':checkpoint['parent_checkpoint_sha256'],
                'native_checkpoint_sha256':checkpoint['native_checkpoint_sha256'],
                'frozen_encoder_source':'unchanged_original_bound_S0_FP32_masters',
                'training_BF16_encoder_reused':False,'original_parent_modified':False,
                'optimizer_resume_claim':False,'training_feedback_required':False,'all_parameters_frozen':True,
                'strict_action_load':True,'master_dtype':'float32',
                'action_buffers':'unchanged_bound_parent_values_verified_against_native'}
            if any(audit.get(key)!=value for key,value in exact.items()):raise ValueError('Missing strict FP32-master restoration audit')
            return True
    return False


def gpu_monitor(route,owned):
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                  '--format=csv,noheader,nounits'],text=True,timeout=15)
    for line in rows.splitlines():
        if not line.strip():continue
        uuid,pid,memory=(field.strip() for field in line.split(','))
        if uuid not in route['gpu_uuids']:continue
        stamp=process_tree(int(pid)).get(int(pid))
        if int(pid) not in owned or not stamp or stamp['start_ticks']!=owned[int(pid)]['start_ticks']:
            raise ValueError('Unknown/foreign process on endpoint GPU; stopping owned work only')
        int(memory) # Refuse unobservable usage rather than guessing zero.
    rows=subprocess.check_output(['nvidia-smi','--id='+','.join(map(str,route['gpus'])),
        '--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=15)
    used={}
    for line in rows.splitlines():
        uuid,memory=(field.strip() for field in line.split(','))
        if uuid not in route['gpu_uuids'] or uuid in used:raise ValueError('Endpoint GPU UUID changed')
        used[uuid]=int(memory)*2**20
    if set(used)!=set(route['gpu_uuids']):raise ValueError('Missing endpoint GPU driver statistics')
    if any(value>LIMITS['GPU_driver_bytes'] for value in used.values()):raise ValueError('Endpoint GPU driver usage cap')
    return used


def stop_owned(jobs):
    """Only sessions this supervisor created; never unknown root/GPU processes."""
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def active(job):
        rows=session_members(job['child'].pid,job.get('start_ticks'))
        job.setdefault('owned',{}).update(rows)
        return {pid:value for pid,value in rows.items() if value['state']!='Z'}
    def send(job,signum):
        groups={row['pgrp'] for row in active(job).values()}
        for group in groups:
            try:os.killpg(group,signum)
            except ProcessLookupError:pass
    try:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        for job in jobs:
            send(job,signal.SIGTERM)
        deadline=time.monotonic()+60
        while any(active(job) for job in jobs) and time.monotonic()<deadline:time.sleep(.1)
        for job in jobs:
            send(job,signal.SIGKILL)
            try:job['child'].wait(timeout=5)
            except subprocess.TimeoutExpired:job['cleanup_error']='OWNED_ROOT_EXIT_NOT_CONFIRMED'
            job['cleanup_complete']=job['child'].poll() is not None and not active(job)
    finally:
        for sig,value in previous.items():signal.signal(sig,value)


def monitor(jobs,phase,route,*,pending,launch,check_stop=lambda:None):
    """At most two heavy loaders; finish this invocation once, never auto retry."""
    peak=0;seen={};start=time.monotonic();last_sample=0.;last_output_scan=0.;last_GPU_scan=0.;why=None;gpu={}
    try:
        while pending or any(job['child'].poll() is None for job in jobs):
            check_stop()
            for job in jobs:
                if job['child'].poll() is not None:
                    remaining=session_members(job['child'].pid,job.get('start_ticks'))
                    if any(row['state']!='Z' for row in remaining.values()):
                        raise ValueError('Owned root exited with surviving session descendants')
                    if job['child'].returncode!=0:raise ValueError('Owned child failed; no additional dispatch')
            live=[job for job in jobs if job['child'].poll() is None];trees={}
            for job in live:
                tree=session_members(job['child'].pid,job.get('start_ticks'));trees[job['unit']]=tree;seen.update(tree)
                job.setdefault('owned',{}).update(tree)
                rss=sum(row['rss'] for row in tree.values())
                limit=LIMITS['export_rank_RSS'] if phase=='export' else LIMITS[phase+'_RSS']
                if rss>limit:raise ValueError('Owned process-tree RSS cap')
                if phase=='export' and not job['loaded'] and load_finished(job['rank'],tree,
                        after=job.get('started_unix',0.),expected=job['load_expected']):job['loaded']=True
            aggregate=sum(row['rss'] for tree in trees.values() for row in tree.values());peak=max(peak,aggregate)
            if phase=='export' and aggregate>LIMITS['export_all_RSS']:raise ValueError('Aggregate export RSS cap')
            if time.monotonic()-last_output_scan>=10:
                if owned_output_bytes()>LIMITS['output_bytes']:raise ValueError('Endpoint output cap')
                last_output_scan=time.monotonic()
            if phase=='export' and time.monotonic()-last_GPU_scan>=5:
                gpu=gpu_monitor(route,seen);last_GPU_scan=time.monotonic()
            loading=sum(not job['loaded'] for job in live)
            while pending and (phase!='export' or loading<LIMITS['loaders']):
                check_stop()
                rank=pending.pop(0);job=launch(rank)
                if job not in jobs:jobs.append(job)
                check_stop()
                loading+=1
                if phase!='export':break
            if time.monotonic()-last_sample>=5:
                sample={'unix':time.time(),'phase':phase,'aggregate_RSS':aggregate,'GPU_driver_bytes':gpu,
                    'live_units':[job['unit'] for job in live],'heavy_loaders':loading,
                    'live_and_waited_subtree_CPU_seconds':sum(row['cpu_seconds'] for tree in trees.values() for row in tree.values())}
                with (OUTPUT/'control/RESOURCE_SAMPLES.jsonl').open('a') as stream:stream.write(json.dumps(sample)+'\n')
                last_sample=time.monotonic()
            time.sleep(.25)
        check_stop()
        if owned_output_bytes()>LIMITS['output_bytes']:raise ValueError('Endpoint output cap')
        if any(job['child'].returncode!=0 for job in jobs):why='CHILD_PARTIAL_OR_FAILED_NO_AUTO_RETRY'
    except BaseException as error:
        why=type(error).__name__+': '+str(error)
    finally:
        try:stop_owned(jobs)
        except BaseException as error:why='OWNED_CLEANUP_ERROR: '+repr(error)
    if any(not job.get('cleanup_complete',False) for job in jobs):why=why or 'OWNED_CLEANUP_UNCONFIRMED'
    for job in jobs:
        immutable(job['attempt']/'EXIT.json',{'returncode':job['child'].returncode,'stop_reason':why,
            'finished_unix':time.time(),'owned_PID':job['child'].pid,
            'owned_IDs':{str(pid):{'start_ticks':row['start_ticks']} for pid,row in job.get('owned',{}).items()},
            'owned_cleanup_complete':job.get('cleanup_complete',False)})
    return {'status':'PROCESSES_EXITED' if why is None else 'PARTIAL_OR_FAILED','stop_reason':why,
        'wall_seconds':time.monotonic()-start,'peak_aggregate_sampled_RSS_bytes':peak,
        'owned_PID_count':len(seen),'invocations_started':len(jobs)}


def ensure_exports_exited(control):
    """CPU phases release OUR export sessions; unrelated new GPU users are safe."""
    for admission in (Path(control)/'invocations').glob('export_rank*/*/ADMISSION.json'):
        exit_path=admission.with_name('EXIT.json')
        if not exit_path.is_file():raise ValueError('Unresolved owned export invocation; inspect before CPU scoring')
        exited=small(exit_path)
        if not exited.get('owned_cleanup_complete'):raise ValueError('Owned export cleanup unconfirmed')
        for raw_pid,old in exited.get('owned_IDs',{}).items():
            pid=int(raw_pid);current=process_tree(pid).get(pid)
            if current and current['state']!='Z' and current['start_ticks']==old['start_ticks']:
                raise ValueError('Owned exporter still alive; GPU resources not released')


def execute(phase,receipt,binding,documents,*,resume):
    helpers=mature_helpers();route=route_checks(binding,phase,helpers)
    if owned_output_bytes()>LIMITS['output_bytes']:raise ValueError('Endpoint output cap already exceeded')
    control=OUTPUT/'control';control.mkdir(parents=True,exist_ok=True)
    locks=[]
    try:
        for path in [control/'SUPERVISOR.lock']+([Path(f'/tmp/foresight_gpu_{gpu}.lock') for gpu in range(4)] if phase=='export' else []):
            handle=path.open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        route_checks(binding,phase,helpers)
        immutable(control/'BINDING.json',stable_binding(binding))
        resource_admission=control/('RESOURCE_ADMISSION_'+str(time.time_ns())+'.json')
        immutable(resource_admission,receipt)
        native=pinned(binding['checkpoint']['complete'])
        expected_load={'registration':binding['training']['registration']['path'],'tag':binding['checkpoint']['tag'],
            'run_identity':pinned(binding['training']['identity'])['identity'],
            'receipt_identity':native['receipt_identity'],'native_receipt_sha256':binding['checkpoint']['complete']['sha256'],
            'native_checkpoint_sha256':digest({key:value['sha256'] for key,value in native['files'].items()}),
            'manifest_sha256':documents['fixed_inputs']['pins']['DEV_manifest']['sha256'],
            'selected_scenes_sha256':digest(pinned(documents['fixed_inputs']['pins']['DEV_manifest'])['scenes'])}
        if (OUTPUT/'export/IDENTITY.json').exists():require_export_identity(small(OUTPUT/'export/IDENTITY.json'),expected_load)
        if phase=='export':
            for gpu,uuid in zip(route['gpus'],route['gpu_uuids']):helpers['fresh_gpu_v3']({'gpu':gpu,'gpu_uuid':uuid})
            pending=[rank for rank in range(4) if not (OUTPUT/f'export/COMPLETE_rank{rank}.json').exists()]
            if not pending:raise ValueError('All ranks already complete; no duplicate export')
        else:
            # Mature exact manifest/receipt verification before a single scorer.
            from tools.st_world_policy.score_world import read_complete_export
            export,scenes=read_complete_export(OUTPUT/'export')
            require_export_identity(export,expected_load)
            if (export['world_size']!=4 or len(scenes)!=1696 or export['source_sha']!=SOURCE
                    or export['checkpoint']['completed']!=4096):raise ValueError('Incomplete endpoint export')
            ensure_exports_exited(control)
            completion=OUTPUT/'official_score/COMPLETE.json' if phase=='score' else ANALYSIS/'ANALYSIS.json'
            if completion.exists():raise ValueError('Stage already complete; no duplicate execution')
            if phase=='analysis' and not (OUTPUT/'official_score/COMPLETE.json').is_file():raise ValueError('Scoring not complete')
            pending=[None]
        jobs=[]
        def launch(rank):
            unit=f'export_rank{rank}' if phase=='export' else phase
            argv=argv_for(phase,documents,binding,rank)
            helpers['check_route_time_v3'](route,time.time(),{'export':7860,'score':4260,'analysis':1860}[phase])
            attempt=claim_invocation(control,unit,resume=resume,binding_sha=digest(stable_binding(binding)),argv=argv)
            started_unix=time.time()
            env={**os.environ,**THREAD_ENV,'CUDA_VISIBLE_DEVICES':route['gpu_uuids'][rank] if phase=='export' else ''}
            try:log=(attempt/'OUTPUT.log').open('xb')
            except OSError as error:
                immutable(attempt/'EXIT.json',{'returncode':None,'stop_reason':'LOG_OPEN_OS_ERROR_BEFORE_POPEN: '+str(error),
                    'owned_cleanup_complete':True,'owned_IDs':{},'finished_unix':time.time()})
                raise
            try:
                try:
                    child=subprocess.Popen(argv,cwd=RUNTIME,env=env,stdin=subprocess.DEVNULL,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(f.fileno() for f in locks))
                except OSError as error:
                    immutable(attempt/'EXIT.json',{'returncode':None,'stop_reason':'Popen_OS_ERROR: '+str(error),
                        'owned_cleanup_complete':True,'owned_IDs':{},'finished_unix':time.time()})
                    raise
                # Register before log.close or any other post-Popen operation.
                # All subsequent failures are visible to owned cleanup.
                job={'child':child,'attempt':attempt,'unit':unit,'rank':rank,'loaded':False,
                    'start_ticks':None,'owned':{},'started_unix':started_unix,'load_expected':expected_load}
                jobs.append(job)
            finally:log.close()
            tree=session_members(child.pid);ticks=tree.get(child.pid,{}).get('start_ticks')
            job.update(start_ticks=ticks,owned=tree)
            immutable(attempt/'LAUNCH.json',{'pid':child.pid,'start_ticks':ticks,'argv':argv,'cwd':str(RUNTIME),
                'source_sha':SOURCE,'controller_sha256':sha(__file__),'admission_sha256':digest(receipt),
                'resource_admission':{'path':str(resource_admission),'sha256':sha(resource_admission)},
                'started_unix':started_unix,'rank':rank,'GPU_UUID':env['CUDA_VISIBLE_DEVICES']})
            return job
        before=resource.getrusage(resource.RUSAGE_CHILDREN)
        previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        stop_request=[]
        def interrupted(signum,frame):
            # Do not raise between fork/Popen and receiving/registering the
            # owned child handle. Defer to a monitor boundary after ownership
            # transfer, without blocking signals in this or the child process.
            if not stop_request:stop_request.append(signum)
        def check_stop():
            if stop_request:raise RuntimeError('OPERATOR_SIGNAL_'+str(stop_request[0]))
        try:
            for sig in previous:signal.signal(sig,interrupted)
            result=monitor(jobs,phase,route,pending=pending,launch=launch,check_stop=check_stop)
        finally:
            for sig,value in previous.items():signal.signal(sig,value)
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        result['waited_child_CPU_seconds']=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime)
        # Validate end artifacts, never report exit0 alone as completed scoring.
        if result['status']=='PROCESSES_EXITED':
            try:
                if phase=='export':
                    from tools.st_world_policy.score_world import read_complete_export
                    read_complete_export(OUTPUT/'export');ensure_exports_exited(control)
                elif phase=='score':
                    complete=small(OUTPUT/'official_score/COMPLETE.json')
                    if (complete['requested_scenes']!=1696 or complete['valid_scored']!=1696
                            or complete['invalid']!=0 or complete['unresolved']!=0):raise ValueError('Incomplete official score')
                else:
                    complete=small(ANALYSIS/'ANALYSIS.json')
                    if complete['status']!='COMPLETE' or complete['validity']!={'requested':1696,'paired_valid':1696,'invalid':0,'unresolved':0}:
                        raise ValueError('Incomplete paired analysis result')
            except (OSError,ValueError,KeyError) as error:
                result.update(status='PARTIAL_OR_FAILED',stop_reason='OUTPUT_VALIDATION: '+str(error))
        immutable(control/(phase+'_RESULT_'+str(time.time_ns())+'.json'),result)
        return result
    finally:
        for handle in reversed(locks):handle.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',action='store_true',help='Report fixed scope without any checkpoint/model/GPU read')
    parser.add_argument('--admission',type=Path);parser.add_argument('--admission-sha256')
    parser.add_argument('--phase',choices=('export','score','analysis'))
    parser.add_argument('--execute',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.status:
            if args.admission or args.phase or args.execute or args.resume:raise ValueError('Status takes no dispatch arguments')
            print(json.dumps({'status':'WAITING_CHECKPOINT_AND_FRESH_LEAD_ADMISSION','source_sha':SOURCE,
                'run_id':RUN_ID,'completed_required':4096,'limits':LIMITS,'dispatch_started':False}));return 0
        if not args.admission or not args.admission_sha256 or not args.phase:raise ValueError('Actual admission/hash/phase required')
        receipt,binding,documents=validate_admission(args.admission,args.admission_sha256)
        validate_endpoint(binding,documents)
        if not args.execute:
            if args.resume:raise ValueError('Resume is execution only')
            print(json.dumps({'status':'ENDPOINT_METADATA_VERIFIED_EXECUTION_NOT_STARTED','binding_sha256':digest(binding)}));return 0
        result=execute(args.phase,receipt,binding,documents,resume=args.resume)
        print(json.dumps(result));return 0 if result['status']=='PROCESSES_EXITED' else 2
    except WaitingCheckpoint as error:
        print(json.dumps({'status':'WAITING_CHECKPOINT','error':str(error)}));return 3
    except (ValueError,AssertionError,KeyError,TypeError,OSError,RuntimeError,subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
        print(json.dumps({'status':'BLOCKED','error':str(error)}));return 2


if __name__=='__main__':raise SystemExit(main())
