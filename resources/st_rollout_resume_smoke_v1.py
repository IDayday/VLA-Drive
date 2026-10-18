"""Lead-only runtime751 native4 versus native2+2 SMOKE; no automatic dispatch.

An isolated observer hashes native logical state without changing trainer math.
--status/plan use metadata only. Fresh independent admission is required for
prepare/initial/resume; compare reads only this experiment's audit artifacts.
"""
from __future__ import annotations
import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

CAMPAIGN=Path('/mnt/project/st-world-policy-20261016')
RUNTIME=Path('/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018')
SOURCE='751fcfa2b41549ae16f7b3204aef8dd18b912016'
ROOT=CAMPAIGN/'smoke/rollout_scale_native_resume_v3'
PRIOR_ROOT=CAMPAIGN/'smoke/rollout_scale_native_resume_v2'
OWNERSHIP_MONITOR=CAMPAIGN/'resources/monitor_rollout_scale_formal_v1.py'
OWNERSHIP_MONITOR_SHA='3d01323f9b6bfa970bdd4abde5d4b6a37925d5eacd0813d4d996368010fecc38'
_MATURE_STAT=None
FORMAL=CAMPAIGN/'formal/consequence/registrations/C1_rollout_std_fulltrain_seed42_official_4096_v1/REGISTRATION.json'
FORMAL_SHA='928b9788404c4dcb9316b8457dc1a627010f892b85308f52aacf07b2cbb0783f'
MANIFEST=CAMPAIGN/'audits/flow_policy/training_integration_v1/real32/TRAINING_MANIFEST.json'
MANIFEST_SHA='19a4429f6341ee9f39c05118545c816433231b397217da887a81e4ad38cffc5b'
MATURE=CAMPAIGN/'resources/qualify_official_native_resume_flow_v2.py'
ENDPOINT_SHA='a2e24537b9749258417e390be705d4d5d4e6f775986ca479a1c67c4330759a3e'
IMPLEMENTER='/root/road_failure_direction_runner'
PYTHON='/root/miniconda3/envs/ddp/bin/python'
PAUSE_MARKER='owned observer control after engine native2 save, before COMPLETE and next optimizer\n'
LIMITS={'scope':'SMOKE','world':4,'global_batch':32,'micro_batch':4,'accumulation':2,
    'max_updates':4096,'warmup':100,'stop_after':4,'total_new_updates':8,'primary_calls':1024,
    'retry_calls':0,'global_calls':1024,'arm_calls':512,'max_invocations':4,'prior_invocations':1,'new_invocations':3,
    'wall_seconds':3540,'prior_wall_seconds':60,'grace_seconds':60,'group_RSS_bytes':256*2**30,
    'aggregate_RSS_bytes':512*2**30,'GPU_driver_bytes':48*2**30,
    'fresh_RAM_bytes':640*2**30,'output_bytes':104*2**30,'native_directories':8,
    'native_directory_bytes':12*2**30,'monitor_read_bytes':1024*2**30,
    'launch_shared_free_bytes':1634*2**30,'runtime_shared_free_bytes':1530*2**30}
COMPARE_LIMITS={'CPU_seconds':120,'RSS_bytes':2*2**30,'input_bytes':64*2**20,'output_bytes':16*2**20}


def helper():
    path=Path(__file__).with_name('st_endpoint_supervisor_v1.py')
    if sha(path)!=ENDPOINT_SHA:raise ValueError('Reviewed owned-process helper changed')
    spec=importlib.util.spec_from_file_location('mature_endpoint_lifecycle',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*2**20),b''):result.update(chunk)
    return result.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    with Path(path).open('rb') as stream:raw=stream.read(2*2**20+1)
    if len(raw)>2*2**20:raise ValueError('Oversized SMOKE control')
    return json.loads(raw)


def pin(path):return {'path':str(path),'sha256':sha(path)}


def pinned(ref):
    if set(ref)!={'path','sha256'} or not Path(ref['path']).is_absolute() or sha(ref['path'])!=ref['sha256']:
        raise ValueError('Changed SMOKE control pin')
    return read(ref['path'])


def write(path,value):helper().immutable(path,value)


def process_record(pid):
    """Mature formal PPID/start/RSS reader plus needed lifecycle metadata."""
    global _MATURE_STAT
    if _MATURE_STAT is None:
        if sha(OWNERSHIP_MONITOR)!=OWNERSHIP_MONITOR_SHA:raise ValueError('Mature PPID source changed')
        tree=ast.parse(OWNERSHIP_MONITOR.read_text());nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='stat']
        if len(nodes)!=1:raise ValueError('Mature process stat helper absent')
        namespace={'Path':Path,'os':os};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OWNERSHIP_MONITOR),'exec'),namespace)
        _MATURE_STAT=namespace['stat']
    row=_MATURE_STAT(pid);fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    if row['start_ticks']!=int(fields[19]):raise ProcessLookupError('PID changed during ownership read')
    return {**row,'rss':row['rss_bytes'],'state':fields[0],'pgrp':int(fields[2]),'sid':int(fields[3]),
        'cpu_seconds':sum(int(fields[index]) for index in (11,12,13,14))/os.sysconf('SC_CLK_TCK')}


def process_table():
    table={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:table[int(path.name)]=process_record(int(path.name))
            except (OSError,ValueError,IndexError):pass
    return table


def owned_processes(job):
    """PPID closure, then retained PID/start and proven child-session ownership.

    Torch elastic ranks call setsid. SID equality to timeout is insufficient.
    A new SID is retained only after its leader was proven in the owned PPID
    tree. No NVML PID is used as an ownership seed. Unobserved unknowns fail.
    """
    table=process_table();root=job['child'].pid;known=job.setdefault('owned',{});sessions=job.setdefault('owned_sessions',{})
    if job.get('start_ticks') is None and job['child'].poll() is None and root in table:job['start_ticks']=table[root]['start_ticks']
    owned={pid:row for pid,row in table.items() if pid in known and row['start_ticks']==known[pid]['start_ticks']}
    if root in table and table[root]['start_ticks']==job.get('start_ticks'):owned[root]=table[root]
    while True:
        for pid,row in owned.items():
            if pid==row['sid']:sessions[pid]=row['start_ticks']
        added={}
        for pid,row in table.items():
            if pid in owned:continue
            parent=owned.get(row['ppid']);leader=table.get(row['sid']);stamp=sessions.get(row['sid'])
            via_parent=parent is not None and row['start_ticks']>=parent['start_ticks']
            session_continuous=(leader is not None and leader['start_ticks']==stamp) or any(
                member['sid']==row['sid'] and member['state']!='Z' for member in owned.values())
            via_session=stamp is not None and row['start_ticks']>=stamp and session_continuous
            if via_parent or via_session:added[pid]=row
        if not added:break
        owned.update(added)
    known.update(owned);return owned


def stop_smoke_owned(jobs):
    """Signal only individually reverified known descendants, across setsid."""
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def active(job):return {pid:row for pid,row in owned_processes(job).items() if row['state']!='Z'}
    def send(job,signum):
        for pid,row in active(job).items():
            try:
                current=process_record(pid)
                if current['start_ticks']==row['start_ticks']:os.kill(pid,signum)
            except (FileNotFoundError,ProcessLookupError):pass
    try:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        for job in jobs:send(job,signal.SIGTERM)
        deadline=time.monotonic()+60
        while any(active(job) for job in jobs) and time.monotonic()<deadline:time.sleep(.1)
        for job in jobs:
            send(job,signal.SIGKILL)
            try:job['child'].wait(timeout=5)
            except subprocess.TimeoutExpired:job['cleanup_error']='OWNED_ROOT_EXIT_NOT_CONFIRMED'
            job['cleanup_complete']=job['child'].poll() is not None and not active(job)
    finally:
        for sig,value in previous.items():signal.signal(sig,value)


def monitor_smoke_gpu(ep,route,jobs,currently_owned):
    """One ownership-only refresh closes the snapshot-to-NVML fork race.

    Never seed ownership from the reported GPU PID, command, or GPU UUID.
    A still-unknown PID after fresh PPID/start discovery is rejected unchanged.
    This is not an official scoring/trainer retry.
    """
    try:return ep.gpu_monitor(route,currently_owned)
    except ValueError as error:
        if not str(error).startswith('Unknown/foreign process on endpoint GPU'):raise
        refreshed={}
        for job in jobs:refreshed.update(owned_processes(job))
        return ep.gpu_monitor(route,refreshed)


def prior_failed_launch():
    records=list(PRIOR_ROOT.glob('control/invocations/*/*/ADMISSION.json'))
    result=read(PRIOR_ROOT/'control/initial_RESULT.json')
    if (len(records)!=1 or result['invocations_started']!=1 or result['status']!='PARTIAL_OR_FAILED'
            or 'Unknown/foreign process' not in result['stop_reason']):raise ValueError('Prior failed launch accounting changed')
    path=records[0];exited=read(path.with_name('EXIT.json'))
    if not exited['owned_cleanup_complete']:raise ValueError('Prior source did not confirm owned cleanup; Lead must inspect')
    if any(PRIOR_ROOT.glob('*/runs/*')):raise ValueError('Prior failed launch has real run state; never assume zero updates/calls')
    prior_exited(PRIOR_ROOT/'control')
    return {'consumed_trainer_invocations':1,'consumed_optimizer_updates':0,'consumed_official_calls':0,
        'wall_seconds_reserved':60,'initial_result':pin(PRIOR_ROOT/'control/initial_RESULT.json'),
        'admission':pin(path),'launch':pin(path.with_name('LAUNCH.json')),'exit':pin(path.with_name('EXIT.json')),
        'prior_IO_is_SID_sampled_lower_bound':True}


def claim_smoke_invocation(control,unit,*,binding_sha,argv):
    if unit not in ('initial_continuous','initial_interrupted','resume_interrupted'):raise ValueError('Unknown SMOKE invocation unit')
    claimed=[p for p in Path(control).glob('invocations/*/*') if p.is_dir()]
    if LIMITS['prior_invocations']+len(claimed)>=LIMITS['max_invocations']:
        raise ValueError('SMOKE total trainer invocation cap including prior failed launch exhausted')
    return helper().claim_invocation(control,unit,resume=False,binding_sha=binding_sha,argv=argv)


def inspect_native(folder,*,identity=None,completed=None):
    """Metadata/stat only; real trainer verifies all bytes before strict load."""
    if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))
    from tools.st_world_policy.training_state import require_native_inventory
    folder=Path(folder);receipt=read(folder/'COMPLETE.json');state=receipt['state']
    if (receipt['schema']!='st_native_checkpoint_v1' or receipt['world_size']!=4
            or digest({k:v for k,v in receipt.items() if k!='receipt_identity'})!=receipt['receipt_identity']
            or state['tag']!=folder.name or (identity is not None and state['identity']!=identity)
            or (completed is not None and state['completed']!=completed)
            or state['pending_feedback'] is not None or state['pending_old_policy'] is not None):
        raise ValueError('Foreign/incomplete SMOKE native boundary')
    require_native_inventory(receipt['files'],4)
    if set(receipt['files'])!={p.name for p in folder.glob('*.pt')}:raise ValueError('Native inventory changed')
    for name,info in receipt['files'].items():
        if Path(name).name!=name or (folder/name).stat().st_size!=info['bytes']:raise ValueError('Native shard size differs')
    return receipt


def selected_native(run,step,*,require_latest=True):
    valid=[];rejected=[]
    identity=read(Path(run)/'identity.json')['identity']
    for path in (Path(run)/'checkpoints').glob('step_*'):
        try:
            receipt=inspect_native(path,identity=identity)
            valid.append((receipt['state']['completed'],path.name,path,receipt))
        except (OSError,ValueError,KeyError) as error:rejected.append({'folder':str(path),'error':repr(error)})
    if not valid:raise ValueError('No complete SMOKE boundary')
    population=valid if require_latest else [row for row in valid if row[0]==step]
    if not population:raise ValueError('Requested complete boundary absent')
    completed,_,folder,receipt=max(population,key=lambda row:row[:2])
    if completed!=step:raise ValueError('Latest complete boundary is not the requested exact step'+str(step))
    return folder,receipt,rejected


def plan():
    formal=pinned({'path':str(FORMAL),'sha256':FORMAL_SHA});cfg=pinned(formal['artifacts']['config'])
    manifest=pinned({'path':str(MANIFEST),'sha256':MANIFEST_SHA})
    if (manifest['data_identity']['split']!='train' or len(manifest['entries'])!=32
            or len({row['scene_id'] for row in manifest['entries']})!=32):raise ValueError('Exact fixed32 TRAIN required')
    run=Path(cfg['runtime']['output_root'])/formal['run_id'];sizes=[]
    for folder in sorted((run/'checkpoints').glob('step_000000_*')):
        receipt=inspect_native(folder,completed=0)
        sizes.append({'folder':str(folder),'receipt':pin(folder/'COMPLETE.json'),
            'bytes':sum(value['bytes'] for value in receipt['files'].values()),'payload_read':False})
    return {'status':'WAITING_FRESH_LEAD_ADMISSION','source_sha':SOURCE,'limits':LIMITS,
        'fixed_train_manifest':pin(MANIFEST),'scenes':32,'formal_registration':pin(FORMAL),
        'actual_step0_native_stats':sizes,'mature_controller':pin(MATURE),
        'BC_noise_time':'NOT_STORED_BY_RUNTIME751','actual_new_updates':0,'actual_official_calls':0}


def draft_binding(output,layout_ref,valid_seconds):
    """Metadata-only on the actual host; no model construction or GPU query."""
    if not 3900<=valid_seconds<=7200:raise ValueError('Explicit3900–7200 second lease window required')
    layout=pinned(layout_ref);host=socket.gethostname()
    if (layout['hostname']!=host or layout['host']!='training-vlawm-tsinghua-zt6'
            or layout['runtime_source']!=SOURCE or layout['global_call_cap_including_retries']!=1024
            or layout['retry_headroom_total']!=0):raise ValueError('Actual host/frozen zero-retry resource layout required')
    output=Path(output)
    if not output.is_absolute() or not output.is_relative_to(CAMPAIGN) or output==ROOT or ROOT in output.parents or output.exists():
        raise ValueError('Fresh separate campaign admission directory required')
    if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))
    from tools.st_world_policy.training_config import environment_record
    formal=pinned({'path':str(FORMAL),'sha256':FORMAL_SHA});current=environment_record()
    if current!=pinned(formal['artifacts']['environment']):raise ValueError('Actual software differs from exact formal runtime')
    ep=helper();policy=read(ep.DISPATCH_POLICY);authorization=read(ep.AUTHORIZATION)
    if host not in policy['allowed_aliases'] or layout['host'] not in authorization['allowed_hosts']:raise ValueError('Host not currently authorized')
    groups=[layout['groups'][name] for name in ('A_continuous4','B_pause2_resume2')]
    if [row['GPU_indices'] for row in groups]!=[list(range(4)),list(range(4,8))]:raise ValueError('Wrong8-card layout')
    uuids=sum([row['GPU_UUIDs'] for row in groups],[])
    if len(set(uuids))!=8 or any(not value.startswith('GPU-') for value in uuids):raise ValueError('Actual distinct UUIDs required')
    output.mkdir(parents=True,exist_ok=False);write(output/'ENVIRONMENT.json',current);now=time.time();leases={}
    for arm,group in zip(('continuous','interrupted'),groups):
        lease={'schema':'st_training_resource_lease_v1','hostnames':[host],'world_size':4,
            'cuda_visible_devices_by_host':{host:','.join(group['GPU_UUIDs'])},'gpu_uuids_by_host':{host:group['GPU_UUIDs']},
            'valid_from_unix':now-1,'valid_until_unix':now+valid_seconds,'max_invocation_seconds':1800,
            'purpose':'Isolated instrumented751 native4 vs2+2 SMOKE; no formal initialization',
            'authorization':str(ep.AUTHORIZATION)}
        path=output/(arm+'_RESOURCE_LEASE.json');write(path,lease);leases[arm]=pin(path)
    binding={'source_sha':SOURCE,'controller_sha256':sha(__file__),'limits':LIMITS,'root':str(ROOT),'manifest':pin(MANIFEST),
        'environment':pin(output/'ENVIRONMENT.json'),'dispatch_policy':pin(ep.DISPATCH_POLICY),'authorization':pin(ep.AUTHORIZATION),
        'resource_layout':layout_ref,'leases':leases,'prior_failed_launch':prior_failed_launch(),
        'ownership_monitor':pin(OWNERSHIP_MONITOR),
        'route':{'host':layout['host'],'hostname':host,'gpus':list(range(8)),'gpu_uuids':uuids,
            'valid_from_unix':now-1,'valid_until_unix':now+valid_seconds}}
    path=output/'BINDING_DRAFT.json';write(path,{'schema':'st_rollout_resume_smoke_binding_draft_v1',
        'status':'WAITING_INDEPENDENT_CORRECTNESS_RESOURCES_REVIEWS','binding':binding,'binding_sha256':digest(binding),
        'dispatch_started':False,'metadata_only_no_model_no_GPU_query':True})
    return {'status':'BINDING_DRAFT_ONLY_NOT_ADMITTED','artifact':pin(path),'binding_sha256':digest(binding)}


def admission(path,checksum):
    if sha(path)!=checksum:raise ValueError('Admission checksum differs')
    receipt=read(path);binding=receipt['binding']
    if (receipt['schema']!='st_rollout_resume_smoke_admission_v1' or receipt['status']!='APPROVED'
            or receipt['binding_sha256']!=digest(binding) or binding['controller_sha256']!=sha(__file__)
            or binding['source_sha']!=SOURCE or binding['limits']!=LIMITS
            or binding['root']!=str(ROOT) or binding['manifest']!=pin(MANIFEST)
            or binding['prior_failed_launch']!=prior_failed_launch()
            or binding['ownership_monitor']!={'path':str(OWNERSHIP_MONITOR),'sha256':OWNERSHIP_MONITOR_SHA}):
        raise ValueError('Foreign/stale SMOKE admission')
    names=[]
    for kind in ('correctness','resources'):
        review=pinned(receipt['reviews'][kind]);name=review['reviewer']
        if (review['status']!='PASS' or review['review_kind']!=kind or review['binding_sha256']!=digest(binding)
                or name in (IMPLEMENTER,receipt['issuer']) or not review['evidence_artifacts']):
            raise ValueError('Independent SMOKE review missing')
        for ref in review['evidence_artifacts']:
            if sha(ref['path'])!=ref['sha256']:raise ValueError('Review evidence changed')
        names.append(name)
    if len(set(names))!=2:raise ValueError('Two distinct independent observer reviewers required')
    return receipt,binding


def route_checks(binding,*,initial):
    if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))
    ep=helper();helpers=ep.mature_helpers()
    if (binding['dispatch_policy']['path']!=str(ep.DISPATCH_POLICY)
            or binding['authorization']['path']!=str(ep.AUTHORIZATION)):raise ValueError('Actual live policy/auth paths required')
    policy=pinned(binding['dispatch_policy']);auth=pinned(binding['authorization']);route=binding['route'];host=socket.gethostname()
    aliases=policy['allowed_hosts'].get('training-vlawm-tsinghua-zt6',{}).get('aliases',[])
    if (host not in aliases or host!=route['hostname'] or route['host']!='training-vlawm-tsinghua-zt6'
            or host not in policy['allowed_aliases'] or route['host'] not in auth['allowed_hosts']):raise ValueError('Wrong allowed host')
    for name,row in policy['global_dispatch_prohibitions'].items():
        if host==name or route['host']==name or host in row.get('aliases',[]):raise ValueError('Host withdrawn')
    helpers['check_route_time_v3'](route,time.time(),LIMITS['wall_seconds'] if initial else 1800)
    if route['gpus']!=list(range(8)) or len(set(route['gpu_uuids']))!=8:raise ValueError('Exact8 GPU route required')
    if not set(range(32)).issubset(os.sched_getaffinity(0)):raise ValueError('CPU0–31 required')
    capacity=helpers['cpu_capacity_v3']()
    if capacity['quota_us']>=0 and capacity['quota_us']/capacity['period_us']<16:raise ValueError('Insufficient CPU quota')
    helpers['fresh_environment_v3'](binding['environment']);helpers['shared_lock_mount_v3']()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True,timeout=15).strip()!=SOURCE or subprocess.check_output(['git','status','--porcelain'],cwd=RUNTIME,timeout=15).strip():raise ValueError('Runtime751 changed')
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    if initial and int(mem['MemAvailable'].split()[0])*1024<LIMITS['fresh_RAM_bytes']:raise ValueError('Fresh RAM insufficient')
    import shutil
    if shutil.disk_usage(CAMPAIGN).free<(LIMITS['launch_shared_free_bytes'] if initial else LIMITS['runtime_shared_free_bytes']):raise ValueError('Shared storage reserve insufficient')
    return route,helpers


def prepare(binding):
    """Mature v2 preparation pattern, current formal recipe and fixed TRAIN32."""
    if ROOT.exists():raise ValueError('Preserve existing SMOKE; preparation is not overwrite/resampling')
    if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))
    from tools.st_world_policy.flow_training_config import validate_config,target_contract
    from tools.st_world_policy.check_formal_approval import source_fingerprint,verify_registration
    formal=pinned({'path':str(FORMAL),'sha256':FORMAL_SHA});base=pinned(formal['artifacts']['config'])
    plan() # Strict legal32 and actual initial native stat before any writes.
    ROOT.mkdir(parents=True,exist_ok=False);write(ROOT/'BINDING.json',binding)
    write(ROOT/'BUDGET.json',{'limits':LIMITS,'weights_usable_for_formal_init':False,'source_sha':SOURCE,
        'audit_worker_sha256':sha(__file__),'original_trainer_modified':False,
        'prior_failed_launch':binding['prior_failed_launch'],'maximum_new_trainer_invocations':3,
        'total_wall_accounting_seconds':60+3540+120+180})
    for arm in ('continuous','interrupted'):
        folder=ROOT/arm;folder.mkdir();cfg=json.loads(json.dumps(base))
        cfg['execution'].update(scope='SMOKE',sample_limit=32,stop_after=4)
        cfg['runtime'].update(output_root=str(folder/'runs'),max_seconds=1800,save_every=2)
        cfg['runtime']['official_execution']['adaptation_reason']='Isolated runtime751 SMOKE native4 versus2+2; same4096/warm100/G4/MC50/Euler10/repeat8/rollout-scale definition. No formal initializer.'
        cfg['feedback'].update(max_calls=512,retry_attempts=0)
        validate_config(cfg)
        write(folder/'CONFIG.json',cfg);write(folder/'TARGET_CONTRACT.json',target_contract(cfg))
        lease=pinned(binding['leases'][arm]);host=binding['route']['hostname']
        expected_uuids=binding['route']['gpu_uuids'][:4] if arm=='continuous' else binding['route']['gpu_uuids'][4:]
        if (lease['world_size']!=4 or lease['hostnames']!=[host]
                or lease['cuda_visible_devices_by_host']!={host:','.join(expected_uuids)}
                or lease['gpu_uuids_by_host']!={host:expected_uuids}):raise ValueError('Wrong actual SMOKE resource lease')
        artifacts={k:dict(v) for k,v in formal['artifacts'].items()}
        artifacts.update(config=pin(folder/'CONFIG.json'),target_contract=pin(folder/'TARGET_CONTRACT.json'),
            data_manifest=pin(MANIFEST),environment=binding['environment'],resource_lease=binding['leases'][arm])
        reg={'schema':'st_world_run_v1','run_id':'C1_rollout_std_native_'+arm+'_v1','source_sha':SOURCE,
            'source_fingerprint':source_fingerprint(RUNTIME),'artifacts':artifacts,'execution':cfg['execution'],
            'implementer_agents':[IMPLEMENTER],'weights_usable_for_formal_init':False}
        write(folder/'REGISTRATION.json',reg);verify_registration(folder/'REGISTRATION.json',repo=RUNTIME)
    return {'status':'PREPARED_NOT_STARTED','root':str(ROOT),'weights_usable_for_formal_init':False}


def run_path(arm):
    reg=read(ROOT/arm/'REGISTRATION.json');cfg=pinned(reg['artifacts']['config'])
    return Path(cfg['runtime']['output_root'])/reg['run_id']


def group_ready(run,job):
    run=Path(run)
    if not (all((run/'official_runtime'/f'initial_engine_rank{rank}.json').is_file() for rank in range(4))
            and (run/'initialization.json').is_file() and (run/'status.json').is_file()):return False
    identity=read(run/'identity.json');status=read(run/'status.json')
    if (identity['source_sha']!=SOURCE or digest({k:v for k,v in identity.items() if k!='identity'})!=identity['identity']
            or status['identity']!=identity['identity'] or status['source_sha']!=SOURCE
            or status['started_unix']<job['started_unix']
            or status['pid'] not in owned_processes(job)):
        raise ValueError('Heavy-load gate is not from the actual owned invocation')
    for path in (run/'checkpoints').glob('step_000000_*'):
        if (path/'COMPLETE.json').exists():
            inspect_native(path,identity=identity['identity'],completed=0);return True
    return False


def validate_result(arm,step,updates):
    run=run_path(arm);result=read(run/'RESULT.json');folder,receipt,rejected=selected_native(run,step)
    if (result['completed']!=step or result['new_optimizer_updates']!=updates or result['status']!='PAUSED'
            or not result['frozen_encoder_unchanged'] or result['identity']!=receipt['state']['identity']):
        raise ValueError('Actual stopped SMOKE progress differs; no retry/relabel')
    if step==2 and read(run/'PAUSE_INTENT.json')['reason']!='OPERATOR_PAUSED':raise ValueError('Not the requested operator boundary2')
    snapshot=ROOT/'control'/f'{arm}_step{step}_RESULT_SNAPSHOT.json';write(snapshot,result)
    return {'run':str(run),'result_snapshot':pin(snapshot),'native':str(folder),'receipt':pin(folder/'COMPLETE.json'),'rejected':rejected}


def prior_exited(control):
    ep=helper()
    for path in Path(control).glob('invocations/*/*/ADMISSION.json'):
        exit_path=path.with_name('EXIT.json')
        if not exit_path.is_file():raise ValueError('Previous SMOKE invocation has unknown exit')
        record=read(exit_path)
        if not record['owned_cleanup_complete']:raise ValueError('Previous owned cleanup unconfirmed')
        for raw_pid,stamp in record.get('owned_IDs',{}).items():
            row=ep.process_tree(int(raw_pid)).get(int(raw_pid))
            if row and row['state']!='Z' and row['start_ticks']==stamp:raise ValueError('Owned SMOKE process still live')


def validate_pause_marker(run,folder):
    record=read(ROOT/'PAUSE_REQUEST.json');marker=Path(run)/'STOP_REQUESTED'
    if (Path(run)!=run_path('interrupted') or record['run']!=str(run) or record['completed']!=2
            or record['source_sha']!=SOURCE or record['observer_sha256']!=sha(__file__)
            or record['binding_sha256']!=digest(read(ROOT/'BINDING.json'))
            or record['native_directory']!=str(folder) or record['native_tag']!=Path(folder).name
            or record['marker']!=pin(marker) or marker.read_text()!=PAUSE_MARKER):
        raise ValueError('STOP marker is not this controller exact native2 request; no blind acknowledgement')
    return pin(ROOT/'PAUSE_REQUEST.json')


def archive_owned_pause(run,folder):
    pause=validate_pause_marker(run,folder);marker=Path(run)/'STOP_REQUESTED'
    archive=ROOT/'control/OWNED_STEP2_STOP_REQUESTED.archived'
    if archive.exists():raise ValueError('Own pause already archived; no automatic additional resume')
    marker.rename(archive)
    expected=read(ROOT/'PAUSE_REQUEST.json')['marker']['sha256']
    if archive.read_text()!=PAUSE_MARKER or sha(archive)!=expected:
        write(ROOT/'control/OWNED_PAUSE_ACK_REJECTED.json',{'status':'REJECTED_CHANGED_MARKER','archive':pin(archive),
            'new_STOP_preserved':marker.exists(),'no_launch':True})
        # A replacement moved by the atomic rename is retained, and restored
        # without overwriting any still newer operator STOP at the source path.
        if not marker.exists():
            try:os.link(archive,marker)
            except FileExistsError:pass
        raise ValueError('STOP changed during archive; evidence retained, no launch')
    write(ROOT/'control/OWNED_PAUSE_ACK.json',{'schema':'st_owned_native2_pause_ack_v1','source_sha':SOURCE,
        'observer_sha256':sha(__file__),'binding_sha256':digest(read(ROOT/'BINDING.json')),
        'run':str(run),'completed':2,'native_directory':str(folder),'native_receipt':pin(Path(folder)/'COMPLETE.json'),
        'pause_request':pause,'archived_marker':pin(archive),'trainer_acknowledge_stop':False})


def validate_pause_ack(run,folder):
    ack=read(ROOT/'control/OWNED_PAUSE_ACK.json');archive=ROOT/'control/OWNED_STEP2_STOP_REQUESTED.archived'
    if (ack['schema']!='st_owned_native2_pause_ack_v1' or ack['source_sha']!=SOURCE
            or ack['observer_sha256']!=sha(__file__) or ack['binding_sha256']!=digest(read(ROOT/'BINDING.json'))
            or ack['run']!=str(run) or ack['completed']!=2 or ack['native_directory']!=str(folder)
            or ack['native_receipt']!=pin(Path(folder)/'COMPLETE.json')
            or ack['pause_request']!=pin(ROOT/'PAUSE_REQUEST.json') or ack['archived_marker']!=pin(archive)
            or archive.read_text()!=PAUSE_MARKER or ack['trainer_acknowledge_stop'] is not False):
        raise ValueError('Missing/foreign exact owned pause acknowledgement')
    if (Path(run)/'STOP_REQUESTED').exists():raise ValueError('New operator STOP preserved; never blindly acknowledge')
    return ack


def execute(phase,receipt,binding):
    ep=helper();route,helpers=route_checks(binding,initial=phase=='initial')
    if read(ROOT/'BINDING.json')!=binding:raise ValueError('Prepared SMOKE binding changed')
    control=ROOT/'control';control.mkdir(exist_ok=True);locks=[];jobs=[];stop=[]
    start_path=control/'START.json'
    if phase=='initial':write(start_path,{'started_unix':time.time(),'deadline_unix':time.time()+LIMITS['wall_seconds'],'binding_sha256':digest(binding)})
    start=read(start_path);deadline=start['deadline_unix']
    old_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(signum,frame):
        if not stop:stop.append(signum)
    def check_stop():
        if stop:raise RuntimeError('OPERATOR_SIGNAL_'+str(stop[0]))
        if time.time()>=deadline:raise RuntimeError('SMOKE_GLOBAL_WALL_BOUND')
    def launch(arm,resume):
        check_stop();unit='resume_interrupted' if resume else 'initial_'+arm
        ids=list(range(4)) if arm=='continuous' else list(range(4,8));uuids=[route['gpu_uuids'][i] for i in ids]
        left=max(1,int(deadline-time.time()));cpus='0-15' if arm=='continuous' else '16-31'
        argv=['timeout','--signal=TERM','--kill-after=60s',str(left)+'s','taskset','-c',cpus,PYTHON,'-m','torch.distributed.run',
            '--standalone','--nnodes=1','--nproc_per_node=4',str(Path(__file__).resolve()),'worker',
            '--registration',str(ROOT/arm/'REGISTRATION.json'),'--audit-output',str(ROOT/arm/'observer')]
        if resume:argv+=['--resume']
        attempt=claim_smoke_invocation(control,unit,binding_sha=digest(binding),argv=argv)
        for gpu,uuid in zip(ids,uuids):helpers['fresh_gpu_v3']({'gpu':gpu,'gpu_uuid':uuid})
        check_stop()
        env={**os.environ,**ep.THREAD_ENV,'CUDA_VISIBLE_DEVICES':','.join(uuids)}
        log=None;launched_at=time.time()
        try:
            log=(attempt/'OUTPUT.log').open('xb')
            child=subprocess.Popen(argv,cwd=RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,pass_fds=tuple(handle.fileno() for handle in locks))
            job={'child':child,'attempt':attempt,'unit':unit,'arm':arm,'owned':{},'start_ticks':None,'started_unix':launched_at}
            jobs.append(job) # No raising signal handler or post-Popen gap.
        except OSError as error:
            write(attempt/'EXIT.json',{'status':'KNOWN_NOT_STARTED','error':repr(error),'owned_cleanup_complete':True})
            raise
        finally:
            if log is not None:log.close()
        tree=owned_processes(job)
        write(attempt/'LAUNCH.json',{'pid':child.pid,'argv':argv,'cwd':str(RUNTIME),'source_sha':SOURCE,
            'audit_worker_sha256':sha(__file__),'admission_sha256':digest(receipt),'started_unix':time.time()})
        check_stop()
    prior_stats=read(PRIOR_ROOT/'control/initial_RESULT.json')['stats'] if phase=='initial' else read(control/'initial_RESULT.json')['stats']
    prior_rchar=prior_stats.get('sampled_rchar',0);prior_read=prior_stats.get('sampled_read_bytes',0)
    reason=None;stats={'peak_aggregate_RSS':0,'sampled_rchar':prior_rchar,'sampled_read_bytes':prior_read,
        'IO_scope':'Cumulative sampled /proc rchar and read_bytes; lower bounds for mmap/short-lived/unwaited descendants, not an IO quota'}
    last_IO=last_GPU=0.;seen={};retired={'rchar':0,'read_bytes':0};requested=False
    try:
        for path in [control/'RUN.lock']+[Path('/tmp/foresight_gpu_'+str(i)+'.lock') for i in range(8)]:
            handle=path.open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        for sig in old_handlers:signal.signal(sig,interrupted)
        if phase=='resume':
            if read(control/'initial_RESULT.json')['status']!='EXECUTION_COMPLETE_PENDING_COMPARISON':raise ValueError('Initial group execution incomplete')
            prior_exited(control);paused=validate_result('interrupted',2,2)
            archive_owned_pause(run_path('interrupted'),Path(paused['native']));launch('interrupted',True)
        else:launch('continuous',False)
        while True:
            check_stop()
            if phase=='initial' and len(jobs)==1 and group_ready(run_path('continuous'),jobs[0]):launch('interrupted',False)
            if phase=='initial' and len(jobs)==2 and (ROOT/'PAUSE_REQUEST.json').exists():
                pause=read(ROOT/'PAUSE_REQUEST.json')
                if pause['run']!=str(run_path('interrupted')) or pause['completed']!=2:raise ValueError('Foreign pause request')
                requested=True
            aggregate=0;currently_owned={}
            for job in jobs:
                tree=owned_processes(job);currently_owned.update(tree)
                for pid,row in tree.items():
                    old=seen.get(pid,{})
                    if old.get('start_ticks')==row['start_ticks']:
                        for name in ('rchar','read_bytes'):
                            if name in old:row[name]=old[name]
                    elif old:
                        for name in retired:retired[name]+=old.get(name,0)
                    seen[pid]=row
                rss=sum(row['rss'] for row in tree.values());aggregate+=rss
                if rss>LIMITS['group_RSS_bytes']:raise ValueError('SMOKE group RSS cap')
                if job['child'].poll() is not None:
                    if job['child'].returncode!=0:raise ValueError('SMOKE child failed; no automatic retries')
                    if any(row['state']!='Z' for row in tree.values()):raise ValueError('Owned descendants survived launcher')
            stats['peak_aggregate_RSS']=max(stats['peak_aggregate_RSS'],aggregate)
            if aggregate>LIMITS['aggregate_RSS_bytes']:raise ValueError('SMOKE aggregate RSS cap')
            if time.monotonic()-last_GPU>=5:
                ep.LIMITS['GPU_driver_bytes']=LIMITS['GPU_driver_bytes'];monitor_smoke_gpu(ep,route,jobs,currently_owned);last_GPU=time.monotonic()
            if time.monotonic()-last_IO>=10:
                files=[p for p in ROOT.rglob('*') if p.is_file()]
                if sum(p.stat().st_size for p in files)>LIMITS['output_bytes']:raise ValueError('SMOKE output cap')
                native_dirs=[p for arm in ('continuous','interrupted') for p in (run_path(arm)/'checkpoints').glob('step_*')]
                if len(native_dirs)>8:raise ValueError('SMOKE native directory cap includes partial')
                if any(sum(p.stat().st_size for p in folder.glob('*') if p.is_file())>LIMITS['native_directory_bytes'] for folder in native_dirs):raise ValueError('SMOKE single native cap')
                import shutil
                if shutil.disk_usage(CAMPAIGN).free<LIMITS['runtime_shared_free_bytes']:raise ValueError('Protect shared remaining commitments')
                calls=sum(read(p)['calls'] for p in ROOT.glob('*/runs/*/official_feedback/*/attempt_*/ADMITTED.json'))
                if calls>1024:raise ValueError('SMOKE campaign official-call cap')
                for pid,row in seen.items():
                    try:
                        if ep.process_tree(pid).get(pid,{}).get('start_ticks')!=row['start_ticks']:continue
                        io=dict(line.split(':',1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())
                        row['rchar']=int(io['rchar']);row['read_bytes']=int(io['read_bytes'])
                    except (FileNotFoundError,ProcessLookupError):pass
                stats['sampled_rchar']=prior_rchar+retired['rchar']+sum(row.get('rchar',0) for row in seen.values())
                stats['sampled_read_bytes']=prior_read+retired['read_bytes']+sum(row.get('read_bytes',0) for row in seen.values())
                if stats['sampled_rchar']>LIMITS['monitor_read_bytes']:raise ValueError('SMOKE cumulative sampled logical-rchar guard')
                last_IO=time.monotonic()
            if all(job['child'].poll() is not None for job in jobs) and (phase=='resume' or len(jobs)==2):break
            time.sleep(.1)
        if phase=='initial':
            if not requested:raise ValueError('Never observed owned step2 save; no fabricated pause')
            results={'continuous':validate_result('continuous',4,4),'interrupted':validate_result('interrupted',2,2)}
        else:results={'interrupted':validate_result('interrupted',4,2)}
    except BaseException as error:reason=repr(error);results={}
    finally:
        try:stop_smoke_owned(jobs)
        except BaseException as error:reason='CLEANUP_UNCONFIRMED: '+repr(error)
        if any(not job.get('cleanup_complete',False) for job in jobs):reason=reason or 'OWNED_CLEANUP_UNCONFIRMED'
        for job in jobs:
            write(job['attempt']/'EXIT.json',{'returncode':job['child'].returncode,'owned_cleanup_complete':job.get('cleanup_complete',False),
                'owned_IDs':{str(pid):row['start_ticks'] for pid,row in job['owned'].items()},'stop_reason':reason})
        for sig,value in old_handlers.items():signal.signal(sig,value)
        for handle in reversed(locks):handle.close()
    result={'status':'EXECUTION_COMPLETE_PENDING_COMPARISON' if reason is None else 'PARTIAL_OR_FAILED',
        'phase':phase,'stop_reason':reason,'stats':stats,'results':results,'invocations_started':len(jobs),
        'weights_usable_for_formal_init':False,'new_formal_updates':0,
        'prior_consumed_trainer_invocations':1,'total_trainer_invocation_cap':4}
    write(control/(phase+'_RESULT.json'),result);return result


def logical_state(value):
    """Full logical values, bounded tensor copies, no lossy dtype conversion."""
    import torch
    import numpy as np
    if torch.is_tensor(value):
        h=hashlib.sha256();count=0
        def chunks(tensor):
            if tensor.numel()<=1<<20:yield tensor.detach().contiguous().reshape(-1).cpu()
            elif tensor.is_contiguous():
                flat=tensor.detach().view(-1)
                for at in range(0,flat.numel(),1<<20):yield flat[at:at+(1<<20)].cpu()
            else:
                axis=next(i for i,size in enumerate(tensor.shape) if size>1);half=tensor.shape[axis]//2
                yield from chunks(tensor.narrow(axis,0,half));yield from chunks(tensor.narrow(axis,half,tensor.shape[axis]-half))
        for chunk in chunks(value):
            if chunk.is_floating_point() and not bool(chunk.isfinite().all()):raise ValueError('Nonfinite logical audit tensor')
            raw=chunk.reshape(-1).view(torch.uint8).numpy().tobytes();h.update(raw);count+=len(raw)
        return {'kind':'tensor','dtype':str(value.dtype),'shape':list(value.shape),'bytes':count,'sha256':h.hexdigest()}
    if isinstance(value,np.ndarray):return {'kind':'ndarray','dtype':str(value.dtype),'shape':list(value.shape),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value,dict):return {'kind':'mapping','items':[[str(type(key)),str(key),logical_state(item)] for key,item in value.items()]}
    if isinstance(value,(list,tuple)):return {'kind':'sequence','type':type(value).__name__,'items':[logical_state(item) for item in value]}
    if value is None or isinstance(value,(str,int,float,bool)):return {'kind':'scalar','type':type(value).__name__,'value':value}
    if type(value).__module__=='deepspeed.runtime.fp16.loss_scaler':return {'kind':'loss_scaler','type':type(value).__name__,'state':logical_state(vars(value))}
    raise ValueError('Unknown logical state type, not silently omitted: '+str(type(value)))


def engine_snapshot(engine,client):
    optimizer=engine.optimizer.state_dict()
    required={'single_partition_of_fp32_groups','base_optimizer_state','partition_count','group_paddings','zero_stage'}
    if not required.issubset(optimizer):raise ValueError('Native public ZeRO2 state inventory differs')
    return {'module':logical_state(engine.module.state_dict()),'zero_optimizer':logical_state(optimizer),
        'engine':logical_state({key:getattr(engine,key) for key in ('global_steps','global_samples','skipped_steps','dp_world_size','mp_world_size')}),
        'lr_scheduler':logical_state(None if engine.lr_scheduler is None else engine.lr_scheduler.state_dict()),
        'client_state':client}


def install_observer(trainer,engine_type,root,*,rank,resume):
    """Only native save/load and capture/restore are wrapped; no model hooks."""
    original={name:getattr(trainer,name) for name in ('capture_rng','restore_rng')}
    original.update(save=engine_type.save_checkpoint,load=engine_type.load_checkpoint)
    state={'captured':None,'loaded':None,'loads':0,'restore_verified':False,'save_calls':0,'audit_seconds':{}}
    root=Path(root)
    def voted(function,label):
        tick=time.perf_counter()
        try:return trainer.collective_call(function,'external native observer: '+label)
        finally:state['audit_seconds'][label]=state['audit_seconds'].get(label,0.)+time.perf_counter()-tick
    def persist(path,value):
        write(path,{'observer_sha256':sha(__file__),'runtime_sha':SOURCE,'rank':rank,**value})
    def observed_capture(generators):
        def audit():
            value=original['capture_rng'](generators)
            before=logical_state(value);after=logical_state(original['capture_rng'](generators))
            if before!=after:raise ValueError('Capture observer changed RNG')
            state['captured']=before;state['generators']=generators
            return value
        return voted(audit,'capture RNG identity')
    def observed_save(engine,*args,**kwargs):
        import inspect
        def preflight():
            bound=inspect.signature(original['save']).bind(engine,*args,**kwargs);bound.apply_defaults()
            values=bound.arguments
            return values,json.loads(json.dumps(values['client_state']))
        values,client=voted(preflight,'save signature and deep client state')
        tag=values['tag']
        def audit():
            if state['captured'] is None:raise ValueError('Save lacks original consumed RNG capture')
            snapshot=engine_snapshot(engine,client)
            if logical_state(original['capture_rng'](state['generators']))!=state['captured']:raise ValueError('Save tensor audit changed RNG')
            persist(root/f'SAVE_{tag}_rank{rank}.json',{'kind':'SAVE_BEFORE_NATIVE','snapshot':snapshot,'rng':state['captured']})
        voted(audit,'full native logical save state')
        result=original['save'](engine,*args,**kwargs);state['save_calls']+=1
        def pause_at_committed_save():
            if not resume and rank==0 and root==ROOT/'interrupted/observer' and client['completed']==2:
                run=Path(values['save_dir']).parent
                if run!=run_path('interrupted'):raise ValueError('Pause outside own SMOKE forbidden')
                with (run/'STOP_REQUESTED').open('x') as stream:stream.write(PAUSE_MARKER)
                write(ROOT/'PAUSE_REQUEST.json',{'run':str(run),'completed':2,'unix':time.time(),
                    'native_directory':str(Path(values['save_dir'])/tag),
                    'native_tag':tag,'source_sha':SOURCE,'observer_sha256':sha(__file__),
                    'binding_sha256':digest(read(ROOT/'BINDING.json')),'marker':pin(run/'STOP_REQUESTED'),
                    'COMPLETE_seen':(Path(values['save_dir'])/tag/'COMPLETE.json').exists(),
                    'completion_not_inferred_from_directory':True,'control_operation':'OWNED_STOP_REQUESTED_AFTER_ENGINE_SAVE',
                    'signal_sent':False})
        voted(pause_at_committed_save,'owned SMOKE step2 stop control')
        return result
    def observed_load(engine,*args,**kwargs):
        import inspect
        def preflight():
            bound=inspect.signature(original['load']).bind(engine,*args,**kwargs);bound.apply_defaults();values=bound.arguments
            if (not values['load_module_strict'] or not values['load_optimizer_states'] or not values['load_lr_scheduler_states']
                    or values['load_module_only'] or values['custom_load_fn'] is not None):raise ValueError('Non-strict native observation forbidden')
            return values
        values=voted(preflight,'strict load signature')
        result=original['load'](engine,*args,**kwargs)
        def audit():
            folder=Path(values['load_dir'])/values['tag'];receipt=read(folder/'COMPLETE.json');client=receipt['state']
            if state['loads'] or client['completed']!=2 or not result[0] or any(result[1].get(k)!=v for k,v in client.items()):raise ValueError('Expected exactly one complete native2 strict load')
            saved=read(root/f"SAVE_{values['tag']}_rank{rank}.json")
            actual=engine_snapshot(engine,client)
            if actual!=saved['snapshot']:raise ValueError('Live module/master/Adam/group/client differs from pre-save logical state')
            persist(root/f'LOAD_step2_rank{rank}.json',{'kind':'STRICT_LIVE_NATIVE_MATCH','native_receipt':pin(folder/'COMPLETE.json'),
                'save_audit':pin(root/f"SAVE_{values['tag']}_rank{rank}.json"),'snapshot_sha256':digest(actual),
                'native_client_state':client,'scheduler_object':engine.lr_scheduler is not None})
            state.update(loaded=saved,loads=1)
        voted(audit,'full actual native loaded values')
        return result
    def observed_restore(value,generators):
        def audit():
            result=original['restore_rng'](value,generators)
            supplied=logical_state(value);actual=logical_state(original['capture_rng'](generators))
            after=logical_state(original['capture_rng'](generators))
            if actual!=supplied or actual!=after:raise ValueError('Actual restored RNG mismatch or observer altered RNG')
            if resume and state['loaded'] is not None and not state['restore_verified']:
                if actual!=state['loaded']['rng']:raise ValueError('Resume RNG differs from consumed SAVE2 state')
                persist(root/f'RESTORE_step2_rank{rank}.json',{'kind':'EXACT_ACTUAL_RESTORED_RNG','rng':actual,
                    'normalization_verified_before_restore_by_source751':True,'BC_noise_time':'NOT_STORED_BY_RUNTIME751'})
                state['restore_verified']=True
            return result
        return voted(audit,'actual post-restore RNG identity')
    trainer.capture_rng=observed_capture;trainer.restore_rng=observed_restore
    engine_type.save_checkpoint=observed_save;engine_type.load_checkpoint=observed_load
    def uninstall():
        trainer.capture_rng=original['capture_rng'];trainer.restore_rng=original['restore_rng']
        engine_type.save_checkpoint=original['save'];engine_type.load_checkpoint=original['load']
    return state,uninstall


def worker(args):
    """Instrumented SMOKE identity; original751 trainer.run remains the trainer."""
    if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))
    import importlib
    from deepspeed.runtime.engine import DeepSpeedEngine
    regpath=Path(args.registration).resolve();reg=read(regpath);cfg=pinned(reg['artifacts']['config'])
    allowed=[(ROOT/arm/'REGISTRATION.json').resolve() for arm in ('continuous','interrupted')]
    if (regpath not in allowed or reg['source_sha']!=SOURCE or reg['weights_usable_for_formal_init'] is not False
            or cfg['execution']['scope']!='SMOKE' or cfg['execution']['stop_after']!=4 or cfg['execution']['max_updates']!=4096
            or cfg['optimizer']['warmup']!=100 or cfg['fpo']['advantage']!='scene_center_rollout_std_v1'):
        raise ValueError('Observer is only this isolated native SMOKE')
    if read(ROOT/'BINDING.json')['controller_sha256']!=sha(__file__):raise ValueError('Unreviewed observer execution identity')
    if args.resume:
        selected,_,_=selected_native(run_path('interrupted'),2)
        validate_pause_ack(run_path('interrupted'),selected)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True,timeout=15).strip()!=SOURCE:raise ValueError('Wrong trainer source')
    trainer=importlib.import_module('tools.st_world_policy.train_flow_policy')
    if not Path(trainer.__file__).resolve().is_relative_to(RUNTIME):raise ValueError('Foreign trainer imported')
    rank=int(os.environ['RANK']);root=Path(args.audit_output).resolve()
    if root!=regpath.parent/'observer':raise ValueError('Audit output outside owned experiment')
    state,uninstall=install_observer(trainer,DeepSpeedEngine,root,rank=rank,resume=args.resume)
    try:
        result=trainer.run(SimpleNamespace(registration=str(regpath),approval=None,resume=args.resume,acknowledge_stop=False))
        if args.resume and not (state['loads']==1 and state['restore_verified']):raise ValueError('Native load audit did not execute')
        expected=1 if args.resume else (3 if regpath.parent.name=='continuous' else 2)
        if state['save_calls']!=expected:raise ValueError('Unexpected native save count; no extra recovery snapshots allowed')
        write(root/f"{'RESUMED' if args.resume else 'INITIAL'}_OBSERVER_COMPLETE_rank{rank}.json",{
            'runtime_sha':SOURCE,'observer_sha256':sha(__file__),'registration':pin(regpath),'rank':rank,
            'completed':result['completed'],'save_calls':state['save_calls'],'load_calls':state['loads'],
            'actual_resume_RNG_verified':state['restore_verified'],'audit_seconds':state['audit_seconds'],
            'scheduler_object':'none_original_completed_based_schedule','BC_noise_time':'NOT_STORED_BY_RUNTIME751',
            'audit_cost_is_not_training_throughput':True})
        return result
    finally:uninstall()


def consumed_receipt(run,step,rank):
    native,receipt,_=selected_native(run,step,require_latest=False)
    path=Path(run)/'rollouts'/f'boundary_{step-1:06d}'/f'rank_{rank}_COMPLETE.json'
    record=read(path);normal=record['identity']['advantage_normalization']
    if (normal['common_sha256']!=receipt['state']['last_advantage_scale_common_sha256']
            or digest(normal['common'])!=normal['common_sha256'] or normal['local']['rank']!=rank):
        raise ValueError('Last consumed normalizer mismatch')
    return pin(path)


def verify_load_witness(root,run,rank):
    root=Path(root);folder,_,_=selected_native(run,2,require_latest=False)
    load_path=root/f'LOAD_step2_rank{rank}.json';restore_path=root/f'RESTORE_step2_rank{rank}.json'
    loaded=read(load_path);restore=read(restore_path);save_path=root/f'SAVE_{folder.name}_rank{rank}.json'
    for row in (loaded,restore):
        if row['observer_sha256']!=sha(__file__) or row['runtime_sha']!=SOURCE or row['rank']!=rank:
            raise ValueError('Foreign observer load/RNG witness')
    if (loaded['kind']!='STRICT_LIVE_NATIVE_MATCH' or restore['kind']!='EXACT_ACTUAL_RESTORED_RNG'
            or loaded['native_receipt']!=pin(folder/'COMPLETE.json') or loaded['save_audit']!=pin(save_path)):
        raise ValueError('Actual native/load/save identity mismatch')
    saved=pinned(loaded['save_audit'])
    if (saved['observer_sha256']!=sha(__file__) or saved['runtime_sha']!=SOURCE or saved['rank']!=rank
            or digest(saved['snapshot'])!=loaded['snapshot_sha256'] or saved['rng']!=restore['rng']):
        raise ValueError('Saved logical state or actual restored RNG differs')
    return {'load':pin(load_path),'restore':pin(restore_path),'consumed_normalization':consumed_receipt(run,2,rank)}


def compare():
    """Audit trees plus small original rollout tensors, never GB native rereads."""
    usage=resource.getrusage(resource.RUSAGE_SELF);cpu_start=usage.ru_utime+usage.ru_stime
    def guard():
        usage=resource.getrusage(resource.RUSAGE_SELF)
        if usage.ru_utime+usage.ru_stime-cpu_start>COMPARE_LIMITS['CPU_seconds'] or usage.ru_maxrss*1024>COMPARE_LIMITS['RSS_bytes']:
            raise ValueError('Comparison CPU/RSS budget')
    inputs=[p for arm in ('continuous','interrupted') for pattern in ('observer/*.json','runs/*/rollouts/*/rank_*.pt','runs/*/rollouts/*/rank_*_COMPLETE.json') for p in (ROOT/arm).glob(pattern)]
    if sum(path.stat().st_size for path in inputs)>COMPARE_LIMITS['input_bytes']:raise ValueError('Comparison audit-input byte budget')
    if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))
    import torch
    from tools.foresight.compare_resume import differences
    torch.set_num_threads(2);report={'status':'COMPARED','BC_noise_time':'NOT_STORED_BY_RUNTIME751',
        'scheduler':'No scheduler object: original learning_rate(completed,4096,warm100) recomputation',
        'source_sha':SOURCE,'observer_sha256':sha(__file__),'pairs':{},'rollouts':{},'strict_load':{}}
    for rank in range(4):
        guard()
        report['strict_load'][str(rank)]=verify_load_witness(ROOT/'interrupted/observer',run_path('interrupted'),rank)
    for step in (2,4):
        exact=True;rows=[]
        for rank in range(4):
            guard()
            records=[]
            for arm in ('continuous','interrupted'):
                folder,_,_=selected_native(run_path(arm),step,require_latest=False)
                records.append(read(ROOT/arm/'observer'/f'SAVE_{folder.name}_rank{rank}.json'))
            compared=[{key:row['snapshot'][key] for key in ('module','zero_optimizer','engine','lr_scheduler')}|{'rng':row['rng']} for row in records]
            diff=differences(*compared);exact=exact and not diff
            cursors=[{key:row['snapshot']['client_state'][key] for key in ('completed','epoch','offset','exposure','policy_version','counts','rng_stream_base_seed')} for row in records]
            if cursors[0]!=cursors[1]:raise ValueError('Different consumed sampler state')
            rows.append({'rank':rank,'bitwise':not diff,'differences':diff,'cursor':cursors[0]})
        report['pairs'][str(step)]={'bitwise':exact,'ranks':rows}
    # Compare raw numerical values/IDs, never cross-run rollout/comparison IDs.
    for boundary in range(4):
        rows=[]
        for rank in range(4):
            guard()
            values=[]
            for arm in ('continuous','interrupted'):
                folder=run_path(arm)/'rollouts'/f'boundary_{boundary:06d}';receipt=read(folder/f'rank_{rank}_COMPLETE.json')
                path=Path(receipt['payload']['path'])
                if path!=folder/f'rank_{rank}.pt' or path.stat().st_size>2**20 or sha(path)!=receipt['payload']['sha256']:raise ValueError('Unexpected/changed small rollout audit payload')
                payload=torch.load(path,map_location='cpu',weights_only=False)
                if payload['identity']!=receipt['identity']:raise ValueError('Rollout payload identity differs within its own run')
                values.append({'scene_ids':payload['rollout']['scene_ids'],'initial_noise':payload['rollout']['initial_noise'],
                    'raw_actions':payload['rollout']['raw_actions'],'samples':payload['comparison']['samples'],
                    'old_cfm':payload['comparison']['old_cfm'],'rewards':payload['feedback']['rewards'],
                    'valid':payload['feedback']['valid'],'advantages':payload['normalized_advantages']})
            diff=differences(*values);rows.append({'rank':rank,'exact':not diff,'differences':diff})
        report['rollouts'][str(boundary)]={'exact':all(row['exact'] for row in rows),'ranks':rows}
    prefix=report['pairs']['2']['bitwise'] and all(report['rollouts'][str(i)]['exact'] for i in (0,1))
    suffix=report['pairs']['4']['bitwise'] and all(report['rollouts'][str(i)]['exact'] for i in (2,3))
    report['prefix_A_A_bitwise']=prefix;report['resumed_suffix_bitwise']=suffix
    report['verdict']='EXACT_REPLAY' if prefix and suffix else ('REPLAY_DIVERGENCE' if prefix else 'A_A_NONBITWISE_INCONCLUSIVE_NO_POSTHOC_TOLERANCE')
    report['strict_native_load_and_RNG_match']=True
    steps={}
    for arm in ('continuous','interrupted'):
        path=run_path(arm)/'steps.jsonl'
        if path.stat().st_size>2**20:raise ValueError('Unexpected step audit size')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if [row['step'] for row in rows]!=[1,2,3,4]:raise ValueError('Missing/duplicated actual optimizer update')
        steps[arm]=rows
    from tools.foresight.student_state import learning_rate
    cfg=read(ROOT/'continuous/CONFIG.json');opt=cfg['optimizer'];step_rows=[]
    for step in range(1,5):
        left,right=steps['continuous'][step-1],steps['interrupted'][step-1]
        expected_lr=learning_rate(step-1,opt['lr'],opt['minimum_lr'],100,4096)
        ids=[[rank['sample_ids'] for rank in row['per_rank']] for row in (left,right)]
        if ids[0]!=ids[1] or left['lr']!=expected_lr or right['lr']!=expected_lr:raise ValueError('Actual sample sequence/original scheduler differs')
        step_rows.append({'step':step,'sample_ids':ids[0],'lr':expected_lr,'losses_continuous':left['losses'],
            'losses_resumed':right['losses'],'losses_exact':left['losses']==right['losses'],
            'original_sampled_master_update_evidence':{arm:{key:steps[arm][step-1][key] for key in
                ('sampled_changed_elements','maximum_sampled_change','fp32_master_update_verified')
                if key in steps[arm][step-1]} for arm in ('continuous','interrupted')}})
    report['actual_next_step3_4_and_losses']=step_rows
    prefix=prefix and all(row['losses_exact'] for row in step_rows[:2])
    suffix=suffix and all(row['losses_exact'] for row in step_rows[2:])
    report['prefix_A_A_bitwise']=prefix;report['resumed_suffix_bitwise']=suffix
    report['verdict']='EXACT_REPLAY' if prefix and suffix else ('REPLAY_DIVERGENCE' if prefix else 'A_A_NONBITWISE_INCONCLUSIVE_NO_POSTHOC_TOLERANCE')
    report['update_numerical_boundary']='Full logical states are byte-hashed. Original sampled update evidence is referenced. Full FP64 master-update delta/cosine and cross-fresh-run nonbitwise master envelope are NOT_COMPUTED; no post-hoc tolerance is inferred.'
    guard();report['comparison_limits']=COMPARE_LIMITS
    if len(json.dumps(report,allow_nan=False).encode())>COMPARE_LIMITS['output_bytes']:raise ValueError('Comparison output budget')
    write(ROOT/'COMPARISON.json',report);return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    subs.add_parser('status');subs.add_parser('plan');subs.add_parser('compare')
    sub=subs.add_parser('draft-binding');sub.add_argument('--output',type=Path,required=True);sub.add_argument('--resource-layout',type=Path,required=True);sub.add_argument('--resource-layout-sha256',required=True);sub.add_argument('--valid-for-seconds',type=int,default=7200)
    for name in ('prepare','initial','resume'):
        sub=subs.add_parser(name);sub.add_argument('--admission',type=Path,required=True);sub.add_argument('--admission-sha256',required=True)
    sub=subs.add_parser('worker');sub.add_argument('--registration',required=True);sub.add_argument('--audit-output',required=True);sub.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='status':result={'status':'WAITING_FRESH_LEAD_ADMISSION','source_sha':SOURCE,'limits':LIMITS,'dispatch_started':False}
        elif args.command=='plan':result=plan()
        elif args.command=='draft-binding':result=draft_binding(args.output,{'path':str(args.resource_layout),'sha256':args.resource_layout_sha256},args.valid_for_seconds)
        elif args.command=='worker':worker(args);return 0
        elif args.command=='compare':result=compare()
        else:
            receipt,binding=admission(args.admission,args.admission_sha256)
            if args.command=='prepare':result=prepare(binding)
            else:result=execute(args.command,receipt,binding)
        print(json.dumps(result));return 2 if result.get('status')=='PARTIAL_OR_FAILED' or result.get('verdict')=='REPLAY_DIVERGENCE' else 0
    except (ValueError,KeyError,OSError,RuntimeError,subprocess.SubprocessError) as error:
        print(json.dumps({'status':'BLOCKED','error':repr(error)}));return 2


if __name__=='__main__':raise SystemExit(main())
