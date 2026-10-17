#!/usr/bin/env python3
"""One independently admitted dev/train metadata index; no automatic retry."""
with open('/proc/self/io') as _io:
    BOOT_IO={k:int(v) for k,v in (line.split(':') for line in _io)}
import time
START=time.monotonic()
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from types import ModuleType

FINALIZER_RUNTIME=Path('/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017')
FINALIZER_COMMIT='3df052ea422c4ba042ef3be33dea19438c5702dc'
CORE_SHA='4b2f0fccfd4bf3d655198f61179c88f6b2a51840731465c720b8addb6169b1a2'
CORE_PATH=FINALIZER_RUNTIME/'reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py'
raw=CORE_PATH.read_bytes()
if hashlib.sha256(raw).hexdigest()!=CORE_SHA:raise ValueError('Reviewed wait-boundary controller changed')
core=ModuleType('fixed_future_index_monitor');core.__file__=str(CORE_PATH)
exec(compile(raw,str(CORE_PATH),'exec'),core.__dict__)
core.BOOT_IO=BOOT_IO;core.START=START
BASE=core.BASE;MiB=core.MiB;GiB=core.GiB
COUNTS={'dev':1696,'train':101592}
CONTRACT={'path':str(BASE/'audits/fixed_teacher_batch64_runtime_run_v1/RESOLVED_PLAN.json'),
    'sha256':'a1de94f3e51b3e8ed050fa7f63b55d819cfa6453af7c1268b04d20c51f5e1d32'}
INDEX_BUILDER_SHA='ea8fe604c747d3a227f927c81c60851289b17c7f6fa1bfd36a95a8f9897d545f'
FIELDS=('identity','request_identity','cache_identity','scene_count','record_count','database_bytes',
    'builder_peak_rss_bytes','tensor_files_opened','image_files_opened','tensor_disk_bytes')
MONITORS={**core.MONITORS,'supervise_once.py':'d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902'}
PREPARER_PATH=Path(__file__).absolute().parent.parent/'full_future_index_preparation_v1/prepare_index_controls.py'
PREPARER_SHA='360a52ab7b8cf94bb1e8b1c7d2d6068a7ffe1acd1eec3f7761834a519d5a04d0'


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_checkpoint_commitments(cp,valid_until):
    # Original 2183 GiB obligations remain, plus the separately admitted T1
    # 120 GiB. A later larger, independently pinned CURRENT receipt is safe;
    # silently reusing the older smaller commitment is not.
    if (cp.get('schema')!='st_checkpoint_storage_commitments_v1' or cp.get('status')!='CURRENT'
            or type(cp.get('remaining_bytes')) is not int or cp['remaining_bytes']<2303*GiB
            or cp.get('free_floor_bytes')!=1<<40
            or type(cp.get('valid_until_unix')) not in (int,float) or not math.isfinite(cp['valid_until_unix'])
            or cp['valid_until_unix']<valid_until):raise ValueError('Fresh full checkpoint/floor commitment required')


def limits(role):
    if role not in COUNTS:raise ValueError('Only dev/train index roles')
    return {'wall_seconds':900 if role=='dev' else 3600,'cleanup_seconds':40,
        'read_bytes':(4 if role=='dev' else 64)*GiB,'rss_bytes':4*GiB,'cpu_affinity':[0,1],
        'root_bytes':(128*MiB if role=='dev' else 2*GiB),
        'stdout_bytes':MiB,'stderr_bytes':MiB,'attempts':1,'automatic_retry':False,'gpu':False}


def validate_admission(a,now,*,child=False):
    role=a.get('role');budget=limits(role)
    if (a.get('schema')!='st_full_future_index_resource_admission_v1'
            or a.get('status')!='PASS_BOUNDED_'+role.upper()+'_INDEX_ONLY'
            or json.dumps(a.get('limits'),sort_keys=True,allow_nan=False)!=json.dumps(budget,sort_keys=True)
            or a.get('hostname')!=socket.gethostname() or a.get('reviewer') in (None,'','/root/st_indexed_composition')
            or a.get('monitor_core')!={'path':str(CORE_PATH),'sha256':CORE_SHA}):
        raise ValueError('Exact independently reviewed index admission required')
    for name in ('issued_at_unix','latest_start_unix','valid_until_unix'):
        if type(a.get(name)) not in (int,float) or not math.isfinite(a[name]):raise ValueError('Finite index lease required')
    if not a['issued_at_unix']<=now<=a['latest_start_unix'] or a['valid_until_unix']<now+(0 if child else budget['wall_seconds']+40):
        raise ValueError('Expired/insufficient index resource window')
    if not {0,1}.issubset(os.sched_getaffinity(0)):raise ValueError('Admitted CPUs unavailable')
    return role


def expected_command(role,request_identity):
    requests=BASE/('caches/full_future_metadata_train_finalized_v1/TEACHER_REQUESTS.json' if role=='train'
                   else 'caches/full_future_metadata_dev_v1/TEACHER_REQUESTS.json')
    return [core.PYTHON,'-m','tools.st_world_policy.index_teacher_metadata','build',
        '--requests',str(requests),
        '--expected-request-identity',request_identity,'--contract-receipt',CONTRACT['path'],
        '--contract-receipt-sha256',CONTRACT['sha256'],
        '--output',str(BASE/('caches/full_future_request_index_'+role+'_v1')),'--max-scenes',str(COUNTS[role]),
        '--max-disk-bytes',str((112 if role=='dev' else 2016)*MiB),'--max-builder-rss-bytes',str(4*GiB)]


def validate_preparation(p,role,reg,done):
    if (p.get('schema')!='st_full_future_metadata_controls_v1' or p.get('phase')!='index'
            or p.get('index_phase')!='EXACT_REQUEST_BOUND' or p.get('dispatch_authorized') is not False
            or p.get('build_executed') is not False or p.get('new_teacher_tensor_bytes')!=0):
        raise ValueError('Actual index PREPARATION required')
    bindings=p.get('actual_completion_bindings',{})
    if set(bindings)!=set(COUNTS) or any(b.get('status')!='ACTUAL_COMPLETE_BOUND' for b in bindings.values()):
        raise ValueError('Both actual future COMPLETE inputs are required')
    binding=bindings[role];inp=p['source_inputs']['roles'][role]
    schema='st_future_metadata_finalization_complete_v1' if role=='train' else 'st_future_metadata_complete_v1'
    if (inp['future_complete']!=binding['complete'] or done.get('schema')!=schema
            or done.get('status')!='COMPLETE' or done.get('scope')!='full_population'
            or done.get('request_identity')!=binding.get('request_identity')
            or type(done.get('scenes')) is not int or done['scenes']!=COUNTS[role]
            or binding.get('scenes')!=COUNTS[role] or done.get('training_admission') is not False
            or done.get('files',{}).get('TEACHER_REQUESTS.json')!=binding.get('request_file_sha256')):
        raise ValueError('Missing or changed actual role COMPLETE/request population')
    if role=='train' and (Path(inp['future_complete']['path'])!=BASE/'caches/full_future_metadata_train_finalized_v1/FINALIZATION_COMPLETE.json'
            or binding.get('requests')!=expected_command(role,done['request_identity'])[5]
            or done.get('raw_log_deserialized') is not False or done.get('source_shards_reused') is not True):
        raise ValueError('Explicit finalized TRAIN receipt/root required')
    for name in ('teacher_extractions','tensor_disk_bytes'):
        if type(done.get(name)) is not int or done[name]!=0:raise ValueError('Metadata-only future input required')
    size=(56 if role=='dev' else 1008)*MiB
    requirements={'output_bytes':size,'staging_bytes':size,'metadata_bytes':(16 if role=='dev' else 32)*MiB}
    if (reg.get('schema')!='st_metadata_build_registration_v1' or reg.get('writer_kind')!='future_index'
            or reg.get('callback_protocol')!='st_metadata_storage_check_v1'
            or reg.get('command')!=expected_command(role,done['request_identity'])
            or reg.get('storage_requirements')!=requirements
            or reg.get('limits',{}).get('max_disk_bytes')!=limits(role)['root_bytes']
            or reg.get('limits',{}).get('max_rss_bytes')!=4*GiB
            or reg.get('limits',{}).get('max_seconds')!=limits(role)['wall_seconds']):
        raise ValueError('Exact bounded request-only index registration required')
    return binding


def validate_finalized_input(p,done):
    """Re-use the reviewed new receipt/terminal/old-failure binding exactly."""
    raw=core.pinned({'path':str(PREPARER_PATH),'sha256':PREPARER_SHA},decode=False)
    preparer=ModuleType('reviewed_finalized_index_preparation');preparer.__file__=str(PREPARER_PATH)
    exec(compile(raw,str(PREPARER_PATH),'exec'),preparer.__dict__)
    helper=preparer.load_helper();item=p['source_inputs']['roles']['train']
    terminal_pin=p['actual_completion_bindings']['train']['build_result']
    terminal=core.pinned(terminal_pin)
    reg=preparer.bind_finalized_train(helper,item['plan'],done,item['future_complete'],terminal,terminal_pin)
    if reg['source_code_root']!=item['original_source_code_root']:
        raise ValueError('Original frozen source runtime lineage changed')


def runtime_binding(role,source):
    if role=='train':
        if (source.get('schema')!='st_future_finalized_index_runtime_pins_v1'
                or source.get('runtime')!=str(FINALIZER_RUNTIME) or source.get('runtime_commit')!=FINALIZER_COMMIT):
            raise ValueError('Exact finalized TRAIN index runtime required')
        return str(FINALIZER_RUNTIME),FINALIZER_COMMIT
    runtime,commit=core.RUNTIMES['dev']
    if source.get('runtime')!=runtime or not source.get('source_commit','').startswith(commit):
        raise ValueError('Original immutable DEV index source required')
    return runtime,source['source_commit']



def validate_additive_policy(p,old,new):
    if {k:v for k,v in old.items() if k!='claims'}!={k:v for k,v in new.items() if k!='claims'}:
        raise ValueError('Index declaration may not change existing policy/core/reservations')
    old_claims=old.get('claims');new_claims=new.get('claims')
    if not isinstance(old_claims,list) or not isinstance(new_claims,list):raise ValueError('Explicit policy claims required')
    before={v['root']:v for v in old_claims};after={v['root']:v for v in new_claims}
    if len(before)!=len(old_claims) or len(after)!=len(new_claims) or any(after.get(k)!=v for k,v in before.items()):
        raise ValueError('Existing claim changed, missing or duplicated')
    expected={}
    for role in COUNTS:
        pin=p['registrations'][role];root=str(BASE/('caches/full_future_request_index_'+role+'_v1'))
        size=(56 if role=='dev' else 1008)*MiB
        expected[root]={'root':root,'claim_id':'full-future-'+role+'-index-v1','identity_kind':'st_metadata_build_registration_v1',
            'cache_identity':pin['sha256'],'registration_path':pin['path'],'registration_sha256':pin['sha256'],
            'output_bytes':size,'staging_bytes':size,'metadata_bytes':(16 if role=='dev' else 32)*MiB}
    if {k:v for k,v in after.items() if k not in before}!=expected:
        raise ValueError('Only exact two index declarations may be added')

def controls(args,*,child=False):
    a=core.pinned({'path':args.admission,'sha256':args.admission_sha256})
    role=validate_admission(a,time.time(),child=child)
    if a['controller']['path']!=str(Path(__file__).absolute()):raise ValueError('Wrong index controller path')
    core.pinned(a['controller'],decode=False)
    preparation_path=Path(a['preparation']['path'])
    if not preparation_path.is_relative_to(BASE/'audits/full_future_index_preparation_v1') or preparation_path.name!='PREPARATION.json':
        raise ValueError('Actual isolated index preparation path required')
    p=core.pinned(a['preparation']);regpin=p['registrations'][role];reg=core.pinned(regpin)
    done=core.pinned(p['source_inputs']['roles'][role]['future_complete'])
    validate_preparation(p,role,reg,done)
    source=core.pinned(p['source_inputs']['roles'][role]['source_pins'])
    runtime,commit=runtime_binding(role,source)
    if reg['cwd']!=runtime:
        raise ValueError('Original immutable index source required')
    for pin in source['files'].values():core.pinned(pin,decode=False)
    if role=='train':validate_finalized_input(p,done)
    if a['policy']!=p['proposed_policy'] or a['registration']!=regpin:raise ValueError('Changed index policy/registration binding')
    if a.get('base_policy')!=p['source_inputs']['old_policy']:raise ValueError('Exact additive base policy required')
    validate_additive_policy(p,core.pinned(a['base_policy']),core.pinned(a['policy']))
    cp=core.pinned(a['checkpoint_receipt'])
    validate_checkpoint_commitments(cp,a['valid_until_unix'])
    if set(a['monitor_helpers'])!=set(MONITORS):raise ValueError('Reviewed monitor helpers required')
    parent=None
    for name,sha in MONITORS.items():
        pin=a['monitor_helpers'][name]
        if pin['sha256']!=sha or Path(pin['path']).name!=name:raise ValueError('Changed monitor helper')
        core.pinned(pin,decode=False)
        if parent is None:parent=Path(pin['path']).parent
        if Path(pin['path']).parent!=parent:raise ValueError('Mixed monitor helpers')
    sys.path.insert(0,str(parent));import supervise_once as s;import observed_cli as obs
    sys.path.insert(0,runtime)
    from tools.st_world_policy import metadata_storage as m,cache_storage_budget as b
    guard={'policy_path':a['policy']['path'],'policy_sha256':a['policy']['sha256'],
        'registration_path':regpin['path'],'registration_sha256':regpin['sha256'],
        'checkpoint_receipt':a['checkpoint_receipt']['path'],'checkpoint_sha256':a['checkpoint_receipt']['sha256']}
    root=BASE/('caches/full_future_request_index_'+role+'_v1')
    if reg['output_root']!=str(root):raise ValueError('Exact role index root required')
    return a,p,reg,root,guard,s,obs,m,b


def read_small_actual(path):
    # No SQLite/index/request-body read: only the newly emitted small COMPLETE.
    before=path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size>MiB
            or any(x.is_symlink() for x in (path,*path.parents))):raise ValueError('Small non-aliased COMPLETE required')
    version=lambda v:(v.st_dev,v.st_ino,v.st_mode,v.st_nlink,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        opened=os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink!=1 or version(opened)!=version(before):
            raise ValueError('Index COMPLETE changed before descriptor read')
        raw=bytearray()
        while len(raw)<=MiB:
            block=os.read(fd,MiB+1-len(raw))
            if not block:break
            raw.extend(block)
        after_fd=os.fstat(fd);after=path.lstat()
        if (len(raw)>MiB or version(before)!=version(after_fd) or version(before)!=version(after)
                or len(raw)!=before.st_size or any(x.is_symlink() for x in (path,*path.parents))):
            raise ValueError('Index COMPLETE changed or exceeded bound during read')
    finally:os.close(fd)
    return json.loads(raw),{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def validate_result(role,p,output,done,contract):
    if not isinstance(output,dict) or output.get('status')!='GUARDED_METADATA_RETURN':raise ValueError('Guarded index result required')
    artifact=output.get('artifact')
    if not isinstance(artifact,dict) or set(artifact)!=set(FIELDS):raise ValueError('Original index CLI result required')
    binding=p['actual_completion_bindings'][role]
    if (output.get('registration_sha256')!=p['registrations'][role]['sha256']
            or done.get('schema')!='st_teacher_metadata_index_v1' or done.get('status')!='COMPLETE'
            or done.get('encoding')!='canonical_json_zlib_level1_v1'
            or done.get('identity')!=digest({k:v for k,v in done.items() if k!='identity'})
            or artifact!={k:done.get(k) for k in FIELDS}
            or done.get('request_identity')!=binding['request_identity']
            or type(done.get('scene_count')) is not int or done['scene_count']!=COUNTS[role]
            or done.get('cache_identity') is not None or done.get('cache_root') is not None
            or done.get('cache_header') is not None or done.get('complete_header') is not None
            or done.get('builder_source_sha256')!=INDEX_BUILDER_SHA
            or done.get('coverage')!='exact_original_manifest' or done.get('training_admission') is not False
            or done.get('contract')!=contract['contract']):raise ValueError('Actual COMPLETE/source/request contract mismatch')
    database_hash=done.get('database_sha256')
    if not isinstance(database_hash,str) or len(database_hash)!=64 or any(c not in '0123456789abcdef' for c in database_hash):
        raise ValueError('Original database content hash required')
    for name in ('tensor_files_opened','image_files_opened','tensor_disk_bytes'):
        if type(done.get(name)) is not int or done[name]!=0:raise ValueError('Metadata index opened forbidden payload')
    for name in ('record_count','database_bytes','builder_peak_rss_bytes'):
        if type(done.get(name)) is not int or done[name]<=0:raise ValueError('Invalid index counts/resources')
    if (done['database_bytes']>(112 if role=='dev' else 2016)*MiB
            or done['builder_peak_rss_bytes']>4*GiB
            or done.get('max_disk_bytes')!=(112 if role=='dev' else 2016)*MiB
            or done.get('max_builder_rss_bytes')!=4*GiB):raise ValueError('Index exceeds fixed role budget')
    request={'path':expected_command(role,binding['request_identity'])[5],'sha256':binding['request_file_sha256']}
    if done.get('dependencies',{}).get('requests')!=request:raise ValueError('Original request file pin changed')
    return done['identity']


def run(args):
    a,p,reg,root,guard,s,obs,m,b=controls(args);role=a['role'];budget=a['limits'];runtime=reg['cwd']
    source=core.pinned(p['source_inputs']['roles'][role]['source_pins'])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()!=runtime_binding(role,source)[1]:
        raise ValueError('Immutable index source HEAD changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=runtime).strip():raise ValueError('Immutable index source dirty')
    dispatch=core.pinned(a['dispatch_policy'])
    if not any(socket.gethostname()==h or socket.gethostname().startswith(h+'-worker-') for h in dispatch['allowed_hosts']):
        raise ValueError('Current host is not permitted')
    contract=core.pinned(CONTRACT);s.shared_mount();os.sched_setaffinity(0,[0,1])
    started=root/'CONTROL_INDEX_STARTED.json';result_path=root/'CONTROL_INDEX_RESULT.json'
    with ExitStack() as stack:
        lock=s.acquire(s.LOCK,stack);validate_admission(a,time.time())
        start={'schema':'st_full_future_index_invocation_v1','role':role,'admission':{'path':args.admission,'sha256':args.admission_sha256},
            'controller':a['controller'],'monitor_core':a['monitor_core'],'preparation':a['preparation'],
            'registered_unix':time.time(),'limits':budget,'automatic_retry':False}
        core.publish_guarded(m,b,guard,started,start)
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONPATH=runtime,ST_PROBE_SUPERVISOR_LOCK_FD=str(lock.fileno()))
            command=[core.PYTHON,str(Path(__file__).absolute()),'--child','--admission',args.admission,'--admission-sha256',args.admission_sha256]
            observed=core.monitor(command,runtime=runtime,env=env,fd=lock.fileno(),root=root,s=s,m=m,obs=obs,budget=budget)
            envelope=observed['child_accounting'];output=envelope.get('cli_result') if envelope else None
            if observed['returncode']!=0 or observed['failure'] is not None:raise ValueError('Bounded index child failed: '+str(observed['failure']))
            done,complete_pin=read_small_actual(root/'COMPLETE.json');validate_result(role,p,output,done,contract)
            total=core.charge_after_reap(observed['read_accounting'],obs.proc_io())
            if total>budget['read_bytes'] or observed['invocation_seconds']>budget['wall_seconds']:raise ValueError('Index invocation exceeded admitted resources')
            result={**start,'status':'INDEX_COMPLETE','resources':observed,'cli_result':output,'index_complete':complete_pin,
                'charged_parent_child_rchar_bytes':total,'training_authorized':False,'new_teacher_tensor_bytes':0}
            core.publish_guarded(m,b,guard,result_path,result);print(json.dumps({'status':'INDEX_COMPLETE','result':str(result_path)}));return 0
        except BaseException as exc:
            result={**start,'status':'FAILED_NO_AUTO_RETRY','error_type':type(exc).__name__,'error':str(exc)[:2048],
                'resources':locals().get('observed'),'charged_source_bytes':budget['read_bytes']}
            core.publish_guarded(m,b,guard,result_path,result);return 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    if args.child:
        core.controls=controls
        return core.child_run(args)
    return run(args)


if __name__=='__main__':raise SystemExit(main())
