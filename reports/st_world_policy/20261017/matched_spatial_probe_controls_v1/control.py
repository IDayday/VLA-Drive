#!/usr/bin/env python3
"""Fixed SP1/2/3 controls over the original probe; no automatic admission/retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import types

HERE=Path(__file__).resolve().parent
UP=HERE.parent/'spatial_probe_actual_preparation_v1'
INVENTORY=HERE.parent/'matched_spatial_probe_handoff_v1/ASSET_INVENTORY.json'
INVENTORY_SHA='d4ee7ce482342985aed21c9024d5d3912f6a5f50dba67583223e4e73dec2c7f5'
RUNTIME=Path('/mnt/project/VLA-Drive-st-w-storage-only-581399eb-v1-20261017')
SOURCE='2a1e140a64d2817247ceb6214f7f2affa20d56c6'
OLD_SOURCE='581399eb309677c20fe7464791b4f18685e947a6'
OLD_RUNTIME=Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017')
PINS={'prepare_bindings.py':'31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84',
      'supervise_once.py':'d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902',
      'observed_cli.py':'c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643'}
STORAGE={'tools/st_world_policy/metadata_storage.py':'68ffeb4862c6ea0536e34e9c14f36ee189625e02ead806775cd397e179f7e5ca',
         'tools/st_world_policy/spatial_w_storage.py':'0d2c16be1e7ef98bf5b1c9ab6abc10f6b4fdd23ff722d8e0e41f0b280f957987'}
READ_TOTAL=0


def raw(path, expected=None, limit=2<<20):
    global READ_TOTAL
    path=Path(path)
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        before=os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size>limit:raise ValueError('Small regular source/control required')
        READ_TOTAL+=before.st_size
        if READ_TOTAL>64<<20:raise ValueError('64MiB control/source read budget exhausted')
        data=f.read(limit+1);after=os.fstat(f.fileno())
    version=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if len(data)!=before.st_size or version(before)!=version(after) or version(after)!=version(path.stat()):raise ValueError('Changed control file')
    digest=hashlib.sha256(data).hexdigest()
    if expected is not None and digest!=expected:raise ValueError('Source/control hash mismatch')
    return data


def document(path, expected=None, limit=2<<20):
    data=raw(path,expected,limit)
    return json.loads(data),{'path':str(path),'sha256':hashlib.sha256(data).hexdigest()}


def arm_row(arm):
    if arm not in ('SP1','SP2','SP3'):raise ValueError('Only fixed SP1/SP2/SP3 are permitted')
    inventory,_=document(INVENTORY,INVENTORY_SHA)
    return next(a for a in inventory['arms'] if a['arm']=='S-P'+arm[-1])


def patch_exit_race(source):
    old="                if done and active:\n                    reason = reason or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'"
    new="                if done and active:\n                    active = owner.discover()  # recheck after WNOWAIT, as reviewed 4b2f monitor\n                    if active:\n                        reason = reason or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'"
    if source.count(old)!=1:raise ValueError('Original exit-boundary source differs')
    return source.replace(old,new)


def module(name, transform=None):
    path=UP/(name+'.py');source=raw(path,PINS[name+'.py']).decode()
    if transform:source=transform(source)
    result=types.ModuleType(name);result.__file__=str(path);result.__package__=''
    sys.modules[name]=result;exec(compile(source,str(path),'exec'),result.__dict__)
    return result


def controls(arm, *, supervisor=False):
    row=arm_row(arm)
    def prepare_source(source):
        assert source.count("'S-P0'")==3 and source.count('memory_tokens=144')==1
        source=source.replace("'S-P0'",repr(row['arm'])).replace('memory_tokens=144','memory_tokens='+str(row['memory_tokens']))
        old="protocol_path = runtime / 'reports/st_world_policy/20261017/spatial_probe/PROTOCOL.json'"
        assert source.count(old)==1
        return source.replace(old,"protocol_path = Path("+repr(str(OLD_RUNTIME/'reports/st_world_policy/20261017/spatial_probe/PROTOCOL.json'))+")")
    p=module('prepare_bindings',prepare_source)
    p.TAG=row['checkpoint']['tag'];p.REG=Path(row['registration']['path']);p.REG_SHA=row['registration']['sha256']
    p.EXPORT=Path(row['export_identity']['path']);p.EXPORT_SHA=row['export_identity']['sha256']
    p.RECEIPT_SHA=row['native_receipt']['sha256'];p.JOB=Path(row['proposed_job_root'])
    p.ROOTS={s:Path(row['cache_root'])/s for s in ('dev','train')}
    p.read=lambda path,expected=None,limit=1<<20:document(path,expected,limit)
    if not supervisor:return p,row
    s=module('supervise_once',patch_exit_race)
    def no_retries(admission,stage):
        if any(k in admission for k in ('manual_retry','previous_stage_result','previous_artifact_reconciliation','storage_compatibility')):
            raise ValueError('No old retry/reconciliation/storage bridge is valid for new matched arms')
        return None
    s.check_manual_fields=no_retries
    original_validate=s.validate
    def validate(admission,preparation,stage,now):
        if admission.get('matched_arm')!=arm or admission.get('automatic_retry') is not False:raise ValueError('Exact matched arm and no automatic retry required')
        if admission.get('reviewer') in ('/root/st_indexprep_boundary_fix',None,''):raise ValueError('Independent reviewer required')
        if admission.get('matched_controller')!=self_pin():raise ValueError('Matched controller changed or absent')
        if preparation['runtime']!={'path':str(RUNTIME),'source_sha':SOURCE}:raise ValueError('Wrong fixed storage-only runtime')
        if preparation['checkpoint_identity']!=row['checkpoint']['sha256']:raise ValueError('Wrong fixed native checkpoint')
        return original_validate(admission,preparation,stage,now)
    s.validate=validate
    original_supervise=s.supervise
    def supervise(argv,**kwargs):
        expected=[s.PYTHON,str(UP/'observed_cli.py')]
        if argv[:2]!=expected or len(argv)!=10 or argv[2::2]!=['--admission','--admission-sha256','--attempt','--read-cap']:
            raise ValueError('Only original observer invocation is permitted')
        return original_supervise([s.PYTHON,str(HERE/'control.py'),'observe','--arm',arm,*argv[2:]],**kwargs)
    s.supervise=supervise
    return p,row,s


def self_pin():
    return {'path':str(HERE/'control.py'),'sha256':hashlib.sha256(raw(HERE/'control.py')).hexdigest(),
            'inventory_sha256':INVENTORY_SHA}


def verify_runtime():
    def git(*args):return subprocess.check_output(['git',*args],cwd=RUNTIME,timeout=15).decode().strip()
    if git('rev-parse','HEAD')!=SOURCE or git('status','--porcelain'):raise ValueError('Storage-only runtime must remain clean and pinned')
    if sorted(git('diff','--name-only',OLD_SOURCE,SOURCE).splitlines())!=sorted(STORAGE):raise ValueError('Only two approved storage files may differ')
    for path,pin in STORAGE.items():raw(RUNTIME/path,pin)
    # The whole-tree delta proves every compute/readout/CLI file remains original.
    return {'runtime':str(RUNTIME),'source_sha':SOURCE,'compute_parent':OLD_SOURCE,'only_changed_files':STORAGE,
            'probe_mathematics_changed':False,'labels_changed':False}


def prepare(arm):
    provenance=verify_runtime();p,row=controls(arm)
    output=HERE/'prepared'/arm
    pin=p.build(RUNTIME,SOURCE,output)
    prep,_=document(pin['path'],pin['sha256'])
    p.write_once(output/'RUNTIME_COMPATIBILITY.json',provenance)
    for stage in ('dev_plan','dev_extract','train_plan','train_extract','fit'):
        action=stage.rsplit('_',1)[-1]
        wall={'plan':600,'extract':960,'fit':1260}[action]
        template={'schema':'st_SP0_probe_stage_admission_v1','status':'DRAFT_NOT_APPROVED',
            'matched_arm':arm,'matched_controller':self_pin(),'automatic_retry':False,'reviewer':None,
            'stage':stage,'hostname':None,'valid_until_unix':None,'runtime':prep['runtime'],'preparation':pin,
            'helpers':PINS,'cpu_affinity':[2,3],'gpu':{'index':None,'uuid':None} if action=='extract' else None,
            'storage_policy':None,'storage_migration':None,'checkpoint_commitments':None,
            'typed_storage_approved':False,'outer_wall_seconds':wall,'cleanup_coverage_seconds':40,
            'observed_RSS_limit_bytes':(2 if action=='fit' else 64)<<30,
            'observed_GPU_peak_limit_bytes':24<<30 if action=='extract' else 0,
            'read_limit_bytes':(2 if action=='fit' else 256)<<30,
            'scope':'Only this fixed one-shot stage; Lead dispatch after independent resource admission'}
        p.write_once(output/(stage+'_ADMISSION_TEMPLATE.json'),template)
    print(json.dumps({'status':'PREPARED_NOT_APPROVED','arm':arm,'preparation':pin,'read_bytes':READ_TOTAL,'dispatch':False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','run','observe'):
        q=sub.add_parser(name);q.add_argument('--arm',required=True,choices=('SP1','SP2','SP3'))
        if name!='prepare':q.add_argument('--admission',required=True);q.add_argument('--admission-sha256',required=True)
        if name=='observe':q.add_argument('--attempt',required=True);q.add_argument('--read-cap',required=True,type=int)
    args=parser.parse_args()
    if args.command=='prepare':
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded prepare exhausted')))
        signal.alarm(120)
        prepare(args.arm);return 0
    p,row,s=controls(args.arm,supervisor=True)
    if args.command=='run':
        sys.argv=[str(UP/'supervise_once.py'),'--admission',args.admission,'--admission-sha256',args.admission_sha256]
        return s.main()
    admission,_=document(args.admission,args.admission_sha256)
    preparation,_=document(admission['preparation']['path'],admission['preparation']['sha256'])
    s.validate(admission,preparation,admission['stage'],time.time())
    fd=int(os.environ['ST_PROBE_SUPERVISOR_LOCK_FD']);os.fstat(fd)
    if os.getpgrp()!=os.getpid():raise ValueError('Dedicated supervised session required')
    observer=module('observed_cli')
    return observer.run(preparation,admission['stage'],args.attempt,admission.get('storage_policy'),
                        admission.get('checkpoint_commitments'),args.read_cap)


if __name__=='__main__':raise SystemExit(main())
