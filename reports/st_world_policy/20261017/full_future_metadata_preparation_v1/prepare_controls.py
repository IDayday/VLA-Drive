"""Prepare exact same-ledger future metadata proposals; never migrate or build."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

MiB=1<<20
GiB=1<<30
BASE=Path('/mnt/project/st-world-policy-20261016')
PYTHON='/root/miniconda3/envs/ddp/bin/python'
READ_BYTES=0
WRITE_BYTES=0


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def safe(path):
    p=Path(path)
    if not p.is_absolute() or '..' in p.parts or any(q.is_symlink() for q in (p,*p.parents)):
        raise ValueError('Absolute nonsymlink metadata path required')
    return p


def read(pin):
    global READ_BYTES
    path=safe(pin['path']);s=path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_size>4*MiB or READ_BYTES+s.st_size>32*MiB:
        raise ValueError('Small metadata-only preparation budget exceeded')
    READ_BYTES+=s.st_size;raw=path.read_bytes();a=path.stat()
    version=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    if version(s)!=version(a) or hashlib.sha256(raw).hexdigest()!=pin['sha256']:
        raise ValueError('Pinned input changed')
    return raw


def document(pin):return json.loads(read(pin))


def write(root,name,value):
    global WRITE_BYTES
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    # Leave 256 KiB of the one-MiB preparation allowance for directories/blocks.
    if len(raw)>MiB or WRITE_BYTES+len(raw)>768*1024:raise ValueError('One-MiB preparation output budget exceeded')
    WRITE_BYTES+=len(raw)
    path=root/name
    with path.open('xb') as f:f.write(raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def extend_policy(old,claims):
    new=copy.deepcopy(old);package=new['metadata_package']
    if package['schema']=='st_metadata_storage_package_v1':
        if package['maximum_bytes']!=4*GiB:raise ValueError('Exact original four-GiB envelope required')
        previous=digest(package)
        package.update(schema='st_metadata_storage_package_v2',maximum_bytes=8*GiB,previous_package_sha256=previous)
    elif package['schema']!='st_metadata_storage_package_v2' or package['maximum_bytes']!=8*GiB:
        raise ValueError('Only the explicit original v1 to v2 transition exists')
    roots={c['root'] for c in old['claims']}
    if any(c['root'] in roots for c in claims) or len({c['root'] for c in claims})!=len(claims):
        raise ValueError('No old claim replacement or duplicate new root')
    new['claims'].extend(claims)
    if new.get('frozen_w_package')!=old.get('frozen_w_package') or new['legacy_reservations']!=old['legacy_reservations']:
        raise AssertionError('Existing W/legacy identity changed')
    return new


def callback_pins(source,module):
    runtime=Path(source['runtime']);entry=module.replace('.','/')+'.py'
    tree=ast.parse(read(source['files'][entry]))
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id.startswith('METADATA_STORAGE_CALLBACK_'):
                    constants[target.id]=ast.literal_eval(node.value)
    if constants['METADATA_STORAGE_CALLBACK_PROTOCOL']!='st_metadata_storage_check_v1':
        raise ValueError('Original bounded metadata callback required')
    names=[entry]+[v.replace('.','/')+'.py' for v in constants['METADATA_STORAGE_CALLBACK_DEPENDENCIES']]
    pins=[]
    for name in names:
        pin=source['files'][name]
        if safe(pin['path'])!=runtime/name:raise ValueError('Runtime dependency path mismatch')
        read(pin);pins.append({'path':pin['path'],'sha256':pin['sha256']})
    return pins


def existing_pins(root):
    """No deletion/archive; caller verifies the separately reviewed 16-pin API."""
    global READ_BYTES
    files=[];directories=[]
    for directory,dirs,names in os.walk(root,followlinks=False):
        for name in dirs:
            p=safe(Path(directory)/name)
            if not p.is_dir():raise ValueError('Nonregular preexisting directory')
            directories.append(p.relative_to(root).as_posix())
        for name in names:
            p=safe(Path(directory)/name);s=p.stat()
            if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_size>4*MiB:
                raise ValueError('Only bounded original plan/control files may be adopted')
            if p.suffix not in ('.json','.lock','.log'):raise ValueError('Payload forbidden in plan namespace')
            if READ_BYTES+s.st_size>32*MiB:raise ValueError('Metadata budget exhausted')
            READ_BYTES+=s.st_size;raw=p.read_bytes();after=p.stat()
            version=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
            if version(s)!=version(after):raise ValueError('Plan control changed during preparation')
            files.append({'relative_path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()})
            if len(files)>16:raise ValueError('Preexisting population exceeds reviewed 16-pin bound')
    return sorted(files,key=lambda x:x['relative_path']),sorted(directories)


def storage_registration(role,phase,root,source,command,requirements,preexisting=(),directories=()):
    prep=phase=='preparation';total=sum(requirements.values())
    patterns=[r'METADATA_ADMISSION\.json',r'BUILD\.lock',r'COMPLETE\.json(?:\.[a-z0-9_]+\.tmp)?',
        r'CONTROL_[A-Z0-9_]+\.json(?:\.[a-z0-9_]+\.tmp)?']
    if prep:
        patterns += [r'PLAN\.(?:json|lock)',r'(?:TEACHER_REQUESTS|RAW_LOG_PROVENANCE|SHARD_RECEIPTS|COMPLETE)\.json(?:\.[0-9a-f]+\.partial)?',
            r'log_shards/[0-9a-f]{64}\.json(?:\.[0-9a-f]+\.partial)?']
    else:patterns += [r'metadata\.(?:sqlite(?:-journal)?|[0-9a-f]+\.partial\.sqlite(?:-journal)?|orphan\.[0-9a-f]+\.sqlite)']
    patterns += [re.escape(p['relative_path']) for p in preexisting]
    module='tools.st_world_policy.'+('prepare_future_metadata' if prep else 'index_teacher_metadata')
    reg={'schema':'st_metadata_build_registration_v1','writer_kind':'future_'+phase,
        'callback_protocol':'st_metadata_storage_check_v1','cwd':source['runtime'],'command':command,
        'source_pins':callback_pins(source,module),'output_root':str(root),'storage_requirements':requirements,
        'limits':{'max_disk_bytes':total,'max_rss_bytes':4*GiB,
            'max_seconds':(1800 if role=='dev' else 14400) if prep else (900 if role=='dev' else 3600),
            'max_depth':1 if prep else 0,'max_entries':2048 if prep else 100},
        'file_patterns':patterns,'directory_patterns':(['log_shards'] if prep else [])+list(directories)}
    if preexisting:reg['preexisting_pins']=list(preexisting)
    return reg


def prepare(inputs,root):
    old=document(inputs['old_policy']);ledger=document(inputs['old_ledger']);resource_plan=document(inputs['resource_plan'])
    core_names={'schema','campaign','state_dir','allowed_roots','cache_cap_bytes','free_reserve_bytes',
        'ledger_reserve_bytes','legacy_reservations','metadata_package','frozen_w_package'}
    if ledger['core']!={k:v for k,v in old.items() if k in core_names}:
        raise ValueError('Actual old policy/core mismatch')
    if any(ledger['claims'][c['root']]['claim']!=c for c in old['claims']):raise ValueError('Old claim mismatch')
    phase=inputs['phase']
    if (phase=='preparation' and resource_plan['old_ledger_sha256']!=inputs['old_ledger']['sha256']) or resource_plan['final_metadata_reserved_mib']!=5616:
        raise ValueError('Resource proposal identity mismatch')
    if phase not in ('preparation','index'):raise ValueError('Only original two metadata writer phases')
    claims=[];registrations={};requirements={}
    for role in ('dev','train'):
        b=inputs['roles'][role]
        if not b.get('source_pins'):raise ValueError('WAITING_REVIEWED_BUILD_RUNTIME: source pin set missing')
        source=document(b['source_pins']);plan=document(b['plan'])
        if plan['identity']!=digest({k:v for k,v in plan.items() if k!='identity'}):raise ValueError('PLAN self identity mismatch')
        for name,sha in plan['implementation'].items():
            pin=source['files'][name]
            if pin['sha256']!=sha:raise ValueError('Build runtime changes original PLAN implementation: '+name)
            read(pin)
        role_budget=resource_plan['roles'][role];count=role_budget['scenes']
        if plan['status']!='PLAN_ONLY' or plan['scope']!='full_population' or plan['selected_scene_count']!=count:
            raise ValueError('Exact completed full population plan required')
        if any(plan[k] is not False for k in ('gpu_authorized','teacher_extraction_authorized','training_authorized')):
            raise ValueError('Plan grants forbidden model scope')
        if phase=='preparation':
            out=Path(role_budget['prep_output_root'])
            if Path(b['plan']['path'])!=out/'PLAN.json':raise ValueError('PLAN namespace differs')
            completed=document(b['plan_result'])
            if (completed.get('status')!='PLAN_COMPLETE' or completed['cli_result']['identity']!=plan['identity']
                    or completed['cli_result']['selected_scene_count']!=count):
                raise ValueError('Actual successful bounded PLAN controller evidence required')
            storage_ast=ast.parse(read(source['files']['tools/st_world_policy/metadata_storage.py']))
            bounds=[n.comparators[0].value for n in ast.walk(storage_ast) if isinstance(n,ast.Compare)
                and isinstance(n.left,ast.Call) and isinstance(n.left.func,ast.Name) and n.left.func.id=='len'
                and len(n.left.args)==1 and isinstance(n.left.args[0],ast.Name) and n.left.args[0].id=='pins'
                and len(n.ops)==1 and isinstance(n.ops[0],ast.Gt) and isinstance(n.comparators[0],ast.Constant)]
            if 16 not in bounds:raise ValueError('BLOCKED_PREEXISTING_LIMIT: new reviewed 16-pin build runtime required')
            preexisting,dirs=existing_pins(out)
            body={'schema':'st_future_metadata_registration_v1','phase':'data_preparation','approved_scope':'full_population',
                'plan_sha256':b['plan']['sha256'],'plan_identity':plan['identity'],
                'gpu_authorized':False,'teacher_extraction_authorized':False,'training_authorized':False}
            internal=write(root,role.upper()+'_FUTURE_PREPARATION_REGISTRATION.json',{**body,'identity':digest(body)})
            command=[PYTHON,'-m','tools.st_world_policy.prepare_future_metadata','build','--plan',b['plan']['path'],
                '--registration',internal['path'],'--registration-sha256',internal['sha256']]
            req={'output_bytes':(56 if role=='dev' else 1008)*MiB,'staging_bytes':(56 if role=='dev' else 1008)*MiB,
                'metadata_bytes':(16 if role=='dev' else 32)*MiB}
        else:
            done=document(b['future_complete']);contract=document(inputs['contract_receipt'])
            if (done['schema']!='st_future_metadata_complete_v1' or done['status']!='COMPLETE'
                    or done['plan_identity']!=plan['identity'] or done['scenes']!=count or done['tensor_disk_bytes']!=0):
                raise ValueError('WAITING_REQUEST_ID: actual matching future COMPLETE required')
            request_identity=done['request_identity']
            if not re.fullmatch('[0-9a-f]{64}',request_identity):raise ValueError('Malformed actual request identity')
            out=Path(role_budget['index_output_root']);preexisting=[];dirs=[]
            if out.exists():raise ValueError('Do not overwrite an existing index namespace')
            req={'output_bytes':(56 if role=='dev' else 1008)*MiB,'staging_bytes':(56 if role=='dev' else 1008)*MiB,
                'metadata_bytes':(16 if role=='dev' else 32)*MiB}
            cp=inputs['contract_receipt']
            command=[PYTHON,'-m','tools.st_world_policy.index_teacher_metadata','build',
                '--requests',str(Path(role_budget['prep_output_root'])/'TEACHER_REQUESTS.json'),
                '--expected-request-identity',request_identity,'--contract-receipt',cp['path'],
                '--contract-receipt-sha256',cp['sha256'],'--output',str(out),'--max-scenes',str(count),
                '--max-disk-bytes',str(role_budget['index']['database_page_limit_bytes']),
                '--max-builder-rss-bytes',str(4*GiB)]
        reg=storage_registration(role,phase,out,source,command,req,preexisting,dirs)
        pin=write(root,role.upper()+'_STORAGE_REGISTRATION.json',reg);registrations[role]=pin
        claims.append({'root':str(out),'claim_id':'full-future-'+role+'-'+phase+'-v1',
            'identity_kind':'st_metadata_build_registration_v1','cache_identity':pin['sha256'],
            'registration_path':pin['path'],'registration_sha256':pin['sha256'],**req})
        requirements[role]=sum(req.values())
    proposal=extend_policy(old,claims);policy_pin=write(root,'PROPOSED_POLICY.json',proposal)
    inventory=write(root,'NO_PILOT_ADOPTIONS.json',{'schema':'st_metadata_pilot_inventory_v1','roots':[]})
    # This is a blocked template, never a signed claim of live quiescence.
    template={'schema':'st_metadata_writer_quiescence_v1','status':'PENDING_LEAD_LIVE_WRITER_CENSUS',
        'owner_role':'Lead','old_policy':inputs['old_policy'],'new_policy':policy_pin,'old_ledger':inputs['old_ledger'],
        'pilot_inventory':inventory,'active_or_unknown_writers':None,'queued_writers_disabled':None,
        'future_admission_owner':'Lead','process_evidence':[],'checked_at_unix':None,'valid_until_unix':None}
    write(root,'QUIESCENCE_NOT_APPROVED.json',template)
    return {'schema':'st_full_future_metadata_controls_v1','status':'PREPARED_NOT_MIGRATED_NOT_AUTHORIZED',
        'phase':phase,'source_inputs':inputs,'proposed_policy':policy_pin,'registrations':registrations,
        'pilot_inventory':inventory,'new_claim_bytes':sum(requirements.values()),
        'administrative_metadata_bytes_not_claims':128*MiB,'final_metadata_budget_bytes':5616*MiB,
        'metadata_envelope_bytes':8*GiB,'index_phase':'WAITING_REQUEST_ID' if phase=='preparation' else 'EXACT_REQUEST_BOUND',
        'new_teacher_tensor_bytes':0,'live_ledger_written':False,'build_executed':False,'dispatch_authorized':False,
        'metadata_read_bytes':READ_BYTES}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True);p.add_argument('--inputs-sha256',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    inputs=document({'path':a.inputs,'sha256':a.inputs_sha256});out=safe(a.output)
    if not out.is_relative_to(BASE/'audits/full_future_metadata_preparation_v1'):
        raise ValueError('Only isolated new metadata proposal namespace allowed')
    out.mkdir(parents=True,exist_ok=False)
    result=prepare(inputs,out);pin=write(out,'PREPARATION.json',result)
    print(json.dumps({'status':result['status'],'preparation':pin}))


if __name__=='__main__':main()
