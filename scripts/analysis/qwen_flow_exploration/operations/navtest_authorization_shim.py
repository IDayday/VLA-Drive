"""CPU-only compatibility owner for immutable evaluation admission345.

Never modifies global policy, model code, a qualification, GPU assignments or
lease duration. It can refresh only this study's full-policy host fingerprints
after a sealed user-permission-semantic comparison. Unknown changes fail closed.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

TASK='qwen_flow_three_routes_20261024'
GRANT='qwen_sourceflow_navtest_resources_20261024'
LEGACY_TASK='qwen_sourceflow_native_20261018'
GLOBAL_OPERATION_FIELDS=frozenset(('lease_id','lease_record','scope','new_task_dispatch_allowed','authorized_gpu_indices'))
LEGACY_ALLOWED_KEYS=frozenset(('aliases','lease_id','task_scope','purpose','authorized_gpu_indices'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def checked(value):
    body=copy.deepcopy(value)
    if body.pop('sha256',None)!=fingerprint(body):raise ValueError('Changed/missing seal')
    return value


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.writing-'+uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def full_fingerprint(policy,host):
    # Exact immutable345 implementation, without importing model libraries.
    return fingerprint(dict(standing_grant=policy['task_authorizations'].get(GRANT),
        study_grant=policy['task_authorizations'].get(TASK),
        global_host=policy.get('global_dispatch_prohibitions',{}).get(host),
        permanent_host=policy.get('allowed_hosts',{}).get(host)))


def contains_host(value,aliases):
    if isinstance(value,str):return value in aliases
    if isinstance(value,dict):return any(key in aliases or contains_host(item,aliases) for key,item in value.items())
    if isinstance(value,list):return any(contains_host(item,aliases) for item in value)
    return False


def target_withdrawals(policy,aliases):
    return {key:copy.deepcopy(value) for key,value in policy.items()
            if (('withdraw' in key.lower() or 'revok' in key.lower() or key=='latest_host_restriction')
                and contains_host(value,aliases))}


def record_files(value):
    """Bind referenced actual permission/revocation records, including content."""
    result={}
    def visit(x):
        if isinstance(x,dict):
            for key,item in x.items():
                if isinstance(item,str) and (key=='record' or key.endswith('_record')) and item.startswith('/'):
                    path=Path(item)
                    result[item]=(hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'MISSING')
                else:visit(item)
        elif isinstance(x,list):
            for item in x:visit(item)
    visit(value)
    return result


def permission_semantics(policy,host,mode):
    global_host=copy.deepcopy(policy.get('global_dispatch_prohibitions',{}).get(host))
    aliases={host,*((global_host or {}).get('aliases',[]))}
    stripped=({key:value for key,value in global_host.items() if key not in GLOBAL_OPERATION_FIELDS}
              if global_host is not None else None)
    withdrawals=target_withdrawals(policy,aliases)
    value=dict(host=host,mode=mode,standing_grant=copy.deepcopy(policy['task_authorizations'].get(GRANT)),
        study_grant=copy.deepcopy(policy['task_authorizations'].get(TASK)),global_host_permission_fields=stripped,
        target_withdrawals=withdrawals,
        permanent_host=copy.deepcopy(policy.get('allowed_hosts',{}).get(host)) if mode=='ordinary' else None)
    value['actual_permission_record_hashes']=record_files(dict(global_host_permission_fields=stripped,target_withdrawals=withdrawals))
    return value


def capability_invariant(lease):
    value=copy.deepcopy(lease);value.pop('sha256',None)
    for host in value['hosts'].values():host.pop('authorization_sha256',None)
    return value


def seal_semantics(baseline,anchor,helper_path):
    checked(anchor)
    if anchor['status']!='ACTIVE' or anchor.get('allow_TRAIN_or_DEV') is not False or anchor.get('allow_optimizer') is not False:
        raise PermissionError('Only the original finite frozen Navtest capability may anchor permissions')
    hosts={}
    for host,entry in anchor['hosts'].items():
        if (any(not isinstance(i,int) or isinstance(i,bool) or not 0<=i<8 for i in entry['gpu_indices']) or
                len(set(entry['gpu_indices']))!=len(entry['gpu_indices'])):
            raise PermissionError('Only exact distinct physical GPU indices0..7 are legal')
        if full_fingerprint(baseline,host)!=entry['authorization_sha256']:
            raise PermissionError('Baseline is not the original already-authorized full fingerprint')
        standing=baseline['task_authorizations'].get(GRANT,{})
        ordinary=(host in baseline['allowed_hosts'] and host in baseline['task_authorizations'][TASK]['allowed_hosts'] and
                  baseline.get('global_dispatch_prohibitions',{}).get(host,{}).get('new_task_dispatch_allowed') is not False)
        mode='ordinary' if ordinary else 'standing_navtest_only'
        if not ordinary:
            if host not in standing.get('allowed_hosts',[]) or not set(entry['gpu_indices'])<=set(standing['authorized_gpu_indices'][host]):
                raise PermissionError('No original standing host/GPU permission')
            if baseline['allowed_hosts'].get(host) is not None:
                raise PermissionError('Anchor a Navtest-only host at a closed legacy lease boundary')
        if host.endswith('tsinghua-zt6') and 0 in entry['gpu_indices']:
            raise PermissionError('zt6 GPU0 remains unauthorized')
        hosts[host]=dict(mode=mode,semantics=permission_semantics(baseline,host,mode),
            closed_global_host=copy.deepcopy(baseline.get('global_dispatch_prohibitions',{}).get(host)),
            closed_allowed_host=copy.deepcopy(baseline.get('allowed_hosts',{}).get(host)))
    helper=Path(helper_path)
    result=dict(schema='sealed_Navtest_user_permission_semantics_v1',created_unix=time.time(),
        original_anchor_capability_sha256=anchor['sha256'],original_anchor_host_full_hashes={h:e['authorization_sha256'] for h,e in anchor['hosts'].items()},
        hosts=hosts,known_legacy_helper_path=str(helper.resolve()),known_legacy_helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        stripped_global_operational_fields=sorted(GLOBAL_OPERATION_FIELDS),
        no_global_policy_write=True,no_GPU_dispatch=True,unknown_permission_changes='refuse; never auto-rebase the sealed semantic receipt')
    result['sha256']=fingerprint(result)
    return result


def validate_legacy_state(policy,host,receipt,*,depth=0,seen=None):
    """Allow exact closed baseline or proven original-helper lease enter/close.

    A bare false new_task flag, new scope or smaller GPU list under an active
    lease is NOT discarded as metadata. It fails the exact active-shape check.
    """
    if depth>8:raise PermissionError('Legacy lease ancestry too deep')
    row=receipt['hosts'][host];global_host=policy.get('global_dispatch_prohibitions',{}).get(host)
    allowed=policy.get('allowed_hosts',{}).get(host)
    if global_host==row['closed_global_host'] and allowed==row['closed_allowed_host']:return 'CLOSED_BASELINE'
    if row['mode']=='ordinary':raise PermissionError('Ordinary permanent host admission changed')
    if not isinstance(global_host,dict) or not isinstance(allowed,dict):raise PermissionError('Unrecognized host admission transition')
    if set(allowed)!=LEGACY_ALLOWED_KEYS:raise PermissionError('Unknown temporary allowed-host fields')
    standing=policy['task_authorizations'][GRANT];indices=standing['authorized_gpu_indices'][host]
    expected=dict(aliases=row['closed_global_host']['aliases'],lease_id=global_host.get('lease_id'),
        task_scope=LEGACY_TASK,purpose='this registered Navtest only',authorized_gpu_indices=indices)
    if (not global_host.get('lease_id') or allowed!=expected or global_host.get('new_task_dispatch_allowed') is not True or
            global_host.get('scope')!='this registered native Navtest only' or global_host.get('authorized_gpu_indices')!=indices):
        raise PermissionError('Active lease shape changed, including withdrawal/permission reduction')
    lease_record=Path(global_host.get('lease_record',''))
    helper_root=Path(receipt['known_legacy_helper_path']).parent.resolve()
    if lease_record.name!='lease.json' or not lease_record.parent.resolve().is_relative_to(helper_root):
        raise PermissionError('Unexpected original-helper lease record location')
    if seen is None:seen=set()
    marker=str(lease_record.resolve())
    if marker in seen:raise PermissionError('Cyclic lease ancestry')
    seen.add(marker)
    if lease_record.exists():
        lease=read(lease_record)
        if (lease.get('lease_id')!=global_host['lease_id'] or lease.get('standing_grant')!=GRANT or
                lease.get('hosts',{}).get(host)!=indices):raise PermissionError('Legacy lease record does not prove this exact scope')
    previous=read(lease_record.parent/'policy_before.json')
    if permission_semantics(previous,host,row['mode'])!=row['semantics']:
        raise PermissionError('Legacy ancestry contains changed real permissions')
    validate_legacy_state(previous,host,receipt,depth=depth+1,seen=seen)
    return 'KNOWN_LEGACY_ACTIVE' if lease_record.exists() else 'KNOWN_LEGACY_PUBLISHING'


def validate_current(policy,receipt,lease,template):
    checked(receipt);checked(lease);checked(template)
    if capability_invariant(lease)!=capability_invariant(template):raise PermissionError('Capability scope/expiry/budget/configuration changed')
    if lease['status']!='ACTIVE' or time.time()>=lease['expires_unix']:raise PermissionError('Expired or inactive capability is not renewable by this shim')
    if set(lease['hosts'])!=set(receipt['hosts']):raise PermissionError('Capability host set changed')
    helper=Path(receipt['known_legacy_helper_path'])
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=receipt['known_legacy_helper_sha256']:
        raise PermissionError('Original legacy writer implementation changed')
    states={}
    for host,entry in lease['hosts'].items():
        row=receipt['hosts'][host]
        if permission_semantics(policy,host,row['mode'])!=row['semantics']:
            raise PermissionError('True permission/withdrawal/unknown-field change: '+host)
        if row['mode']=='standing_navtest_only':
            grant=policy['task_authorizations'][GRANT]
            if host not in grant['allowed_hosts'] or not set(entry['gpu_indices'])<=set(grant['authorized_gpu_indices'][host]):
                raise PermissionError('Standing permission was reduced')
        if host.endswith('tsinghua-zt6') and 0 in entry['gpu_indices']:raise PermissionError('zt6 GPU0 is forbidden')
        states[host]=validate_legacy_state(policy,host,receipt)
    return states


def refresh(policy_path,lease_path,receipt_path,template_path,*,dry_run=False,invalidate_on_refusal=False):
    policy_path=Path(policy_path);lease_path=Path(lease_path)
    with Path(str(policy_path)+'.lock').open('a') as policy_lock:
        fcntl.flock(policy_lock,fcntl.LOCK_EX);raw=policy_path.read_bytes();policy=json.loads(raw)
        lease=checked(read(lease_path));before=copy.deepcopy(lease);receipt=checked(read(receipt_path));template=checked(read(template_path))
        try:states=validate_current(policy,receipt,lease,template)
        except Exception:
            if invalidate_on_refusal and not dry_run and lease.get('status')=='ACTIVE':
                blocked=copy.deepcopy(lease);blocked['status']='BLOCKED_PERMISSION_CHANGE';blocked.pop('sha256');blocked['sha256']=fingerprint(blocked)
                atomic(lease_path,blocked)  # Revoke our capability; never refresh an authorization hash.
            raise
        updates={}
        for host in lease['hosts']:
            new=full_fingerprint(policy,host)
            if lease['hosts'][host]['authorization_sha256']!=new:
                updates[host]=dict(before=lease['hosts'][host]['authorization_sha256'],after=new)
                lease['hosts'][host]['authorization_sha256']=new
        if capability_invariant(lease)!=capability_invariant(before):raise AssertionError('Shim attempted more than a fingerprint refresh')
        if policy_path.read_bytes()!=raw:raise RuntimeError('Global policy changed during its exclusive lock')
        lease.pop('sha256');lease['sha256']=fingerprint(lease)
        if updates and not dry_run:atomic(lease_path,lease)
        return dict(status='SEMANTICS_UNCHANGED',at_unix=time.time(),states=states,updated_hosts=updates,
            dry_run=dry_run,capability_sha256=lease['sha256'],permission_semantics_sha256=receipt['sha256'],global_policy_unchanged=True)


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='command',required=True)
    seal=sub.add_parser('seal');seal.add_argument('--baseline-policy',required=True);seal.add_argument('--anchor-lease',required=True)
    seal.add_argument('--legacy-helper',required=True);seal.add_argument('--output',required=True)
    for name in ('once','serve'):
        q=sub.add_parser(name);q.add_argument('--policy',required=True);q.add_argument('--lease',required=True)
        q.add_argument('--semantics',required=True);q.add_argument('--template',required=True);q.add_argument('--output',required=True)
        q.add_argument('--dry-run',action='store_true')
        if name=='serve':q.add_argument('--interval',type=float,default=.2)
    a=p.parse_args()
    if a.command=='seal':
        if Path(a.output).exists():raise FileExistsError('Never replace an existing permission semantic seal')
        result=seal_semantics(read(a.baseline_policy),read(a.anchor_lease),a.legacy_helper);atomic(a.output,result);print(json.dumps(result));return
    if a.command=='once':
        result=refresh(a.policy,a.lease,a.semantics,a.template,dry_run=a.dry_run,invalidate_on_refusal=not a.dry_run)
        atomic(a.output,result);print(json.dumps(result));return
    if a.dry_run or not .02<=a.interval<=10:raise ValueError('Finite CPU owner interval only; use once for dry-run')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(a.lease)+'.semantic_owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                result=refresh(a.policy,a.lease,a.semantics,a.template,invalidate_on_refusal=True)
                result['owner_pid']=os.getpid();atomic(output,result)
            except Exception as error:
                atomic(output,dict(status='REFUSED_NO_AUTH_REFRESH',at_unix=time.time(),owner_pid=os.getpid(),error=repr(error)))
                raise
            time.sleep(a.interval)


if __name__=='__main__':main()
