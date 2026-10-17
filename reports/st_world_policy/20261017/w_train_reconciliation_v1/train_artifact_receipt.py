"""Exact SP0 TRAIN64 artifact reconciliation; no relaunch, tensor reads or resource approval."""
import argparse
import hashlib
import json
from pathlib import Path
import stat

SCHEMA='st_SP0_train_w_reconciliation_v1'
REVIEW_SCHEMA='st_SP0_train_w_artifact_reconciliation_review_v1'
REVIEW_SHA='725d3427d042f47882e89a5814937b10e3a7c49b5ea58da690e374c7b37dbfad'
FAILED_SHA='4d24e4ed9a97744fe0d977c21284cb5fd64ec14c1348d71f6edadeff61245b27'
REASON='OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
JOB=Path('/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1')
FAILED_PATH=JOB/'stages/train_extract/RESULT.json'
TRAIN_ROOT=Path('/mnt/project/st-world-policy-20261016/caches/learned_w_probe_SP0_5000_v1/train')
PREPARATION_PIN={'path':str(JOB/'controls_581399eb_v2_metadata_only/PREPARATION.json'),
    'sha256':'34c174a6aa0c859a1b0a5f8b7dd800bbe189d51c770211b49c35d061e92b6431'}


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pin_shape(pin):
    if (not isinstance(pin,dict) or set(pin)!={'path','sha256'} or not Path(pin['path']).is_absolute()
            or not isinstance(pin['sha256'],str) or len(pin['sha256'])!=64
            or any(x not in '0123456789abcdef' for x in pin['sha256'])):
        raise ValueError('Exact bounded reconciliation pin required')


def file_version(path):
    path=Path(path)
    for p in (path,*path.parents):
        if p.is_symlink():raise ValueError('Symlink artifact forbidden')
    s=path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or not 0<s.st_size<=2<<20:
        raise ValueError('Original small regular W payload required; no payload read occurs')
    return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,
            'ctime_ns':s.st_ctime_ns,'nlink':s.st_nlink}


def evidence(review_pin,failed_pin,preparation_pin,cache_root,read):
    """Reuse one fixed independent validation. Recheck metadata and payload stat only."""
    for pin in (review_pin,failed_pin,preparation_pin):pin_shape(pin)
    if (review_pin['sha256']!=REVIEW_SHA or failed_pin!={'path':str(FAILED_PATH),'sha256':FAILED_SHA}
            or preparation_pin!=PREPARATION_PIN or Path(cache_root)!=TRAIN_ROOT):
        raise ValueError('Only the reviewed exact failed attempt/artifact population can reconcile')
    review,_=read(review_pin['path'],review_pin['sha256'])
    failed,_=read(failed_pin['path'],failed_pin['sha256'])
    cache_root=Path(cache_root)
    if (review.get('schema')!=REVIEW_SCHEMA or review.get('status')!='COMMITTED_ARTIFACTS_INDEPENDENTLY_VALIDATED'
            or review.get('original_stage_status_unchanged')!='STOPPED_NO_AUTO_RETRY'
            or review.get('original_supervisor_reason')!=REASON
            or review.get('cache_validity_does_not_rewrite_failed_launch') is not True
            or review.get('matching_old_session_processes_at_review')!=[]
            or review.get('gpu_initialized') is not False or review.get('launch_approved') is not False
            or review.get('downstream_transition_approved') is not False or review.get('automatic_retry') is not False):
        raise ValueError('Independent review is incomplete or grants an unsupported transition')
    res=failed.get('resources',{})
    cleanup=res.get('process_cleanup',{})
    if (failed.get('schema')!='st_SP0_probe_stage_result_v1' or failed.get('stage')!='train_extract'
            or failed.get('status')!='STOPPED_NO_AUTO_RETRY' or failed.get('preparation')!=preparation_pin
            or type(res.get('exit_code')) is not int or res['exit_code']!=0 or res.get('limit_reason')!=REASON
            or res.get('first_monitor_error') is not None
            or res.get('counters',{}).get('limit_exceeded') is not None
            or type(cleanup.get('live_owned_remaining')) is not int or cleanup['live_owned_remaining']!=0
            or type(cleanup.get('verified_processes')) is not int or cleanup['verified_processes']<1
            or review.get('compute_resources')!=res):
        raise ValueError('Only completed computation with verified zero remaining cleanup is reconcilable')
    checked={}
    for path,pin in review['input_pins_unchanged'].items():
        value,actual=read(path,pin['sha256'])
        checked[path]=(value,actual)
    if checked.get(failed_pin['path'],(None,None))[1]!=failed_pin:
        raise ValueError('Failed attempt absent from independent reviewed inputs')
    prep=checked[preparation_pin['path']][0]
    signature,sig_pin=checked[str(cache_root/'identity.json')]
    complete,complete_pin=checked[str(cache_root/'COMPLETE.json')]
    cli,cli_pin=checked[str(Path(failed_pin['path']).parent/'CLI_RESULT.json')]
    if (prep['caches']['train']!={'root':str(cache_root),'identity':review['signature_identity']}
            or signature['identity']!=review['signature_identity'] or complete!=cli or cli!=failed['cli_result']
            or complete!={'completed':64,'requested':64,'identity':signature['identity'],
                'scientific_result':False,'status':'COMPLETE'}
            or signature.get('targets_read') is not False or signature.get('schema')!='st_frozen_spatial_w_v1'
            or signature.get('precision')!='FP32_native_master_TF32_off'
            or signature['binding']['checkpoint_identity']!=review['checkpoint_identity']
            or signature['binding']['checkpoint_tag']!=review['checkpoint_tag']):
        raise ValueError('Successful CLI, COMPLETE, exact cache and independent review disagree')
    rows=signature['rows'];records=review.get('records',[])
    if review.get('scenes')!=64 or len(records)!=64 or len(rows)!=64:
        raise ValueError('Incomplete independently validated cache population')
    snapshots=[]
    for row,r in zip(rows,records):
        if (r.get('token')!=row['token'] or r.get('log')!=row['log'] or r.get('shape')!=[144,2048]
                or r.get('mask_shape')!=[144] or r.get('dtype')!='torch.float32'
                or r.get('finite') is not True or r.get('mask_all_true') is not True
                or r.get('feature_elements')!=294912 or r.get('mask_elements')!=144):
            raise ValueError('Missing full-record original-validator evidence')
        receipt,receipt_pin=checked[str(cache_root/(row['token']+'.json'))]
        if receipt!={'identity':signature['identity'],'token':row['token'],'log':row['log'],'sha256':r['payload_sha256']}:
            raise ValueError('Reviewed record receipt changed')
        payload=cache_root/(row['token']+'.safetensors')
        snapshot=file_version(payload)
        if (r.get('receipt')!=receipt_pin or r.get('payload_path')!=str(payload)
                or r.get('file_version')!=snapshot):
            raise ValueError('Payload/receipt changed since independent original-validator review')
        snapshots.append({'token':row['token'],'receipt':receipt_pin,'payload_sha256':r['payload_sha256'],
            'path':str(payload),'stat_after_independent_review':snapshot})
    if len({r['token'] for r in snapshots})!=64:raise ValueError('Duplicate reviewed scene')
    counters=res['counters']
    values=[counters['hash_logical_bytes_admitted'],counters['torch_load_nominal_bytes'],counters['proc_self_io_delta']['rchar']]
    if any(type(v) is not int or v<0 for v in values):raise ValueError('Incomplete original resource counters')
    charged=max(values[0]+values[1],values[2]);account=failed['read_accounting']
    if (review['extract_charged_read_bytes']!=charged or review['prior_role_read_bytes']!=account['prior_read_bytes']
            or review['remaining_role_read_bytes_before_review']!=account['role_limit_bytes']-account['prior_read_bytes']-charged
            or account['role_limit_bytes']!=256<<30):
        raise ValueError('Original consumed role budget changed')
    return {'schema':SCHEMA,'status':'ARTIFACTS_ACCEPTED_SUPERVISOR_FAILED','eligible_consumer':'fit',
        'supervisor_status':'STOPPED_NO_AUTO_RETRY','supervisor_reason':REASON,
        'original_failed_result':failed_pin,'successful_cli':cli_pin,'cache_complete':complete_pin,
        'cache_signature':sig_pin,'cache_identity':signature['identity'],'preparation':preparation_pin,
        'independent_review':review_pin,'payload_snapshots':snapshots,
        'upstream_accounting':{'prior_role_read_bytes':account['prior_read_bytes'],'extract_charged_read_bytes':charged,
            'role_limit_bytes':account['role_limit_bytes'],'review_payload_hash_plus_mmap_nominal_bytes':review['payload_hash_plus_mmap_nominal_bytes'],
            'remaining_role_read_bytes_upper_bound_before_small_metadata':review['remaining_role_read_bytes_less_review_payload_upper'],
            'attempt_count_preserved':1,'additional_extraction_attempts':0},
        'launch_approved':False,'automatic_retry':False,'new_resource_authorization':False,
        'scientific_result':False,'payload_content_revalidated_here':False,
        'later_fit_still_requires_original_read_world_hash_validation':True}


def create_receipt(review_pin,failed_pin,preparation_pin,cache_root,read):
    body=evidence(review_pin,failed_pin,preparation_pin,cache_root,read)
    return {**body,'identity':identity(body)}


def validate_receipt(receipt_pin,*,failed_pin,preparation_pin,cache_root,read):
    pin_shape(receipt_pin)
    receipt,_=read(receipt_pin['path'],receipt_pin['sha256'])
    if not isinstance(receipt,dict) or receipt.get('identity')!=identity({k:v for k,v in receipt.items() if k!='identity'}):
        raise ValueError('Reconciliation identity changed')
    expected=create_receipt(receipt['independent_review'],failed_pin,preparation_pin,cache_root,read)
    if receipt!=expected:raise ValueError('Reconciliation evidence/status/files changed')
    return receipt


def main():
    from supervise_fit import load_original
    load_original()
    from prepare_bindings import JOB,ROOTS,read,write_once
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--independent-review',required=True);p.add_argument('--review-sha256',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    failed_pin={'path':str(JOB/'stages/train_extract/RESULT.json'),'sha256':FAILED_SHA}
    failed,_=read(failed_pin['path'],failed_pin['sha256'])
    receipt=create_receipt({'path':str(Path(args.independent_review).resolve()),'sha256':args.review_sha256},
        failed_pin,failed['preparation'],ROOTS['train'],read)
    output=write_once(Path(args.output).resolve(),receipt)
    print(json.dumps({'status':receipt['status'],'receipt':output,'launch_approved':False,'payload_read':False}))


if __name__=='__main__':main()
