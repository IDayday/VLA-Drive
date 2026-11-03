"""Validate an AUG full-result event and notify the existing report watcher.

One-shot CPU callback only: no polling, model/scorer imports, external messages,
training mutation, or evaluation-state writes. The Eval owner retains notified_sha.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


ROOT=Path('/mnt/project/d46-priority-rl-20261103')
REQUEST=Path('/mnt/project/qwen-world-rl-optimization-20261030/reporting/refresh_request.json')
PARENT='386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
ALGORITHM='d46_augustus_native_v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sealed(value):
    body=dict(value);digest=body.pop('sha256',None)
    return bool(digest) and fingerprint(body)==digest


def validated_point(root,case):
    """Read compact seals/identities, never scene rows or optimizer tensors."""
    root=Path(root).resolve()
    if case.get('status')!='COMPLETE' or case.get('world')!=32:
        raise ValueError('Only a COMPLETE world32 evaluation event may publish')
    b=case['binding'];exp=b['experiment'];cp=Path(b['checkpoint']).resolve()
    version='_v2'if exp in('AUG3_FULL_ALIGNED','AUG4_LORA_ALIGNED')else'_v1'
    if (exp not in('AUG1_FULL','AUG2_LORA','AUG3_FULL_ALIGNED','AUG4_LORA_ALIGNED') or cp.parent!=root/'runs'/(exp+version)/'checkpoints'
            or b.get('parent_model_sha256')!=PARENT or b.get('algorithm_schema')!=ALGORITHM):
        raise ValueError('Foreign AUG experiment/parent/checkpoint binding')
    native=read(cp/'identity.json');files=read(cp/'files.json');seal=read(cp/'COMPLETE')
    if (not sealed(native) or native.get('qualification')is not False
            or native.get('algorithm_schema')!=ALGORITHM or native.get('experiment_id')!=exp
            or native.get('parent_model_sha256')!=PARENT or native.get('sha256')!=b['identity_sha256']
            or files.get('identity.json')!=hashlib.sha256((cp/'identity.json').read_bytes()).hexdigest()
            or seal.get('files_sha256')!=hashlib.sha256((cp/'files.json').read_bytes()).hexdigest()
            or seal.get('files_sha256')!=b['inventory_sha256'] or files.get('optimizer.pt')!=b['payload_sha256']
            or seal.get('updates')!=b['updates'] or seal.get('phase')not in('READY','COMPLETE')
            or seal.get('next_inner')!=0):
        raise ValueError('Native AUG seal/parent/qualification binding failed')
    output=Path(case['output']).resolve();expected=root/'evaluations'/case['id']/'evaluation'
    if output!=expected or Path(case['summary']).resolve()!=output.parent/'SUMMARY.json':
        raise ValueError('Summary is outside the registered evaluation case')
    identity=read(output/'identity.json');summary=read(case['summary'])
    signature=fingerprint(dict(parent=PARENT,trainable_payload=files['optimizer.pt'],scope=native['scope']))
    if (not sealed(identity) or identity.get('checkpoint')!=str(cp)
            or identity.get('D44_training_identity')!=native['sha256']
            or identity.get('model_sha256')!=signature or b.get('policy_signature')!=signature
            or identity.get('updates')!=b['updates'] or identity.get('cohort')!='NAVTEST'
            or identity.get('qualification')is not False or identity.get('seed')!=2101
            or identity.get('count')!=12146 or identity.get('inference')!='single_candidate_cleanx_K10_shift5'
            or not sealed(summary) or summary.get('status')!='COMPLETE'
            or summary.get('scenes')!=12146 or summary.get('scene_denominator')!=12146
            or summary.get('physical_logs')!=44 or summary.get('invalid_scenes')!=0
            or summary.get('qualification')is not False or summary.get('seed')!=2101
            or summary.get('cohort')!='NAVTEST' or summary.get('architecture_id')!='D'
            or summary.get('evaluation')!=str(output)
            or summary.get('evaluation_identity_sha256')!=identity['sha256']
            or summary.get('model_sha256')!=signature
            or summary.get('protocol_sha256')!=identity.get('protocol',{}).get('sha256')
            or summary.get('token_sha256')!=identity.get('token_sha256')):
        raise ValueError('Full12146/44/invalid0 official summary/identity binding failed')
    score=summary.get('PDMS');raw=summary.get('means',{}).get('score')
    if (not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=100
            or not isinstance(raw,(int,float)) or not math.isfinite(raw) or abs(score-100*raw)>1e-8):
        raise ValueError('Invalid full PDMS units/value')
    return dict(id=case['id'],experiment=exp,updates=b['updates'],PDMS=score,
        checkpoint=str(cp),parent_model_sha256=PARENT,summary=case['summary'],summary_sha256=summary['sha256'])


def completed_points(root=ROOT,state_path=None):
    path=Path(state_path)if state_path else Path(root)/'operations/AUGUSTUS_CAMPAIGN_20261103/EVAL_STATE.json'
    if not path.is_file():return []
    state=read(path)
    if state.get('schema')!='d46_augustus_official_evaluation_queue_v1':return []
    points=[]
    for case in state.get('cases',{}).values():
        if case.get('status')!='COMPLETE':continue
        try:points.append(validated_point(root,case))
        except(OSError,ValueError,KeyError,TypeError):continue
    return points


def notify(state_path,case_id,expected_summary_sha,*,root=ROOT,request_path=REQUEST):
    state=read(state_path)
    if state.get('schema')!='d46_augustus_official_evaluation_queue_v1':raise ValueError('Foreign evaluation state')
    point=validated_point(root,state['cases'][case_id])
    if point['summary_sha256']!=expected_summary_sha:raise ValueError('Complete-event summary SHA changed')
    path=Path(request_path);event_id='AUG-full-'+case_id+'-'+point['summary_sha256']
    if path.is_file() and read(path).get('id')==event_id:
        return dict(status='ALREADY_NOTIFIED',case=case_id,notified_sha=point['summary_sha256'])
    payload=dict(id=event_id,at=time.time(),scientific_result=False,event='evaluation_complete',
        reason='Validated full AUG NAVTEST result; render all current results and best checkpoints now',
        evidence=point['summary'],validated_point=point)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.complete-event-'+str(os.getpid()))
    temporary.write_text(json.dumps(payload,ensure_ascii=True,indent=2)+'\n');os.replace(temporary,path)
    return dict(status='NOTIFIED',case=case_id,notified_sha=point['summary_sha256'],request=str(path))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--case',required=True)
    p.add_argument('--expected-summary-sha',required=True);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--request',type=Path,default=REQUEST)
    a=p.parse_args()
    print(json.dumps(notify(a.state,a.case,a.expected_summary_sha,root=a.root,request_path=a.request),sort_keys=True))


if __name__=='__main__':main()
