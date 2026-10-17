"""Fixed independent reissue after revoked shallow-copy approval; no dispatch."""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import time

HERE=Path(__file__).resolve().parent
PROPOSAL=Path('/mnt/project/st-world-policy-20261016/resources/history32_actual_20261017T2345/ADMISSION_W_COMPAT_PROPOSAL.json')
PROPOSAL_SHA='debf89c03c09a82f48613bbb59fb122c6de4a28a6b27a5a6e20cb8a5f871e8c8'
WRAPPER_SHA='7bf6934f0efcf0abc21c5f8be5cfa08e6b210ecde1aa3f648bc8a91534df8a22'
REVIEWER='/root/temporal_narrow_review'
STATUS='APPROVED_HISTORY32_PILOT_ONLY'

def digest(raw):return hashlib.sha256(raw).hexdigest()
def fresh_proposal():
    raw=PROPOSAL.read_bytes()
    if digest(raw)!=PROPOSAL_SHA:raise ValueError('Proposal content changed')
    return json.loads(raw)

def authorized_transform(proposal):
    result=deepcopy(proposal)
    result.update(status=STATUS,reviewer=REVIEWER)
    return result

def verify_transform(proposal,signed):
    if signed != authorized_transform(proposal):
        raise ValueError('Signed output differs from verified proposal beyond status/reviewer')
    if signed['controller_sources'][-1]['sha256'] != WRAPPER_SHA:
        raise ValueError('Signed wrapper pin is not the reviewed current wrapper')

def verify_dependencies(signed):
    path=Path(signed['controller_sources'][-1]['path'])
    if digest(path.read_bytes())!=WRAPPER_SHA:raise ValueError('Actual wrapper source changed')
    spec=importlib.util.spec_from_file_location('verified_history32_wrapper_v2',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    checked=[]
    pins=[*signed['controller_sources'],*signed['w_source_compatibility'].values(),
          *[{'path':str(m.RUNTIME/name),'sha256':sha} for name,sha in {**m.PINS,**m.BUILDER_PINS}.items()]]
    for pin in pins:
        m.pinned(pin,decode=False);checked.append(deepcopy(pin))
    cp=m.pinned(signed['checkpoint_receipt']);m.pinned(signed['policy'])
    reg=m.pinned(signed['registration']);m.validate_registration(reg,signed,[2,3])
    m.validate_binding(signed,cp,now=time.time())
    selected=m.pinned({'path':str(m.SELECTED),'sha256':m.SELECTED_SHA})
    if len(selected['scenes'])!=32:raise ValueError('Wrong fixed population')
    ctl=Path(signed['control_root'])
    if (not ctl.is_dir() or list(ctl.iterdir()) or any(p.is_symlink() for p in (ctl,*ctl.parents))
            or Path(signed['output_root']).exists()):raise ValueError('Control/output already used or unsafe')
    v=os.statvfs('/mnt/project');free=v.f_bavail*v.f_frsize
    if free<=cp['remaining_bytes']+(1024+768)*(1<<30)+(32<<20):raise ValueError('Storage floor changed')
    return dict(checked_pins=checked,current_checkpoint=signed['checkpoint_receipt'],free_bytes=free,checked_unix=time.time())

def atomic_new(path,obj):
    raw=(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode();tmp=path.with_suffix('.new.tmp')
    with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.link(tmp,path);tmp.unlink()
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return digest(raw)

def main():
    os.sched_setaffinity(0,{8,9})
    original=HERE/'ACTUAL_ADMISSION.json';oldraw=original.read_bytes()
    if digest(oldraw)!='22248e44f2dffd5a171be4a2b1314cdd9ad0317ac0e598b259588b5cde4c224d':raise ValueError('Preserved v1 evidence changed')
    revoke=dict(status='REVOKED_DO_NOT_USE',artifact={'path':str(original),'sha256':digest(oldraw)},reviewer=REVIEWER,reason='Reviewer signing script used dict(a) shallow copy; modifying nested controller_sources for comparison mutated signed a after validation, replacing wrapper7bf6934 with old14f70bab.',builder_started=False,detected_by='Lead pre-dispatch review',original_artifact_preserved=True,replaces_approval_review='APPROVAL_REVIEW.json',recorded_unix=time.time())
    atomic_new(HERE/'REVOCATION_V1.json',revoke)
    proposal=fresh_proposal();signed=authorized_transform(proposal)
    verify_transform(fresh_proposal(),signed)
    before=verify_dependencies(signed)
    verify_transform(fresh_proposal(),signed)
    target=HERE/'ACTUAL_ADMISSION_V2.json';sha=atomic_new(target,signed)
    actual=json.loads(target.read_bytes())
    verify_transform(fresh_proposal(),actual)
    after=verify_dependencies(actual)
    report=dict(status='PASS',reviewer=REVIEWER,proposal={'path':str(PROPOSAL),'sha256':PROPOSAL_SHA},actual_admission={'path':str(target),'sha256':sha},wrapper_sha256=WRAPPER_SHA,changes_from_disk_proposal=['status','reviewer'],prewrite_validation=before,postwrite_validation=after,deepcopy_used=True,checked_output_equals_fresh_disk_proposal_except_two_fields=True,v1_revoked=True,builder_started=False,ledger_accessed=False,GPU=False,scientific_result=False,scope='ONE_EXACT32_METADATA_ATTEMPT_ONLY')
    atomic_new(HERE/'APPROVAL_REVIEW_V2.json',report)
    print(json.dumps({'admission_path':str(target),'admission_sha256':sha,'wrapper_sha256':actual['controller_sources'][-1]['sha256']}))

if __name__=='__main__':main()
