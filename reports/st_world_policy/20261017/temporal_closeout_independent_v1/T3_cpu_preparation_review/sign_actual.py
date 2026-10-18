"""Sign only the exact reviewed T3 proposal; never dispatch or touch a ledger."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import time

HERE=Path(__file__).resolve().parent
PROPOSAL=Path('/mnt/project/st-world-policy-20261016/resources/T3_cpu_preparation_actual_20261018T0020/ADMISSION_CURRENT_PROPOSAL.json')
PROPOSAL_SHA='b7e021726069d66c222d94e31a74cbb34adceb97c7cc8cac285c208a9b9f1ce5'
AUTHOR=Path('/mnt/project/VLA-Drive-st-teacher-channel-control-20261017/reports/st_world_policy/20261017/temporal_T3_cpu_preparation_v1')
CP=Path('/mnt/project/st-world-policy-20261016/resources/history32_actual_20261017T2345/CHECKPOINT_COMMITMENTS_2703G.json')
CP_SHA='015e7f6dddf7b1cab6823f88310db4297ed9ffd83653e9b6fab8cc666cc82856'
REVIEWER='/root/temporal_narrow_review'
STATUS='APPROVED_T3_METADATA_PREPARATION_ONLY'

def raw(path):
    p=Path(path);assert p.stat().st_size<=4<<20
    return p.read_bytes()
def sha(path):return hashlib.sha256(raw(path)).hexdigest()
def readpin(pin):
    b=raw(pin['path']);assert hashlib.sha256(b).hexdigest()==pin['sha256'],pin
    return json.loads(b)
def pin(path):return {'path':str(path),'sha256':sha(path)}
def proposal():return readpin({'path':str(PROPOSAL),'sha256':PROPOSAL_SHA})
def assert_signed_matches(candidate, source):
    assert candidate['status']==STATUS and candidate['reviewer']==REVIEWER
    restored=deepcopy(candidate)
    restored['status']=source['status'];restored['reviewer']=source['reviewer']
    assert restored==source, 'Signed object differs from validated input beyond status/reviewer'
def verify_all(a):
    p=readpin(a['plan'])
    for item in a['controller_sources']:
        assert sha(item['path'])==item['sha256']
    for item in p['source_files'].values():assert sha(item['path'])==item['sha256']
    for item in p['inputs'].values():readpin(item)
    for key in ('prior_evidence','history_control_adoption'):readpin(p['control_reservation'][key])
    assert a['control_reservation']==p['control_reservation']
    assert a['hostname']==socket.gethostname() and {2,3}.issubset(os.sched_getaffinity(0))
    assert a['issued_at_unix']<=time.time()<=a['latest_start_unix']
    assert a['valid_until_unix']>=time.time()+340
    assert not Path(a['output_root']).exists()
    return p
def atomic_new(path, document):
    b=(json.dumps(document,indent=2,sort_keys=True)+'\n').encode()
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.admission-',delete=False) as f:
        temp=Path(f.name);f.write(b);f.flush();os.fsync(f.fileno())
    try:os.link(temp,path)
    finally:temp.unlink()
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def main():
    a=proposal()
    prior=readpin({'path':str(AUTHOR/'ADMISSION_PROPOSAL.json'),'sha256':'99c50b3de157ed22f97c22a8653b04f89f0452d3aac9d046c10676d4afbc5bd4'})
    restored=deepcopy(a)
    for k in ('issued_at_unix','latest_start_unix','valid_until_unix'):restored[k]=prior[k]
    assert restored==prior
    review=readpin({'path':str(HERE/'REVIEW.json'),'sha256':'5cad5b6c50f43c489c6f9f7b8d92b32df4a57898c91831fd27c685680c2719e5'})
    assert review['status']=='PASS' and review['admission_issued'] is False
    p=verify_all(a)
    cp=readpin({'path':str(CP),'sha256':CP_SHA})
    print('CP_KEYS',sorted(cp))
    # The exact CURRENT receipt is independently reviewed; retain all2703GiB.
    assert cp['status']=='CURRENT'
    assert cp['valid_until_unix']>=a['valid_until_unix']
    free=os.statvfs('/mnt/project').f_bavail*os.statvfs('/mnt/project').f_frsize
    required=(2703+1024+768)*(1<<30)+(8<<20)
    assert free>required
    signed=deepcopy(a);signed['status']=STATUS;signed['reviewer']=REVIEWER
    assert_signed_matches(signed,proposal())
    bad=deepcopy(signed);bad['controller_sources'][-1]['sha256']='0'*64
    try:assert_signed_matches(bad,proposal())
    except AssertionError:negative_nested=True
    else:raise AssertionError('Nested mutation was not rejected')
    assert a==proposal(), 'Negative test contaminated input'
    # Exercise the actual bound preflight only: no main/run/bootstrap/CLI calls.
    spec=importlib.util.spec_from_file_location('t3_actual_readonly',AUTHOR/'prepare_once.py')
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    c=w.controller()
    class Reads:
        def raw(self,path,expected):
            b=raw(path);assert hashlib.sha256(b).hexdigest()==expected;return b
        def document(self,ref):return readpin(ref)
    c.validate(signed,p,None,Reads(),time.time())
    assert_signed_matches(signed,proposal());verify_all(signed)
    target=HERE/'ACTUAL_ADMISSION.json';atomic_new(target,signed)
    emitted=json.loads(raw(target))
    assert_signed_matches(emitted,proposal());verify_all(emitted)
    result={'status':'PASS','reviewer':REVIEWER,'signed_admission':pin(target),'source_proposal':pin(PROPOSAL),
      'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'review':pin(HERE/'REVIEW.json'),'checkpoint_commitments':pin(CP),
      'source_to_current_proposal_changes':['issued_at_unix','latest_start_unix','valid_until_unix'],
      'validated_to_signed_changes':['status','reviewer'],
      'post_write_fresh_disk_comparison_and_pins':'PASS',
      'nested_mutation_negative_test':negative_nested,'original_bound_validate':'PASS',
      'shared_free_bytes_observed':free,'separate_retained_reserves_plus_T3_bytes':required,
      'scope':'One CPU-only T3 exact32 metadata preparation; Lead alone executes; no future4-update training approval.',
      'actual_preparation_started':False,'ledger_read_or_write':False,'gpu_or_ssh_access':False}
    atomic_new(HERE/'ACTUAL_ADMISSION_REVIEW.json',result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
