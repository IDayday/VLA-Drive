"""Bounded retry resource precheck; deliberately does not issue an admission."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
B=Path('/mnt/project/st-world-policy-20261016')
J=B/'audits/spatial_probe_actual_SP0_5000_v1'
H=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1')
sys.path.insert(0,str(H))
spec=importlib.util.spec_from_file_location('pending_retry_supervisor',H/'supervise_once.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
pins=[]
def read(path,expected=None):
    p=Path(path);assert not p.is_symlink() and p.stat().st_size<1<<20
    raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert expected is None or expected==sha
    pin={'path':str(p),'sha256':sha};pins.append(pin);return json.loads(raw),pin
prep,pp=read(J/'controls_581399eb_v2_metadata_only/PREPARATION.json','34c174a6aa0c859a1b0a5f8b7dd800bbe189d51c770211b49c35d061e92b6431')
proposal,_=read(H.parent/'sp0_dev_extract_manual_retry_v1/METADATA_PREPARATION.json')
attempt,accounting,previous=m.stage_inputs({'preparation':pp,**proposal['admission_fields_to_add_after_review']},'dev_extract')
assert not attempt.exists() and accounting['prior_read_bytes']==68166209851 and accounting['remaining_read_bytes']==206711697093
policy,polpin=read(J/'storage_preparation_dev_v1/PROPOSED_POLICY_V8_SP0_DEV_W.json','0bf0dc467a83a378e02ab4c52e00c7f2836011985edc5bf76d4fb2135957f191')
ledger,ledgerpin=read(B/'resources/cache_storage_ledger_v1/CLAIMS.json')
member=next(iter(policy['frozen_w_package']['members'].values()))
assert len(policy['frozen_w_package']['members'])==1
amount=sum(member[k] for k in ('output_bytes','staging_bytes','metadata_bytes'));assert amount==26085776
assert member['root']==prep['caches']['dev']['root']
cp,cppin=read(HERE.parent/'indexed_profile64_training_admission_v1/CHECKPOINT_COMMITMENTS_2183G_V1.json','d5bce1fec28d6ce3dab496287c9be41852ca154495ecf4cf686c7e780ebf4d52')
assert cp['remaining_bytes']==2183<<30 and cp['valid_until_unix']>time.time()+1000
stat=os.statvfs(B);free=stat.f_bavail*stat.f_frsize;assert free>(2183<<30)+(1<<40)+amount+(128<<20)
paths=[];stack=[J]
while stack:
    p=stack.pop();assert not p.is_symlink();paths.append(p)
    assert len(paths)<256
    if p.is_dir():stack.extend(p.iterdir())
charge=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in paths)
assert charge+(5<<20)<32<<20
proc_locks=Path('/proc/locks').read_text().splitlines()
locks=[]
for p in (B/'resources/SPATIAL_PROBE_ONE_LOADER.lock',Path('/tmp/foresight_gpu_5.lock')):
    item={'path':str(p),'exists':p.exists(),'held_rows':[]}
    if p.exists():
        st=p.stat()
        for line in proc_locks:
            for token in line.split():
                bits=token.split(':')
                if len(bits)!=3:continue
                try:found=(int(bits[0],16),int(bits[1],16),int(bits[2]))==(os.major(st.st_dev),os.minor(st.st_dev),st.st_ino)
                except ValueError:continue
                if found:item['held_rows'].append(line)
    locks.append(item)
gpu=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
apps=subprocess.check_output(['nvidia-smi','-i','5','--query-compute-apps=pid,process_name,used_memory,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
helpers={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [H/'supervise_once.py',H/'observed_cli.py',H/'prepare_bindings.py']}
assert helpers['supervise_once.py']=='747873b72ce6bcb2cfd3163bd89ce1560e4538d1069952d8deed176c349a67ef'
assert helpers['observed_cli.py']=='c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643'
assert helpers['prepare_bindings.py']=='31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84'
result={'status':'PENDING_INDEPENDENT_CORRECTNESS_NO_ADMISSION','observed_at_unix':time.time(),'pins':pins,'helpers':helpers,'read_accounting':accounting,'attempt':str(attempt),'attempt_exists':False,'gpu5':gpu,'gpu5_compute_processes':apps,'lock_observations':locks,'actual_free_bytes':free,'checkpoint_commitments':cppin,'metadata_job_root_charged_bytes':charge,'metadata_job_root_paths':len(paths),'retry_control_maximum_bytes':5<<20,'existing_metadata_control_reservation_bytes':32<<20,'effective_campaign_metadata_reserved_mib':1264,'typed_W_bytes':amount,'new_typed_W_claim_required':False,'existing_W_root_exists':Path(member['root']).exists(),'live_ledger':ledgerpin,'live_ledger_rows':len(ledger['claims']),'runtime_unchanged':prep['runtime'],'dispatch':False,'ledger_write':False,'model_or_image_or_tensor_reads':False,'new_plan':False,'notes':['New 2183GiB commitment includes running PROFILE GPU0 80GiB','Original failed attempt remains spent and unmodified','Only exact fixed retry directory may be newly created by Lead after approval','No train plan, train extraction, fit or additional claim is authorized']}
(HERE/'PRECHECK.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','gpu5','lock_observations','actual_free_bytes','metadata_job_root_charged_bytes','existing_W_root_exists','live_ledger','live_ledger_rows','read_accounting')},sort_keys=True,indent=2))
