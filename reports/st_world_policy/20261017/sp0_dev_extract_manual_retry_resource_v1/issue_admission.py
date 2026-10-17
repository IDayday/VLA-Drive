"""Issue the single fixed retry after independently reported correctness PASS."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
B=Path('/mnt/project/st-world-policy-20261016');J=B/'audits/spatial_probe_actual_SP0_5000_v1'
H=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1')
pin=lambda p:{'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
sys.path.insert(0,str(H))
spec=importlib.util.spec_from_file_location('reviewed_manual_retry',H/'supervise_once.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old=HERE.parent/'sp0_dev_extract_admission_v1/DEV_EXTRACT_ADMISSION.json'
a=json.loads(old.read_text())
helpers={name:pin(H/name)['sha256'] for name in m.HELPERS}
assert helpers=={'supervise_once.py':'747873b72ce6bcb2cfd3163bd89ce1560e4538d1069952d8deed176c349a67ef','observed_cli.py':'c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643','prepare_bindings.py':'31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84'}
proposal=m.read(H.parent/'sp0_dev_extract_manual_retry_v1/METADATA_PREPARATION.json')[0]
cp_path=HERE.parent/'indexed_profile64_training_admission_v1/CHECKPOINT_COMMITMENTS_2183G_V1.json'
cp,cppin=m.read(cp_path,'d5bce1fec28d6ce3dab496287c9be41852ca154495ecf4cf686c7e780ebf4d52')
assert cp['remaining_bytes']==2183<<30
now=time.time();until=min(now+3600,cp['valid_until_unix']);assert until-now>1000
a.update(helpers=helpers,checkpoint_commitments=cppin,issued_at_unix=now,valid_until_unix=until,latest_start_for_40s_cleanup_unix=until-1000,**proposal['admission_fields_to_add_after_review'])
a['limits'].update(plan_consumed_read_bytes=37533999575,failed_extract_consumed_read_bytes=30632210276,prior_consumed_read_bytes=68166209851,remaining_extract_read_bytes=206711697093)
a['prior_admission_preserved']=pin(old)
a['independent_correctness']={'status':'PASS_REPORTED_BY_LEAD','source_commit_prefix':'4a9ef2be','owner_cpu_tests_passed':13,'independent_cpu_tests_passed':12,'scope':'Fixed manual retry and cumulative read accounting; exact supervisor helper pinned separately'}
a['allowed_attempt_directory']=str(m.DEV_RETRY_DIRECTORY)
a['excluded']=['additional retry','new dev plan','train plan','train extraction','fit','formal temporal training','additional cache members','automatic retry']
prep=m.read(a['preparation']['path'],a['preparation']['sha256'])[0]
assert m.validate(a,prep,'dev_extract',now)==(960,64<<30)
attempt,accounting,_=m.stage_inputs(a,'dev_extract')
assert not attempt.exists() and accounting['remaining_read_bytes']==206711697093
ledgerpin=pin(B/'resources/cache_storage_ledger_v1/CLAIMS.json')
assert ledgerpin==a['observed_current_ledger']
m.fresh_gpu(a['gpu'])  # Hardware/owner query only; no lock, CUDA context or dispatch.
free=os.statvfs(B).f_bavail*os.statvfs(B).f_frsize
assert free>cp['remaining_bytes']+(1<<40)+26085776+(128<<20)
target=HERE/'ADMISSION.json';assert not target.exists()
target.write_text(json.dumps(a,sort_keys=True,indent=2)+'\n')
result={'status':'PASS_ONE_EXPLICIT_DEV_EXTRACT_RETRY_RESOURCE_ADMISSION','admission':pin(target),'checkpoint_commitments':cppin,'helpers':helpers,'actual_free_bytes':free,'actual_stage_inputs':accounting,'actual_validate':[960,64<<30],'actual_gpu_fresh_check':'PASS','new_attempt_created':False,'existing_W_claim_reused':True,'model_or_tensor_reads':False,'dispatch':False,'ledger_write':False,'valid_until_unix':until,'latest_start_unix':until-1000}
(HERE/'ADMISSION_EVIDENCE.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(result,sort_keys=True,indent=2))
