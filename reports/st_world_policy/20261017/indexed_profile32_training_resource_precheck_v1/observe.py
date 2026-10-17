"""Read-only, narrow local PROFILE resource snapshot; grants no admission."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

HERE=Path(__file__).resolve().parent
BASE=Path('/mnt/project/st-world-policy-20261016')
PARENT=BASE/'smoke/world_indexed_provider_native2_v1'
FACTORY=PARENT/'profile32_v1'
OUTPUT=BASE/'profile/world_indexed_provider_profile32_updates64_v1'
def small(path,expected=None):
    path=Path(path);assert not path.is_symlink() and path.stat().st_size<1<<20
    raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    assert expected is None or sha==expected
    return json.loads(raw),{'path':str(path),'sha256':sha,'bytes':len(raw)}
receipt,rpin=small(FACTORY/'preparation_control_v1/RESULT.json','6407b37cd7f457c57f60717c71ef47bcedc9e2d40c090e51239cdc2fcf8be6d1')
assert receipt['status']=='COMPLETE' and receipt['resources']['exit_code']==0 and receipt['new_teacher_tensor_bytes']==0
config,configpin=small(FACTORY/'TRAINING_CONFIG.json')
cp,cppin=small(BASE/'audits/spatial_probe_actual_SP0_5000_v1/storage_preparation_dev_v1/CHECKPOINT_COMMITMENTS_2103G_V1.json','60606edc286a0c34016842a03f0de08f2422936e347364c7373defe9efcc5259')
policy,polpin=small('/mnt/project/server_dispatch_policy.json')
auth,authpin=small(BASE/'resources/AUTHORIZATION_20261016T203442.json')
host=socket.gethostname()
assert host in policy['allowed_hosts']['training-vla-zt']['aliases'] and host in policy['allowed_aliases']
assert 'training-vla-zt' in auth['allowed_hosts']
for key,value in policy.get('global_dispatch_prohibitions',{}).items():
    if key in (host,'training-vla-zt') or host in value.get('aliases',[]):
        assert value.get('new_task_dispatch_allowed') is True
query=lambda args:subprocess.check_output(['nvidia-smi',*args],text=True,timeout=10).strip()
gpu=query(['-i','0','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
gpu_process=query(['-i','0','--query-compute-apps=pid,process_name,used_memory,gpu_uuid','--format=csv,noheader,nounits'])
all_gpus=query(['--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
lock=Path('/tmp/foresight_gpu_0.lock')
lock_observation={'path':str(lock),'exists':lock.exists(),'acquired_by_reviewer':False,'held_rows':[]}
if lock.exists():
    st=lock.stat(); lock_observation['inode']=st.st_ino
    for row in Path('/proc/locks').read_text().splitlines():
        fields=row.split()
        for item in fields:
            parts=item.split(':')
            if len(parts)==3:
                try:matches=(int(parts[0],16),int(parts[1],16),int(parts[2]))==(os.major(st.st_dev),os.minor(st.st_dev),st.st_ino)
                except ValueError:continue
                if matches:lock_observation['held_rows'].append(row)
paths=[PARENT,*PARENT.rglob('*')]
assert len(paths)<256 and not any(p.is_symlink() for p in paths)
charged=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in paths)
disk=os.statvfs(BASE);free=disk.f_bavail*disk.f_frsize
mem={line.split(':')[0]:int(line.split()[1])<<10 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
now=time.time();total=(2183<<30)+(1<<40)
result={'status':'PRECHECK_ONLY_NOT_TRAINING_ADMISSION','observed_at_unix':now,'host':host,'factory':rpin,'factory_actual_resources':receipt['resources'],'factory_parent_charged_bytes_after_receipt':charged,'factory_parent_cap_bytes':32<<20,'config':configpin,'execution':config['execution'],'runtime':config['runtime'],'authorization':authpin,'dispatch_policy':polpin,'host_allowed_at_observation':True,'output_root':str(OUTPUT),'output_root_exists':OUTPUT.exists(),'checkpoint_commitments':cppin,'old_full_commitments_bytes':2103<<30,'new_profile_reservation_bytes':80<<30,'combined_full_commitments_bytes':2183<<30,'additional_free_floor_bytes':1<<40,'required_free_bytes':total,'observed_free_bytes':free,'free_above_requirement_bytes':free-total,'quota_verified':False,'prior_commitment_expires_unix':cp['valid_until_unix'],'prior_commitment_seconds_remaining':cp['valid_until_unix']-now,'gpu0':gpu,'gpu0_compute_processes':gpu_process,'gpu0_lock':lock_observation,'other_gpu_snapshot':all_gpus,'memory':mem,'loadavg':Path('/proc/loadavg').read_text().strip(),'held_boundaries':['No training admission before exact launcher review','Refresh storage commitments through full 1500s run plus save/cleanup window','No creation of output/claim/ledger','Protect all existing GPU1-4 users','Do not reuse old hard-coded GPU1 lock/1020s timeout'],'reviewer_dispatch':False,'reviewer_gpu_compute':False,'reviewer_tensor_reads':False,'reviewer_ledger_writes':False}
(HERE/'OBSERVATION.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(result,sort_keys=True,indent=2))
