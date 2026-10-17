from pathlib import Path
import hashlib,json,os,resource,signal,stat,time
signal.alarm(60)
os.sched_setaffinity(0,{8,9})
resource.setrlimit(resource.RLIMIT_AS,(256<<20,256<<20))
t0=time.monotonic(); total=0; pins=[]
def read(path,expected=None):
 global total
 p=Path(path); st=p.stat(); assert stat.S_ISREG(st.st_mode) and st.st_size < 1<<20
 with p.open('rb') as f: b=f.read((1<<20)+1)
 assert len(b)==st.st_size and p.stat().st_mtime_ns==st.st_mtime_ns
 total+=len(b);assert total<=4<<20
 h=hashlib.sha256(b).hexdigest();assert expected is None or h==expected
 pin={'path':str(p),'sha256':h,'bytes':len(b)};pins.append(pin)
 return json.loads(b),pin
parent,pp=read('/mnt/project/st-world-policy-20261016/smoke/official_history_T1_native4_v1/controls/CHECKPOINT_COMMITMENTS_2303G_V1.json','13f3f4b282db2aff24528bc5434db51a9351cf43cf02dcdea1fbef38d5434cb3')
prior,priorpin=read(parent['basis']['previous_2183GiB']['path'],parent['basis']['previous_2183GiB']['sha256'])
assert parent['remaining_bytes']==2303*(1<<30) and prior['remaining_bytes']==2183*(1<<30)
assert parent['free_floor_bytes']==1<<40 and parent['new_checkpoint_reservation_bytes']==120*(1<<30)
now=time.time();observations=[]
for i in range(1,5):
 d,pin=read(f'/mnt/project/st-world-policy-20261016/formal/consequence/runs/C{i}_fulltrain_seed42_official_4096_v1/status.json')
 assert d['status']=='RUNNING' and 0<=now-d['updated_unix']<=600 and 0<d['completed']<=4096
 observations.append({'experiment':f'C{i}_fulltrain_seed42_official_4096_v1','status':d['status'],'completed':d['completed'],'registered_endpoint':4096,'identity':d['identity'],'source_sha':d['source_sha'],'updated_unix':d['updated_unix'],'status_age_seconds':now-d['updated_unix'],'evidence':pin})
v=os.statvfs('/mnt/project'); free=v.f_bavail*v.f_frsize
assert free>parent['remaining_bytes']+parent['free_floor_bytes']
receipt={k:parent[k] for k in ('schema','remaining_bytes','free_floor_bytes','new_checkpoint_reservation_bytes','new_checkpoint_root','reservation_kind','quota_verified')}
receipt.update({'status':'CURRENT','checked_at_unix':now,'valid_until_unix':now+21600,'refresh_window_seconds':21600,'parent_receipt':pp,'basis':{'previous_2183GiB':priorpin,'current_shared_C_status':observations},'actual_free_bytes':free,'filesystem_stat_path':'/mnt/project','filesystem_block_size':v.f_frsize,'existing_checkpoint_bytes_not_deleted':True,'cache_ledger_modified':False,'old_profile_reservation_reused':False,'upper_reservation_unchanged':True,'remaining_not_reduced_by_progress':True,'scope':'Refresh unchanged conservative2303GiB upper checkpoint commitment and1TiB free floor; preserve original2183GiB plus separate120GiB T1 reservation. No C5, no new training or dispatch permission.','verification_boundary':'Fresh shared status-file snapshots and statvfs only; no remote process liveness inference, checkpoint inventory/hash/read, deletion, or quota verification. All older obligations remain reserved regardless of current observed progress.','reviewer':'/root/st_owned_descendants_review_v2','new_training_authorized':False})
out=Path(__file__).parent/'CHECKPOINT_COMMITMENTS_2303G_V2.json'
with out.open('x') as f: f.write(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
hwm=next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:'))
cost={'status':'PASS_BOUNDED_READ_ONLY_REFRESH','elapsed_seconds':time.monotonic()-t0,'explicit_JSON_read_bytes':total,'actual_self_VmHWM_bytes':hwm,'cpu_affinity':sorted(os.sched_getaffinity(0)),'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_pins':pins,'no_model_payload_read':True,'no_ledger_access':True,'no_dispatch':True}
assert cost['elapsed_seconds']<60 and hwm<=256<<20
with (out.parent/'REFRESH_EVIDENCE.json').open('x') as f:f.write(json.dumps(cost,indent=2,sort_keys=True)+'\n')
print(json.dumps({'receipt':str(out),'sha256':cost['receipt_sha256'],'read_bytes':total,'free_bytes':free,'checked':now,'until':now+21600,'C_completed':[o['completed'] for o in observations]}))
