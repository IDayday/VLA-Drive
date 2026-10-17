"""Prepare immutable SP0 dev W controls only: no migration, locks or extraction."""
from pathlib import Path
import copy,hashlib,json,os,resource,sys,time
START=time.monotonic()
R=Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017')
B=Path('/mnt/project/st-world-policy-20261016')
JOB=B/'audits/spatial_probe_actual_SP0_5000_v1'
OUT=JOB/'storage_preparation_dev_v1'
OLD=B/'resources/CACHE_STORAGE_BUDGET_POLICY_v7_index_followup.json'
LEDGER=B/'resources/cache_storage_ledger_v1/CLAIMS.json'
SIGNATURE=JOB/'stages/dev_plan/SIGNATURE.json'
blocked_suffixes=('.pt','.pth','.safetensors','.bin','.jpg','.jpeg','.png','.npy','.npz','.pkl','.sqlite')
read_entries=[]
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  name=os.fsdecode(args[0]);mode=args[1];flags=args[2]
  if name.lower().endswith(blocked_suffixes):raise RuntimeError('Forbidden payload/index open:'+name)
  writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
  if writing and not str(Path(name).absolute()).startswith(str(OUT)+'/'):raise RuntimeError('Write outside owned preparation:'+name)
  if not writing:
   try:size=Path(name).stat().st_size
   except OSError:size=0
   read_entries.append((name,size))
   if sum(x[1] for x in read_entries)>128<<20:raise RuntimeError('Bounded metadata read allowance exhausted')
 if time.monotonic()-START>120 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2<<30:raise RuntimeError('Preparation CPU/RAM budget')
def pin(p):
 p=Path(p); assert not p.is_symlink(); raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest())
def read(p,sha=None):
 q=pin(p);assert sha is None or q['sha256']==sha;return json.loads(Path(p).read_text())
def write(name,x):
 p=OUT/name;raw=(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();assert len(raw)<1<<20
 with p.open('xb') as f:f.write(raw)
 return pin(p)
assert not OUT.exists(); OUT.mkdir()
sys.addaudithook(audit)
sys.path.insert(0,str(R))
from tools.st_world_policy import cache_storage_budget as b
from tools.st_world_policy import spatial_w_storage as w
from tools.st_world_policy.metadata_storage import validate_package
old=read(OLD,'b7062c4da878519c435db2e88b899bd476e5cd6a0e9f16dca191fe61522bb831')
ledger=read(LEDGER,'7fcfc0c18fa69fc65a125ee36acc1204f9976e62926bede016269136f79cebbe')
assert len(old['claims'])==len(ledger['claims'])==7
assert all(ledger['claims'][c['root']]['claim']==c for c in old['claims'])
sig=read(SIGNATURE,'3de51e64aec7f555f8d5c053695656346e1549796dad08c2c5bdafa88a26cff4')
result=read(JOB/'stages/dev_plan/RESULT.json','dd765726eff1c937c31a1aca0f3cc47db8515f53790f84646ae1b94218857520')
assert result['status']=='COMPLETE' and sig['identity']=='bccc958c07f1ff9ce1618221abe8763337cab742243975f2f7ae47180819b673'
claim=w.claim_from_signature(root=B/'caches/learned_w_probe_SP0_5000_v1/dev',claim_id='SP0-frozen-W-dev16-native5000-v1',signature_path=SIGNATURE,signature_sha256=pin(SIGNATURE)['sha256'])
assert not Path(claim['root']).exists()
claimpin=write('DEV_TYPED_W_CLAIM.json',claim)
new=copy.deepcopy(old);new['claims'].append(claim)
new['frozen_w_package']={'schema':'st_frozen_spatial_w_storage_package_v1','maximum_bytes':1<<30,'baseline_policy_sha256':pin(OLD)['sha256'],'baseline_ledger_sha256':pin(LEDGER)['sha256'],'baseline_claims':{c['root']:c for c in old['claims']},'baseline_legacy_reservations':old['legacy_reservations'],'metadata_package_sha256':b._digest(old['metadata_package']),'members':{claim['root']:claim}}
newpin=write('PROPOSED_POLICY_V8_SP0_DEV_W.json',new)
# Reuse exactly the constructor's read-only validation without its state-dir mkdir.
def parsed(path):
 obj=b.StorageBudget.__new__(b.StorageBudget);obj.policy_path=Path(path);obj.policy,obj.core,obj.declarations=obj._read_policy();return obj
ob=parsed(OLD);nb=parsed(newpin['path'])
w.validate_migration(ob,nb,old_policy_sha256=pin(OLD)['sha256'],old_ledger_sha256=pin(LEDGER)['sha256'])
meta=validate_package(nb.core,nb.declarations);assert meta==1136<<20
teacher_reports=B/'resources/QUALIFICATION_REPORT_BUDGET_v1.json'
assert pin(teacher_reports)['sha256']=='1ab0e8a8673b354ec81d99311444c68a502ac8d4d0f86e66e103373f90c0744e'
review=Path('/mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017')
sp0=review/'spatial_probe_dev_plan_admission_v1/DEV_PLAN_ADMISSION.json'
factory=review/'indexed_smoke_preparation_resource_v1/WRAPPER_RESOURCE_APPROVAL.json'
admin={'schema':'st_administrative_metadata_budget_carryforward_v1','status':'PRESERVED_ADMINISTRATIVE_RESERVATIONS_NOT_LEDGER_CLAIMS','base_metadata_package_bytes':meta,'reservations':[{'category':'qualification_reports','bytes':64<<20,'evidence':pin(teacher_reports)},{'category':'SP0_controls_including_this_preparation','bytes':32<<20,'evidence':pin(sp0)},{'category':'indexed_provider_factory','bytes':32<<20,'evidence':pin(factory)}],'effective_metadata_bytes':1264<<20,'metadata_maximum_bytes':4<<30,'cache_ledger_claimed':False,'no_double_count_on_future_adoption':True}
adminpin=write('ADMINISTRATIVE_METADATA_CARRYFORWARD.json',admin)
total=nb.core['ledger_reserve_bytes']+sum(x['reserved_bytes'] for x in nb.core['legacy_reservations'])+sum(b.StorageBudget._total(x) for x in nb.declarations.values())
assert total+(128<<20)<768<<30
oldcp=B/'audits/metadata_migration_preparation_v1/CHECKPOINT_COMMITMENTS_V1.json'
original=read(oldcp,'1bf12212b29202cd3f5cf80e73425ab5d663c778e7ba35df43a83bb2c27289ad')
lease=B/'smoke/world_indexed_provider_training2_v1/RESOURCE_LEASE.json';smoke=read(lease)
assert original['remaining_bytes']==1983<<30 and smoke['checkpoint_reservation_bytes']==120<<30
now=time.time();sv=os.statvfs(B);free=sv.f_bavail*sv.f_frsize;remaining=2103<<30
assert free>remaining+(1<<40)+b.StorageBudget._total(claim)+(128<<20)
cp={'schema':'st_checkpoint_storage_commitments_v1','status':'CURRENT','checked_at_unix':now,'valid_until_unix':now+3600,'actual_free_bytes':free,'remaining_bytes':remaining,'free_floor_bytes':1<<40,'quota_verified':False,'existing_checkpoint_bytes_not_deleted':True,'scope':'All nine original ST run budgets1983GiB plus the separately registered indexed student smoke120GiB. Conservative full allocation: no subtraction for already written bytes; not a consumed-byte measurement.','basis':{'original_complete_budget':pin(oldcp),'indexed_smoke_resource_lease':pin(lease)},'origin':'fresh statvfs plus conservative authorized full budgets; prior receipt expiry not reused'}
cppin=write('CHECKPOINT_COMMITMENTS_2103G_V1.json',cp)
invpin=write('PILOT_INVENTORY_NO_ADOPTIONS.json',{'schema':'st_metadata_pilot_inventory_v1','roots':[],'scope':'No completed administrative roots adopted in this minimal W transition; all prior reservations and128MiB separate administrative headroom retained'})
# A truthfully blocked template. Lead alone replaces this with a fresh complete evidence record.
bindings={'old_policy_sha256':pin(OLD)['sha256'],'new_policy_sha256':newpin['sha256'],'old_ledger_sha256':pin(LEDGER)['sha256'],'pilot_inventory_sha256':invpin['sha256'],'checkpoint_sha256':cppin['sha256']}
q={'schema':'st_metadata_writer_quiescence_v1','status':'PENDING_LEAD_LIVE_WRITER_CENSUS','owner_role':'Lead','bindings':bindings,'checked_writer_roots':sorted(set(ob.declarations)|{x['root'] for x in ob.core['legacy_reservations']}),'active_or_unknown_writers':None,'queued_writers_disabled':None,'future_admission_owner':'Lead','process_evidence':[],'checked_at_unix':None,'valid_until_unix':None}
qpin=write('QUIESCENCE_TEMPLATE_NOT_APPROVED.json',q)
used=max(result['resources']['counters']['hash_logical_bytes_admitted']+result['resources']['counters']['torch_load_nominal_bytes'],result['resources']['counters']['proc_self_io_delta']['rchar'])
prep={'schema':'st_SP0_dev_W_migration_preparation_v1','status':'PREPARED_VALIDATED_NOT_MIGRATED','source_sha':'581399eb309677c20fe7464791b4f18685e947a6','source_runtime':str(R),'old_policy':pin(OLD),'old_ledger':pin(LEDGER),'new_policy':newpin,'actual_signature':pin(SIGNATURE),'claim':claimpin,'checkpoint_commitments':cppin,'pilot_inventory':invpin,'administrative_carryforward':adminpin,'pending_quiescence_template':qpin,'old_claims_preserved':7,'new_exact_W_members':1,'dev_W_total_bytes':b.StorageBudget._total(claim),'metadata_core_bytes':meta,'effective_metadata_bytes':1264<<20,'all_declared_cache_bytes':total,'all_declared_plus_administrative_bytes':total+(128<<20),'cache_cap_bytes':768<<30,'extraction':{'stage':'dev_extract','gpu_count':1,'gpu_uuid':'NOT_ASSIGNED','cpu_affinity':[0,1],'outer_wall_seconds':960,'cooperative_seconds':900,'owned_cleanup_seconds':40,'rss_observed_limit_bytes':64<<30,'gpu_allocator_peak_limit_bytes':24<<30,'split_lifecycle_read_bytes':256<<30,'already_planned_read_bytes':used,'remaining_extract_read_bytes':(256<<30)-used,'output_root':claim['root'],'control_root':str(JOB/'stages/dev_extract'),'control_logical_cap_bytes':5<<20,'automatic_retry':False,'approved':False},'pending':['Fresh exact Lead writer/queue census and QUIESCENT receipt','Lead-only same-ledger migration; preserve administrative headroom','Fresh exact idle GPU/lease and separate dev_extract admission'], 'payload_reads':0,'live_ledger_policy_writes':0,'migration_executed':False,'extraction_executed':False,'metadata_read_stat_sum_bytes':sum(x[1] for x in read_entries),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'elapsed_seconds':time.monotonic()-START}
assert pin(LEDGER)['sha256']==bindings['old_ledger_sha256']
pp=write('PREPARATION.json',prep)
print(json.dumps({'preparation':pp,'claim':claim,'new_policy':newpin,'checkpoint_commitments':cppin,'elapsed_seconds':prep['elapsed_seconds'],'metadata_read_stat_sum_bytes':prep['metadata_read_stat_sum_bytes'],'peak_rss_bytes':prep['peak_rss_bytes']},sort_keys=True))
