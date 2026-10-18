"""Independent metadata/stat-only reconciliation; never open checkpoint payloads."""
from pathlib import Path
import copy,hashlib,json,os,stat,sys,time,shutil
HERE=Path(__file__).resolve().parent
OWNER=Path('/mnt/project/VLA-Drive-st-flow-resume-efficiency-inheritance-20261018/reports/st_world_policy/20261018/checkpoint_remaining_reconciliation_v1')
reads=0
refs=[]
def load(path,sha=None):
 global reads
 p=Path(path); assert p.is_file() and not p.is_symlink() and p.stat().st_size < 2<<20
 raw=p.read_bytes(); reads+=len(raw); h=hashlib.sha256(raw).hexdigest()
 if sha:assert sha==h,(str(p),'hash')
 ref={'path':str(p),'sha256':h}
 if ref not in refs: refs.append(ref)
 return json.loads(raw)
def pin(r):return load(r['path'],r['sha256'])
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
s=load(OWNER/'CONSUMED_REMAINING_SNAPSHOT.json','86dfb91df58672f3af483f57a08dd505f93a0eef2e0a83ae14119f169c3da0c7')
c=load(OWNER/'CHECKPOINT_COMMITMENTS_REMAINING_V2.json','7b88fcbf0a4b4505aa251fdbf2e146c6b94dd3fcb15cef8269cee4d1989fa60c')
orig=pin(s['original_budget_basis']);orig1983=pin(s['original1983_budget']);gross=pin(c['parent_gross_receipt'])
assert orig['reservation_basis']=='Retain all original 4*240 +187 +4*209 GiB commitments; do not subtract existing files'
assert orig['suggested_remaining_checkpoint_reserve_bytes']==1983<<30
assert c['retained_prior_allocation_bytes']==2703<<30
assert gross['remaining_bytes']==c['total_retained_allocation_bytes']==4143<<30
assert c['new20k_allocation_bytes']==1440<<30
assert s['other_or_new20k_checkpoint_credit_bytes']==0
seen=set();total=0;cpcount=0;runrows=[]
for k,r in enumerate(s['runs']):
 assert r['run_id']==f'SP{k}_fulltrain_seed42_official_5k_v1'
 assert any(x.get('root')==r['root'] or x.get('run')==r['root'] or x.get('run_id')==r['run_id'] for x in orig['runs']), (r['run_id'],'not original')
 reg=pin(r['registration']);cfg=pin(r['config']);b=pin(r['boundary'])
 assert reg['run_id']==r['run_id'] and cfg['execution']['stop_after']==5000
 assert b['first_wave_per_run_native_space_reserve_bytes']==r['original_allocation_bytes']==240<<30
 assert b['planned_periodic_checkpoints']==5
 assert [cp['completed'] for cp in r['checkpoints']]==[1000,2000,3000,4000,5000]
 subtotal=0
 for cp in r['checkpoints']:
  d=pin(cp['COMPLETE']);folder=Path(cp['COMPLETE']['path']).parent
  assert folder.parent.parent==Path(r['root'])
  assert d['schema']=='st_native_checkpoint_v1' and d['world_size']==4
  assert d['state']['completed']==cp['completed'] and d['state']['tag']==folder.name
  assert d['receipt_identity']==digest({k:v for k,v in d.items() if k!='receipt_identity'})
  assert {x['name'] for x in cp['files']}==set(d['files']) and len(d['files'])==9
  cpcredit=0
  for f in cp['files']:
   p=folder/f['name'];st=p.lstat();assert stat.S_ISREG(st.st_mode) and st.st_nlink==1
   key=(st.st_dev,st.st_ino);assert key not in seen;seen.add(key)
   assert st.st_dev==f['device'] and st.st_ino==f['inode']
   assert st.st_size==f['bytes']==d['files'][f['name']]['bytes']
   assert st.st_blocks*512==f['allocated_bytes']
   assert st.st_mtime_ns==f['mtime_ns'] and st.st_ctime_ns==f['ctime_ns']
   assert f['recorded_sha256']==d['files'][f['name']]['sha256']
   credit=min(st.st_size,st.st_blocks*512);assert credit==f['credited_bytes'];cpcredit+=credit
  assert cpcredit==cp['credited_bytes'];subtotal+=cpcredit;cpcount+=1
 assert subtotal==r['credited_already_on_disk_bytes']
 assert r['original_allocation_bytes']-subtotal==r['remaining_original_allocation_bytes']
 total+=subtotal;runrows.append({'run_id':r['run_id'],'credited_bytes':subtotal})
assert len(seen)==180 and cpcount==20
assert total==747345660520==c['credited_already_on_disk_bytes']
assert c['remaining_bytes']==(4143<<30)-total==3701166716312
assert c['commitment_release_bytes']==0 and c['original_total_retention_limit_changed'] is False
sys.path.insert(0,'/mnt/project/VLA-Drive-st-full-history-v3-runtime-20261018')
from tools.st_world_policy.cache_storage_budget import StorageBudget
lp=Path('/mnt/project/st-world-policy-20261016/resources/cache_storage_ledger_v1/CLAIMS.json');state=load(lp)
b=StorageBudget.__new__(StorageBudget);b.core=state['core'];b.state_dir=lp.parent
reg=load('/mnt/project/VLA-Drive-st-full-history-controller-20261018/reports/st_world_policy/20261018/full_causal_history_build_preparation_v1/TRAIN_STORAGE_REGISTRATION.json')
pol=load('/mnt/project/VLA-Drive-st-full-history-controller-20261018/reports/st_world_policy/20261018/full_causal_history_build_preparation_v1/FULL_ROLES_POLICY_PROPOSAL.json')
claims=pol['claims']
if isinstance(claims,dict): claims=list(claims.values())
claim=next(x for x in claims if x['root']==reg['output_root'])
pros=copy.deepcopy(state)
if claim['root'] not in pros['claims']:
 pros['claims'][claim['root']]={'claim':claim,'observed_bytes':0,'charged_bytes':b._total(claim),'device':lp.parent.stat().st_dev}
outstanding=b.core['ledger_reserve_bytes']+sum(x['reserved_bytes'] for x in b.core['legacy_reservations'])+sum(b._total(x['claim'])-x['observed_bytes'] for x in pros['claims'].values())
assert sum(x['reserved_bytes'] for x in b.core['legacy_reservations'])==692580253696
b._check_disk(pros,lp.parent,free_reserve_bytes=(1<<40)+c['remaining_bytes'])
old_failed=False
try:b._check_disk(pros,lp.parent,free_reserve_bytes=(1<<40)+(4143<<30))
except ValueError:old_failed=True
free=shutil.disk_usage(lp.parent).free
strong=c['remaining_bytes']+(1024+768+10)*(1<<30); assert free>=strong
report={'schema':'st_independent_consumed_remaining_check_v1','reviewer_agent':'/root/spatial20k_efficiency','status':'PASS','checked_unix':time.time(),'checkpoint_count':cpcount,'unique_payload_inodes':len(seen),'credited_bytes':total,'gross_retained_bytes':4143<<30,'remaining_bytes':c['remaining_bytes'],'per_run':runrows,'metadata_bytes_read':reads,'payload_bytes_read':0,'ledger_mutated':False,'source_changed':False,'native_outstanding_with_train':outstanding,'native_required_free':(1<<40)+c['remaining_bytes']+outstanding,'native_readonly_check':'PASS','old_gross_native_check_failed':old_failed,'free_bytes':free,'conservative_768GiB_plus_10GiB_required':strong,'conservative_headroom_bytes':free-strong,'evidence_artifacts':refs,'limitations':['Payload content hashes inherited, not rehashed; existing complete receipt and exact stat version checked.','Per-project quota not independently visible.','Live disk capacity and source must be rechecked at dispatch; no deletion of retained files approved.']}
(HERE/'STORAGE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='evidence_artifacts'},indent=2))
