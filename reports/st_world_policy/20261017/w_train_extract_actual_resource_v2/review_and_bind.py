from pathlib import Path
import hashlib,importlib.util,json,math,os,resource,signal,socket,subprocess,sys,time
signal.alarm(120);os.sched_setaffinity(0,{8,9});resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20))
t0=time.monotonic();pins=[];total=0
HERE=Path(__file__).parent;LEAD=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017');JOB=Path('/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1')
def read(path,expected=None):
 global total
 p=Path(path);st=p.stat();assert p.is_file() and st.st_size<=4<<20
 b=p.read_bytes();assert len(b)==st.st_size and p.stat().st_mtime_ns==st.st_mtime_ns
 total+=len(b);assert total<64<<20
 h=hashlib.sha256(b).hexdigest();assert expected is None or h==expected,(str(p),h,expected)
 pin={'path':str(p),'sha256':h};pins.append(dict(pin,bytes=len(b)));return json.loads(b),pin
prep,prep_pin=read(JOB/'controls_581399eb_v2_metadata_only/PREPARATION.json','34c174a6aa0c859a1b0a5f8b7dd800bbe189d51c770211b49c35d061e92b6431')
fixed,_=read(LEAD/'w_storage_bridge_v1/FIXED_INPUTS.json','c4305f8d459694b2e89e546638c10e976074a2301b967eea32b4edc4f4bcb61f')
sc,sc_pin=read(LEAD/'correctness/w_storage_bridge_review_v2/SOURCE_CHECK.json')
for name,h in sc['source_hashes'].items():
 p=LEAD/'w_storage_bridge_v1'/name;b=p.read_bytes();total+=len(b);assert hashlib.sha256(b).hexdigest()==h
policy,policy_pin=read(JOB/'storage_preparation_train_v2/PROPOSED_POLICY.json','b69506569a24f1057d975d3a94c8cfaee75af1affc0c27279a97458d67e70cf2')
migration,migration_pin=read(JOB/'storage_preparation_train_v2/MIGRATION_RECEIPT.json','f3fdcb64fc33b088013f50a3fb410c3a3f124b3a9379edc6ac6243be9d97990c')
compat,compat_pin=read(JOB/'storage_preparation_train_v2/STORAGE_COMPATIBILITY.json','484292108ec085f7e512ccbfd8bd5aac15824b21f6420a5efe1cad1b8443e7f6')
old,oldpin=read('/mnt/project/VLA-Drive-st-future-finalization-fix-20261017/reports/st_world_policy/20261017/future_finalization_actual_proposal_v1/DRAFT_PROPOSED_POLICY.json','266a253d266fecf20057b2a79988c28cd3de1fbcb7d9293a7daa90b21300cfdb')
assert migration['old_policy_sha256']==oldpin['sha256'] and migration['new_policy_sha256']==policy_pin['sha256'] and migration['status']=='MIGRATED'
assert len(old['claims'])==12 and len(policy['claims'])==13
newclaim=fixed['train_claim'];assert sum(newclaim[k] for k in ('output_bytes','staging_bytes','metadata_bytes'))==86254736
expected=json.loads(json.dumps(old));expected['claims'].append(newclaim);expected['frozen_w_package']['members'][newclaim['root']]=newclaim
assert expected==policy,'nonadditive policy change'
ledger,ledgerpin=read(Path(policy['state_dir'])/'CLAIMS.json',migration['new_ledger_sha256'])
cp,cppin=read('/mnt/project/st-world-policy-20261016/smoke/official_history_T1_native4_v1/controls/CHECKPOINT_COMMITMENTS_2303G_V1.json','13f3f4b282db2aff24528bc5434db51a9351cf43cf02dcdea1fbef38d5434cb3')
q,qp=read(JOB/'storage_preparation_train_v2/QUIESCENCE.json','0907977448d86cbad7b53316635da51165b4a26729c679d41dcce0d490cfe3e6')
previous,prevpin=read(JOB/'stages/train_plan/RESULT.json','001952cd9d5f608ba1761056d41159135ec3e61853d9aff5d3da90f1590aef15')
assert previous['status']=='COMPLETE' and previous['resources']['process_cleanup']['live_owned_remaining']==0
assert not (JOB/'stages/train_extract').exists(),'attempt already exists; no retry authorized'
assert not Path(prep['caches']['train']['root']).exists(),'train cache already exists; inspect instead of duplicate'
now=time.time();assert q['status']=='QUIESCENT' and not q['active_or_unknown_writers'] and now<q['valid_until_unix']
assert cp['remaining_bytes']==2303<<30 and cp['free_floor_bytes']==1<<40
v=os.statvfs('/mnt/project');free=v.f_bavail*v.f_frsize
assert free>cp['remaining_bytes']+cp['free_floor_bytes']+(1<<30)
mem={a.split(':')[0]:int(a.split()[1])*1024 for a in Path('/proc/meminfo').read_text().splitlines() if a.startswith('MemAvailable:')};assert mem['MemAvailable']>=64<<30
row=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
assert row[0].strip()=='GPU-f679bff0-a4a2-ae4f-0b85-74c519b45a9c' and int(row[1])<=64 and int(row[2])==0
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
assert not any(x.split(',')[0].strip()==row[0].strip() for x in apps.splitlines())
now=time.time();latest=now+600;valid=latest+1000;assert valid<cp['valid_until_unix']
ad={'schema':'st_SP0_probe_stage_admission_v1','status':'APPROVED','reviewer':'/root/st_owned_descendants_review_v2','stage':'train_extract','hostname':socket.gethostname(),'reviewed_at_unix':now,'latest_dispatch_start_unix':latest,'valid_until_unix':valid,'outer_wall_seconds':960,'cleanup_coverage_seconds':40,'observed_RSS_limit_bytes':64<<30,'observed_GPU_peak_limit_bytes':24<<30,'cpu_affinity':[2,3],'gpu':{'index':0,'uuid':row[0].strip()},'read_limit_bytes':256<<30,'remaining_read_bytes':237302801889,'control_output_total_maximum_bytes':32<<20,'runtime':prep['runtime'],'preparation':prep_pin,'helpers':{k:fixed['upstream'][k]['sha256'] for k in ('prepare_bindings.py','observed_cli.py','supervise_once.py')},'storage_policy':policy_pin,'storage_migration':migration_pin,'checkpoint_commitments':cppin,'storage_compatibility':compat_pin,'typed_storage_approved':True,'launch_approved':True,'formal_training_approved':False,'automatic_retry':False,'scope':'One original fixed SP0 train64 frozen-W extraction only using exact reviewed storage bridge; no standalone plan, DEV rerun, fit, training, or automatic continuation. Lead alone dispatches.','independent_bridge_correctness':sc_pin,'current_ledger_at_review':ledgerpin,'fresh_quiescence':qp,'previous_plan_evidence':prevpin,'training_or_scientific_PASS':False}
# Original helpers and bridge validation only: no install, cache_plan, torch, reservation, or subprocess workload.
sys.path.insert(0,str(LEAD/'w_storage_bridge_v1'));import supervise_bridge
original=supervise_bridge.build_supervisor()
wall,rss=original.validate(ad,prep,'train_extract',time.time());assert wall==960 and rss==64<<30
attempt,accounting,pin=original.stage_inputs(ad,'train_extract');assert accounting['remaining_read_bytes']==237302801889 and pin==prevpin
assert attempt==JOB/'stages/train_extract' and not attempt.exists()
assert 'torch' not in sys.modules
out=HERE/'ADMISSION.json'
with out.open('x') as f:f.write(json.dumps(ad,indent=2,sort_keys=True)+'\n')
adpin={'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
command=['/root/miniconda3/envs/ddp/bin/python',str(LEAD/'w_storage_bridge_v1/supervise_bridge.py'),'--admission',adpin['path'],'--admission-sha256',adpin['sha256']]
with (HERE/'COMMAND.json').open('x') as f:f.write(json.dumps({'argv':command,'cwd':prep['runtime']['path'],'dispatch_owner':'Lead','executed':False},indent=2)+'\n')
hwm=next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:'))
rchar=int(next(x.split(':')[1] for x in Path('/proc/self/io').read_text().splitlines() if x.startswith('rchar:')))
assert total<=64<<20 and rchar<=64<<20 and hwm<=512<<20 and time.monotonic()-t0<120
result={'status':'APPROVED_ONE_TRAIN64_EXTRACTION_ONLY','reviewer':ad['reviewer'],'admission':adpin,'source_pins':pins,'original_claims_exactly_preserved':12,'only_new_claim_bytes':86254736,'whole_typed_W_package_reserved_bytes':migration['frozen_w_package_reserved_bytes'],'metadata_reserved_bytes_unchanged':migration['metadata_package_reserved_bytes'],'fresh_free_bytes':free,'MemAvailable_bytes':mem['MemAvailable'],'GPU_fresh_at_review':{'index':0,'uuid':row[0].strip(),'used_MiB_upper_bound_verified':64,'utilization_percent':int(row[2]),'compute_owners':0},'original_stage_inputs':accounting,'no_model_image_teacher_payload_reads':True,'no_ledger_mutation':True,'no_dispatch':True,'torch_imported':False,'actual_original_validator_and_stage_inputs':'PASS','inherited_correctness_scope':'Exact bridge45 focused CPU cases, not rerun in this read-only admission','remaining_boundary':'Current admission neither proves actual GPU extraction success nor fit/training; Lead must fresh-check lease/host/locks at start; original whole-stage monitor remains unchanged. Review only observes GPU inventory, not allocation.','review_cost':{'seconds':time.monotonic()-t0,'explicit_read_bytes':total,'process_rchar_total_bytes':rchar,'self_VmHWM_bytes':hwm,'cpu_affinity':sorted(os.sched_getaffinity(0))}}
with (HERE/'REVIEW.json').open('x') as f:f.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'admission':adpin,'status':result['status'],'cost':result['review_cost']}))
