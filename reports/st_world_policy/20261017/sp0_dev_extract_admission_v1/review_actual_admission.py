"""Independent small-control check and one-stage resource token; no execution."""
import hashlib,importlib.util,json,os,socket,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=Path('/mnt/project/st-world-policy-20261016');J=B/'audits/spatial_probe_actual_SP0_5000_v1';S=J/'storage_preparation_dev_v1'
R=Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017');H=R/'reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1'
reads=[]
def read(p,sha=None):
 p=Path(p);assert not p.is_symlink() and p.stat().st_size<1<<20
 raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert sha is None or h==sha
 ref={'path':str(p),'sha256':h};reads.append({**ref,'bytes':len(raw)});return json.loads(raw),ref
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
prep,pp=read(J/'controls_581399eb_v2_metadata_only/PREPARATION.json','34c174a6aa0c859a1b0a5f8b7dd800bbe189d51c770211b49c35d061e92b6431')
policy,polpin=read(S/'PROPOSED_POLICY_V8_SP0_DEV_W.json','0bf0dc467a83a378e02ab4c52e00c7f2836011985edc5bf76d4fb2135957f191')
ledger,lp=read(B/'resources/cache_storage_ledger_v1/CLAIMS.json','433852d03c46345c5418880693f0575a2111a30e1c1292c01b1161e366c5daca')
mig,mp=read(B/'resources/cache_storage_ledger_v1/MIGRATION_10df383020d32441a7c3cfe617c89e3b8a651f2af720b7dd8fe8918cee221b34.complete.json','4ba5fecdd8c241e7d525827317d724dbbb69a8f7e9c8c7791ab49ff87ba61c15')
act,ap=read(S/'MIGRATION_ACTUAL.json','42629787a300676b2ba05027028038a74057923e91f2a2547589e610e25cb3fe');assert act['result']==mig
snap,_=read(mig['snapshot_path']);old=json.loads(bytes.fromhex(snap['ledger_hex']));assert old['claims']==ledger['claims'] and len(ledger['claims'])==7
assert hashlib.sha256(bytes.fromhex(snap['ledger_hex'])).hexdigest()=='7fcfc0c18fa69fc65a125ee36acc1204f9976e62926bede016269136f79cebbe'
assert mig['status']=='MIGRATED' and mig['new_ledger_sha256']==lp['sha256'] and mig['new_policy_sha256']==polpin['sha256']
assert digest(ledger['core'])==mig['new_core_sha256']
q,qp=read(mig['completion_quiescence_receipt'],mig['completion_quiescence_sha256']);assert q['status']=='QUIESCENT' and q['owner_role']=='Lead' and q['active_or_unknown_writers']==[] and q['queued_writers_disabled'] is True
cp,cppin=read(S/'CHECKPOINT_COMMITMENTS_2103G_V1.json','60606edc286a0c34016842a03f0de08f2422936e347364c7373defe9efcc5259')
admin,adminpin=read(S/'ADMINISTRATIVE_METADATA_CARRYFORWARD.json','1b3a22906bfebaa99bd425e855d9ec73550a4b34013490bb815814d1847aa530');assert admin['effective_metadata_bytes']==1264<<20
plan,planpin=read(J/'stages/dev_plan/RESULT.json','dd765726eff1c937c31a1aca0f3cc47db8515f53790f84646ae1b94218857520')
sig,sigpin=read(J/'stages/dev_plan/SIGNATURE.json','3de51e64aec7f555f8d5c053695656346e1549796dad08c2c5bdafa88a26cff4')
assert plan['status']=='COMPLETE' and sig['identity']==prep['caches']['dev']['identity']
assert len(policy['frozen_w_package']['members'])==1 and policy['frozen_w_package']['maximum_bytes']==1<<30
claim=next(iter(policy['frozen_w_package']['members'].values()));assert claim['root']==prep['caches']['dev']['root'] and claim['signature_sha256']==sigpin['sha256']
amount=sum(claim[x] for x in ('output_bytes','staging_bytes','metadata_bytes'));assert amount==26085776==mig['frozen_w_package_reserved_bytes']
assert not (J/'stages/dev_extract').exists() and not Path(claim['root']).exists()
now=time.time();until=min(now+1500,cp['valid_until_unix']);assert until-now>1000
assert cp['remaining_bytes']==2103<<30 and policy['free_reserve_bytes']==1<<40 and policy['cache_cap_bytes']==768<<30
free=os.statvfs(B).f_bavail*os.statvfs(B).f_frsize;assert free>cp['remaining_bytes']+(1<<40)+amount+(128<<20)
helpers={n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in ('supervise_once.py','observed_cli.py','prepare_bindings.py')}
used=max(plan['resources']['counters']['hash_logical_bytes_admitted']+plan['resources']['counters']['torch_load_nominal_bytes'],plan['resources']['counters']['proc_self_io_delta']['rchar'])
a={'schema':'st_SP0_probe_stage_admission_v1','status':'APPROVED','reviewer':'/root/st_metadata_resource_review','stage':'dev_extract','hostname':socket.gethostname(),'cpu_affinity':[0,1],'issued_at_unix':now,'valid_until_unix':until,'latest_start_for_40s_cleanup_unix':until-1000,'preparation':pp,'runtime':prep['runtime'],'helpers':helpers,'gpu':{'index':5,'uuid':'GPU-cba744b9-96c8-c3e2-f836-2f0126c038a0'},'storage_policy':polpin,'storage_migration':mp,'checkpoint_commitments':cppin,'typed_storage_approved':True,'observed_current_ledger':lp,'administrative_carryforward':adminpin,'prior_plan':planpin,'actual_signature':sigpin,'limits':{'cooperative_seconds':900,'outer_wall_seconds':960,'owned_cleanup_seconds':40,'rss_observed_limit_bytes':64<<30,'gpu_allocator_peak_limit_bytes':24<<30,'role_plan_plus_extract_read_bytes':256<<30,'plan_consumed_read_bytes':used,'remaining_extract_read_bytes':(256<<30)-used,'typed_W_total_bytes':amount,'stage_control_logical_bytes':5<<20,'effective_metadata_reserved_bytes':1264<<20},'automatic_retry':False,'protected_other_tasks':True,'excluded':['train extraction','fit','formal temporal training','additional cache members','automatic retry']}
sys.path.insert(0,str(H));spec=importlib.util.spec_from_file_location('reviewed_sp0_supervisor',H/'supervise_once.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);assert m.validate(a,prep,'dev_extract',now)==(960,64<<30)
assert 'torch' not in sys.modules
ad=HERE/'DEV_EXTRACT_ADMISSION.json';ad.write_text(json.dumps(a,sort_keys=True,indent=2)+'\n');adpin={'path':str(ad),'sha256':hashlib.sha256(ad.read_bytes()).hexdigest()}
e={'status':'PASS_BOUNDED_DEV_EXTRACT_ADMISSION','admission':adpin,'pins':reads,'helper_sha256':helpers,'current_free_bytes':free,'old_claim_rows_exactly_preserved':7,'single_actual_typed_W_member':True,'effective_metadata_reserved_bytes':1264<<20,'positive_existing_controller_validate':True,'metadata_bytes_read':sum(x['bytes'] for x in reads),'gpu_query_observation':{'index':5,'uuid':a['gpu']['uuid'],'memory_used_mib':0,'utilization_percent':0,'requires_fresh_controller_recheck':True},'native_model_image_tensor_reads':0,'reviewer_gpu_execution':False,'reviewer_migration_or_ledger_write':False,'reviewer_dispatch':False}
(HERE/'EVIDENCE.json').write_text(json.dumps(e,sort_keys=True,indent=2)+'\n');print(json.dumps({'admission':adpin,'valid_until_unix':until,'latest_start_for_40s_cleanup_unix':until-1000,'metadata_bytes_read':e['metadata_bytes_read'],'remaining_extract_read_bytes':a['limits']['remaining_extract_read_bytes']},sort_keys=True))
