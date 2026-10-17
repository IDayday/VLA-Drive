"""Independent small-control review. Does not import builders, launch or write ledgers."""
import ast,hashlib,json,math,os,pathlib,resource,socket,subprocess,time
from pathlib import Path
start=time.monotonic(); io0=int(Path('/proc/self/io').read_text().split('rchar: ')[1].splitlines()[0]); inputs=[];read_bytes=0
out=Path(__file__).resolve().parent

def read(p,sha=None,js=True):
 global read_bytes
 p=Path(p);assert p.is_file() and not any(x.is_symlink() for x in (p,*p.parents));assert p.stat().st_size<=1<<20
 b=p.read_bytes();read_bytes+=len(b);assert read_bytes<16<<20
 h=hashlib.sha256(b).hexdigest();assert sha is None or h==sha,(str(p),h,sha)
 inputs.append({'path':str(p),'sha256':h,'bytes':len(b)})
 return json.loads(b) if js else b

cond=read(out/'CONDITIONS.json');a=dict(cond['fixed_fields_from_template'])
lead=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/history64_actual_admission_v1')
e=read(lead/'LIVE_QUIESCENCE_EVIDENCE.json','759aaae25e60455018cebcd05a5146045923dabf9128069af32b89b19e1ed109')
q=read(lead/'QUIESCENCE.json','c8b5a17faf7fc1f471c521bd7605fdd855d201e22c33138743f14ed5d9d34f6b')
c=read(lead/'CONTROL_RESERVATION_ADOPTED.json','a9abca6fb8492b100557597b22169fae0897bfa780918b9e4bf1f0ab3ef5a8e8')
cp=read(**dict(p=a['checkpoint_receipt']['path'],sha=a['checkpoint_receipt']['sha256']))
policy=read(a['policy']['path'],a['policy']['sha256']);reg=read(a['registration']['path'],a['registration']['sha256'])
state=Path(policy['state_dir']); migration=read(state/'MIGRATION_1a729d88423b0f6643c33304aed715bff49d2b4f4d57c8fbfc1f13d4b438d6b7.complete.json','3ea4f8ce5c6a01842b4360acd7dec88c72bf6500f748f4b2a2b81fa0ad90ef04')
ledger=read(state/'CLAIMS.json','d81af1a046ce83765964935ec6cf9d9d876a21476e6f4e575bbfa85e95b6f526')
snap=read(migration['snapshot_path']);oldraw=bytes.fromhex(snap['ledger_hex']);assert hashlib.sha256(oldraw).hexdigest()==snap['old_sha256']==migration['old_ledger_sha256'];old=json.loads(oldraw)
assert ledger['claims']==old['claims'],'Prior claims/accounting must be unchanged'
assert migration['status']=='MIGRATED' and migration['new_policy_sha256']==a['policy']['sha256']
assert migration['completion_quiescence_sha256']==inputs[2]['sha256']
assert migration['checkpoint_sha256']==a['checkpoint_receipt']['sha256']
assert migration['migration_id'] in ledger['metadata_migrations']
assert ledger['core']['cache_cap_bytes']==policy['cache_cap_bytes']==768<<30
assert ledger['core']['free_reserve_bytes']==policy['free_reserve_bytes']==1<<40
for k in ('schema','campaign','state_dir','allowed_roots','legacy_reservations','ledger_reserve_bytes','metadata_package','frozen_w_package'):
 assert ledger['core'][k]==policy[k],k
assert q['schema']=='st_metadata_writer_quiescence_v1' and q['status']=='QUIESCENT'
assert q['owner_role']==q['future_admission_owner']=='Lead' and q['queued_writers_disabled'] and not q['active_or_unknown_writers']
assert q['bindings']['old_ledger_sha256']==snap['old_sha256'] and q['bindings']['new_policy_sha256']==a['policy']['sha256']
expectedroots={x['root'] for x in policy['claims'][:-1]}|{x['root'] for x in policy['legacy_reservations']}
assert set(q['checked_writer_roots'])==expectedroots
assert not e['matching_writers'] and all(x['exclusive_nonblocking_acquired'] for x in e['lockchecks'])
assert e['hostname']==socket.gethostname()
for pin in q['process_evidence']:read(pin['path'],pin['sha256'])
terminal=read(e['train_terminal']['path'],e['train_terminal']['sha256']);assert terminal['status']=='FAILED_NO_AUTO_RETRY'
for pid in e['known_train_pids_absent']: assert not Path(f'/proc/{pid}').exists(),pid
# Independent exit review is immutable git evidence, not another filesystem census.
rpath='reports/st_world_policy/20261017/full_future_train_exit_review_v2/REVIEW.json'
rb=subprocess.check_output(['git','show','2e8240ee34b9ee9a03c408af8ef146db6a4cb5cc:'+rpath]);read_bytes+=len(rb);exit_review=json.loads(rb)
inputs.append({'git_commit':'2e8240ee34b9ee9a03c408af8ef146db6a4cb5cc','path':rpath,'bytes':len(rb),'sha256':hashlib.sha256(rb).hexdigest()})
assert c['status']=='ADOPTED_CONTROL_BUDGET_ONLY' and c['reserved_bytes']==1<<20 and c['combined_artifact_control_bytes']==32<<20
assert c['not_borrowed_from_existing_monitor_4MiB'] and c['existing_monitor_addendum_bytes']==4<<20 and c['existing_admin_bytes_preserved']==128<<20
read(c['separate_existing_monitor_addendum']['path'],c['separate_existing_monitor_addendum']['sha256'])
assert c['new_planned_metadata_bytes']<=c['metadata_envelope_bytes']==8<<30
for pin in a['controller_sources']: read(pin['path'],pin['sha256'],False)
rt=Path(a['runtime']);assert subprocess.check_output(['git','-C',str(rt),'rev-parse','HEAD'],text=True).strip()=='b847f5a99dd44ee5256362d380d9d96efb47d35b'
for p in (Path(a['output_root']),Path(a['control_root'])):
 assert not p.exists() and not any(x.is_symlink() for x in (p,*p.parents)),str(p)
dispatch=read(e['policy']['path'],e['policy']['sha256']);assert socket.gethostname() in dispatch['allowed_hosts']['training-vla-zt']['aliases']
free=os.statvfs('/mnt/project');freebytes=free.f_bavail*free.f_frsize
assert freebytes>cp['remaining_bytes']+cp['free_floor_bytes']+(32<<20)
now=time.time();assert q['checked_at_unix']<=now<=q['valid_until_unix'];assert now-e['checked_at_unix']<900
assert cp['status']=='CURRENT' and cp['remaining_bytes']==2303<<30 and cp['free_floor_bytes']==1<<40
assert cp['checked_at_unix']<=now and now+1500<=cp['valid_until_unix']
a.update(status='APPROVED_HISTORY64_PILOT_ONLY',reviewer='/root/st_calibration_independent_correctness_v2',hostname=socket.gethostname(),issued_at_unix=now,latest_start_unix=now+300,valid_until_unix=now+1500)
assert a['latest_start_unix']+940<=a['valid_until_unix']
# Execute only the existing pure validation function AST; no module imports or controller top-level IO.
src=read(a['controller_sources'][0]['path'],a['controller_sources'][0]['sha256'],False).decode();node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='validate')
ns={'LIMITS':a['limits'],'CP':a['checkpoint_receipt'],'ROOT':Path(a['output_root']),'RUNTIME':rt,'CONTROL_ROOT':Path(a['control_root']),'CONTROL_CAP':1<<20,'socket':socket,'math':math}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<existing-pure-admission-validator>','exec'),ns);ns['validate'](a,now)
summary={'schema':'st_history64_independent_actual_admission_review_v1','status':'PASS_NARROW_ACTUAL_ADMISSION_ONLY','reviewer':a['reviewer'],'checked_at_unix':now,'inputs':inputs,'free_bytes_independently_observed':freebytes,'old_claims_and_accounting_unchanged':True,'old_train_failure_preserved':terminal['status'],'migration_status':migration['status'],'control_budget_separate':True,'output_and_control_roots_absent':True,'known_train_pids_absent':e['known_train_pids_absent'],'exit_review':exit_review,'live_descendant_authority':'Lead fresh census plus independent exit review2e8240ee; no additional descendant scanning here','cpu6_7_reservation':'Lead current dispatch message; final freshcheck remains Lead responsibility','checkpoint_quota_verified':cp.get('quota_verified'),'admission_scope':'One fixed64 CPU-only metadata pilot,900+40s; no GPU,training,teacher/cachefullbuild,automaticretry','launch_performed':False,'ledger_mutated':False,'runtime_mutated':False,'prior_44_14_15_suites_rerun':False,'limitations':['No artifact production or real packed run success asserted','No GPU,raw,image,teacher,checkpoint tensor read','Lead must recheck lease,free,CP,roots,no writer andCPU6/7 immediately before one launch','Advisory CP reserve is not a verified filesystem quota','Stopped TRAIN retains FAILED_NO_AUTO_RETRY and all charged claims'],'cost':{'elapsed_seconds':time.monotonic()-start,'explicit_small_file_read_bytes':read_bytes,'logical_read_delta':int(Path('/proc/self/io').read_text().split('rchar: ')[1].splitlines()[0])-io0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_affinity':sorted(os.sched_getaffinity(0))}}
assert summary['cost']['elapsed_seconds']<60 and summary['cost']['logical_read_delta']<16<<20
for name,o in [('ACTUAL_ADMISSION.json',a),('ACTUAL_REVIEW.json',summary)]:
 p=out/name;assert not p.exists();p.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n');print(name,hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps({'status':summary['status'],'cost':summary['cost'],'issued':now,'latest_start':now+300,'valid_until':now+1500}))
