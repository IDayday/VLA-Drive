"""Independent narrow attempt2 admission. No dispatch or ledger mutation."""
from pathlib import Path
import ast,copy,fcntl,hashlib,importlib.util,json,os,shutil,socket,stat,sys,time
HERE=Path(__file__).resolve().parent
P=Path('/mnt/project/VLA-Drive-st-flow-resume-efficiency-inheritance-20261018/reports/st_world_policy/20261018/full_causal_history_train_attempt2_preparation_v1')
OLD=Path('/mnt/project/VLA-Drive-st-full-history-controller-20261018/reports/st_world_policy/20261018/full_causal_history_build_preparation_v1')
refs=[]
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def load(r):
 p=Path(r['path']);assert p.is_file() and not p.is_symlink() and p.stat().st_size<2<<20
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'];refs.append(r);return json.loads(raw)
h=load({'path':str(P/'HANDOFF.json'),'sha256':'c95ad9c51b716ce10833a1c7fb67e280e1ca752008dace01cb9c0475d2bb3cb9'})
a=load(h['admission_proposal']);plan=load(h['plan']);old=load(plan['versioned_attempt']['previous_plan'])
assert (P/'run_full_history_once.py').read_bytes()==(OLD/'run_full_history_once.py').read_bytes().replace(b'f6c96c1cb48d20b1143fc4fd38451e1ba4d6ae02ba71fe1ac7d8fae8ff1a2079',b'a6ea02aa66ff98a51284272f024136d51867e1e999a04a78f9f03ecee4219ed3')
assert (P/'storage_binding.py').read_bytes()==(OLD/'storage_binding.py').read_bytes()
assert ref(P/'run_full_history_once.py')['sha256']=='09b526a9375c5c1c4df6a2a67e07a5f5ab91ca6ea3a061365f0460c435922063'
for p in [P/'run_full_history_once.py',P/'storage_binding.py']: ast.parse(p.read_text());refs.append(ref(p))
normalized=copy.deepcopy(plan);normalized.pop('versioned_attempt');normalized['roles']['train']['control_root']=old['roles']['train']['control_root'];normalized['conservative_metadata_MiB']=old['conservative_metadata_MiB'];assert normalized==old
assert plan['conservative_metadata_MiB']['total']==10098 and plan['conservative_metadata_MiB']['maximum']==10240 and plan['conservative_metadata_MiB']['remaining']==142
assert h['typed_train_2047MiB_not_double_counted'] and h['new_control_reserved_bytes']==1<<20 and h['old_control_reserved_bytes_retained']==1<<20
failed=load(h['failed_result']);load(h['failed_invocation']);assert failed['status']=='FAILED_NO_AUTO_RETRY' and failed['phase']=='START_PUBLICATION' and failed['cleanup']['status']=='NOT_STARTED'
assert failed['charged_rchar_bytes']==h['cumulative_old_fallback_plus_new_authorized_read_bytes']//2==512<<30
assert plan['versioned_attempt']['old_fallback_is_measured_read_bytes'] is False
assert not Path(a['output_root']).exists(), 'Unexpected prior artifact must be separately reconciled'
newcontrol=Path(a['control_root']);assert newcontrol.is_dir() and not newcontrol.is_symlink() and not list(newcontrol.iterdir())
assert socket.gethostname()==a['hostname']=='training-vla-zt-worker-0';assert {10,11} <= os.sched_getaffinity(0)
lock=Path(plan['versioned_attempt']['global_lock_unchanged']);fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
try:assert stat.S_ISREG(os.fstat(fd).st_mode);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(fd,fcntl.LOCK_UN)
finally:os.close(fd)
assert a['limits']==dict(cleanup_seconds=40,cpu_affinity=[10,11],max_disk_bytes=2147483648,max_invocations=1,read_bytes=512<<30,rss_bytes=4<<30,wall_seconds=14400)
cp=load(a['checkpoint_receipt']);assert cp['remaining_bytes']==3701166716312 and cp['total_retained_allocation_bytes']==4143<<30
for r in a['controller_sources']:assert ref(r['path'])==r;refs.append(r)
for k in ['builder_budget','registration','policy']:load(a[k])
load(plan['roles']['train']['population_complete'])
dev=Path('/mnt/project/st-world-policy-20261016/audits/history_data/full_causal_population_v1/dev/COMPLETE.json')
load({'path':str(dev),'sha256':'dd2e999fc66f386335a3bb71bf342bf3bd83a029dee49cd4a8dc791bab3fdecb'})
devc=Path('/mnt/project/st-world-policy-20261016/audits/history_data/full_causal_population_v1/dev_controls/RESULT.json')
d=load({'path':str(devc),'sha256':'3d6f800d2e041ef728853e21f756d0b6d16f4980c8171ae34a6203252aefa227'});assert d['status']=='COMPLETE' and d['cleanup']['live_owned_remaining']==0
sys.path.insert(0,a['runtime'])
from tools.st_world_policy.cache_storage_budget import StorageBudget
lp=Path('/mnt/project/st-world-policy-20261016/resources/cache_storage_ledger_v1/CLAIMS.json');state=load(ref(lp));b=StorageBudget.__new__(StorageBudget);b.core=state['core'];b.state_dir=lp.parent
policy=load(a['policy']);claim=next(x for x in policy['claims'] if x['root']==a['output_root'])
assert a['output_root'] not in state['claims']
pros=copy.deepcopy(state);pros['claims'][a['output_root']]={'claim':claim,'observed_bytes':0,'charged_bytes':b._total(claim),'device':lp.parent.stat().st_dev}
b._check_disk(pros,lp.parent,free_reserve_bytes=(1<<40)+cp['remaining_bytes'])
free=shutil.disk_usage(lp.parent).free;required=cp['remaining_bytes']+(1024+768+10)*(1<<30);assert free>=required
now=time.time();a.update(status='APPROVED_FULL_CAUSAL_HISTORY_METADATA_ONLY',reviewer='/root/spatial20k_efficiency',issued_at_unix=now,latest_start_unix=now+900,valid_until_unix=now+18000)
sys.path.insert(0,str(P));spec=importlib.util.spec_from_file_location('reviewed_fullhistory2',P/'run_full_history_once.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
inherited=module.inherited_binding();assert module.validate(a,cp,'train',plan['roles']['train'],inherited,now=now)==[10,11]
bound=module.bound_source(module.source_bytes(module.ORIGINAL,inherited.PINS[str(module.ORIGINAL.relative_to(module.RUNTIME))]));ast.parse(bound)
assert bound.count('BOUND_CP_REMAINING')==2
assert plan['roles']['train']['scenes']==101592 and plan['roles']['train']['logs']==1176
out=HERE/'ACTUAL_ADMISSION.json';assert not out.exists();out.write_text(json.dumps(a,indent=2)+'\n')
cmd=copy.deepcopy(h['command_template']);cmd['argv']=cmd.pop('argv_prefix')+['--admission',str(out),'--admission-sha256',ref(out)['sha256']];cmd.pop('remaining_args');cmd.update(schema='st_reviewed_single_command_v1',run_by_reviewer=False,no_automatic_retry=True)
(HERE/'COMMAND.json').write_text(json.dumps(cmd,indent=2)+'\n')
report={'schema':'st_full_causal_history_attempt2_independent_efficiency_review_v1','status':'PASS_FOR_SINGLE_TRAIN_METADATA_ATTEMPT','reviewer_agent':'/root/spatial20k_efficiency','checked_unix':now,'native_disk_check':'PASS','free_bytes':free,'conservative_required_free':required,'headroom_bytes':free-required,'wrapper_change':'Only exact PLAN_SHA literal; helper byte identical; native builders/monitor unchanged.','plan_change':'Only TRAIN control namespace plus explicit failed-attempt accounting; same typed claim, observation population, reader and output root.','old_attempt_preserved':True,'old_fallback_charged_bytes':512<<30,'old_actual_read_bytes':None,'new_read_allowance_bytes':512<<30,'combined_old_conservative_plus_new_allowance_bytes':1024<<30,'metadata_total_MiB':10098,'metadata_cap_MiB':10240,'admission':ref(out),'command':ref(HERE/'COMMAND.json'),'evidence_artifacts':refs,'actions':{'dispatch':False,'GPU':False,'large_payload_read':False,'live_ledger_mutation':False,'source_mutation':False,'new_model_test':False},'boundaries':['Lead must launch before latest_start using exact command, CUDA hidden, CPU10/11 and retained global lock.','One4h attempt only; no auto retry, teacher tensors, GPU, scoring or training.','Real DEV completed; whole TRAIN completion within4h remains unproven.','Native source448GiB32MiB/s, logical512GiB,RSS4GiB and2GiBartifact/control budgets remain. Logical IO counters are not physical disk quota.','Existing failed allowance and controls retained; no claim that exception fallback is measured512GiB consumption.','No temporal formal training authorization follows from metadata admission.']}
(HERE/'REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'admission':ref(out),'review':ref(HERE/'REVIEW.json'),'command':ref(HERE/'COMMAND.json'),'latest_start_unix':a['latest_start_unix']},indent=2))
