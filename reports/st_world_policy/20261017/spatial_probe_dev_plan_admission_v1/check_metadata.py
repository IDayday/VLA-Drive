"""Independent bounded controls review. Never runs a plan or storage admission."""
import copy, hashlib, importlib.util, json, os, pathlib, socket, sys, time
D=pathlib.Path(__file__).resolve().parent
R=pathlib.Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017')
B=pathlib.Path('/mnt/project/st-world-policy-20261016')
H=R/'reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1'
P=B/'audits/spatial_probe_actual_SP0_5000_v1/controls_581399eb_v2_metadata_only/PREPARATION.json'
reads=[]
def read(p, expected=None):
 p=pathlib.Path(p); s=p.lstat(); assert not p.is_symlink() and 0<s.st_size<=2<<20
 data=p.read_bytes(); sha=hashlib.sha256(data).hexdigest()
 assert expected is None or sha==expected, str(p)
 reads.append(dict(path=str(p),bytes=len(data),sha256=sha));return json.loads(data)
def pin(p):
 p=pathlib.Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
p=read(P,'34c174a6aa0c859a1b0a5f8b7dd800bbe189d51c770211b49c35d061e92b6431')
for row in p['evidence']+[p['fit_registration']]+[v for c in p['controls'].values() for v in c.values()]: read(row['path'],row['sha256'])
pol=B/'resources/CACHE_STORAGE_BUDGET_POLICY_v7_index_followup.json';ledger=B/'resources/cache_storage_ledger_v1/CLAIMS.json'
policy=read(pol,'b7062c4da878519c435db2e88b899bd476e5cd6a0e9f16dca191fe61522bb831')
state=read(ledger,'7fcfc0c18fa69fc65a125ee36acc1204f9976e62926bede016269136f79cebbe')
report=B/'resources/QUALIFICATION_REPORT_BUDGET_v1.json'; reportdoc=read(report,'1ab0e8a8673b354ec81d99311444c68a502ac8d4d0f86e66e103373f90c0744e')
cp=B/'audits/metadata_migration_preparation_v1/CHECKPOINT_COMMITMENTS_V1.json';cpdoc=read(cp,'1bf12212b29202cd3f5cf80e73425ab5d663c778e7ba35df43a83bb2c27289ad')
assert len(state['claims'])==len(policy['claims'])==7
assert all(state['claims'][x['root']]['claim']==x for x in policy['claims'])
assert policy['cache_cap_bytes']==768<<30 and policy['free_reserve_bytes']==1<<40
assert policy['metadata_package']['maximum_bytes']==4<<30
assert reportdoc['effective_metadata_reserved_bytes']==1200<<20
job=pathlib.Path(p['job_root']); assert not (job/'stages/dev_plan').exists()
entries=[job,*job.iterdir(),*job.glob('*/*')]; assert len(entries)<128 and all(not x.is_symlink() for x in entries)
used=sum(max(x.stat().st_size,x.stat().st_blocks*512) for x in entries)
assert used<1<<20
now=time.time(); deadline=min(now+1200,cpdoc['valid_until_unix']); assert deadline-now>=640
free=os.statvfs(B).f_bavail*os.statvfs(B).f_frsize
assert free>cpdoc['remaining_bytes']+(1<<40)+(32<<20)
helpers={name:hashlib.sha256((H/name).read_bytes()).hexdigest() for name in ('supervise_once.py','observed_cli.py','prepare_bindings.py')}
a=dict(schema='st_SP0_probe_stage_admission_v1',status='APPROVED',reviewer='/root/st_metadata_resource_review',stage='dev_plan',hostname=socket.gethostname(),cpu_affinity=[0,1],valid_until_unix=deadline,issued_at_unix=now,preparation=pin(P),runtime=p['runtime'],helpers=helpers,gpu=None,storage_policy=None,typed_storage_approved=False,
 limits=dict(wall_seconds=600,cleanup_seconds=40,observed_rss_bytes=64<<30,role_plan_plus_extract_read_bytes=256<<30,gpu_count=0,stage_control_logical_bytes=5<<20),
 preserved_policy=pin(pol),preserved_ledger=pin(ledger),preserved_report_budget=pin(report),checkpoint_commitments=pin(cp),
 report_reservation=dict(root=str(job),maximum_bytes=32<<20,scope='all SP0 controls/stages/logs/temp; excludes separately gated fit and W outputs',cache_ledger_claim=False,prior_metadata_effective_bytes=1200<<20,effective_metadata_bytes=1232<<20,metadata_limit_bytes=4<<30,preserve_on_next_migration=True),
 excluded=['GPU extraction','train_plan','fit','ledger or policy migration','formal temporal training','automatic retry'])
sys.path.insert(0,str(H));spec=importlib.util.spec_from_file_location('independent_sp0_supervisor',H/'supervise_once.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.validate(a,p,'dev_plan',now)==(600,64<<30)
negative=[]
for name,modify in [('expired',lambda x:x.update(valid_until_unix=now+629)),('gpu',lambda x:x.update(gpu={'index':0,'uuid':'GPU-test'})),('wrong_runtime',lambda x:x['runtime'].update(source_sha='0'*40)),('wrong_helper',lambda x:x['helpers'].update({'supervise_once.py':'0'*64})),('one_cpu',lambda x:x.update(cpu_affinity=[0]))]:
 x=copy.deepcopy(a);modify(x)
 try:m.validate(x,p,'dev_plan',now)
 except ValueError:negative.append(name)
 else:raise AssertionError(name)
assert 'torch' not in sys.modules
(D/'DEV_PLAN_ADMISSION.json').write_text(json.dumps(a,sort_keys=True,indent=2)+'\n')
e=dict(status='PASS_BOUNDED_DEV_PLAN_ADMISSION',runtime=p['runtime'],checked_at_unix=now,metadata_bytes_read=sum(x['bytes'] for x in reads),pins=reads,helper_sha256=helpers,ledger_rows=7,sp0_controls_observed_charged_bytes=used,shared_free_bytes=free,checkpoint_remaining_bytes=cpdoc['remaining_bytes'],effective_metadata_reserved_bytes=1232<<20,negative_controls=negative,positive_validate=True,native_payload_reads=0,image_reads=0,tensor_reads=0,gpu_calls=0,live_policy_ledger_writes=0,actual_plan_run=False,admission=pin(D/'DEV_PLAN_ADMISSION.json'))
(D/'REVIEW_EVIDENCE.json').write_text(json.dumps(e,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in e.items() if k not in ('pins',)},sort_keys=True))
