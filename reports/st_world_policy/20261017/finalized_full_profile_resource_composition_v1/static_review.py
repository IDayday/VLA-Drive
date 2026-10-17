from pathlib import Path
import ast,hashlib,json,os,resource,signal,subprocess,time
signal.alarm(120);os.sched_setaffinity(0,{8,9});resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30));t=time.monotonic();readbytes=0
runtime=Path('/mnt/project/VLA-Drive-st-finalized-full-profile-runtime-v2-20261017');draft=Path('/mnt/project/VLA-Drive-st-finalized-profile-prep-20261017/reports/st_world_policy/20261017/finalized_full_profile_source_preparation_v1');out=Path(__file__).parent
pins=[]
def raw(p):
 global readbytes
 p=Path(p);assert p.stat().st_size<4<<20;b=p.read_bytes();readbytes+=len(b);assert readbytes<32<<20;pins.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)});return b
def git(*args):return subprocess.check_output(['git',*args],cwd=runtime,text=True,timeout=15).strip()
assert git('rev-parse','HEAD')=='a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a' and not git('status','--porcelain=v1')
changed=git('diff','--name-only','bccebb00319512049e93df58d18cb5ead49575f0','HEAD').splitlines()
assert changed==['tools/st_world_policy/finalize_future_metadata.py','tools/st_world_policy/teacher_provider.py']
finalizer=raw(runtime/changed[0]);provider=raw(runtime/changed[1]);assert hashlib.sha256(finalizer).hexdigest()=='e3443e703e40f16747a0d0bf307dd895045b4406d0e69a9a0b06e7300415e98d'
# Exact reader implementation copied from reviewed finalizer runtime, not merely same function name.
donor=Path('/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017');assert finalizer==raw(donor/changed[0])
tree=ast.parse(finalizer);functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
validator=functions['validate_completion'];calls={n.func.id for n in ast.walk(validator) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
assert not calls.intersection({'finalize','shard_census','CompressedScenes','main'})
reads=[n for n in ast.walk(validator) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_read'];assert len(reads)==2
assert all(isinstance(n.args[0],ast.BinOp) and isinstance(n.args[0].right,ast.Constant) and n.args[0].right.value in ('RAW_LOG_PROVENANCE.json','SHARD_RECEIPTS.json') for n in reads)
# Keep original population streaming implementation. The only provider delta is the advertised new receipt branch.
diff=git('diff','--numstat','bccebb00319512049e93df58d18cb5ead49575f0','HEAD','--','tools/st_world_policy/teacher_provider.py');assert diff.startswith('10\t2\t')
config=json.loads(raw(draft/'DRAFT_TRAINING_CONFIG.json'));consumption=json.loads(raw(draft/'DRAFT_CONSUMPTION.json'));inputs=json.loads(raw(draft/'INPUT_BINDINGS.json'))
e=config['execution'];assert (e['scope'],e['sample_limit'],e['stop_after'],e['world_size'],e['micro_batch'],e['global_batch'],e['max_updates'])==('PROFILE',101592,64,2,1,2,20000)
assert config['optimizer']['warmup']==1000 and config['runtime']['loader_workers']==0 and config['runtime']['max_seconds'] is None and config['runtime']['save_every'] is None
assert consumption['cumulative_optimizer_updates']==64 and consumption['cumulative_scene_exposures']==128 and consumption['per_rank_scene_exposures']==[64,64] and consumption['resume_budget']=='same_run_cumulative_not_per_invocation'
assert inputs['actual_full_future_index'] is None and inputs['new_teacher_tensor_disk_bytes']==0 and inputs['formal_approved'] is False
receipt={'status':'PASS_STATIC_RESOURCE_COMPOSITION_ONLY','reviewer':'/root/st_owned_descendants_review_v2','runtime':{'path':str(runtime),'commit':git('rev-parse','HEAD'),'clean':True},'source_base':'bccebb00319512049e93df58d18cb5ead49575f0','implementation_package_commit':'5327cd32823ca2e69f56efbb4d0118edff5618b1','changed_files':changed,'source_pins':pins,'new_teacher_tensor_disk_bytes':0,'new_target_precision_changes':False,'new_full_population_or_per_batch_scans_added':False,'existing_startup_scan_boundary':'Unchanged full population ordered scan occurs during provider metadata preview, bound PCA scope validation, and create_provider (about3/rank/invocation); resumes incur fresh startup. MetadataIndex initial dependency hashing and existing native data checks remain. These are not zero-cost and remain inside actual whole-job admission.','new_consumer_cost':'Strict fixed receipts/lineage plus outputRAW_LOG_PROVENANCE andSHARD_RECEIPTS; per-file8MiB bound and originalcontroller<=1MiB. Iterates receipt entries but does not read shard bodies, rawlogs, requestsbody, teacher tensors, or invoke new writer.','draft_consumption':consumption,'actual_training_approval':False,'actual_throughput_verified':False,'remaining':['actual full TRAIN request index and dependency verification','new source migration/provider/PCA application scope','filled finite runtime max_seconds/save_every/output and new run identity','fresh whole-job read/RSS/GPU/checkpoint/storage bounds including all startup validations and restarts','actual bounded mixed-layout two-rank profile and resume evidence'],'review_cost':{'wall_seconds':time.monotonic()-t,'explicit_source_control_read_bytes':readbytes,'actual_self_VmHWM_bytes':next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')),'cpu_affinity':sorted(os.sched_getaffinity(0))},'Torch_imported':False,'real_payload_or_ledger_read':False,'dispatch':False}
(out/'REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'sha256':hashlib.sha256((out/'REVIEW.json').read_bytes()).hexdigest(),'cost':receipt['review_cost']}))
