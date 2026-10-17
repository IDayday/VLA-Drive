"""Bounded static composition proof; no framework imports or actual metadata bodies."""
import ast,copy,hashlib,json,os,resource,time
from pathlib import Path
START=time.monotonic();OUT=Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30))
ROOT=Path('/mnt/project/VLA-Drive-st-finalized-full-profile-runtime-v2-20261017')
OLD=Path('/mnt/project/VLA-Drive-st-full-index-profile-20261017')
EXACT=Path('/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017')
DONOR=Path('/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017')
PKG=OUT.parent/'finalized_full_profile_source_preparation_v1'
CACHE={};ROWS=[];N=0;CHECKS=[]
def read(p):
 global N
 p=Path(p)
 if p not in CACHE:
  before=p.stat();assert before.st_size<1<<20;assert N+before.st_size<=32<<20
  data=p.read_bytes();after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  N+=len(data);CACHE[p]=data;ROWS.append({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 return CACHE[p]
def parsed(p):return ast.parse(read(p))
def check(name,condition):
 assert condition,name
 CHECKS.append(name)
q=parsed(ROOT/'tools/st_world_policy/qualify_indexed_provider.py')
math=next(ast.literal_eval(n.value) for n in q.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MATH_PATHS' for t in n.targets))
check('exact_original_15_math_paths',len(math)==15)
for rel in math:check('preserved:'+rel,read(ROOT/rel)==read(OLD/rel)==read(EXACT/rel))
for rel in ['tools/st_world_policy/train_world.py','tools/st_world_policy/training_config.py','tools/st_world_policy/qualify_indexed_provider.py','starVLA/dataloader/st_world_dataset.py','tools/st_world_policy/official_runtime.py','tools/ddpolicy_vehicle/training_state.py','tools/foresight/student_state.py','tools/st_world_policy/prepare_history.py','tools/st_world_policy/metadata_storage.py']:
 check('original_runtime:'+rel,read(ROOT/rel)==read(OLD/rel))
for rel in ['tools/st_world_policy/prepare_future_metadata.py','tools/st_world_policy/prepare_current_teacher.py','starVLA/dataloader/st_world_metadata_index.py','starVLA/dataloader/st_world_future_provenance.py']:
 check('read_dependency:'+rel,read(ROOT/rel)==read(OLD/rel)==read(DONOR/rel))
for rel in ['tools/st_world_policy/finalize_future_metadata.py','tools/st_world_policy/teacher_provider.py']:
 check('reviewed_donor_exact:'+rel,read(ROOT/rel)==read(DONOR/rel))
new=parsed(ROOT/'tools/st_world_policy/teacher_provider.py');old=parsed(OLD/'tools/st_world_policy/teacher_provider.py')
function=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_validate_full_population')
class RestoreOld(ast.NodeTransformer):
 def visit_Assign(self,node):
  if any(isinstance(t,ast.Name) and t.id in ('future_name','future_schema') for t in node.targets):return None
  return self.generic_visit(node)
 def visit_If(self,node):
  if 'st_future_metadata_finalization_complete_v1' in ast.unparse(node.test):return None
  return self.generic_visit(node)
 def visit_Name(self,node):
  if node.id=='future_name':return ast.Constant(value='COMPLETE.json')
  if node.id=='future_schema':return ast.Constant(value='st_future_metadata_complete_v1')
  return node
normalized=RestoreOld().visit(copy.deepcopy(new))
check('all_legacy_provider_semantics_identical_after_only_new_branch_removed',ast.dump(normalized,include_attributes=False)==ast.dump(old,include_attributes=False))
branch=next(n for n in function.body if isinstance(n,ast.If) and 'st_future_metadata_finalization_complete_v1' in ast.unparse(n.test))
call=next(n for n in ast.walk(branch) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='validate_completion')
check('new_branch_real_validator_import',any(isinstance(n,ast.ImportFrom) and n.module=='tools.st_world_policy.finalize_future_metadata' for n in branch.body))
check('new_branch_passes_all_verified_index_dependencies', {x.arg for x in call.keywords}=={'request_sha256','request_identity','manifest_header','scene_count'})
newtext=ast.unparse(function)
for expected in ['entire_original_index_in_original_order','selected/request/current order, observation or time differs','request.check()','source.check()','next(future_rows, None)','next(current_rows, None)','selection_hash.hexdigest()','token_hash.hexdigest()']:
 check('full_population_retains:'+expected,expected in newtext)
h=parsed(ROOT/'tools/st_world_policy/finalize_future_metadata.py');functions={n.name:n for n in h.body if isinstance(n,ast.FunctionDef)}
seen=set();todo=['validate_completion'];attrs=set()
while todo:
 name=todo.pop()
 if name in seen:continue
 seen.add(name)
 for node in ast.walk(functions[name]):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in functions:todo.append(node.func.id)
  if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='f':attrs.add(node.attr)
f=parsed(ROOT/'tools/st_world_policy/prepare_future_metadata.py');exports=set()
for n in f.body:
 if isinstance(n,(ast.FunctionDef,ast.ClassDef)):exports.add(n.name)
 elif isinstance(n,(ast.Import,ast.ImportFrom)):
  for alias in n.names:exports.add(alias.asname or alias.name.split('.')[0])
 elif isinstance(n,ast.Assign):
  for t in n.targets:
   exports.update(x.id for x in ast.walk(t) if isinstance(x,ast.Name))
check('consumer_helper_f_symbols_present_in_unchanged_prepare_future',attrs<=exports)
check('consumer_no_finalizer_writer_execution',not({'finalize','run','main','build'}&seen))
lineage=functions['_lineage'];check('consumer_lineage_check_code_default_false',len(lineage.args.kw_defaults)==1 and ast.literal_eval(lineage.args.kw_defaults[0]) is False)
consumer=functions['validate_completion'];lineagecalls=[n for n in ast.walk(consumer) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_lineage']
check('consumer_lineage_uses_default_read_only_branch',len(lineagecalls)==1 and not lineagecalls[0].keywords)
check('helper_has_no_packed_history_name',b'packed_log' not in read(ROOT/'tools/st_world_policy/finalize_future_metadata.py'))
config=json.loads(read(PKG/'DRAFT_TRAINING_CONFIG.json'));budget=json.loads(read(PKG/'DRAFT_CONSUMPTION.json'))
check('draft_not_launchable_unassigned_fields',config['runtime']['max_seconds'] is None and config['runtime']['save_every'] is None and config['runtime']['output_root'].startswith('__UNASSIGNED'))
check('cumulative_stop64_128_exposures',config['execution']['stop_after']==64 and budget['cumulative_optimizer_updates']==64 and budget['cumulative_scene_exposures']==128 and budget['resume_budget']=='same_run_cumulative_not_per_invocation')
check('history_delta_off_original_time_axis',not config['space_time']['history'] and not config['space_time']['delta'] and config['execution']['max_updates']==20000 and config['optimizer']['warmup']==1000)
check('no_torch_import','torch' not in __import__('sys').modules)
result={'status':'PASS_NARROW_STATIC_COMPOSITION_ONLY','runtime_commit':'a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a','proposal_commit':'5327cd32823ca2e69f56efbb4d0118edff5618b1','checks':CHECKS,'helper_consumer_functions':sorted(seen),'helper_prepare_future_attributes':sorted(attrs),'source_pins':ROWS,'measurement':{'source_control_bytes':N,'wall_seconds':time.monotonic()-START,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_affinity':sorted(os.sched_getaffinity(0))},'actual_population_payload_read':False,'actual_full_population_order_verified':False,'torch_imported':False,'training_or_dispatch_permission':False}
(OUT/'REVIEW.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['source_pins','checks']},indent=2));print('CHECKS_PASSED',len(CHECKS))
