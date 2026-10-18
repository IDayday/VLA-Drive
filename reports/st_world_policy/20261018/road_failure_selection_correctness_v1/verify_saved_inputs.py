"""Independent identity-only check; no cache, proxy, simulator or model call."""
import hashlib,json,os,resource,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tools.st_world_policy import run_road_direction as runner
from tools.st_world_policy import road_failure_selection as selection
import torch
started=time.monotonic();cpu=resource.getrusage(resource.RUSAGE_SELF)
allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:2]))
signal.alarm(120)
folder=Path('/mnt/project/st-world-policy-20261016/audits/consequence/road_failure_direction_v1')
reg_raw=(folder/'REGISTRATION.json').read_bytes(); reg=json.loads(reg_raw)
assert hashlib.sha256(reg_raw).hexdigest()=='c9c76ffd513fcf3cf59e1753358bd552b0635ae6394399e509d15c09dd9cda0a'
assert reg['source_commit']=='864619b0130fb3b220b577dd34b993ffeac97e27'
assert reg['source_hashes']==runner.source_hashes()
prepared=selection.prepare(reg['evidence']['path'])
assert prepared['evidence']==reg['evidence'] and prepared['scene_count']==32
assert prepared['selection_hash']==reg['selection_hash']
evidence=json.loads(Path(reg['evidence']['path']).read_bytes())
limits={**runner.FAILURE_LIMITS,'asset_read_bytes':240*2**20,'rss_and_address_space_bytes':2*2**30,'wall_seconds':120,'cpu_seconds':90}
class ReadOnlyBudget(runner.Budget):
 def read(self,path,*args,**kwargs):
  assert not str(path).endswith('metric_cache.pkl'), 'No metric context content allowed in correctness preparation'
  return super().read(path,*args,**kwargs)
 def write(self,*args,**kwargs): raise AssertionError('Identity verifier cannot write runtime assets')
budget=ReadOnlyBudget(Path(__file__).parent,limits)
loaded=selection.load_inputs(evidence,budget)
assert len(loaded)==32
counts={'failed_DAC_0':0,'native_t0_false':0,'exact_saved_raw':0,'full_8x4_raw':0}
for item,raw,request,record in loaded:
 assert record['metrics']['drivable_area_compliance']==0.;counts['failed_DAC_0']+=1
 assert selection.native_t0(record) is False;counts['native_t0_false']+=1
 assert raw.dtype==torch.float32 and tuple(raw.shape)==(8,4);counts['full_8x4_raw']+=1
 assert hashlib.sha256(raw.numpy().tobytes()).hexdigest()==item['failed']['raw_sha256'];counts['exact_saved_raw']+=1
 runner.decode_discrepancy_gate(raw,request.trajectory)
assert not torch.cuda.is_initialized()
usage=resource.getrusage(resource.RUSAGE_SELF)
result={'status':'PASS','scope':'actual execution reader + frozen frame/hash/request/context/reference/raw/native_t0; no proxy or official scores',
 'registration_sha256':hashlib.sha256(reg_raw).hexdigest(),'bound_selection_sha256':reg['evidence']['sha256'],
 'requested_scenes':32,'loaded_scenes':32,'unique_logs':len({x[0]['source_log'] for x in loaded}),
 'counts':counts,'new_official_calls':0,'proxy_calls':0,'model_loads':0,'metric_cache_content_reads':0,'cuda_initialized':False,
 'wall_seconds':time.monotonic()-started,'cpu_seconds':usage.ru_utime+usage.ru_stime-cpu.ru_utime-cpu.ru_stime,
 'logical_asset_bytes':budget.read_bytes,'peak_rss_bytes':usage.ru_maxrss*1024,'asset_reads':budget.reads}
Path(__file__).with_name('REAL_INPUT_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='asset_reads'},sort_keys=True))
