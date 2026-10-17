"""Three approved synthetic cases under actual ddp CPython; no real inputs."""
import contextlib,hashlib,io,json,os,platform,resource,signal,sys,time,zlib
from pathlib import Path
start=time.monotonic();report_root=Path(__file__).resolve().parent
runtime=Path('/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017')
assert Path.cwd()==runtime
sys.path.insert(0,str(runtime));sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']=''
def counters():return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
base=counters()
def deadline(*_):raise RuntimeError('60second runtime review budget exhausted')
signal.signal(signal.SIGALRM,deadline);signal.alarm(60);resource.setrlimit(resource.RLIMIT_FSIZE,(16<<20,16<<20))
import pytest,torch
from tools.st_world_policy import finalize_future_metadata as final
from tools.st_world_policy import metadata_storage,teacher_provider
from tests.st_world import test_finalize_future_metadata
import_end=counters();import_seconds=time.monotonic()-start
pins=[]
for p in [Path(final.__file__),Path(metadata_storage.__file__),Path(teacher_provider.__file__),Path(test_finalize_future_metadata.__file__)]:
 b=p.read_bytes();pins.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
help_output=io.StringIO()
with contextlib.redirect_stdout(help_output):
 try:final.parser().parse_args(['--help'])
 except SystemExit as e:assert e.code==0
assert 'finalize-existing' in help_output.getvalue();print('ACTUAL_PARSER_HELP\n'+help_output.getvalue())
strict=[]
for argv in [[],['build'],['finalize-existing'],['finalize-existing','--registration','/synthetic/r.json','--registration-sha256','a'*64,'--output','/synthetic/out','--unknown-option']]:
 with contextlib.redirect_stderr(io.StringIO()):
  try:final.parser().parse_args(argv)
  except SystemExit as e:assert e.code==2;strict.append(argv)
  else:raise AssertionError('invalid args accepted')
parsed=final.parser().parse_args(['finalize-existing','--registration','/synthetic/r.json','--registration-sha256','a'*64,'--output','/synthetic/out'])
assert parsed.command=='finalize-existing'
# Parser-only paths above are not opened. Fixtures below are local and synthetic.
fixture_root=Path('/tmp/st-finalizer-ddp310-narrow-20261017')
code=pytest.main(['-vv','-p','no:cacheprovider','--basetemp',str(fixture_root),
 'tests/st_world/test_finalize_future_metadata.py::test_finalize_reads_each_shard_once_and_preserves_rows_bytes_and_failed_attempt',
 'tests/st_world/test_finalize_future_metadata.py::test_bounded_decompression_rejects_corruption_and_expansion[bomb]',
 'tests/st_world/test_finalize_future_metadata.py::test_population_consumer_keeps_current_order_checks_and_strict_new_receipt[finalized]'])
end=counters();size=sum(p.stat().st_size for p in fixture_root.rglob('*') if p.is_file() and not p.is_symlink())
cost={'elapsed_seconds':time.monotonic()-start,'import_seconds':import_seconds,'import_logical_read_bytes':import_end['rchar']-base['rchar'],'test_and_parser_logical_read_bytes':end['rchar']-import_end['rchar'],'total_process_logical_read_bytes':end['rchar']-base['rchar'],'test_and_parser_write_bytes':end['wchar']-import_end['wchar'],'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'fixture_retained_bytes':size,'cpu_affinity':sorted(os.sched_getaffinity(0))}
bounded=cost['elapsed_seconds']<60 and cost['total_process_logical_read_bytes']<256<<20 and cost['test_and_parser_logical_read_bytes']<16<<20 and cost['test_and_parser_write_bytes']<16<<20 and cost['peak_RSS_bytes']<2<<30
review={'schema':'st_future_finalization_ddp310_runtime_review_v1','status':'PASS_RUNTIME_SYNTHETIC_ONLY' if code==0 and bounded else 'FAIL','reviewer':'/root/st_calibration_independent_correctness_v2','source_runtime_commit':'3df052ea','source_runtime':str(runtime),'executable':sys.executable,'python':sys.version,'torch':torch.__version__,'zlib_build':zlib.ZLIB_VERSION,'zlib_runtime':zlib.ZLIB_RUNTIME_VERSION,'module_imports':'PASS','parser_help':help_output.getvalue(),'parser_negative_cases':strict,'pytest_exitcode':code,'selected_tests':3,'source_pins':pins,'cuda_initialized':torch.cuda.is_initialized(),'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'actual_data_read':False,'actual_ledger_mutation':False,'dispatch':False,'cost':cost,'cost_bounds_satisfied':bounded,'scope':'Only imported fixed runtime, actual parser, bounded zlib bomb rejection, original-order two-shard finalization and finalized provider receipt consumption. No new controller-adapter review or real build claim.'}
assert torch.cuda.is_initialized() is False
(report_root/'REVIEW.json').write_text(json.dumps(review,indent=2,sort_keys=True)+'\n');print('RUNTIME_REVIEW',json.dumps(review));signal.alarm(0)
raise SystemExit(code if bounded else 99)
