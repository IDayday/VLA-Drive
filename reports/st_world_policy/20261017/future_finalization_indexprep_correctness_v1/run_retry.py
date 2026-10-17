import time,os,resource,sys,json,importlib.util,unittest,threading
from pathlib import Path
start=time.monotonic();root=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']='';resource.setrlimit(resource.RLIMIT_FSIZE,(16<<20,16<<20))
def io():return {k:int(v) for k,v in (x.split(':') for x in Path('/proc/self/io').read_text().splitlines())}
def rss_watch():
 while True:
  data=Path('/proc/self/status').read_text()
  rss=int(next(v.split()[1] for v in data.splitlines() if v.startswith('VmRSS:')))*1024
  if rss>2<<30:
   os.write(2,b'RSS_BUDGET_EXCEEDED\n');os._exit(97)
  time.sleep(.05)
threading.Thread(target=rss_watch,daemon=True).start()
boot=io();sys.path.insert(0,'/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017')
spec=importlib.util.spec_from_file_location('independent_route',root/'test_narrow.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
loaded=io();startup=time.monotonic()-start
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in [m.RealSharedValidatorAdapterTests]])
result=unittest.TextTestRunner(verbosity=2).run(suite);end=io()
cost={'wall_seconds':time.monotonic()-start,'startup_seconds':startup,'startup_read_bytes':loaded['rchar']-boot['rchar'],'test_read_bytes':end['rchar']-loaded['rchar'],'total_read_bytes':end['rchar']-boot['rchar'],'test_write_bytes':end['wchar']-loaded['wchar'],'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_affinity':sorted(os.sched_getaffinity(0))}
bounds=cost['wall_seconds']<60 and cost['RSS_bytes']<2<<30 and cost['total_read_bytes']<145<<20 and cost['test_read_bytes']<64<<20 and cost['test_write_bytes']<64<<20
value={'status':'PASS_SYNTHETIC_INDEXPREP_ONLY' if result.wasSuccessful() and bounds else 'FAIL','source_commit':'99ff3af1b255cf3b37b13954850e8b657c6b493f','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'bounds_pass':bounds,'cost':cost,'torch_imported':'torch' in sys.modules,'CUDA_initialized':sys.modules['torch'].cuda.is_initialized() if 'torch' in sys.modules else False,'actual_payload_or_ledger_access':False,'launch':False}
(root/'RETRY_TEST_RESULT.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');print(json.dumps(value));raise SystemExit(0 if result.wasSuccessful() and bounds else 1)
