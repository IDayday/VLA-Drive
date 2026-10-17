import importlib.util,json,os,resource,signal,sys,time,unittest
from pathlib import Path
start=time.monotonic();root=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']=''
def io():return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
boot=io()
def timeout(*_):raise RuntimeError('180s bounded correctness control review deadline')
signal.signal(signal.SIGALRM,timeout);signal.alarm(180);resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30));resource.setrlimit(resource.RLIMIT_FSIZE,(16<<20,16<<20))
spec=importlib.util.spec_from_file_location('independent_control_tests',root/'test_narrow.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
startup=io();startup_seconds=time.monotonic()-start
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(m.owner.Controls),unittest.defaultTestLoader.loadTestsFromTestCase(m.ActualDraftWiring)])
result=unittest.TextTestRunner(verbosity=2).run(suite);end=io()
cost={'elapsed_seconds':time.monotonic()-start,'startup_seconds':startup_seconds,'startup_rchar':startup['rchar']-boot['rchar'],'test_rchar':end['rchar']-startup['rchar'],'total_rchar':end['rchar']-boot['rchar'],'test_wchar':end['wchar']-startup['wchar'],'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_affinity':sorted(os.sched_getaffinity(0))}
limits=cost['elapsed_seconds']<180 and cost['peak_RSS_bytes']<2<<30 and cost['total_rchar']<32<<20 and cost['test_wchar']<64<<20
value={'schema':'st_future_finalization_control_correctness_test_v1','code_commit':'7d698b6d6f04661833f03f73ce4f0e92f23820cb','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'status':'PASS_CONTROL_SYNTHETIC_ONLY' if result.wasSuccessful() and limits else 'FAIL','cost':cost,'bounds_pass':limits,'imports_torch': 'torch' in sys.modules,'actual_admission_created':False,'actual_ledger_read_or_mutated':False,'actual_source_payload_read':False,'launch':False}
(root/'TEST_RESULT.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps(value));signal.alarm(0);raise SystemExit(0 if result.wasSuccessful() and limits else 1)
