import time,os,resource,sys,json,importlib.util,unittest
from pathlib import Path
start=time.monotonic();root=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']=''
def io():return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
boot=io();resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30));resource.setrlimit(resource.RLIMIT_FSIZE,(16<<20,16<<20))
spec=importlib.util.spec_from_file_location('independent_index_controller',root/'test_narrow.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
loaded=io();startup=time.monotonic()-start
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in [m.owner.Increment,m.NarrowController,m.PreparationWrapper]])
r=unittest.TextTestRunner(verbosity=2).run(suite);end=io()
cost={'wall_seconds':time.monotonic()-start,'startup_seconds':startup,'startup_read_bytes':loaded['rchar']-boot['rchar'],'test_read_bytes':end['rchar']-loaded['rchar'],'total_read_bytes':end['rchar']-boot['rchar'],'test_write_bytes':end['wchar']-loaded['wchar'],'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_affinity':sorted(os.sched_getaffinity(0))}
bounds=cost['wall_seconds']<120 and cost['RSS_bytes']<1<<30 and cost['total_read_bytes']<32<<20 and cost['test_write_bytes']<16<<20
v={'status':'PASS_SYNTHETIC_CONTROLLER_ONLY' if r.wasSuccessful() and bounds else 'FAIL','source_commit':'bf67fad2b77e443395635f8871eb757cbbb8c596','tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'bounds_pass':bounds,'cost':cost,'torch_imported':'torch' in sys.modules,'actual_payload_or_ledger_access':False,'real_dispatch':False}
(root/'TEST_RESULT.json').write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps(v));raise SystemExit(0 if r.wasSuccessful() and bounds else 1)
