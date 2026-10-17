import time,os,resource,sys,json,importlib.util,unittest
from pathlib import Path
START=time.monotonic();ROOT=Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30));resource.setrlimit(resource.RLIMIT_FSIZE,(16<<20,16<<20))
def io():return {k:int(v) for k,v in (x.split(':') for x in Path('/proc/self/io').read_text().splitlines())}
boot=io();spec=importlib.util.spec_from_file_location('independent_train_fit',ROOT/'test_independent.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);loaded=io()
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in [m.owner.Consumer,m.Independent]])
r=unittest.TextTestRunner(verbosity=2).run(suite);end=io()
cost={'wall_seconds':time.monotonic()-START,'startup_rchar_bytes':loaded['rchar']-boot['rchar'],'test_rchar_bytes':end['rchar']-loaded['rchar'],'total_rchar_bytes':end['rchar']-boot['rchar'],'writes_bytes':end['wchar']-boot['wchar'],'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_affinity':sorted(os.sched_getaffinity(0))}
assert cost['wall_seconds']<120 and cost['total_rchar_bytes']<64<<20 and cost['peak_RSS_bytes']<1<<30 and cost['writes_bytes']<16<<20
v={'status':'PASS_NARROW_TRAIN64_FIT_CONSUMER' if r.wasSuccessful() else 'FAIL','source_commit':'875e1437339d06863d978e1f72c62b0a9ca9f7af','tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'cost':cost,'actual_W_payload_read':False,'Torch_imported':'torch' in sys.modules,'real_fit_or_dispatch':False,'resource_admission':False,'original_failed_status_preserved':True,'artifact_review_inherited_sha256':'725d3427d042f47882e89a5814937b10e3a7c49b5ea58da690e374c7b37dbfad'}
(ROOT/'REVIEW.json').write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps(v));raise SystemExit(0 if r.wasSuccessful() else 1)
