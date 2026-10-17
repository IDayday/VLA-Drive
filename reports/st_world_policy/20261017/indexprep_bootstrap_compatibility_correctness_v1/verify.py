"""Narrow regular-source/argv preservation check; no real preparation or ledger."""
import ast,hashlib,importlib.util,json,os,resource,subprocess,sys,tempfile,time,unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
START=time.monotonic();OUT=Path(__file__).resolve().parent
ROOT=OUT.parent;NEW=ROOT/'future_index_sequential_control_v1/prepare_once.py'
OLD=Path('/mnt/project/VLA-Drive-st-finalized-index-controller-review-20261017/reports/st_world_policy/20261017/future_index_sequential_control_v1/prepare_once.py')
BIND=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/future_index_preparation_actual_v2/BINDINGS.json')
def io():return {k:int(v) for k,v in (x.split(':') for x in Path('/proc/self/io').read_text().splitlines())}
startio=io();new=NEW.read_bytes();old=OLD.read_bytes();bindingsraw=BIND.read_bytes();binding=json.loads(bindingsraw)
assert hashlib.sha256(new).hexdigest()=='49b21405d1dacee17405598c1a7165727db9855af4de66c2feff70ab30b93ce5'
assert hashlib.sha256(old).hexdigest()=='c749b4f9181b407cdb0b0684d7d26aeb52db8c31664b3b6f353def90f629ad1e'
assert hashlib.sha256(bindingsraw).hexdigest()=='c15e726202a788eb027a8ce42f3c484dee1320dfb9338c9abd79f9fbf94a1cd0'
spec=importlib.util.spec_from_file_location('reviewed_indexprep_bootstrap',NEW);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
subprocess_rchar=0
class Checks(unittest.TestCase):
 def test_existing_run_options_load_and_constants_unchanged(self):
  a,b=ast.parse(old),ast.parse(new)
  for name in ['run','options','load']:
   oldnode=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name)
   newnode=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(oldnode,include_attributes=False),ast.dump(newnode,include_attributes=False))
  assignments=lambda tree:[ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.Assign)]
  self.assertEqual(assignments(a),assignments(b))
 def test_regular_source_exact_bytes_and_wrong_hash(self):
  with tempfile.TemporaryDirectory(prefix='index-bootstrap-',dir='/tmp') as d:
   p=Path(d)/'x.py';v=b'x = 1\n';p.write_bytes(v)
   self.assertEqual(m.read_source(p,hashlib.sha256(v).hexdigest()),v)
   with self.assertRaises(ValueError):m.read_source(p,'0'*64)
 def test_initial_alarm_then_same_deadline_finite_run(self):
  events=[];args=object()
  @contextmanager
  def initial(deadline):events.append(('initial_enter',deadline));yield;events.append(('initial_exit',deadline))
  @contextmanager
  def finite_alarm(deadline):events.append(('finite_enter',deadline));yield;events.append(('finite_exit',deadline))
  finite=SimpleNamespace(whole_invocation_alarm=finite_alarm)
  def load(*a):events.append(('load',a));return finite
  def run(a,f):self.assertIs(a,args);self.assertIs(f,finite);events.append(('run',));return 17
  with patch.object(m,'first_source_alarm',initial),patch.object(m,'options',return_value=args),patch.object(m.os,'sched_setaffinity'),patch.object(m,'load',load),patch.object(m,'run',run):self.assertEqual(m.main(),17)
  self.assertEqual([e[0] for e in events],['initial_enter','load','initial_exit','finite_enter','run','finite_exit'])
  self.assertEqual(events[0][1],events[3][1]);self.assertEqual(events[0][1],m.START+m.LIMITS['wall_seconds'])
 def test_actual_help_and_bad_args(self):
  global subprocess_rchar
  # Real --help path exits before assigning worker affinity or loading the adapter.
  wrapper="import runpy,sys,pathlib,json; p=sys.argv[1]; sys.argv=sys.argv[1:]; before=int(pathlib.Path('/proc/self/io').read_text().split('rchar: ')[1].splitlines()[0]); code=0\ntry: runpy.run_path(p,run_name='__main__')\nexcept SystemExit as e: code=e.code\nafter=int(pathlib.Path('/proc/self/io').read_text().split('rchar: ')[1].splitlines()[0]); print('CHILD_RCHAR='+str(after-before),file=sys.stderr); raise SystemExit(code)"
  for args,code in [(['--help'],0),(['--unknown'],2)]:
   p=subprocess.run([sys.executable,'-c',wrapper,str(NEW),*args],capture_output=True,text=True,timeout=5)
   self.assertEqual(p.returncode,code);subprocess_rchar+=int(p.stderr.rsplit('CHILD_RCHAR=',1)[1].strip())
   if code==0:
    for arg in ['--wrapper-sha256','--preparer','--policy','--ledger-sha256','--output','--train-index-source-pins']:self.assertIn(arg,p.stdout)
    (OUT/'HELP.txt').write_text(p.stdout)
 def test_actual_bindings_match_code_and_preserve_budget(self):
  pins=binding['pins'];argv=binding['argv'];options=dict(zip(argv[2::2],argv[3::2]))
  self.assertEqual(pins['wrapper']['sha256'],hashlib.sha256(new).hexdigest());self.assertEqual(options['--wrapper-sha256'],pins['wrapper']['sha256'])
  for name,arg in [('preparer','preparer'),('train_source_pins','train-index-source-pins')]:
   self.assertEqual(options['--'+arg],pins[name]['path']);self.assertEqual(options['--'+arg+'-sha256'],pins[name]['sha256'])
  self.assertEqual(options['--output'],'/mnt/project/st-world-policy-20261016/audits/full_future_index_preparation_v1/finalized_phase_v1')
  limits=binding['limits'];self.assertEqual((limits['wall_seconds'],limits['cleanup_seconds'],limits['rss_bytes'],limits['total_logical_read_bytes'],limits['total_new_output_bytes']),(120,40,1<<30,384<<20,1<<20))
  self.assertFalse(binding['ledger_mutation_authorized']);self.assertFalse(binding['index_build_authorized']);self.assertFalse(binding['model_execution_authorized']);self.assertFalse(binding['automatic_retry']);self.assertEqual(binding['env']['CUDA_VISIBLE_DEVICES'],'')
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));end=io()
cost={'wall_seconds':time.monotonic()-START,'parent_rchar_bytes':end['rchar']-startio['rchar'],'child_rchar_bytes':subprocess_rchar,'total_measured_rchar_bytes':end['rchar']-startio['rchar']+subprocess_rchar,'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_affinity':sorted(os.sched_getaffinity(0))}
assert cost['wall_seconds']<60 and cost['total_measured_rchar_bytes']<8<<20 and cost['maxrss_bytes']<1<<30
value={'status':'PASS_NARROW_BOOTSTRAP_COMPATIBILITY_ONLY' if r.wasSuccessful() else 'FAIL','source_commit':'e8ac21550ae0d38bf8414486282c0b41908d3db3','wrapper_sha256':hashlib.sha256(new).hexdigest(),'bindings_sha256':hashlib.sha256(bindingsraw).hexdigest(),'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'cost':cost,'torch_imported':'torch' in sys.modules,'resource_FIFO_test':'SEPARATE_REVIEW','actual_launch_approved':False,'ledger_or_payload_read':False}
(OUT/'REVIEW.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps(value));raise SystemExit(0 if r.wasSuccessful() else 1)
