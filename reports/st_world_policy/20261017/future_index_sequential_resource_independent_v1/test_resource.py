import ast,hashlib,importlib.util,os,signal,tempfile,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
PATH=ROOT/'reports/st_world_policy/20261017/future_index_sequential_control_v1/prepare_once.py'
spec=importlib.util.spec_from_file_location('independent_index_prepare',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Resource(unittest.TestCase):
 def test_exact_limits(self):self.assertEqual(m.LIMITS,dict(wall_seconds=120,cleanup_seconds=40,read_bytes=384<<20,rss_bytes=1<<30))
 def test_fixed_monitor_pin(self):self.assertEqual(hashlib.sha256((m.LIB/'full_future_metadata_preparation_v1/run_build_once.py').read_bytes()).hexdigest(),m.CORE_SHA)
 def test_fixed_finite_adapter_pin(self):self.assertEqual(hashlib.sha256(m.ADAPTER.read_bytes()).hexdigest(),m.ADAPTER_SHA)
 def test_regular_pin_pass(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.py';p.write_bytes(b'x=1\n');self.assertEqual(m.read_source(p,hashlib.sha256(b'x=1\n').hexdigest()),b'x=1\n')
 def test_wrong_hash_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.py';p.write_bytes(b'x=1\n')
   with self.assertRaises(ValueError):m.read_source(p,'0'*64)
 def test_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.py';p.write_bytes(b'x=1\n');q=Path(d)/'b.py';q.symlink_to(p)
   with self.assertRaises(ValueError):m.read_source(q,'0'*64)
 def test_relative_rejected(self):
  with self.assertRaises(ValueError):m.read_source('badrelative','0'*64)
 def test_fifo_rejected_without_wait(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'source.py';os.mkfifo(p)
   previous=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('independent200msdeadline: FIFO blocked before source verification')))
   signal.setitimer(signal.ITIMER_REAL,.2)
   try:
    with self.assertRaises(ValueError):m.read_source(p,'0'*64)
   finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
 def test_first_source_load_is_under_absolute_deadline(self):
  tree=ast.parse(PATH.read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
  unguarded=[x for x in main.body if isinstance(x,ast.Assign) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id=='load']
  self.assertEqual(unguarded,[],'first adapter load executes before alarm context')
 def test_boot_accounting_before_nontrivial_imports(self):
  s=PATH.read_text();self.assertLess(s.index('BOOT_IO='),s.index('import argparse'));self.assertIn('core.BOOT_IO=BOOT_IO;core.START=START',s)
 def test_monitor_and_wait_reuse(self):
  s=PATH.read_text();self.assertIn('observed=core.monitor(',s);self.assertIn('charged=core.charge_after_reap(',s);self.assertNotIn('subprocess.Popen(',s);self.assertIn("'automatic_retry':False",s)
if __name__=='__main__':unittest.main(verbosity=2)
