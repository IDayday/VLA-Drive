"""CPU-only regression for bootstrap source reads; never invokes preparation."""
import time
START=time.monotonic()
with open('/proc/self/io') as f:BOOT={k:int(v) for k,v in (line.split(':') for line in f)}
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'future_index_sequential_control_v1/prepare_once.py'
spec=importlib.util.spec_from_file_location('boundary_subject',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Boundary(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.p=Path(self.tmp.name)/'source.py';self.p.write_bytes(b'x=1\n')
        self.sha=hashlib.sha256(b'x=1\n').hexdigest()

    def test_regular_roundtrip_and_exact_limits(self):
        self.assertEqual(m.read_source(self.p,self.sha),b'x=1\n')
        self.assertEqual(m.LIMITS,dict(wall_seconds=120,cleanup_seconds=40,read_bytes=384<<20,rss_bytes=1<<30))

    def test_wrong_hash_and_relative_rejected(self):
        with self.assertRaises(ValueError):m.read_source(self.p,'0'*64)
        with self.assertRaises(ValueError):m.read_source('relative.py',self.sha)

    def test_symlink_file_and_parent_rejected(self):
        link=self.p.with_name('link');link.symlink_to(self.p)
        with self.assertRaises(ValueError):m.read_source(link,self.sha)
        directory=self.p.with_name('directory');directory.symlink_to(self.p.parent,target_is_directory=True)
        with self.assertRaises(ValueError):m.read_source(directory/self.p.name,self.sha)

    def test_directory_and_oversize_rejected_before_open(self):
        with self.assertRaises(ValueError):m.read_source(self.p.parent,self.sha)
        with self.p.open('wb') as f:f.truncate((1<<20)+1)
        with patch.object(m.os,'open',side_effect=AssertionError('must not open oversize source')):
            with self.assertRaises(ValueError):m.read_source(self.p,self.sha)

    def test_fifo_has_immediate_type_rejection(self):
        self.p.unlink();os.mkfifo(self.p)
        with m.first_source_alarm(time.monotonic()+.2):
            with self.assertRaisesRegex(ValueError,'regular'):m.read_source(self.p,self.sha)

    def test_lstat_open_fifo_race_is_nonblocking(self):
        original=os.open
        def replace(path,flags):
            self.assertTrue(flags&os.O_NONBLOCK);self.assertTrue(flags&os.O_NOFOLLOW)
            self.p.unlink();os.mkfifo(self.p)
            return original(path,flags)
        with m.first_source_alarm(time.monotonic()+.2),patch.object(m.os,'open',side_effect=replace):
            with self.assertRaisesRegex(ValueError,'before open'):m.read_source(self.p,self.sha)

    def test_lstat_open_symlink_race_rejected(self):
        target=self.p.with_name('target');target.write_bytes(b'x=1\n');original=os.open
        def replace(path,flags):
            self.p.unlink();self.p.symlink_to(target);return original(path,flags)
        with patch.object(m.os,'open',side_effect=replace):
            with self.assertRaises(OSError):m.read_source(self.p,self.sha)

    def test_descriptor_and_path_replacement_checked(self):
        original=os.read;changed=False
        def read(fd,size):
            nonlocal changed
            data=original(fd,size)
            if not changed:
                changed=True;self.p.unlink();self.p.write_bytes(b'x=1\n')
            return data
        with patch.object(m.os,'read',side_effect=read):
            with self.assertRaises(ValueError):m.read_source(self.p,self.sha)

    def test_growing_source_read_is_bounded(self):
        requested=[]
        def read(fd,size):requested.append(size);return b'x'*size
        with patch.object(m.os,'read',side_effect=read):
            with self.assertRaises(ValueError):m.read_source(self.p,self.sha)
        self.assertEqual(sum(requested),(1<<20)+1)
        self.assertLessEqual(max(requested),64<<10)

    def test_initial_load_is_under_same_absolute_deadline(self):
        received=[]
        @contextlib.contextmanager
        def finite_alarm(deadline):received.append(deadline);yield
        finite=type('Finite',(),{'whole_invocation_alarm':staticmethod(finite_alarm)})()
        def load(*args):
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0],0)
            return finite
        with patch.object(m,'options',return_value=object()),patch.object(m.os,'sched_setaffinity'),patch.object(m,'load',side_effect=load),patch.object(m,'run',return_value=17):
            self.assertEqual(m.main(),17)
        self.assertEqual(received,[m.START+m.LIMITS['wall_seconds']])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))

    def test_expired_start_does_not_load_any_source(self):
        with patch.object(m,'START',time.monotonic()-121),patch.object(m,'load') as load:
            with self.assertRaises(TimeoutError):m.main()
            load.assert_not_called()

    def test_stalled_first_load_times_out_and_restores_handler(self):
        previous=signal.getsignal(signal.SIGALRM)
        with patch.object(m,'START',time.monotonic()),patch.object(m,'LIMITS',{'wall_seconds':.03}),patch.object(m,'options',return_value=object()),patch.object(m.os,'sched_setaffinity'),patch.object(m,'load',side_effect=lambda *a:time.sleep(.3)):
            with self.assertRaises(TimeoutError):m.main()
        self.assertIs(signal.getsignal(signal.SIGALRM),previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))

    def test_existing_alarm_is_preserved_and_rejected(self):
        signal.setitimer(signal.ITIMER_REAL,2)
        try:
            with self.assertRaises(ValueError):
                with m.first_source_alarm(time.monotonic()+1):self.fail('must not enter')
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0],0)
        finally:signal.setitimer(signal.ITIMER_REAL,0)

    def test_reviewed_sources_and_monitor_unchanged(self):
        m.read_source(m.ADAPTER,m.ADAPTER_SHA)
        m.read_source(m.LIB/'full_future_metadata_preparation_v1/run_build_once.py',m.CORE_SHA)
        text=SOURCE.read_text()
        self.assertIn('observed=core.monitor(',text)
        self.assertIn('charged=core.charge_after_reap(',text)
        self.assertNotIn('subprocess.Popen(',text)
        self.assertIn('core.BOOT_IO=BOOT_IO;core.START=START',text)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Boundary))
    with open('/proc/self/io') as f:final={k:int(v) for k,v in (line.split(':') for line in f)}
    cost={'wall_seconds':time.monotonic()-START,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          'logical_read_once_bytes':final['rchar']-BOOT['rchar'],'kernel_read_bytes':final['read_bytes']-BOOT['read_bytes'],
          'cpu_affinity':sorted(os.sched_getaffinity(0)),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
          'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'fixture_logical_write_upper_bound_bytes':2<<20,'real_payload_read':False,'ledger_access':False,'actual_preparation':False}
    print('COST_JSON='+json.dumps(cost,sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
