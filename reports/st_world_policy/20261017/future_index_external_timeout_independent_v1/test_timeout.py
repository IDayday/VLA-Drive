import time
START=time.monotonic()
import hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
EVENTS=[]
PEAK=0
def tree_rss(pid):
    total=0;pending=[pid];seen=set()
    while pending:
        pid=pending.pop()
        if pid in seen:continue
        seen.add(pid)
        try:
            total+=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            pending.extend(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
        except (FileNotFoundError,ProcessLookupError):pass
    return total
def bounded(mode,root,seconds):
    global PEAK
    argv=['/usr/bin/timeout','--signal=TERM','--kill-after=40s',str(seconds)+'s',sys.executable,str(D/'fixture.py'),mode,str(root)]
    before=time.monotonic()
    with (root/'stdout.log').open('wb') as out,(root/'stderr.log').open('wb') as err:
        p=subprocess.Popen(argv,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        while p.poll() is None:
            PEAK=max(PEAK,tree_rss(p.pid));assert PEAK<1<<30
            assert time.monotonic()-before<seconds+42
            time.sleep(.01)
    return p.returncode,time.monotonic()-before,argv
class TimeoutComposition(unittest.TestCase):
    def test_actual_index_module_fifo_is_bounded_before_core_import(self):
        with tempfile.TemporaryDirectory(prefix='index-timeout-fifo-') as d:
            root=Path(d);os.mkfifo(root/'core.fifo')
            rc,elapsed,argv=bounded('fifo',root,.3)
            self.assertEqual(rc,124);self.assertTrue((root/'fifo_reached.json').exists());self.assertLess(elapsed,1)
            marker=json.loads((root/'fifo_reached.json').read_text());self.assertFalse(Path(f'/proc/{marker["pid"]}').exists())
            EVENTS.append({'case':'actual_import_fifo','returncode':rc,'elapsed_seconds':elapsed,'argv':argv,'status':'PASS'})
    def test_timeout_term_reaches_real_monitor_owned_cleanup(self):
        with tempfile.TemporaryDirectory(prefix='index-timeout-owned-') as d:
            root=Path(d);rc,elapsed,argv=bounded('monitor',root,1)
            self.assertEqual(rc,124)
            result=json.loads((root/'monitor_result.json').read_text())
            self.assertEqual(result['failure'],'SUPERVISOR_SIGNAL_15')
            self.assertEqual(result['cleanup']['live_owned_remaining'],0)
            self.assertGreaterEqual(result['cleanup']['verified_processes'],2)
            for row in json.loads((root/'owned_fixture_pids.json').read_text()):
                self.assertFalse(Path(f'/proc/{row["pid"]}').exists())
            self.assertLess(elapsed,3)
            EVENTS.append({'case':'real_monitor_external_term','returncode':rc,'elapsed_seconds':elapsed,'argv':argv,
                'failure_preserved':result['failure'],'cleanup':result['cleanup'],'status':'PASS',
                'resource_sampling_in_monitor':'fixture-local RSS only; actual ownership/signal/cleanup helpers unmodified'})
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TimeoutComposition))
    status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    io=dict((k,int(v)) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines()))
    cost={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'wall_seconds':time.monotonic()-START,
          'cpu_affinity':sorted(os.sched_getaffinity(0)),'harness_hwm_bytes':int(status['VmHWM'].split()[0])*1024,
          'sampled_fixture_tree_peak_rss_bytes':PEAK,'process_rchar_including_reaped_children_once':io['rchar'],'events':EVENTS,
          'actual_task_signals':False,'real_payload_reads':False,'ledger_access':False,'gpu':False}
    assert cost['wall_seconds']<120 and io['rchar']<256<<20
    (D/'COST.json').write_text(json.dumps(cost,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
