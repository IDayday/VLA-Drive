import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import navtest_expand_active as m


class ExpansionGateTest(unittest.TestCase):
    def test_exact_process_and_watchdog_resume(self):
        with tempfile.TemporaryDirectory(prefix='navtest-gate-', dir='/mnt/project') as directory:
            root = Path(directory); tool = root / 'cpu_probe.py'
            tool.write_text('import time\nwhile True: time.sleep(.1)\n')
            child = subprocess.Popen([sys.executable, str(tool), '--serve', '--output', str(root)])
            try:
                stat = Path('/proc', str(child.pid), 'stat')
                start = stat.read_text().rsplit(')', 1)[1].split()[19]
                self.assertTrue(m.process_matches(child.pid, start, tool, root))
                self.assertFalse(m.process_matches(child.pid, str(int(start)+1), tool, root))
                self.assertFalse(m.process_matches(child.pid, start, tool, root / 'foreign'))
                os.kill(child.pid, signal.SIGSTOP)
                for _ in range(50):
                    if stat.read_text().rsplit(')', 1)[1].split()[0] == 'T': break
                    time.sleep(.01)
                m.resume_exact(dict(pid=child.pid,start=str(int(start)+1),tool=str(tool),output=str(root)))
                self.assertEqual(stat.read_text().rsplit(')', 1)[1].split()[0], 'T')
                gate = dict(pid=child.pid,start=start,tool=str(tool),output=str(root),
                            destination=str(root),deadline=time.time()-1)
                path=root/'GATE.json';path.write_text(json.dumps(gate))
                m.watchdog(path)
                for _ in range(50):
                    if stat.read_text().rsplit(')', 1)[1].split()[0] != 'T': break
                    time.sleep(.01)
                self.assertNotEqual(stat.read_text().rsplit(')', 1)[1].split()[0], 'T')
                self.assertEqual(json.loads((root/'INFERENCE_FAILED.json').read_text())['classification'],
                                 'supplemental_dispatch_watchdog')
            finally:
                os.kill(child.pid, signal.SIGCONT)
                child.terminate(); child.wait(timeout=5)

    def test_clean_release_does_not_resume_or_fail(self):
        with tempfile.TemporaryDirectory(prefix='navtest-gate-', dir='/mnt/project') as directory:
            root=Path(directory);path=root/'GATE.json'
            path.write_text(json.dumps(dict(deadline=time.time()+20)))
            (root/'RELEASED.json').write_text('{}')
            m.watchdog(path)
            self.assertFalse((root/'INFERENCE_FAILED.json').exists())


if __name__ == '__main__':
    unittest.main()
