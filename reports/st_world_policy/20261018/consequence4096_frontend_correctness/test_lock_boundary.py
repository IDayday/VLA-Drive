"""Tiny actual FD inheritance checks; no campaign payload or worker execution."""
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class LockBoundary(unittest.TestCase):
    def retained_locks(self, inherited):
        with tempfile.TemporaryDirectory(prefix='st-common4096-review-') as root:
            paths = [Path(root) / 'SERIAL.lock', Path(root) / 'arm.RUN.lock']
            owners = [p.open('a+') for p in paths]
            for owner in owners:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fd = owners[inherited].fileno()
            child = subprocess.Popen(
                [sys.executable, '-c',
                 'import os,sys;os.fstat(int(sys.argv[1]));print("ready",flush=True);sys.stdin.readline()', str(fd)],
                pass_fds=(fd,), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), 'ready')
                # Losing the parent copies models controller termination. A live
                # child retains only the FD actually supplied to pass_fds.
                for owner in owners:
                    owner.close()
                locked = []
                for path in paths:
                    with path.open('a+') as probe:
                        try:
                            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            locked.append(True)
                        else:
                            locked.append(False)
                return locked
            finally:
                for owner in owners:
                    owner.close()
                child.communicate('\n', timeout=5)
                self.assertEqual(child.returncode, 0)

    def test_original_arm_fd_does_not_retain_global_serial_lock(self):
        self.assertEqual(self.retained_locks(inherited=1), [False, True])

    def test_serial_fd_retains_cross_arm_exclusion_after_parent_loss(self):
        self.assertEqual(self.retained_locks(inherited=0), [True, False])


if __name__ == '__main__':
    unittest.main(verbosity=2)
