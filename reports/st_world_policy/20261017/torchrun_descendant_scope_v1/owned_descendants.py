"""PPID-witnessed ownership for a future, explicitly reviewed bounded launcher.

Reuses the existing pidfd, subreaper and unreaped-Popen primitives. No process
is admitted by command line, SID, GPU occupancy or a historical PID receipt.
This is sampled /proc accounting, not a kernel cgroup hard limit. A descendant
which forks and becomes orphaned before its ancestry is observed is unknown;
it is reported, never guessed at or signalled.
"""
import math
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace


def ancestry_supervisor(legacy):
    """Return an isolated facade; never monkeypatch a loaded/active helper."""
    class OwnedDescendants(legacy.OwnedSession):
        @staticmethod
        def same(a, b):
            # setsid() is allowed for an already witnessed child. PID reuse is
            # still rejected; pidfds remain the only signalling capability.
            return all(a[k] == b[k] for k in ('pid', 'start_ticks'))

        def __init__(self, child):
            self.evidence = []
            self.unknown_adopted = []
            self.supervisor = legacy.proc_identity(os.getpid())
            self.preexisting_children = baseline_children
            actual = legacy.proc_identity(child.pid)
            if actual['ppid'] != os.getpid():
                raise RuntimeError('Ownership requires this supervisor\'s actual Popen child')
            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            super().__init__(child)
            self.evidence.append(dict(child=dict(self.anchor), parent=dict(self.supervisor),
                                      basis='this_unreaped_Popen'))

        @staticmethod
        def _snapshot():
            entries = list(Path('/proc').iterdir())
            if len(entries) > 32768:
                raise RuntimeError('Bounded process metadata scan exceeded')
            result = {}
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                try:
                    record = legacy.proc_identity(int(entry.name))
                    result[record['pid']] = record
                except (FileNotFoundError, ProcessLookupError):
                    continue
            return result

        def _admit_child(self, record, parent):
            """Require a current PPID edge, bracketed by exact parent identity."""
            try:
                parent_before = legacy.proc_identity(parent['pid'])
                current = legacy.proc_identity(record['pid'])
                if (not self.same(parent, parent_before)
                        or not self.same(record, current)
                        or current['ppid'] != parent['pid']
                        or current['start_ticks'] < parent['start_ticks']):
                    return False
                fd = legacy.pidfd_open(current['pid'])
                try:
                    after = legacy.proc_identity(current['pid'])
                    parent_after = legacy.proc_identity(parent['pid'])
                    if (not self.same(current, after) or after['ppid'] != parent['pid']
                            or not self.same(parent, parent_after)):
                        return False
                    old = self.members.get(current['pid'])
                    if self.verified >= 4096:
                        raise RuntimeError('Bounded owned-process history exceeded')
                    if old is not None:
                        if self.same(old[0], current):
                            return False
                        # A recycled numeric PID is never the old capability.
                        os.close(old[1])
                    self.members[current['pid']] = (current, fd)
                    fd = None
                    self.verified += 1
                    self.evidence.append(dict(child=dict(current), parent=dict(parent_after),
                                              basis='live_ppid_bracketed_pidfd'))
                    return True
                finally:
                    if fd is not None:
                        os.close(fd)
            except (FileNotFoundError, ProcessLookupError):
                return False

        def discover(self):
            if not self.same(self.anchor, legacy.proc_identity(self.child.pid)):
                raise RuntimeError('Unreaped Popen anchor identity changed')
            snapshot = self._snapshot()
            # Transitive closure admits rank -> loader even when both have new
            # sessions. Every edge is rechecked live, not trusted from snapshot.
            pending = list(snapshot.values())
            while pending:
                progress = False
                rest = []
                for record in pending:
                    old = self.members.get(record['pid'])
                    if old is not None and self.same(old[0], record):
                        continue
                    parent = self.members.get(record['ppid'])
                    if parent is not None and self._admit_child(record, parent[0]):
                        progress = True
                    else:
                        rest.append(record)
                if not progress:
                    break
                pending = rest
            # Newly adopted but unwitnessed children cannot safely be attributed
            # to this Popen. Fail qualification, without signalling those PIDs.
            witnessed = {(r['child']['pid'], r['child']['start_ticks']) for r in self.evidence}
            unknown = [r for r in snapshot.values()
                if r['ppid'] == os.getpid()
                and (r['pid'], r['start_ticks']) not in self.preexisting_children
                and (r['pid'], r['start_ticks']) not in witnessed
                and not (r['pid'] in self.members
                         and self.same(self.members[r['pid']][0], r))]
            seen = {(r['pid'], r['start_ticks']) for r in self.unknown_adopted}
            self.unknown_adopted.extend(r for r in unknown
                                       if (r['pid'], r['start_ticks']) not in seen)
            active = []
            for pid, (known, fd) in list(self.members.items()):
                try:
                    current = legacy.proc_identity(pid)
                    if self.same(known, current) and current['state'] not in ('Z', 'X'):
                        active.append((known, fd))
                        continue
                    if (pid != self.child.pid and self.same(known, current)
                            and current['ppid'] == os.getpid()):
                        os.waitpid(pid, os.WNOHANG)
                except (FileNotFoundError, ProcessLookupError, ChildProcessError):
                    pass
                if pid != self.child.pid:
                    os.close(fd)
                    del self.members[pid]
            return active

        def assert_scope(self):
            if self.unknown_adopted:
                raise RuntimeError('Unwitnessed adopted child: ownership incomplete; no signal authorised')

        def finish(self, *, terminate, deadline=None):
            if type(deadline) not in (int, float) or not math.isfinite(deadline):
                raise ValueError('An explicit finite absolute cleanup deadline is required')
            if terminate:
                self.send(signal.SIGTERM)
            kill_at = deadline - min(10., max(0., deadline-time.monotonic())/2)
            while self.discover():
                now = time.monotonic()
                if now >= kill_at:
                    self.send(signal.SIGKILL)
                elif terminate:
                    self.send(signal.SIGTERM)
                if now >= deadline:
                    raise RuntimeError('Owned descendants exceed absolute cleanup deadline')
                time.sleep(min(.05, max(0., deadline-now)))
            # The first wait absorbs the root's reaped descendant IO. Previously
            # adopted children have already been reaped into supervisor IO.
            code = self.child.wait(timeout=max(0., deadline-time.monotonic()))
            self.assert_scope()
            return code

    # Capture BEFORE the one Popen; never classify a newly adopted orphan as a
    # preexisting foreign child merely because it appeared before first poll.
    baseline_children = {(r['pid'], r['start_ticks']) for r in OwnedDescendants._snapshot().values()
                         if r['ppid'] == os.getpid()}
    return SimpleNamespace(**{**vars(legacy), 'OwnedSession': OwnedDescendants})
