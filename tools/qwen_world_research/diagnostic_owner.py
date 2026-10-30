"""Finite TRAIN4096 comparison, independent of adaptive feedback and training."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import threading

from . import dispatch
from .campaign_owner import atomic_json, native_checkpoint, read_json, sha_file

ROOT = Path('/mnt/project/qwen-world-rl-optimization-20261030')
ARMS = ('IL70', 'PPO_FULL', 'PPO_ADAPTIVE')


def now():
    return datetime.now(timezone.utc).isoformat()


class DiagnosticOwner:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.owner = self.root / 'owners/TRAIN4096'
        self.owner.mkdir(exist_ok=True)
        self.lock = (self.owner / 'owner.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.mutex = threading.RLock()
        self.state = read_json(self.owner / 'STATE.json', dict(status='RUNNING', runs={}))
        atomic_json(self.owner / 'OWNER.json', dict(pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            started_at=now(), source_file=__file__, source_sha256=sha_file(__file__)))
        self.state.update(status='RUNNING', training_scene_selection_uses_this_panel=False,
            diagnostic='old feedback1024 vs extra3072, not a heldout Test',
            adaptive_recipe_changed=False, arms=list(ARMS))
        self.save()

    def save(self):
        with self.mutex:
            self.state['updated_at'] = now()
            atomic_json(self.owner / 'STATE.json', self.state)

    def capacity_event(self):
        """Wait for actual claimed transports to exit, with a bounded heartbeat."""
        descriptors = []
        claims = read_json(self.root / 'resources/EVALUATION_CLAIMS.json', {})
        for row in claims.values():
            job = Path(row['receipt'])
            launch = read_json(job / 'LAUNCH.json', {})
            if (job / 'RESULT.json').exists() or not dispatch._process_alive(launch):
                continue
            pid = launch['transport_pid']
            try:
                fd = os.pidfd_open(pid) if hasattr(os, 'pidfd_open') else ctypes.CDLL(None, use_errno=True).syscall(434, pid, 0)
                if fd >= 0:
                    descriptors.append(fd)
            except OSError:
                pass
        try:
            select.select(descriptors, [], [], 1800)
        finally:
            for fd in descriptors:
                os.close(fd)

    def run_arm(self, arm):
        with self.mutex:
            node = self.state['runs'].setdefault(arm, dict(status='PREPARING'))
            self.save()
        if node.get('summary'):
            return node['summary']
        try:
            checkpoint = None
            if arm != 'IL70':
                row = next(r for r in dispatch.runs(self.root) if r['id'] == arm)
                try:
                    checkpoint = native_checkpoint(row['root'], 10000)['path']
                except FileNotFoundError:
                    with self.mutex:
                        node.update(status='WAITING_NATIVE_10K', training_job=row['job'])
                        self.save()
                    result = dispatch.wait_job(row['job'])
                    if result['returncode'] != 0:
                        raise RuntimeError('Native stage not complete: ' + str(result))
                    checkpoint = native_checkpoint(row['root'], 10000)['path']
            arguments = dict(root=self.root, label=f'{arm}_u010000_TRAIN4096' if checkpoint else 'IL70_TRAIN4096',
                parent_checkpoint=self.root/'IL_PARENT', rl_checkpoint=checkpoint,
                cohort='TRAIN_DIAGNOSTIC', manifest=self.root/'data/panels/TRAIN_DIAGNOSTIC_004096.json',
                data=self.root/'data/NAVTRAIN')
            quick_retries = 0
            while True:
                try:
                    spec = dispatch.dispatch_evaluation(**arguments)
                except BlockingIOError as error:
                    with self.mutex:
                        node.update(status='WAITING_EVAL_CAPACITY', reason=str(error))
                        self.save()
                    self.capacity_event()
                    continue
                with self.mutex:
                    node.update(status='EVALUATING', spec=spec)
                    self.save()
                results = [dispatch.wait_job(job) for job in spec['jobs']]
                bad = [r for r in results if r['returncode'] != 0]
                if not bad:
                    break
                if any(r['returncode'] != 75 or r.get('child_not_started') is not True for r in bad):
                    raise RuntimeError('Scientific evaluator failed; no retry: ' + str(bad))
                with self.mutex:
                    node.setdefault('resource_refusals', []).extend(bad)
                    self.save()
                quick_retries += 1
                if quick_retries > 2:
                    self.capacity_event()
            summary = dispatch.finish_evaluation(spec)
            with self.mutex:
                node.update(status='COMPLETE', summary=summary, completed_at=now())
                self.save()
            atomic_json(self.root/'reporting/refresh_request.json',
                dict(id=f'TRAIN4096_{arm}_{now()}', scientific_result=True,
                     reason='independent TRAIN4096 complete', evidence=summary['evidence']))
            return summary
        except Exception as error:
            with self.mutex:
                node.update(status='FAILED_REQUIRES_REVIEW', error=repr(error))
                self.save()
            return None

    def run(self):
        with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
            for future in as_completed([pool.submit(self.run_arm, arm) for arm in ARMS]):
                future.result()
        self.state['status'] = ('COMPLETE' if all(v['status']=='COMPLETE' for v in self.state['runs'].values())
                                else 'COMPLETE_WITH_FAILURES_REVIEW_REQUIRED')
        self.save()


if __name__ == '__main__':
    DiagnosticOwner().run()
