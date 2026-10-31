"""Finite single-seed ABCD native training owner; evaluations never block learning.

Own four arms only. Native stages retain Adam/RNG/cursor/unfinished rollout.
Resource refusals before any child are requeued; scientific failures are retained
for diagnosis and do not stop the other architectures. No implicit50k grant.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import traceback
from starVLA.rl.qwen_world_compare.contracts import ROOT, TASK, allowed_hosts, file_sha
from starVLA.rl.qwen_world_compare.resources import probe, MODEL_PYTHON, gpu_activity_issue
from starVLA.rl.flow_grpo.transactions import atomic_json
from .dispatch import launch, job_state


class Owner:
    def __init__(self):
        self.directory = ROOT/'owners'/'ABCD_NATIVE'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory/'owner.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.mutex = threading.RLock()
        self.registry = json.loads((ROOT/'RUNS.json').read_text())
        if self.registry['task'] != TASK or len(self.registry['runs']) != 4:
            raise ValueError('Exactly the registered four architecture runs required')
        self.state_path = self.directory/'STATE.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else dict(runs={})
        atomic_json(self.directory/'OWNER.json', dict(pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            source=str(Path(__file__).resolve()), source_sha256=file_sha(__file__), started_unix=time.time()))

    def update(self, arm, **values):
        with self.mutex:
            self.state['runs'].setdefault(arm, {}).update(values, heartbeat_unix=time.time())
            self.state['heartbeat_unix'] = time.time()
            atomic_json(self.state_path, self.state)

    def stage(self, arm, target):
        with self.mutex:
            return self.state['runs'].setdefault(arm, {}).setdefault('stages', {}).setdefault(str(target), {})

    def candidate(self, run):
        dependency = run.get('startup_dependency_root')
        if dependency:
            journal = Path(dependency)/'metrics_rank0.jsonl'
            if not journal.exists() or not journal.stat().st_size:
                return None
        for route in run['deployment_candidates']:
            if route['host'] not in allowed_hosts(admit_long_job=True):
                continue
            try:
                sample = probe(route['host'])
                if len(sample['gpus']) != 8 or not sample['worker_python_exists']:
                    continue
                if sample['MemAvailable_bytes'] < 100*2**30:
                    continue
                if sample['load1'] > sample['CPU_affinity']*1.5:
                    continue
                if any(gpu_activity_issue(g, sample['owners']) for g in sample['gpus']):
                    continue
                if min(g['free_MiB'] for g in sample['gpus']) < route['minimum_free_MiB']:
                    continue
                # The worker repeats owners/headroom under physical UUID locks.
                return route, sample
            except Exception:
                continue
        return None

    def run_stage(self, run, target, resume):
        arm = run['id']; node = self.stage(arm, target)
        if node.get('complete'):
            return node['checkpoint']
        while True:
            if not node.get('job'):
                self.update(arm, status='WAITING_RESOURCE', stage_target_updates=target)
                chosen = self.candidate(run)
                if chosen is None:
                    time.sleep(30); continue
                route, sample = chosen
                attempt = int(node.get('attempt', 0))+1
                name = f'{arm}/formal_to{target:06}_attempt{attempt:03}'
                command = [MODEL_PYTHON, '-m', 'torch.distributed.run', '--standalone',
                           '--nnodes=1', '--nproc_per_node=8', '-m',
                           'tools.qwen_world_abcd_research.train', '--architecture', run['architecture_id'],
                           '--config', run['config'], '--manifest', self.registry['manifest'],
                           '--il-checkpoint', run['parent_export'],
                           '--parent-selection', self.registry['parent_selection'],
                           '--output', run['root'], '--stop-after-update', str(target), '--save-every', '100']
                if resume:
                    command += ['--resume', resume]
                try:
                    job = launch(route['host'], run['source'], command, name, gpus='0,1,2,3,4,5,6,7',
                        expected_uuids=[g['uuid'] for g in sample['gpus']],
                        share_minimum=route['minimum_free_MiB'], lease_slot=route['lease_slot'])
                except Exception:
                    # A transport error after launch can leave a real child.
                    # Persist exact invocation and inspect its own receipts; never
                    # retry an unproven launch as a second learner.
                    job = ROOT/'jobs'/name
                    with self.mutex:
                        node.update(attempt=attempt, job=str(job), dispatch_error=traceback.format_exc())
                    self.update(arm, status='DISPATCH_TRANSPORT_REVIEW', job=str(job))
                    raise
                with self.mutex:
                    node.update(attempt=attempt, job=str(job), host=route['host'], resume=resume,
                                start_unix=time.time())
                self.update(arm, status='STARTING', job=str(job), host=route['host'])
            receipt = job_state(node['job'])
            latest_path = Path(run['root'])/'latest.json'
            updates = json.loads(latest_path.read_text())['updates'] if latest_path.exists() else 0
            self.update(arm, status='TRAINING' if receipt['status'] == 'RUNNING' else receipt['status'],
                        updates=updates, stage_target_updates=target, job=node['job'], host=node.get('host'))
            if receipt.get('status') == 'REFUSED_RESOURCE':
                if not receipt.get('child_not_started') or receipt.get('returncode') != 75:
                    raise RuntimeError('Ambiguous refusal must not be retried')
                with self.mutex:
                    node.setdefault('refused_jobs', []).append(node.pop('job'))
                self.update(arm, status='RESOURCE_REQUEUE')
                time.sleep(30); continue
            if 'returncode' not in receipt:
                time.sleep(10); continue
            if receipt['returncode'] != 0:
                raise RuntimeError('Native stage failed; retained own receipts: '+node['job'])
            latest = json.loads(latest_path.read_text())
            checkpoint = Path(latest['checkpoint'])
            seal = json.loads((checkpoint/'COMPLETE').read_text())
            if latest['updates'] != target or seal['updates'] != target or seal['world'] != 8:
                raise ValueError('Native stage ended outside its registered full8rank boundary')
            for rank in range(8):
                result = json.loads((Path(run['root'])/f'result_rank{rank}.json').read_text())
                if result['updates'] != target or result['checkpoint'] != str(checkpoint):
                    raise ValueError('Missing actual native rank completion')
            with self.mutex:
                node.update(complete=True, checkpoint=str(checkpoint), end_unix=time.time())
            self.update(arm, status='STAGE_COMPLETE', updates=target)
            return str(checkpoint)

    def run_arm(self, run):
        arm = run['id']
        try:
            resume = None
            for target in run['gates']:
                resume = self.run_stage(run, target, resume)
                if (ROOT/'STOP_REQUESTED').exists():
                    self.update(arm, status='STOPPED_AT_REGISTERED_GATE'); return
            self.update(arm, status='COMPLETED_AT_REGISTERED_CAP', updates=run['target_updates'])
        except Exception:
            self.update(arm, status='FAILED_REVIEW_REQUIRED', error=traceback.format_exc())

    def run(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.run_arm, self.registry['runs']))
        self.state['native_owner_finished_unix'] = time.time()
        atomic_json(self.state_path, self.state)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.parse_args()
    Owner().run()


if __name__ == '__main__':
    main()
