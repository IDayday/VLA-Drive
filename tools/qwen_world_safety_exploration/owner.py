"""Finite single-seed Safety exploration training owner; evaluations never block learning.

Own four arms only. Native stages retain Adam/RNG/cursor/unfinished rollout.
Resource refusals before any child are requeued; scientific failures are retained
for diagnosis and do not stop the other research arms. No implicit50k grant.
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
from starVLA.rl.qwen_world_safety_exploration.contracts import ROOT, TASK, allowed_hosts, file_sha
from starVLA.rl.qwen_world_safety_exploration.resources import probe, MODEL_PYTHON, gpu_activity_issue
from starVLA.rl.flow_grpo.transactions import atomic_json
from .dispatch import launch, job_state
from .notify import refresh


class StopBeforeDispatch(RuntimeError):
    pass


class Owner:
    def __init__(self, *, selected_ids=None, owner_name='RESEARCH_NATIVE'):
        if not owner_name.replace('_','').isalnum():raise ValueError('Explicit local owner name required')
        self.directory = ROOT/'owners'/owner_name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory/'owner.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.mutex = threading.RLock()
        self.registry = json.loads((ROOT/'RUNS.json').read_text())
        if self.registry['task'] != TASK or not 1 <= len(self.registry['runs']) <= 8 or len({r['id']for r in self.registry['runs']})!=len(self.registry['runs']):
            raise ValueError('Exact uniquely registered bounded research arms required')
        ids={r['id']for r in self.registry['runs']}
        if selected_ids is not None and (not selected_ids or not set(selected_ids)<=ids):raise ValueError('Only explicit registered selected arms')
        self.runs=[r for r in self.registry['runs']if selected_ids is None or r['id']in selected_ids]
        if any(Path(r['owner_state']).resolve()!=self.directory/'STATE.json' for r in self.runs):
            raise ValueError('Selected runs must bind this exact owner state, preventing duplicate lineages')
        self.state_path = self.directory/'STATE.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else dict(runs={})
        self.state.update(status='RUNNING')
        self.state.pop('native_owner_finished_unix',None)
        atomic_json(self.directory/'OWNER.json', dict(pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            source=str(Path(__file__).resolve()), source_sha256=file_sha(__file__), started_unix=time.time()))

    def update(self, arm, **values):
        with self.mutex:
            previous = self.state['runs'].get(arm,{}).get('status')
            self.state['runs'].setdefault(arm, {}).update(values, heartbeat_unix=time.time())
            self.state['heartbeat_unix'] = time.time()
            atomic_json(self.state_path, self.state)
            if values.get('status') != previous and values.get('status') in ('FAILED_REVIEW_REQUIRED','STAGE_COMPLETE','COMPLETED_AT_REGISTERED_CAP'):
                refresh(self.state_path,arm+': '+values['status'])

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
            if not node.get('job') and (ROOT/'STOP_REQUESTED').exists():
                raise StopBeforeDispatch('Stopped before launching a new native stage')
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
                           'tools.qwen_world_safety_exploration.train',
                           '--config', run['config'], '--manifest', self.registry['manifest'],
                           '--il-checkpoint', run['parent_checkpoint'], '--data', self.registry['observation_data'],
                           '--parent-selection', self.registry['parent_selection'],
                           '--output', run['root'], '--stop-after-update', str(target), '--save-every', '100']
                if resume:
                    command += ['--resume', resume]
                    if run.get('execution_migration_receipt'):
                        command += ['--migration-receipt',run['execution_migration_receipt']]
                try:
                    job = launch(route['host'], run.get('execution_source',run['source']), command, name, gpus='0,1,2,3,4,5,6,7',
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
        except StopBeforeDispatch:
            self.update(arm,status='STOPPED_BEFORE_DISPATCH')
        except Exception:
            self.update(arm, status='FAILED_REVIEW_REQUIRED', error=traceback.format_exc())

    def run(self):
        with ThreadPoolExecutor(max_workers=min(8,len(self.runs))) as pool:
            list(pool.map(self.run_arm, self.runs))
        self.state['status'] = ('COMPLETED_AT_REGISTERED_CAP' if all(x.get('status')=='COMPLETED_AT_REGISTERED_CAP' for x in self.state['runs'].values()) else 'FINISHED_WITH_REVIEW_REQUIRED')
        self.state['native_owner_finished_unix'] = time.time()
        atomic_json(self.state_path, self.state)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--ids',nargs='+',help='Only these registered arms; never adopt another active owner state')
    parser.add_argument('--owner-name',default='RESEARCH_NATIVE')
    args=parser.parse_args()
    Owner(selected_ids=args.ids,owner_name=args.owner_name).run()


if __name__ == '__main__':
    main()
