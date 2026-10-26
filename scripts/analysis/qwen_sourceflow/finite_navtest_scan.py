"""Finite checkpoint evaluation adapter around qualified native workers.

No actor, sampler, loss, scoring, checkpoint selection or training changes.
Model sources and parents are selected per row, not a shared global parent.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_plan(plan):
    body = dict(plan)
    if body.pop('sha256') != seal(body):
        raise ValueError('Changed finite evaluation registration')
    if (plan['seed'] != 2101 or plan['scenes_each'] != 12146 or plan['logs_each'] != 136 or
        plan['metric'] != 'canonical_NAVSIM_v1.1' or
        plan['inference'] != 'single_candidate_source_shift5_ODE_K10' or
        plan['temporary_eval_host'] != 'training-vla-zt3'):
        raise ValueError('Unregistered population/sampling/host scope')
    extended = plan.get('schema') == 'finite_SCADV3_latest_and_IL_intermediates_full_NAVTEST_v2'
    expected = 14 if extended else 12
    if len(plan['models']) != expected or plan['total_new_single_candidate_inferences'] != expected * 12146:
        raise ValueError('This request has a fixed model list, not a moving checkpoint stream')
    if extended and (not plan.get('original_registration_sha256') or
            {k: plan['models'][k]['updates'] for k in ['IL_080000', 'IL_087000']} !=
            {'IL_080000': 80000, 'IL_087000': 87000}):
        raise ValueError('The explicit extension is IL80k/87k only')
    for label, model in plan['models'].items():
        if Path(label).name != label or model['kind'] not in ('rl', 'il_interim', 'temporal', 'il_parent_control'):
            raise ValueError('Unknown model role')
        if model['kind'] in ('rl', 'il_parent_control') and not model.get('parent_selection'):
            raise ValueError('Each RL/control must bind its own exact IL parent')


def evaluation_arguments(model):
    common = ['--checkpoint', model['parent_checkpoint'] if model['kind'] == 'rl' else model['checkpoint'],
              '--output', model['output'], '--cohort', 'NAVTEST', '--data', model['data']]
    if model['kind'] == 'rl':
        return 'scripts.analysis.qwen_sourceflow.evaluate_policy', common + [
            '--parent-selection', model['parent_selection'], '--rl-export', model['bundle']]
    if model['kind'] == 'il_parent_control':
        return 'scripts.analysis.qwen_sourceflow.evaluate_policy', common + ['--parent-selection', model['parent_selection']]
    if model['kind'] == 'temporal':
        return 'scripts.analysis.qwen_sourceflow.evaluate_temporal_policy', common + ['--registration', model['registration']]
    return 'scripts.analysis.qwen_sourceflow.evaluate_policy', common + ['--interim-navtest-registration', model['registration']]


class Owner:
    def __init__(self, plan):
        self.plan, self.root = plan, Path(plan['root'])
        # Use the exact qualified worker/launch implementation, not a new model runner.
        sys.path.insert(0, plan['source_defaults']['native53']['path'])
        from starVLA.rl.qwen_sourceflow import resources
        from starVLA.rl.flow_grpo.transactions import atomic_json
        self.resource, self.atomic = resources, atomic_json
        self.jobs = {}
        self.state = {label: dict(stage='PENDING') for label in plan['models']}
        self.spare = set()
        self.last_status = 0.
        self.order = ['IL_LATEST', 'DINO_FRESH_LATEST', 'IL77_CONTROL', 'RL77_LATEST',
                      'SCADV3_020000', 'SCADV3_025000', 'SCADV3_030000', 'SCADV3_035000',
                      'SCADV3_040000', 'SCADV3_045000', 'SCADV3_010000', 'SCADV3_005000']

    def enter_lease(self):
        helper = Path('/mnt/project/qwen-sourceflow-native-20261018/native_navtest_spare_lease.py')
        spec = importlib.util.spec_from_file_location('qualified_spare_navtest_lease', helper)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        lease = module.SpareNavtestLease(self.root)
        lease.hosts = {self.plan['temporary_eval_host']: list(range(8))}
        lease.enter()
        receipt = read(self.root / 'lease.json')
        receipt.update(standing_grant='latest explicit user: vla-zt3 for this Navtest only',
                       registration_sha256=self.plan['sha256'], no_training=True)
        self.atomic(self.root / 'lease.json', receipt)
        return lease

    def foreign_reservations(self):
        root = Path('/mnt/project/qwen-sourceflow-native-20261018/parent77k_user20261025/dev_observer/jobs')
        result = set()
        for path in root.glob('*.json'):
            for row in read(path).get('workers', []):
                if not (Path(row['receipt_dir']) / 'result.json').exists():
                    result.update(row['gpu_uuids'])
        return result

    def probe(self):
        routes, inventory = self.resource.available(min_free_MiB=self.plan['minimum_shared_free_MiB'])
        expected = set(self.plan['fixed_hosts']) | {self.plan['temporary_eval_host']}
        routes = [row for row in routes if row['host'] in expected and row['uuid'] not in self.foreign_reservations()]
        for phase in ('first', 'second'):
            inventory[phase] = {h: v for h, v in inventory[phase].items() if h in expected}
        return routes, inventory

    def ready(self, job):
        if job['stage'] != 'eval':
            return False
        directory = Path(self.plan['models'][job['model']]['output']) / 'invocations'
        for path in directory.glob('*.json'):
            row = read(path)
            if row['rank'] == job['rank'] and row.get('physical_inference', 0) > 0:
                return True
        return False

    def active(self):
        return [v for v in self.jobs.values() if not (Path(v['spec']['receipt_dir']) / 'result.json').exists()]

    def outcome(self, name):
        if name not in self.jobs:
            return None
        path = Path(self.jobs[name]['spec']['receipt_dir']) / 'result.json'
        if not path.exists():
            return None
        row = read(path)
        return row.get('status') == 'PROCESS_COMPLETE' and row.get('exit_code') == 0

    def submit(self, model, stage, module, args, source, route=None, rank=None):
        if module not in {'scripts.analysis.qwen_sourceflow.export_policy',
                          'scripts.analysis.qwen_sourceflow.validate_rl_deployment',
                          'scripts.analysis.qwen_sourceflow.evaluate_policy',
                          'scripts.analysis.qwen_sourceflow.evaluate_temporal_policy'}:
            raise PermissionError('Finite evaluator cannot launch a training/controller module')
        name = model + '_' + stage + ('' if rank is None else f'_{rank:02d}')
        directory = self.root / 'jobs' / name
        directory.mkdir(parents=True, exist_ok=False)
        command = [self.resource.MODEL_PYTHON, '-u', '-m', module, *args]
        if rank is not None:
            command += ['--rank', str(rank), '--world', str(self.plan['max_evaluation_workers_per_model'])]
        row = dict(source=source['path'], source_sha=source['sha'], host=route['host'],
                   gpus=[] if stage == 'export' else [route['gpu']],
                   gpu_uuids=[] if stage == 'export' else [route['uuid']],
                   min_free_MiB=0 if stage == 'export' else self.plan['minimum_shared_free_MiB'],
                   module=module, command=command, receipt_dir=str(directory))
        metadata = dict(model=model, stage=stage, rank=rank, spec=row, submitted_unix=time.time())
        # Publish reservations before the CPU model load can hide future CUDA use.
        self.atomic(directory / 'spec.json', row); self.atomic(directory / 'meta.json', metadata)
        self.jobs[name] = metadata
        self.atomic(directory / 'operator.json', self.resource.launch(directory / 'spec.json', source['path']))

    def gpu_route(self, routes, inventory):
        active = self.active()
        for route in sorted(routes, key=lambda x: (-x['free_MiB'], x['host'], x['gpu'])):
            same = [j for j in active if route['uuid'] in j['spec']['gpu_uuids']]
            limit = 2 if route['uuid'] in self.spare else 1
            if len(same) >= limit or any(not self.ready(j) for j in same):
                continue
            loading = sum(j['spec']['host'] == route['host'] and not self.ready(j) for j in active)
            if loading >= 2:
                continue
            host = inventory['second'][route['host']]
            if host['available_memory_bytes'] < 64 * 1024**3:
                continue
            # Unknown GPU load without visible ownership is not an idle device.
            owners = [o for o in host['owners'] if o['uuid'] == route['uuid']]
            if route['uuid'] not in self.spare and not owners:
                continue
            return route
        return None

    def analyze(self, model):
        output = self.root / 'analysis' / (model + '_summary.json')
        command = [self.resource.MODEL_PYTHON, '-m', 'scripts.analysis.qwen_sourceflow.summarize_results',
                   '--evaluation', self.plan['models'][model]['output'], '--output', str(output)]
        directory = self.root / 'analysis'; directory.mkdir(exist_ok=True)
        with (directory / (model + '_summary.log')).open('x') as log:
            subprocess.run(command, cwd=self.plan['source_defaults']['analysis']['path'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        self.state[model] = dict(stage='COMPLETE', summary=str(output), PDMS=read(output)['PDMS'])

    def tick(self):
        routes, inventory = self.probe()
        self.atomic(self.root / 'inventory_latest.json', inventory)
        for label in self.order:
            state = self.state[label]
            if state['stage'] in ('COMPLETE', 'FAILED'):
                continue
            related = [name for name, job in self.jobs.items() if job['model'] == label]
            failed = [name for name in related if self.outcome(name) is False]
            if failed:
                self.state[label] = dict(stage='FAILED', failed_jobs=failed, automatic_retry=False)
                continue
            m = self.plan['models'][label]
            if m['kind'] == 'rl' and not m.get('cached_export'):
                name = label + '_export'
                if name not in self.jobs:
                    if sum(j['stage'] == 'export' for j in self.active()) >= 2:
                        continue
                    hosts = [(h, v) for h, v in inventory['second'].items()
                             if v['available_memory_bytes'] >= 96 * 1024**3 and
                             not any(j['spec']['host'] == h and not self.ready(j) for j in self.active())]
                    if not hosts:
                        continue
                    host, _ = min(hosts, key=lambda x: x[1]['load1'] / x[1]['CPU_affinity'])
                    self.submit(label, 'export', 'scripts.analysis.qwen_sourceflow.export_policy',
                        ['--checkpoint', m['checkpoint'], '--output', m['bundle']], m['source'], dict(host=host))
                    state['stage'] = 'EXPORTING'; continue
                if self.outcome(name) is not True:
                    continue
                if not (Path(m['bundle']) / 'COMPLETE').exists():
                    raise ValueError('Worker exited without sealed export')
                parity = label + '_parity'
                if parity not in self.jobs:
                    route = self.gpu_route(routes, inventory)
                    if route:
                        self.submit(label, 'parity', 'scripts.analysis.qwen_sourceflow.validate_rl_deployment',
                            ['--checkpoint', m['checkpoint'], '--bundle', m['bundle'],
                             '--output', str(self.root / 'parity' / (label + '.json'))], m['source'], route)
                    state['stage'] = 'PARITY'; continue
                if self.outcome(parity) is not True:
                    continue
                if read(self.root / 'parity' / (label + '.json'))['status'] != 'PASS':
                    raise ValueError('Native versus merged ODE parity failed')
            module, args = evaluation_arguments(m)
            if m['kind'] == 'temporal':
                proof = self.root / 'parity' / (label + '_native.json')
                if label + '_eval_00' in self.jobs:
                    if not proof.exists():
                        continue
                    if read(proof)['status'] != 'PASS':
                        raise ValueError('Temporal native predict parity failed')
            for rank in range(self.plan['max_evaluation_workers_per_model']):
                if label + f'_eval_{rank:02d}' in self.jobs:
                    continue
                route = self.gpu_route(routes, inventory)
                if route is None:
                    break
                rank_args = args[:]
                if m['kind'] == 'temporal' and rank == 0:
                    rank_args += ['--parity-output', str(self.root / 'parity' / (label + '_native.json'))]
                self.submit(label, 'eval', module, rank_args, m['source'], route, rank)
                # Fresh probe avoids admitting the same memory twice in a tick.
                routes, inventory = self.probe()
                state['stage'] = 'EVALUATING'
                if m['kind'] == 'temporal' and rank == 0:
                    break
            if all(self.outcome(label + f'_eval_{r:02d}') is True for r in range(self.plan['max_evaluation_workers_per_model'])):
                self.analyze(label)
        if time.time() - self.last_status >= 60:
            counts = {k: len(list((Path(m['output']) / 'scenes').glob('*.json'))) for k, m in self.plan['models'].items()}
            self.atomic(self.root / 'status.json', dict(at=time.time(), models=self.state, scenes=counts,
                requested_each=12146, active_jobs=len(self.active()), training_unchanged=True))
            self.last_status = time.time()

    def run(self):
        for info in self.plan['source_defaults'].values():
            if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=info['path'], text=True).strip() != info['sha']:
                raise ValueError('Changed immutable model/analysis source')
            if subprocess.check_output(['git', 'status', '--porcelain'], cwd=info['path'], text=True).strip():
                raise ValueError('Dirty immutable source')
        lease = self.enter_lease()
        try:
            routes, inventory = self.probe()
            for route in routes:
                host = inventory['second'][route['host']]
                before = inventory['first'][route['host']]
                gpu = next(g for g in host['gpus'] if g['uuid'] == route['uuid'])
                old = next(g for g in before['gpus'] if g['uuid'] == route['uuid'])
                if (route['free_MiB'] >= 79000 and max(gpu['utilization'], old['utilization']) <= 10 and
                    not any(o['uuid'] == route['uuid'] for o in host['owners'] + before['owners'])):
                    self.spare.add(route['uuid'])
            self.atomic(self.root / 'initial_admission.json', dict(routes=routes, inventory=inventory,
                initially_idle_UUIDs=sorted(self.spare), unique_eligible_GPUs=len(routes)))
            for m in self.plan['models'].values():
                if m.get('cached_export'):
                    path = Path(m['reused_parity'])
                    if hashlib.sha256(path.read_bytes()).hexdigest() != m['reused_parity_sha256'] or read(path)['status'] != 'PASS':
                        raise ValueError('Changed previously qualified export parity')
            while not all(v['stage'] in ('COMPLETE', 'FAILED') for v in self.state.values()):
                self.tick(); time.sleep(15)
            while self.active():
                time.sleep(15)
            self.atomic(self.root / 'COMPLETE.json', dict(
                status='COMPLETE' if all(v['stage'] == 'COMPLETE' for v in self.state.values()) else 'COMPLETE_WITH_FAILURES',
                at=time.time(), models=self.state, all_owned_workers_exited=True, registration_sha256=self.plan['sha256']))
        except BaseException:
            self.atomic(self.root / 'failure.json', dict(error=traceback.format_exc(), at=time.time(), automatic_retry=False))
            # Existing native workers own atomic scenes; drain before revoking
            # their temporary host permission. No training receives a signal.
            while self.active():
                time.sleep(15)
            raise
        finally:
            lease.close()


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--registration', required=True)
    args = parser.parse_args(); plan = read(args.registration); validate_plan(plan)
    root = Path(plan['root']); lock = (root / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'owner.json').exists():
        raise FileExistsError('Explicit new attempt required; never duplicate an active/failed owner')
    owner = Owner(plan)
    owner.atomic(root / 'owner.json', dict(pid=os.getpid(), start_ticks=int((Path('/proc') / str(os.getpid()) / 'stat').read_text().split()[21]), at=time.time()))
    (root / 'parity').mkdir(exist_ok=True)
    owner.run()


if __name__ == '__main__':
    main()
