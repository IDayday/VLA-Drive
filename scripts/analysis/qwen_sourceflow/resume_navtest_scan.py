"""Explicit fixed-host continuation after released locks and transport loss.

Reuses native inference, identical checkpoint/source/seed/shard layout and atomic
scene records. Completed workers/scenes are retained, not scored again.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.analysis.qwen_sourceflow.finite_navtest_scan import Owner, read, seal, validate_plan


class FixedContinuation(Owner):
    def __init__(self, plan):
        super().__init__(plan)
        self.order = ['RL77_LATEST', 'DINO_FRESH_LATEST', 'SCADV3_025000', 'SCADV3_040000',
                      'SCADV3_045000', 'IL_080000', 'IL_087000'] + [x for x in self.order if x not in
                      ['RL77_LATEST', 'DINO_FRESH_LATEST', 'SCADV3_025000', 'SCADV3_040000', 'SCADV3_045000']]
        old = Path(plan['previous_control_root'])
        if not (old / 'lease_closed.json').exists(): raise ValueError('Prior admission not closed')
        evidence = read(self.root / 'transport_loss.json'); body = dict(evidence)
        if body.pop('sha256') != seal(body): raise ValueError('Changed handoff evidence')
        if evidence['original_registration_sha256'] != plan['original_registration_sha256']:
            raise ValueError('Unmatched original campaign')
        (self.root / 'analysis').mkdir(exist_ok=True)
        (self.root / 'parity').mkdir(exist_ok=True)
        for path in (old / 'jobs').glob('*/meta.json'):
            outcome = path.parent / 'result.json'
            if not outcome.exists(): continue
            result = read(outcome)
            if result.get('status') == 'PROCESS_COMPLETE' and result.get('exit_code') == 0:
                self.jobs[path.parent.name] = read(path)
        for label in plan['models']:
            summary = old / 'analysis' / (label + '_summary.json')
            if summary.exists():
                row = read(summary); body = dict(row)
                if body.pop('sha256') != seal(body) or row['status'] != 'COMPLETE' or row['scenes'] != 12146:
                    raise ValueError('Invalid inherited full summary')
                shutil.copyfile(summary, self.root / 'analysis' / summary.name)
                self.state[label] = dict(stage='COMPLETE', summary=str(self.root / 'analysis' / summary.name), PDMS=row['PDMS'])
        shutil.copyfile(old / 'parity/DINO_FRESH_LATEST_native.json', self.root / 'parity/DINO_FRESH_LATEST_native.json')

    def enter_lease(self):
        root, atomic = self.root, self.atomic
        class FixedOnly:
            def close(self):
                atomic(root / 'lease_closed.json', dict(at=time.time(), fixed_hosts_only=True,
                    new_owned_workers_exited=True, legacy_remote_exit_unconfirmed=True,
                    no_training_signaled=True, legacy_transport_evidence=str(root / 'transport_loss.json')))
        self.atomic(root / 'lease.json', dict(fixed_hosts_only=True, temporary_hosts=[], at=time.time()))
        return FixedOnly()

    def probe(self):
        routes, inventory = super().probe()
        return [r for r in routes if r['host'] in self.plan['fixed_hosts']], inventory

    def analyze(self, model):
        output = self.root / 'analysis' / (model + '_summary.json')
        command = [self.resource.MODEL_PYTHON, '-m', 'scripts.analysis.qwen_sourceflow.summarize_resumed_navtest',
                   '--evaluation', self.plan['models'][model]['output'], '--output', str(output),
                   '--transport-evidence', str(self.root / 'transport_loss.json')]
        with (self.root / 'analysis' / (model + '_summary.log')).open('x') as log:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[3], stdout=log, stderr=subprocess.STDOUT, check=True)
        self.state[model] = dict(stage='COMPLETE', summary=str(output), PDMS=read(output)['PDMS'])


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--registration', required=True); a = p.parse_args()
    plan = read(a.registration); validate_plan(plan); root = Path(plan['root'])
    lock = (root / 'owner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'owner.json').exists(): raise FileExistsError('Never duplicate a continuation')
    owner = FixedContinuation(plan)
    owner.atomic(root / 'owner.json', dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]), at=time.time()))
    owner.run()


if __name__ == '__main__': main()
