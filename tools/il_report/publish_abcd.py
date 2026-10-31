"""Finite readout publisher for this user-requested four-checkpoint sweep.

Only sealed full results are committed; no model/score execution or Git merge.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .evidence import Evidence, score_row
from .health import atomic

REQUEST = Path('/mnt/project/qwen-world-four-frameworks-20261029/reallocated_NAVTEST_20261031T001000Z')
SOURCE = Path(__file__).resolve().parents[2]
REPORT = SOURCE/'reports/qwen_world_ABCD_navtest_20261031'


def read(path):
    return json.loads(Path(path).read_text())


def git(*args):
    return subprocess.check_output(['git', *args], cwd=SOURCE, text=True).strip()


def cycle():
    data = read(REPORT/'results.json')
    known = {(r['architecture'], r['updates']) for r in data['new_complete']}
    changed = False; pending = []
    for label, pin in read(REQUEST/'PINNED.json')['models'].items():
        root = REQUEST/label
        if not (root/'COMPLETE.json').exists():
            pending.append(label); continue
        done = read(root/'COMPLETE.json')
        if done.get('status') != 'COMPLETE' or done.get('all_owned_workers_exited') is not True:
            raise ValueError('Incomplete native worker ownership')
        if not score_row(Evidence(), root/'SUMMARY.json', label):
            raise ValueError('Full single-candidate protocol or score seal failed')
        if read(root/'native_export_parity.json')['status'] != 'PASS':
            raise ValueError('Missing native/export parity')
        key = (pin['architecture_id'], pin['updates'])
        if key in known: continue
        summary = read(root/'SUMMARY.json'); identity = read(root/'evaluation/identity.json')
        data['new_complete'].append(dict(architecture=key[0], updates=key[1], PDMS=summary['PDMS'],
            submetrics_points={k:100*v for k,v in summary['means'].items()}, summary=str(root/'SUMMARY.json'),
            summary_sha256=summary['sha256'], evaluation_identity_sha256=identity['sha256'],
            protocol_sha256=identity['protocol']['sha256'], token_sha256=identity['token_sha256'],
            seed=2101, scenes=12146, invalid_scenes=0, native_export_parity_status='PASS', owned_workers_exited=True))
        changed = True
    if changed:
        if git('diff', '--cached', '--name-only'):
            return dict(status='WAITING_FOR_EMPTY_GIT_INDEX', pending=pending)
        data['pending'] = pending; data['updated_unix'] = time.time()
        data['new_complete'].sort(key=lambda r:r['architecture'])
        atomic(REPORT/'results.json', data)
        table='\n\n## Latest completed requested checkpoints\n\n| Architecture | Updates | Full Navtest PDMS |\n|---|---:|---:|\n'
        table+=''.join(f'| {r["architecture"]} | {r["updates"]} | {r["PDMS"]:.6f} |\n' for r in data['new_complete'])
        table+='\nPending: '+(', '.join(pending) or 'none')+'. Unequal intermediate budgets do not establish a final architecture ranking.\n'
        path=REPORT/'README.md';text=path.read_text().split('\n\n## Latest completed requested checkpoints')[0]
        text=text.replace('C12k/D7k are still pending; no partial-scene means are published.',
                          'Current completion state is listed below; no partial-scene means are published.')
        path.write_text(text+table)
        prefix=str(REPORT.relative_to(SOURCE))
        git('add','--',prefix)
        staged=git('diff','--cached','--name-only').splitlines()
        if any(not p.startswith(prefix+'/') for p in staged):
            raise RuntimeError('Refusing to commit unrelated staged work')
        git('commit','-m','Publish newly completed full ABCD checkpoint observations')
    # Also retry a transport-only publication failure without making another commit.
    branch=git('branch','--show-current');head=git('rev-parse','HEAD')
    remote=git('ls-remote','origin','refs/heads/'+branch).split()[0]
    if head!=remote:
        prefix=str(REPORT.relative_to(SOURCE))+'/'
        if any(not p.startswith(prefix)for p in git('diff','--name-only',remote,head).splitlines()):
            raise RuntimeError('Publisher will not push unrelated source changes')
        git('push','origin','HEAD:refs/heads/'+branch)
        remote=git('ls-remote','origin','refs/heads/'+branch).split()[0]
    if head!=remote:
        raise RuntimeError('Remote SHA mismatch; no merge or force push attempted')
    return dict(status='PUBLISHED' if changed else 'NO_NEW_COMPLETE_RESULT',
        commit=head,remote_sha=remote,pending=pending,completed=len(data['new_complete']),at=time.time())


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();root=REQUEST/'publication';root.mkdir(exist_ok=True)
    with (root/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(root/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            source=str(SOURCE),only_reports_committed=True,model_updates=0,new_scores=0))
        deadline=time.monotonic()+6*3600
        while True:
            result=cycle();atomic(root/'STATUS.json',result)
            print(json.dumps(result),flush=True)
            if not args.watch or not result['pending']:return
            if time.monotonic()>deadline:
                atomic(root/'BLOCKED.json',dict(reason='Six-hour finite publication window ended; pending measurements retained',pending=result['pending']))
                return
            time.sleep(60)  # Small completion receipts only, never training logs.


if __name__=='__main__':main()
