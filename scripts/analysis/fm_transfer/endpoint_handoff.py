"""Finite B100k SFT -> own screen -> existing eight-GPU gate -> source RL.

This CPU owner only invokes already qualified, immutable 225b entry points.
It never resumes a failed stage, restarts a trainer, selects a checkpoint, or
owns the independent milestone evaluation queue. Ambiguous launches stop for
inspection; a stale status or SSH timeout is not evidence that a worker died.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
import traceback
import uuid

from scripts.analysis.fm_transfer.dispatch_screen import completed_sft
from scripts.analysis.fm_transfer.milestone_controller import worker_allowed, resource_probe
from starVLA.rl.fm_transfer.contracts import ROOT, require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
SOURCE = ROOT / 'source/225b9af660cb0210208e2c46fc4216d0f151275b'
VARIANT = 'B_SOURCE_ACTION_DIT'
PREFIX = VARIANT + '_SFT_update100000'


def read(path):
    return json.loads(Path(path).read_text())


def plan(root, controller_source, screen_hosts, training_host):
    if not screen_hosts or len(set(screen_hosts)) != len(screen_hosts):
        raise ValueError('Distinct, explicitly authorized candidate screen hosts required')
    root = Path(root).resolve()
    checkpoint = root / 'runs' / (VARIANT + '_SFT_seed42_v1') / 'checkpoints/update100000'
    screen = root / 'screens' / PREFIX
    hard = root / 'hard_manifests' / (PREFIX + '.json')
    gate = root / 'qualification' / VARIANT / 'own_SFT_update100000_RL8_v1'
    run = root / 'runs' / (VARIANT + '_R_SOURCE_seed42_v1')
    args = lambda module, *rest: [PYTHON, '-m', 'scripts.analysis.fm_transfer.' + module, *map(str, rest)]
    stages = {
        'sft_summary': args('summarize_sft', '--root', checkpoint.parent.parent,
                            '--output', root / 'analysis' / (VARIANT + '_SFT_complete_training.json')),
        **{f'screen_{i}': args('dispatch_screen', '--source', SOURCE,
              '--sft-checkpoint', checkpoint, '--output', screen, '--shards', '24',
              '--ranks', ','.join(map(str, range(i * 8, (i + 1) * 8)))) for i in range(3)},
        'hard_manifest': args('build_hard_manifest', '--screen', screen, '--output', hard),
        'screen_summary': args('analyze_screen', '--screen', screen, '--hard-manifest', hard,
                               '--output', root / 'analysis' / (PREFIX + '_full_TRAIN_screen.json')),
        'endpoint_gate': args('qualify_rl_endpoint', '--sft-checkpoint', checkpoint,
                              '--manifest', hard, '--output', gate),
        'formal_RL': args('dispatch_rl', '--sft-checkpoint', checkpoint, '--manifest', hard,
                          '--qualification-root', gate, '--baseline-root', root / 'evaluations',
                          '--output', run),
    }
    value = dict(schema='fm_transfer_B_endpoint_handoff_v1', root=str(root),
        controller_source=str(Path(controller_source).resolve()), worker_source=str(SOURCE),
        endpoint=str(checkpoint), screen=str(screen), hard_manifest=str(hard), gate=str(gate), run=str(run),
        screen_hosts=list(screen_hosts), training_host=training_host, stages=stages,
        training_seed=42, evaluation_seed=2101, SFT_updates=100000, RL_updates=20000,
        no_automatic_retry=True, no_qualification_weights_for_formal_RL=True)
    value['sha256'] = fingerprint(value)
    return value


def validate_plan(value):
    expected = plan(value['root'], value['controller_source'], value['screen_hosts'], value['training_host'])
    if value != expected:
        raise ValueError('Changed handoff plan or training source; no implicit migration')
    return value


def remote(host, command):
    worker_allowed(host)
    if host == socket.gethostname().removesuffix('-worker-0'):
        return subprocess.check_output(command, text=True, timeout=30)
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                    host, shlex.join(command)], text=True, timeout=30)


def processes(host, records, module):
    """Exact PID/start/module checks; transport failure propagates, never retries."""
    program = '''import json,sys
from pathlib import Path
out=[]
for r in json.loads(sys.argv[1]):
 p=Path('/proc')/str(r['pid'])
 try:
  stat=(p/'stat').read_text().split();args=(p/'cmdline').read_bytes().decode().split('\\0')
  live=int(stat[21])==r['start_ticks'] and stat[2]!='Z'
  if live and sys.argv[2] not in args:raise ValueError('PID/start has an unexpected module')
 except FileNotFoundError:live=False
 out.append(live)
print(json.dumps(out))'''
    return json.loads(remote(host, [PYTHON, '-c', program, json.dumps(records), module]))


def stage_state(control, name, is_live=processes):
    directory = Path(control) / 'stages' / name
    if not (directory / 'intent.json').exists():
        if directory.exists():
            raise ValueError('Unpublished stage directory requires inspection: ' + name)
        return 'NOT_LAUNCHED'
    intent = read(directory / 'intent.json')
    status = read(directory / 'status.json') if (directory / 'status.json').exists() else None
    if status:
        if status['attempt'] != intent['attempt'] or status['plan_sha256'] != intent['plan_sha256']:
            raise ValueError('Foreign stage attempt: ' + name)
        if status['state'] == 'COMPLETE' and status['exit_code'] == 0:
            return 'COMPLETE'
        if status['state'] != 'RUNNING':
            raise RuntimeError('Stage failed; no automatic restart: ' + name)
        owner = status
    elif (directory / 'launch.json').exists():
        owner = read(directory / 'launch.json')
    else:
        raise RuntimeError('Ambiguous launch; inspect before any retry: ' + name)
    if not is_live(intent['host'], [owner], 'scripts.analysis.fm_transfer.endpoint_handoff')[0]:
        raise RuntimeError('Stage owner exited without success; preserve all evidence: ' + name)
    return 'RUNNING'


def capacity(first, second, required):
    if (len(first['gpus']) != 8 or len(second['gpus']) != 8 or
            len({g['uuid'] for g in second['gpus']}) != 8 or
            [g['uuid'] for g in first['gpus']] != [g['uuid'] for g in second['gpus']]):
        raise ValueError('Changed or aliased physical GPU inventory')
    for sample in (first, second):
        for gpu in sample['gpus']:
            owners = [o for o in sample['owners'] if o['uuid'] == gpu['uuid']]
            if (gpu['free'] < required or (gpu['used'] and not owners) or
                    any(not o['visible'] for o in owners)):
                return False
    return True


def launch(value, control, name, host, inventory=None):
    require_dispatch(); worker_allowed(host); validate_plan(value)
    source = Path(value['controller_source'])
    if not read(source / 'SOURCE_COMPLETE.json')['git_archive_extraction_complete']:
        raise ValueError('Publish the controller snapshot before launching stages')
    directory = Path(control) / 'stages' / name
    directory.mkdir(parents=True, exist_ok=False)
    intent = dict(attempt=uuid.uuid4().hex, stage=name, host=host, plan_sha256=value['sha256'],
        start_unix=time.time(), command=value['stages'][name], inventory=inventory)
    atomic_json(directory / 'intent.json', intent)
    command = [PYTHON, '-m', 'scripts.analysis.fm_transfer.endpoint_handoff',
               '--stage', name, '--control', str(control)]
    # Publish intent before spawning. Losing the launcher response is an ambiguous
    # launch, not permission to launch another copy. The worker also owns a flock.
    program = '''import json,os,subprocess,sys
from pathlib import Path
directory=Path(sys.argv[1]);source=sys.argv[2];command=json.loads(sys.argv[3])
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=source,
 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (directory/'worker.log').open('xb') as log:
 p=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,
  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,start_ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21]))))'''
    receipt = json.loads(remote(host, [PYTHON, '-c', program, str(directory),
                                      value['controller_source'], json.dumps(command)]))
    atomic_json(directory / 'launch.json', receipt)
    return intent


def stage_worker(control, name):
    control = Path(control); value = validate_plan(read(control / 'plan.json'))
    intent = read(control / 'stages' / name / 'intent.json')
    host = require_dispatch()
    if (host != intent['host'] or intent['plan_sha256'] != value['sha256'] or
            intent['command'] != value['stages'][name]):
        raise ValueError('Stage identity/host differs from its published intent')
    directory = control / 'stages' / name
    with (directory / 'owner.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / 'status.json').exists():
            raise ValueError('Stage has already been attempted; no automatic replay')
        status = dict(attempt=intent['attempt'], plan_sha256=value['sha256'], state='RUNNING',
            pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            host=host, start_unix=time.time())
        atomic_json(directory / 'status.json', status)
        try:
            if not endpoint_ready(value):
                raise ValueError('An unfinished SFT cannot enter any endpoint stage')
            source = Path(value['worker_source'])
            if read(source / 'SOURCE_COMPLETE.json') != dict(commit=source.name, git_archive_extraction_complete=True):
                raise ValueError('Immutable qualified source is not sealed')
            env = dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES='',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                VLM_ATTN_IMPLEMENTATION='flash_attention_2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                NO_ALBUMENTATIONS_UPDATE='1', TOKENIZERS_PARALLELISM='false')
            with (directory / 'command.log').open('xb') as log:
                child = subprocess.Popen(intent['command'], cwd=source, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                status['child'] = dict(pid=child.pid,
                    start_ticks=int((Path('/proc') / str(child.pid) / 'stat').read_text().split()[21]))
                atomic_json(directory / 'status.json', status)
                code = child.wait()
            status.update(state='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
        except BaseException:
            status.update(state='FAILED', error=traceback.format_exc()); raise
        finally:
            status.update(end_unix=time.time()); atomic_json(directory / 'status.json', status)


def endpoint_ready(value):
    path = Path(value['endpoint'])
    if not (path / 'COMPLETE').exists():
        return False
    identity, _ = completed_sft(path)
    progress = read(path.parent.parent / 'progress.json')
    if (identity['config']['variant'] != VARIANT or identity['config']['max_updates'] != 100000 or
            identity['config']['seed'] != 42):
        raise ValueError('Only the registered complete B100k endpoint can advance')
    return progress['status'] == 'COMPLETE' and progress['updates'] == 100000


def screen_state(value, is_live=processes):
    screen = Path(value['screen']); receipts = [read(p) for p in sorted(screen.glob('dispatches/*.json'))]
    ranks = [r['rank'] for r in receipts]
    if len(set(ranks)) != len(ranks) or any(r < 0 or r >= 24 for r in ranks):
        raise ValueError('Repeated or foreign screen shard')
    occupied = set(); complete = []
    for host in sorted({r['host'] for r in receipts}):
        subset = [r for r in receipts if r['host'] == host]
        alive = is_live(host, subset, 'scripts.analysis.fm_transfer.screen_train')
        for row, live in zip(subset, alive):
            if row['source'] != value['worker_source'] or row['global_shards'] != 24:
                raise ValueError('Foreign screen source or partition')
            path = screen / f"rank{row['rank']}.json"
            status = read(path) if path.exists() else None
            done = status is not None and status['status'] == 'ASSIGNED_COMPLETE'
            if live: occupied.add(host)
            if done: complete.append(row['rank'])
            elif not live: raise RuntimeError('Screen worker failed; no automatic replay')
    return occupied, len(complete) == 24


def baseline_ready(value):
    root = Path(value['root'])
    for cohort in ('DEV', 'TRAIN_DIAGNOSTIC'):
        directory = root / 'evaluations' / (PREFIX + '_' + cohort)
        analysis = root / 'analysis' / (directory.name + '.json')
        if not analysis.exists(): return False
        meta = read(directory / 'manifest.json'); result = read(analysis)
        if result['status'] != 'COMPLETE_POPULATION' or result['identity'] != meta:
            raise ValueError('Own complete SFT baseline analyzer disagrees with its population')
        if (meta['qualification_only'] or meta['new_SFT']['updates'] != 100000 or
                meta['stage'] != 'SFT' or meta['seed'] != 2101 or meta['modes'] != ['E_NATIVE', 'E_SHIFT5']):
            raise ValueError('An interim/foreign baseline cannot authorize RL')
    return True  # The unchanged formal dispatcher checks parent hashes and every gate again.


def next_actions(states, screen_complete, baselines_complete):
    if states['sft_summary'] != 'COMPLETE':
        return ['sft_summary'] if states['sft_summary'] == 'NOT_LAUNCHED' else []
    screens = [f'screen_{i}' for i in range(3)]
    if any(states[s] != 'COMPLETE' for s in screens):
        # The existing dispatcher holds one shared lock while starting a host
        # partition. Serialize dispatch calls; their scoring workers run concurrently.
        if any(states[s] == 'RUNNING' for s in screens): return []
        return [s for s in screens if states[s] == 'NOT_LAUNCHED'][:1]
    if not screen_complete: return []
    for name in ('hard_manifest', 'screen_summary'):
        if states[name] != 'COMPLETE': return [name] if states[name] == 'NOT_LAUNCHED' else []
    if not baselines_complete: return []
    for name in ('endpoint_gate', 'formal_RL'):
        if states[name] != 'COMPLETE': return [name] if states[name] == 'NOT_LAUNCHED' else []
    return []


def tick(value, control, local_host):
    if not endpoint_ready(value): return dict(status='WAITING_FOR_COMPLETE_B100K_SFT')
    states = {name:stage_state(control, name) for name in value['stages']}
    if states['sft_summary'] == 'COMPLETE':
        summary = read(Path(value['root']) / 'analysis' / (VARIANT + '_SFT_complete_training.json'))
        if (summary['variant'] != VARIANT or summary['logical_updates'] != 100000 or
                summary['logical_scene_exposures'] != 3200000 or summary['RL_candidates'] != 0):
            raise ValueError('Completed SFT journal does not match its registered endpoint budget')
    occupied, complete = screen_state(value)
    occupied.update(read(Path(control) / 'stages' / name / 'intent.json')['host']
        for name, state in states.items() if name.startswith('screen_') and state == 'RUNNING')
    baselines = baseline_ready(value)
    actions = next_actions(states, complete, baselines)
    launched = []
    for name in actions:
        hosts = ([h for h in value['screen_hosts'] if h not in occupied] if name.startswith('screen_')
                 else [value['training_host']] if name in ('endpoint_gate', 'formal_RL') else [local_host])
        for host in hosts:
            inventory = None
            if name.startswith('screen_') or name in ('endpoint_gate', 'formal_RL'):
                worker_allowed(host)
                first = resource_probe(host); time.sleep(2); second = resource_probe(host)
                if not capacity(first, second, 26000 if name.startswith('screen_') else 36000): continue
                inventory = dict(first=first, second=second)
                physical_path = Path(control) / 'physical_hosts.json'
                physical = read(physical_path) if physical_path.exists() else {}
                uuids = [g['uuid'] for g in second['gpus']]
                if (host in physical and physical[host] != uuids) or any(
                        set(other) & set(uuids) for h, other in physical.items() if h != host):
                    raise ValueError('Changed/aliased physical host; do not double reserve GPUs')
                physical[host] = uuids; atomic_json(physical_path, physical)
            launch(value, control, name, host, inventory); occupied.add(host); launched.append(dict(stage=name, host=host))
            break
    status = dict(status='ADVANCING_REGISTERED_B_ENDPOINT', stages=states,
                  screen_complete=complete, baselines_complete=baselines, launched=launched)
    if actions and not launched: status['status'] = 'WAITING_FOR_AUTHORIZED_MEMORY_HEADROOM'
    elif not actions and not complete: status['status'] = 'WAITING_FOR_COMMITTED_FULL_TRAIN_SCREEN'
    elif not actions and not baselines: status['status'] = 'WAITING_FOR_OWN_COMPLETE_SFT_BASELINES'
    if states['formal_RL'] == 'COMPLETE':
        run = Path(value['run']); receipts = [read(p) for p in run.glob('dispatches/*.json')]
        if len(receipts) != 1: raise ValueError('Formal handoff must have exactly one fresh trainer dispatch')
        receipt = receipts[0]
        if (receipt['source'] != value['worker_source'] or receipt['resume'] is not None or
                receipt['qualification_weights_used']): raise ValueError('Changed formal initialization')
        if not processes(receipt['host'], receipts, 'scripts.analysis.fm_transfer.train_rl')[0]:
            raise RuntimeError('Formal trainer exited during startup; inspect, never restart implicitly')
        if (run / 'metrics_rank0.jsonl').exists() and (run / 'metrics_rank0.jsonl').stat().st_size:
            status.update(status='FORMAL_B_RL_STARTED_HANDOFF_COMPLETE', dispatch=receipt,
                          milestone_evaluation_owner_unchanged=True)
    return status


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--control', default=str(ROOT / 'endpoint_handoff_B_v1'))
    parser.add_argument('--stage', help=argparse.SUPPRESS)
    parser.add_argument('--screen-hosts', nargs='+')
    parser.add_argument('--training-host')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(); control = Path(args.control).resolve()
    if args.stage: stage_worker(control, args.stage); return
    local_host = require_dispatch()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only controller required')
    controller_source = Path(__file__).resolve().parents[3]
    if not args.screen_hosts or not args.training_host: parser.error('Explicit screen and training hosts required')
    value = plan(ROOT, controller_source, args.screen_hosts, args.training_host)
    control.mkdir(parents=True, exist_ok=True)
    # One B handoff across control-directory names; do not duplicate the milestone owner.
    with (ROOT / 'B_endpoint_handoff.owner.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (control / 'plan.json').exists() and read(control / 'plan.json') != value:
            raise ValueError('Changed control source/plan: explicit migration required')
        if not (control / 'plan.json').exists():
            foreign = list(Path(value['screen']).glob('dispatches/*.json')) + list(Path(value['run']).glob('dispatches/*.json'))
            if foreign or Path(value['gate']).exists(): raise ValueError('Existing B endpoint work requires explicit adoption, not duplicate dispatch')
        atomic_json(control / 'plan.json', value)
        owner = dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
                     start_unix=time.time(), plan_sha256=value['sha256'], source_sha256=file_sha(__file__))
        atomic_json(control / 'owner.json', owner)
        try:
            while True:
                require_dispatch()
                for host in set(value['screen_hosts'] + [value['training_host']]): worker_allowed(host)
                result = tick(value, control, local_host)
                result.update(**owner, observed_unix=time.time()); atomic_json(control / 'status.json', result)
                if (control / 'STOP_AFTER_TICK').exists():
                    result.update(status='STOPPED_CONTROLLER_ONLY', existing_children_preserved=True)
                    atomic_json(control / 'status.json', result); break
                if args.once or result['status'] == 'FORMAL_B_RL_STARTED_HANDOFF_COMPLETE': break
                time.sleep(30)
        except BaseException:
            atomic_json(control / 'failure.json', dict(**owner, end_unix=time.time(), error=traceback.format_exc()))
            raise


if __name__ == '__main__': main()
