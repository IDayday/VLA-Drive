"""Read-only verification of already-launched fixed-host continuations."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PLANS = json.loads((ROOT / 'LEGAL_RECOVERY_PLANS.json').read_text())['plans']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tail_row(path):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 131072))
        lines = stream.read().splitlines()
    # An active writer may have a partial final line.
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    raise ValueError('No complete recent journal row: ' + str(path))


def verify(name):
    plan = PLANS[name]
    run = Path(plan['run'])
    il = name == 'IL'
    directory = Path('/mnt/project/qwen-world-postcot-20261025/jobs/formal_IL_attempt003') if il else ROOT / name / 'jobs/FORMAL_RESUME'
    if (directory / 'RESULT.json').exists():
        raise ValueError('Continuation already exited: ' + str(directory))
    worker = json.loads((directory / 'RUNNING.json').read_text())
    if worker['host'] != plan['host'] or worker['cwd'] != plan['source']:
        raise ValueError('Changed fixed-host/source binding')
    identity_path = run / ('IDENTITY.json' if il else 'identity.json')
    identity = json.loads(identity_path.read_text())
    mismatches = [rel for rel, expected in identity['source_files'].items()
                  if sha(Path(plan['source']) / rel) != expected]
    if mismatches:
        raise ValueError('Changed native source: ' + repr(mismatches))
    rows = [tail_row(run / f'metrics_rank{rank}.jsonl') for rank in range(8)]
    counts = [row['updates'] for row in rows]
    if min(counts) <= plan['updates']:
        raise ValueError('No actual post-restore update on every rank')
    first = None
    with (run / 'metrics_rank0.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['updates'] == plan['updates'] + 1:
                first = {key: row[key] for key in ('updates', 'collection', 'inner_completed',
                         'invocation', 'ratio_max_abs_from_one', 'gradient_norm', 'optimizer_global_steps',
                         'optimizer_skipped_steps', 'cursor', 'phase') if key in row}
                break
    if first is None:
        raise ValueError('Missing first restored update')
    program = r'''
import json,sys
from pathlib import Path
cfg=json.loads(sys.argv[1]); root=Path('/proc')/str(cfg['worker_pid'])
fields=(root/'stat').read_text().rsplit(')',1)[1].split()
if int(fields[19])!=cfg['worker_start_ticks'] or fields[0]=='Z':raise ValueError('Worker identity changed')
ranks=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=[v.decode() for v in (p/'cmdline').read_bytes().split(bytes([0])) if v]
  if cfg['module'] not in args or str((p/'cwd').resolve())!=cfg['cwd']:continue
  env=dict(v.decode().split('=',1) for v in (p/'environ').read_bytes().split(bytes([0])) if b'=' in v)
  if env.get('WORLD_SIZE')!='8' or 'RANK' not in env:continue
  fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  if fields[0]=='Z':continue
  ranks.append(dict(pid=int(p.name),start_ticks=int(fields[19]),rank=int(env['RANK']),local_rank=int(env['LOCAL_RANK'])))
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
if sorted(v['rank'] for v in ranks)!=list(range(8)):raise ValueError('Native rank closure changed: '+repr(ranks))
print(json.dumps(dict(worker_alive=True,ranks=ranks)))
'''
    cfg = dict(worker_pid=worker['worker_pid'], worker_start_ticks=worker['worker_start_ticks'],
               cwd=worker['cwd'], module='tools.qwen_world_postcot.train_il' if il else 'tools.qwen_world_postcot.train_scope_rl')
    import shlex
    command = ['/usr/bin/python3', '-c', program, json.dumps(cfg)]
    live = json.loads(subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                      plan['host'], shlex.join(command)], text=True, timeout=45))
    return dict(host=plan['host'], status='RUNNING_AND_UPDATING', restored_update=plan['updates'],
                current_common_update=min(counts), rank_update_range=[min(counts), max(counts)],
                lost_unsealed_updates=plan['lost_unsealed_updates'], first_restored_update=first,
                training_source=plan['source'], checked_source_files=len(identity['source_files']),
                source_mismatches=mismatches, native_worker=live, worker_state=str(directory/'RUNNING.json'),
                recovery_inventory_sha256=plan['files_sha256'], identity_sha256=sha(identity_path))


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(zip(PLANS, pool.map(verify, PLANS)))
    receipt = dict(status='PASS', observed_utc=datetime.now(timezone.utc).isoformat(), runs=results,
                   fixed_hosts_only=True, training_math_changed=False,
                   temporal_tail_rollback_disclosed=True,
                   temporary_exit_audit_sha256=sha(ROOT/'TEMPORARY_EXIT_AUDIT.json'))
    path = ROOT / 'MIGRATION_CONFIRMED.json'
    if path.exists():
        raise FileExistsError('Do not overwrite the original migration proof')
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
