"""Read-only endpoint/worker audit; write a separate evidence receipt only."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

import torch

ROOT = Path('/mnt/project/fm-transfer-20261016')
RUN = ROOT / 'runs/A_DDP_TIME_R_SOURCE_seed42_v1'
OUT = ROOT / 'qualification/A_RL20k_analysis'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def probe(host, records, run=None):
    code = '''import json,pathlib
records=RECORDS
out=[]
for r in records:
 p=pathlib.Path('/proc')/str(r['pid'])
 try:
  stat=(p/'stat').read_text().split(') ',1)[1].split()
  out.append(dict(pid=r['pid'],same_process=int(stat[19])==r['start_ticks'],state=stat[0]))
 except FileNotFoundError:out.append(dict(pid=r['pid'],same_process=False,state='ABSENT'))
matches=[]
if RUN:
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   cmd=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0))
   if RUN in cmd:matches.append(dict(pid=int(p.name),command=cmd))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(dict(pinned=out,run_processes=matches)))
'''.replace('RECORDS', repr(records)).replace('RUN', repr(run))
    command = ['python', '-'] if host == socket.gethostname().removesuffix('-worker-0') else [
        'ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, 'python', '-']
    p = subprocess.run(command, input=code, text=True, capture_output=True, timeout=45, check=True)
    result = json.loads(p.stdout)
    assert not any(x['same_process'] for x in result['pinned']), result
    assert not result['run_processes'], result
    return result


def main():
    out = dict(schema='fm_transfer_A_RL20k_completion_audit_v1', observed_unix=time.time(),
               evidence_script_sha256=sha(Path(__file__)), cohorts={})
    for cohort, total in [('DEV', 1696), ('TRAIN_DIAGNOSTIC', 512)]:
        folder = ROOT / f'evaluations/A_DDP_TIME_RL_update020000_{cohort}'
        records = [read(p) for p in sorted((folder / 'dispatches').glob('*_rank*.json'))]
        assert len(records) == 8 and sorted(x['rank'] for x in records) == list(range(8))
        ranks = [read(folder / f'rank{i}.json') for i in range(8)]
        assert all(x['status'] == 'ASSIGNED_COMPLETE' and x['direct_original_check'] for x in ranks)
        assert sum(x['new'] + x['reused'] for x in ranks) == total
        processes = probe(records[0]['host'], records)
        out['cohorts'][cohort] = dict(host=records[0]['host'],
            workers=[{k:x[k] for k in ('rank','pid','start_ticks','source')} for x in records],
            process_probe=processes, all_pinned_workers_exited=True,
            closed_worker_invocations=8, raw_original_checks_passed=8,
            actual_scene_count=total, physical_predictions=sum(x['physical_predictions'] for x in ranks),
            diagnostic_predictions=sum(x['diagnostic_predictions'] for x in ranks))
    dispatch = read(next((RUN / 'dispatches').glob('*.json')))
    out['training_exit'] = probe(dispatch['host'], [dispatch], str(RUN))
    results = [read(RUN / f'result_rank{i}.json') for i in range(8)]
    assert all(x['status']=='COMPLETE' and x['updates']==20000 and x['collections']==5000
               and x['raw_scene_exposures']==40000 and x['new_candidates']==320000 for x in results)
    ck = RUN / 'checkpoints/update020000_inner0'
    seal = read(ck/'COMPLETE')
    assert seal['phase']=='COMPLETE' and seal['updates']==20000 and seal['world']==8
    payload = torch.load(ck/'optimizer.pt', map_location='cpu', weights_only=False)
    optimizer = payload['optimizer']
    steps = [float(x['step']) for x in optimizer['state'].values()]
    assert len(steps)==len(payload['adapters'])==192 and set(steps)=={20000.0}
    assert payload['scheduler']['last_epoch']==20000
    out['optimizer'] = dict(adam_states=len(steps), adam_steps=sorted(set(steps)),
        scheduler_last_epoch=payload['scheduler']['last_epoch'],
        scheduler_last_lr=payload['scheduler']['_last_lr'],
        checkpoint_files_sha256=seal['files_sha256'], optimizer_file_sha256=sha(ck/'optimizer.pt'))
    out['all_eight_training_results_complete']=True
    out['status']='PASS_COMPLETE_A_RL_NOT_FINAL_NAVTEST'
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'completion.json').open('x') as f:
        json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=out['status'],receipt=str(OUT/'completion.json'),optimizer=out['optimizer'])))


if __name__=='__main__':
    main()
