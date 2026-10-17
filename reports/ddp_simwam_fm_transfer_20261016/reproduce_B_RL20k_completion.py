"""Audit the registered B 20k observations; no inference, scoring or update."""
import json
import time
import socket
import subprocess
from pathlib import Path
import torch
from scripts.analysis.fm_transfer.milestone_controller import live_workers
from scripts.analysis.fm_transfer.final_lock import authenticated, validate_rl_summary
from starVLA.rl.simwam_recipe.contracts import file_sha

ROOT = Path('/mnt/project/fm-transfer-20261016')
RUN = ROOT/'runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1'
OUT = ROOT/'qualification/B_RL20k_analysis/completion.json'

def read(path):
    return json.loads(path.read_text())

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
    if OUT.exists():
        raise FileExistsError('Preserve previous evidence')
    result = dict(schema='fm_transfer_B_RL20k_completion_audit_v1', observed_unix=time.time(),
                  evidence_script_sha256=file_sha(__file__), cohorts={})
    for cohort, total in [('DEV', 1696), ('TRAIN_DIAGNOSTIC', 512)]:
        folder = ROOT/f'evaluations/B_SOURCE_ACTION_DIT_RL_update020000_{cohort}'
        manifest = read(folder/'manifest.json')
        assert manifest['seed']==2101 and manifest['modes']==['E_NATIVE','E_SHIFT5']
        records = [read(p) for p in sorted((folder/'dispatches').glob('*_rank*.json'))]
        assert len(records)==8 and sorted(r['rank'] for r in records)==list(range(8))
        ranks = [read(folder/f'rank{i}.json') for i in range(8)]
        assert all(r['status']=='ASSIGNED_COMPLETE' and r['direct_original_check']
                   and r['identity_sha256']==manifest['sha256'] for r in ranks)
        assert sum(r['new']+r['reused'] for r in ranks)==total
        processes = {}
        for host in {r['host'] for r in records}:
            processes[host] = live_workers(host,[r for r in records if r['host']==host])
            assert not any(processes[host]), 'Evaluation workers have not exited yet'
        invocations = [read(p) for p in (folder/'invocations').glob('*/rank*.json')]
        assert len(invocations)==8 and sorted(r['rank'] for r in invocations)==list(range(8))
        assert all(r['status']=='ASSIGNED_COMPLETE' and 'end_unix' in r
                   and r['identity_sha256']==manifest['sha256'] for r in invocations)
        result['cohorts'][cohort] = dict(
            workers=[{k:r[k] for k in ('rank','pid','start_ticks','source','host')} for r in records],
            process_probe=processes, all_pinned_workers_exited=True,
            closed_worker_invocations=8, raw_original_checks_passed=8,
            actual_scene_count=total, new_scenes=sum(r['new'] for r in ranks),
            reused_scenes=sum(r['reused'] for r in ranks),
            physical_predictions=sum(r['physical_predictions'] for r in invocations),
            diagnostic_predictions=sum(r['diagnostic_predictions'] for r in invocations),
            total_worker_wall_seconds=sum(r['total_worker_wall_seconds'] for r in invocations),
            job_wall_seconds=max(r['end_unix'] for r in invocations)-min(r['start_unix'] for r in invocations),
            manifest_sha256=file_sha(folder/'manifest.json'))
    checkpoint = RUN/'checkpoints/update020000_inner0'
    seal = read(checkpoint/'COMPLETE');files=read(checkpoint/'files.json')
    assert seal['phase']=='COMPLETE' and seal['updates']==20000 and seal['world']==8 and seal['next_inner']==0
    assert file_sha(checkpoint/'files.json')==seal['files_sha256']
    assert file_sha(checkpoint/'optimizer.pt')==files['optimizer.pt']
    payload = torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
    steps = [float(x['step']) for x in payload['optimizer']['state'].values()]
    assert len(steps)==len(payload['adapters'])==480 and set(steps)=={20000.0}
    assert payload['scheduler']['last_epoch']==20000
    assert all(x['exp_avg'].dtype==torch.float32 and x['exp_avg_sq'].dtype==torch.float32
               for x in payload['optimizer']['state'].values())
    prefix = read(ROOT/'analysis/B_SOURCE_ACTION_DIT_RL_complete_training.json')
    pop = read(ROOT/'analysis/B_SOURCE_ACTION_DIT_RL_update020000_population.json')
    assert prefix['cumulative']['actual_global_optimizer_updates']==20000
    assert prefix['cumulative']['fresh_global_rollouts']==5000
    assert prefix['cumulative']['scene_exposures']==40000
    assert prefix['cumulative']['new_candidates']==pop['population']['fresh_candidates']==320000
    validate_rl_summary(prefix, authenticated(RUN/'identity.json'))
    ledger=prefix['cost']['invocation_ledger']
    assert ledger['incomplete_invocations']==0 and ledger['unclosed_invocation_worker_wall_seconds']==0
    dispatches = [read(p) for p in (RUN/'dispatches').glob('*.json')]
    assert len(dispatches)==1, 'Unexpected training attempts; inspect lineage before publication'
    dispatch=dispatches[0]
    result['training_exit']=probe(dispatch['host'],[dispatch],str(RUN))
    rows=[read(RUN/f'result_rank{rank}.json') for rank in range(8)]
    for rank,row in enumerate(rows):
        assert row['rank']==rank and not row['qualification']
        assert row['status']=='COMPLETE' and row['updates']==20000 and row['collections']==5000
        assert row['raw_scene_exposures']==40000 and row['new_candidates']==320000
        assert row['next_inner']==0 and row['phase']=='COMPLETE'
    result['all_eight_training_results_complete']=True
    result['optimizer']=dict(adam_states=480,adam_steps=[20000.0],FP32_moments=True,
        scheduler_last_epoch=20000,checkpoint_files_sha256=seal['files_sha256'],
        optimizer_file_sha256=files['optimizer.pt'])
    result['status']='PASS_COMPLETE_B_RL_NOT_FINAL_NAVTEST'
    result['training_endpoint_complete']=True
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],receipt=str(OUT))))

if __name__=='__main__':
    main()
