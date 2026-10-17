"""Read-only native B SFT endpoint audit; no trainer, GPU inference or recovery.

Run after all eight formal workers finish. The separate, existing
analyze_sft_training entry audits every update and the actual scene population.
This receipt verifies native payloads and counters, not planning improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import torch


ROOT = Path('/mnt/project/fm-transfer-20261016')
IDENTITY = '63b2da609f3ab3b225f056a5399050fc217989d25b937876a5a5051a29575af7'
SOURCE = '4e0ce1f45b6778895648689228a5628432bd2616'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def exited(dispatch, run):
    policy = read(Path('/mnt/project/server_dispatch_policy.json'))
    require(dispatch['host'] in policy['allowed_hosts'], 'Host no longer authorized')
    code = '''import json,os,sys
from pathlib import Path
r=json.loads(sys.argv[1]); run=sys.argv[2];p=Path('/proc')/str(r['pid'])
try:
 s=(p/'stat').read_text().rsplit(')',1)[1].split()
 same=int(s[19])==r['start_ticks']; state=s[0]
except FileNotFoundError:same=False;state='ABSENT'
matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 if int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0))
  if run in argv:matches.append(int(p.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(dict(same_trainer_process=same,state=state,run_processes=matches)))
'''
    command = ['python', '-c', code,
               json.dumps({k: dispatch[k] for k in ('pid', 'start_ticks')}), str(run)]
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o',
                             'ConnectTimeout=8', dispatch['host'], shlex.join(command)],
                            capture_output=True, text=True, check=True, timeout=45)
    observation = json.loads(result.stdout)
    require(not observation['same_trainer_process'] and not observation['run_processes'],
            'Recorded trainer or run workers have not exited')
    return observation


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), 'Preserve earlier audit receipts')
    run = ROOT / 'runs/B_SOURCE_ACTION_DIT_SFT_seed42_v1'
    checkpoint = run / 'checkpoints/update100000'
    identity = read(run / 'identity.json')
    require(identity['sha256'] == IDENTITY and not identity['qualification'], 'Foreign SFT identity')
    require(identity['config']['max_updates'] == 100000 and
            identity['config']['global_batch'] == 32 and
            identity['config']['world'] == 8 and identity['config']['seed'] == 42,
            'Changed registered training budget')
    require(read(checkpoint / 'identity.json') == identity, 'Foreign checkpoint identity')
    seal = read(checkpoint / 'COMPLETE')
    require(seal['phase'] == 'READY' and seal['world'] == 8 and
            seal['updates'] == 100000 and seal['cursor'] == 3200000, 'Uncommitted SFT endpoint')
    require(sha(checkpoint / 'files.json') == seal['files_sha256'], 'Changed file manifest')
    files = read(checkpoint / 'files.json')
    require(set(files) == {'model.pt', 'optimizer.pt', 'identity.json'} |
            {f'rank{rank}.pt' for rank in range(8)}, 'Incomplete native payload inventory')
    progress = read(run / 'progress.json')
    require(progress['status'] == 'COMPLETE' and progress['updates'] == 100000 and
            progress['cursor'] == 3200000, 'SFT has not completed')
    results = [read(run / f'result_rank{rank}.json') for rank in range(8)]
    for rank, result in enumerate(results):
        require(result['status'] == 'COMPLETE' and result['end_update'] == 100000 and
                result['cursor'] == 3200000 and result['rank'] == rank and
                result['identity'] == IDENTITY and not result['qualification'],
                f'Rank {rank} lacks a matching formal completion')
    dispatches = [read(path) for path in (run / 'dispatches').glob('*.json')]
    require(len(dispatches) == 1 and dispatches[0]['resume'] is None and
            Path(dispatches[0]['source']).name == SOURCE, 'Unexpected trainer ancestry')
    process_exit = exited(dispatches[0], run)
    # Hash each native payload once, in a bounded streaming buffer. mmap below
    # inspects optimizer counters and metadata without copying all moments.
    verified = {}
    for name, expected in files.items():
        actual = sha(checkpoint / name)
        require(actual == expected, 'Native payload hash mismatch: ' + name)
        verified[name] = actual
    payload = torch.load(checkpoint / 'optimizer.pt', map_location='cpu',
                         weights_only=False, mmap=True)
    states = payload['optimizer']['state']
    parameters = identity['parameters']
    require(len(states) == len(parameters) == 824, 'Unexpected optimizer parameter count')
    require({float(state['step']) for state in states.values()} == {100000.0}, 'Adam step mismatch')
    moment_dtypes = {str(state[key].dtype) for state in states.values()
                    for key in ('exp_avg', 'exp_avg_sq')}
    require(moment_dtypes == {'torch.float32'}, 'Adam moments are not FP32')
    require(payload['scheduler']['last_epoch'] == 100000, 'Scheduler step mismatch')
    model = torch.load(checkpoint / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    require(set(parameters) <= set(model), 'Model is missing a registered trainable tensor')
    require(all(list(model[name].shape) == meta['shape'] and str(model[name].dtype) == meta['dtype']
                for name, meta in parameters.items()), 'Model parameter shape/dtype mismatch')
    rank_states = []
    for rank in range(8):
        value = torch.load(checkpoint / f'rank{rank}.pt', map_location='cpu', weights_only=False)
        require(value['identity'] == IDENTITY and value['state'] ==
                dict(updates=100000, cursor=3200000, phase='READY'), 'Rank saved boundary mismatch')
        require(set(value['rng']) == {'python', 'numpy', 'torch', 'cuda'} and
                value['rng']['cuda'] is not None, 'Incomplete rank RNG state')
        rank_states.append(dict(rank=rank, state=value['state'], rng_keys=sorted(value['rng'])))
    receipt = dict(schema='fm_transfer_B_SFT100k_native_completion_audit_v1',
        status='PASS_NATIVE_SFT_ENDPOINT_NOT_RL_OR_NAVTEST', observed_unix=time.time(),
        script_sha256=sha(Path(__file__)), run=str(run), checkpoint=str(checkpoint),
        identity=IDENTITY, training_source=SOURCE, seal=seal,
        all_eight_training_results_complete=True, process_exit=process_exit,
        payloads_rehashed=verified, optimizer_states=len(states), Adam_steps=[100000],
        moment_dtypes=sorted(moment_dtypes), scheduler_last_epoch=payload['scheduler']['last_epoch'],
        scheduler_last_lr=payload['scheduler']['_last_lr'],
        registered_parameter_tensors=len(parameters),
        additional_model_state_keys=sorted(set(model)-set(parameters)), rank_states=rank_states,
        new_inference=0, new_score_calls=0, new_optimizer_updates=0,
        limits=['Does not replay recovery; use the existing world8 recovery qualification.',
                'Full scene exposure and costs are audited separately by analyze_sft_training.',
                'No claim of improved planning or successful RL.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=receipt['status'], output=str(output),
                         optimizer_states=len(states), Adam_steps=[100000])))


if __name__ == '__main__':
    main()
