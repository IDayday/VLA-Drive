"""Resume a cooperatively stopped native run through the qualified adapter."""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from starVLA.rl.constrained_flow_noise.contracts import file_sha, require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--acceptance', required=True)
    parser.add_argument('--source-lock', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--adapter-module', default='scripts.analysis.constrained_flow_noise.train_efficient_cast',
                        choices=['scripts.analysis.constrained_flow_noise.train_efficient_cast',
                                 'scripts.analysis.constrained_flow_noise.train_efficient_batch'])
    args = parser.parse_args()
    host = require_dispatch()
    run, checkpoint, receipt = map(Path, (args.run, args.checkpoint, args.receipt))
    if receipt.exists():
        raise FileExistsError(receipt)
    guard = run.with_suffix('.launcher.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for path, digest in json.loads(Path(args.source_lock).read_text())['sources'].items():
        if file_sha(path) != digest:
            raise ValueError('Frozen scientific source changed: ' + path)
    proof = json.loads(Path(args.acceptance).read_text())
    execution = importlib.import_module(args.adapter_module).source_identity()
    if proof.get('status') != 'PASS' or proof.get('execution_sha256') != execution['sha256']:
        raise ValueError('Missing matching execution qualification')
    if not (checkpoint / 'COMPLETE').is_file():
        raise ValueError('Native checkpoint incomplete')
    identity = json.loads((run / 'run_identity.json').read_text())
    if json.loads((checkpoint / 'identity.json').read_text()) != identity:
        raise ValueError('Checkpoint scientific identity changed')
    for rank in range(8):
        report = json.loads((run / f'result_rank{rank}.json').read_text())
        if (report['status'] != 'STOPPED_AT_SAVED_BOUNDARY' or
                Path(report['checkpoint']) != checkpoint or report.get('safety_pause')):
            raise ValueError('Every rank must stop at the same safe native boundary')
    for relative, digest in json.loads((checkpoint / 'files.json').read_text()).items():
        if file_sha(checkpoint / relative) != digest:
            raise ValueError('Changed checkpoint file: ' + relative)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                         '--format=csv,noheader'], text=True)
    rows = [line.split(',') for line in inventory.strip().splitlines()]
    if len(rows) != 8 or len({r[1].strip() for r in rows}) != 8 or min(int(r[2].split()[0]) for r in rows) < 35000:
        raise RuntimeError('Insufficient free capacity; no preemption performed')
    original = json.loads(run.with_suffix('.launcher.json').read_text())
    command = list(original['args'])
    index = command.index('scripts.analysis.constrained_flow_noise.train')
    command[index:index + 1] = [args.adapter_module,
                               '--acceptance', str(Path(args.acceptance).resolve()), '--']
    if '--resume' in command:
        index = command.index('--resume')
        del command[index:index + 2]
    command += ['--resume', str(checkpoint)]
    command[0] = sys.executable
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               NO_ALBUMENTATIONS_UPDATE='1', PYTHONUNBUFFERED='1')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    log = receipt.with_suffix('.train.log')
    with log.open('x') as stream:
        process = subprocess.Popen(command, env=env, cwd=Path.cwd(), stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(status='RUNNING', host=host, pid=process.pid, args=command,
                  checkpoint=str(checkpoint), scientific_identity=identity['sha256'],
                  execution_identity=execution, acceptance=str(Path(args.acceptance).resolve()),
                  started_unix=time.time(), inventory=inventory, log=str(log))
    atomic_json(receipt, record)
    code = process.wait()
    record.update(status='TRAIN_PROCESS_EXITED', returncode=code, finished_unix=time.time())
    atomic_json(receipt, record)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
