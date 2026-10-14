"""Resource-only supervision for a fixed Scorer cell on another allowed host."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from iqe.contracts import require
from iqe.io import atomic_json, file_hash, lock, read_json
from .config import load_config
from .resource_guard import check


class ReceiptProcess:
    """Observe a heartbeat from an already assigned worker; never relaunch it."""
    def __init__(self, path, config, cell):
        self.path = Path(path)
        self.expected = {'config_identity': config.identity, 'cell': cell,
                         'Base_binding_hash': file_hash(config.root / 'BASE_BINDING.json')}
        self.result = config.root / 'scorers' / cell / 'result.json'
        self.pid = 0
        self.returncode = None
        self.worker_identity = None
        self.poll()

    def poll(self):
        value = read_json(self.path)
        require(all(value.get(k) == v for k, v in self.expected.items()), 'external worker scientific identity mismatch')
        identity = (value['host'], value['pid'], value['start_ticks'])
        if self.worker_identity is None:
            self.worker_identity = identity
        require(identity == self.worker_identity, 'external worker process identity changed')
        self.pid = value['pid']
        if value['status'] == 'COMPLETE':
            require(self.result.exists() and read_json(self.result)['status'] == 'COMPLETE',
                    'external worker lacks complete training result')
            self.returncode = 0
        elif value['status'] in ('FAILED', 'DEADLINE_PAUSED'):
            self.returncode = 1
        else:
            require(time.time() - value['updated_unix'] < 180, 'external worker heartbeat stale; do not duplicate launch')
        return self.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--cell', choices=('S00', 'S01', 'S10', 'S11'), required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--receipt', required=True)
    p.add_argument('--deadline-hours', type=float, default=24)
    args = p.parse_args()
    require(0 < args.deadline_hours <= 36 and 0 <= args.gpu < 8, 'bounded fixed-cell worker required')
    config = load_config(args.config)
    check(config, 'train-scorer-cell')
    repo = Path(__file__).resolve().parents[2]
    receipt = Path(args.receipt)
    deadline = time.monotonic() + args.deadline_hours * 3600
    identity = {'config_identity': config.identity, 'config_file_sha256': file_hash(args.config),
                'Base_binding_hash': file_hash(config.root / 'BASE_BINDING.json'), 'cell': args.cell,
                'host': os.uname().nodename, 'pid': os.getpid(), 'GPU': args.gpu,
                'start_ticks': Path('/proc/self/stat').read_text().split()[21],
                'training_definition_changed': False}

    def tick(status, **detail):
        atomic_json(receipt, {**identity, 'status': status, 'updated_unix': time.time(), **detail})

    with lock(str(receipt) + '.RUN.lock'):
        try:
            from .capacity import probe
            from .ram_stage import available_bytes
            dependency = config.root / 'banks/rich/scorer_fit/BANK.json'
            while True:
                require(time.monotonic() < deadline, 'external worker admission deadline reached')
                capacity = probe(args.gpu)
                memory = available_bytes()
                if dependency.exists() and capacity['ready'] and memory >= 128 * 1024**3:
                    break
                tick('WAITING_INPUT_OR_CAPACITY', bank_complete=dependency.exists(), capacity=capacity,
                     MemAvailable=memory, reserve_GiB=128)
                time.sleep(10)
            folder = config.root / 'scorers' / args.cell
            result = folder / 'result.json'
            if result.exists() and read_json(result)['status'] == 'COMPLETE':
                tick('COMPLETE', reused=True)
                return
            command = [sys.executable, '-u', '-m', 'iqe.specialist_study.cli', 'train-scorer-cell',
                       '--config', args.config, '--device', 'cuda:0', '--cell', args.cell]
            if list(folder.glob('step_*.pt.COMPLETE.json')):
                command += ['--resume', 'auto']
            env = dict(os.environ, PYTHONPATH=str(repo), CUDA_VISIBLE_DEVICES=str(args.gpu),
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            log = receipt.with_suffix('.log')
            with open(log, 'ab') as stream:
                child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
            launch = {**identity, 'child_pid': child.pid, 'command': command, 'log': str(log),
                      'child_start_ticks': Path(f'/proc/{child.pid}/stat').read_text().split()[21],
                      'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()}
            atomic_json(str(receipt) + f'.LAUNCH_{child.pid}.json', launch, immutable=True)
            paused = False
            while child.poll() is None:
                if not paused and time.monotonic() >= deadline:
                    child.send_signal(signal.SIGTERM)  # Native optimizer-boundary checkpoint, this owned child only.
                    paused = True
                tick('RUNNING', child_pid=child.pid, native_pause_requested=paused)
                time.sleep(10)
            complete = child.returncode == 0 and result.exists() and read_json(result)['status'] == 'COMPLETE'
            tick('COMPLETE' if complete else ('DEADLINE_PAUSED' if paused else 'FAILED'),
                 child_pid=child.pid, returncode=child.returncode)
            require(complete, 'external fixed Scorer cell did not complete; artifacts retained')
        except Exception as exc:
            tick('FAILED', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
