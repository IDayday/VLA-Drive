"""Relay fresh coordinator PID checks to a host without outbound SSH credentials.

The original dispatcher still owns admission, locking and worker launch. Only its
remote liveness lookup is supplied by the coordinator's receipt-bound inventory;
local PIDs are rechecked on the launching host. No inference module is changed.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import socket
import sys
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_proof(path, queue, now=None):
    proof = json.loads(Path(path).read_text())
    body = {k: v for k, v in proof.items() if k != 'sha256'}
    now = time.time() if now is None else now
    if (proof.get('sha256') != digest(body) or proof.get('schema') != 'final_dispatch_pid_relay_v1'
            or Path(proof['queue']).resolve() != Path(queue).resolve()
            or not 0 <= now - proof['finished_unix'] <= 120
            or proof['finished_unix'] - proof['started_unix'] > 120):
        raise ValueError('Wrong, stale or unsealed coordinator PID inventory')
    records = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(queue).glob('workers/*/*.json'))}
    if records != proof['receipt_files']:
        raise ValueError('Queue changed after coordinator PID verification')
    return proof


def checked_live(proof, record, local_live, host=None):
    host = host or socket.gethostname().removesuffix('-worker-0')
    key = record['sha256']
    evidence = proof['checks'].get(key)
    if evidence is None or evidence['record'] != record or type(evidence['alive']) is not bool:
        raise ValueError('PID evidence does not bind this exact dispatch receipt')
    if record['host'] == host:
        return local_live(record)
    return evidence['alive']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--proof', required=True)
    parser.add_argument('--dispatcher-source', required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source = Path(args.dispatcher_source).resolve()
    sys.path.insert(0, str(source))
    dispatch = importlib.import_module('scripts.analysis.fm_transfer.dispatch_final_evaluation')
    original = dispatch.previous_attempts

    def previous_attempts(queue, output, contract, ranks, retry):
        # Called inside the unchanged dispatcher's queue-wide exclusive lock.
        proof = load_proof(args.proof, queue)
        actual = hashlib.sha256(Path(dispatch.__file__).read_bytes()).hexdigest()
        if actual != proof['dispatcher_sha256']:
            raise ValueError('Dispatcher differs from verified coordinator source')
        return original(queue, output, contract, ranks, retry,
                        is_live=lambda record: checked_live(proof, record, dispatch.local_live))

    dispatch.previous_attempts = previous_attempts
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    sys.argv = [str(Path(dispatch.__file__)), *arguments]
    dispatch.main()


if __name__ == '__main__':
    main()
