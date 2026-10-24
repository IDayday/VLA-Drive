"""CPU scheduling adapter: preserve old experiments, honor finite GPU leases.

Imports the original immutable campaign. Only its resource admission is extended;
training/evaluation workers and scientific identities remain their pinned source.
"""
import argparse
import fcntl
import json
from pathlib import Path
import runpy
import sys


def reserved(path):
    p = Path(path)
    if not p.exists():
        return set()
    value = json.loads(p.read_text())
    if value['status'] != 'ACTIVE':
        return set()
    return set(value['gpu_uuids'])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--reservation', required=True)
    p.add_argument('--module', choices=['campaign', 'safe_campaign'], required=True)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from starVLA.rl.qwen_sourceflow import resources
    original = resources.available

    def admitted(exclude=(), min_free_MiB=32000):
        return original(set(exclude) | reserved(a.reservation), min_free_MiB)

    resources.available = admitted
    # Importing after installing the admission adapter binds the same function
    # in Campaign and SafeCampaign, including direct available() calls.
    args = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    sys.argv = ['legacy_resource_owner', *args]
    runpy.run_module('scripts.analysis.qwen_sourceflow.' + a.module,
                     run_name='__main__')


if __name__ == '__main__':
    main()
