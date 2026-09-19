"""Exhaustive fixed-chain comparison; original tolerances, no early abort."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.flow_grpo.comparison import compare_named
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--left', required=True); p.add_argument('--right', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'PASS', 'left': a.left, 'right': a.right,
              'atol': 2e-6, 'rtol': 2e-3, 'script_sha256': file_sha(__file__), 'scenes': {}}
    for rank in (1, 2):
        row = {}
        for kind in ('stats', 'grads'):
            name = f'{kind}_rank{rank}.pt'
            left, right = [torch.load(Path(root)/name, weights_only=True, mmap=True) for root in (a.left, a.right)]
            report = compare_named(left, right)
            atomic_json(out/f'{kind}_rank{rank}.json', report)
            row[kind] = {'status': report['status'], 'modules': report['modules']}
            if report['status'] != 'PASS': result['status'] = 'FAIL'
            del left, right
        result['scenes'][rank] = row
        atomic_json(out/'summary.json', result)
        print(json.dumps({'scene': rank, **row}), flush=True)
    if result['status'] != 'PASS': raise SystemExit(1)

if __name__ == '__main__': main()
