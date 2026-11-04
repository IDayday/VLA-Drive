"""Publish an independent execution root from a complete formal checkpoint."""
import argparse
import json
from starVLA.rl.d46_fpo_engineering.migration import prepare

if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--profile', choices=['exact', 'fast'], required=True)
    p.add_argument('--evidence', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.checkpoint, a.source, a.output, profile=a.profile, evidence=a.evidence)))
