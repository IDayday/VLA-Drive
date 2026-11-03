"""CPU-only typed world16 READY execution upgrade into a completely new root."""
import argparse
import json
from starVLA.rl.d46_world16.execution_upgrade import prepare


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--old-frozen-source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--flags-json',required=True,help='Exactly fast_prefix/fuse_fm_mc/async_scoring/rotary_cache/no_grad_prefix_cache boolean fields')
    p.add_argument('--qualification',action='store_true');a=p.parse_args()
    print(json.dumps(prepare(a.checkpoint,a.output,a.old_frozen_source,execution_flags=json.loads(a.flags_json),qualification=a.qualification),indent=2))


if __name__=='__main__':main()
