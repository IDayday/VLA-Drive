"""Export strict current-only IL/RL policy; no teacher/score/cache dependence."""
import argparse
import json
from starVLA.rl.qwen_world_postcot.deployment import export_checkpoint
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--rl-checkpoint');a=p.parse_args()
    meta=export_checkpoint(a.checkpoint,a.output,a.rl_checkpoint);print(json.dumps(dict(status='COMPLETE',model_sha256=meta['model_sha256'],LoRA_merge=False)))
if __name__=='__main__':main()
