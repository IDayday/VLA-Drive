"""Summarize fresh rollout coverage without inflating four inner updates."""
import argparse
import json
from starVLA.rl.qwen_world_research_data.journal_analysis import save_analysis

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run-root',required=True);p.add_argument('--stream-manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--world-size',type=int,default=8)
    p.add_argument('--window-updates',type=int,default=1000)
    p.add_argument('--current-bank');p.add_argument('--baseline-bank')
    a=p.parse_args();result=save_analysis(a.output,run_root=a.run_root,stream_manifest=a.stream_manifest,
        world_size=a.world_size,window_updates=a.window_updates,current_bank=a.current_bank,baseline_bank=a.baseline_bank)
    print(json.dumps({k:v for k,v in result.items() if k not in ('file_snapshots','per_scene','paired_ODE_regressions')},indent=2))

if __name__=='__main__':main()
