"""Actually construct Base/VQA on CPU and compare every initialized trainable tensor."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent,file_sha256
from navsim.agents.EpisodeDrive.planreg_v2.runtime import load_config
from navsim.agents.EpisodeDrive.planreg_v2.optimizer import logical_group

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base-config',required=True);p.add_argument('--vqa-config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    reports=[]
    for path in (args.base_config,args.vqa_config):
        cfg=load_config(path)
        if not cfg.get('shared_init_path'):raise ValueError('Explicit shared initialization artifact required')
        agent=PlanRegV2Agent(cfg,'cpu')
        state=agent.trainable_state()
        hashes={n:hashlib.sha256(t.numpy().tobytes()).hexdigest() for n,t in state.items()}
        groups={}
        for n,t in state.items():groups[logical_group(n)]=groups.get(logical_group(n),0)+t.numel()
        reports.append(dict(variant=cfg['variant'],hashes=hashes,group_parameters=groups,
                            shared_sha256=file_sha256(cfg['shared_init_path'])))
        del agent,state;gc.collect()
    equal=reports[0]['hashes']==reports[1]['hashes']
    result=dict(trainable_initial_state_bitwise_equal=equal,group_counts_equal=reports[0]['group_parameters']==reports[1]['group_parameters'],reports=reports)
    Path(args.output).write_text(json.dumps(result,indent=2))
    assert equal and result['group_counts_equal']
    print('Base/VQA actual initialized trainable state: bitwise equal')
