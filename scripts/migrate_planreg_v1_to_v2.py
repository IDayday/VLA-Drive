import argparse
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent,file_sha256
from navsim.agents.EpisodeDrive.planreg_v2.checkpoint import warm_start_v1
from navsim.agents.EpisodeDrive.planreg_v2.runtime import load_config

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError('New warm-start artifact required')
    cfg=load_config(args.config)
    agent=PlanRegV2Agent(cfg,'cpu')
    source=torch.load(args.input,map_location='cpu',weights_only=False)
    report=warm_start_v1(agent,source['state_dict'])
    report['source_sha256']=file_sha256(args.input)
    torch.save(dict(schema='planreg_v2_declared_warm_start_v1',model=agent.state_dict(),config=cfg,audit=report),args.output)
    Path(args.output+'.audit.json').write_text(json.dumps(report,indent=2))
