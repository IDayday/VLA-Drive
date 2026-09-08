import argparse
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent,file_sha256
from navsim.agents.EpisodeDrive.planreg_v2.runtime import load_config

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    if Path(a.output).exists(): raise FileExistsError('Do not overwrite any shared initialization')
    cfg = load_config(a.config)
    cfg['shared_init_path'] = None
    torch.manual_seed(a.seed)
    agent = PlanRegV2Agent(cfg,device='cpu')
    state = agent.trainable_state()
    torch.save(dict(schema='planreg_v2_shared_trainable_v1',trainable_state=state,seed=a.seed,config=cfg),a.output)
    print(json.dumps(dict(sha256=file_sha256(a.output),tensors=len(state),parameters=sum(v.numel() for v in state.values()))))
