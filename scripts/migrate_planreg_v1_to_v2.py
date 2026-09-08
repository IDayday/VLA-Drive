import argparse
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent,file_sha256
from navsim.agents.EpisodeDrive.planreg_v2.checkpoint import warm_start_v1
from navsim.agents.EpisodeDrive.planreg_v2.runtime import load_config
from navsim.agents.EpisodeDrive.planreg_v2.runtime import source_fingerprint

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
    # Replay actual serialized V1 generator/head/ego weights. This checks the
    # declared compatible transfer, NOT whole-V2 vs whole-V1 policy equivalence.
    from navsim.agents.EpisodeDrive.transformer_decoder import TransformerDecoder
    from navsim.agents.EpisodeDrive.layers.utils.mlp import MLP
    from navsim.agents.EpisodeDrive.planreg_v2.normalizers import wrap_angle
    state={n.removeprefix('agent.'):v for n,v in source['state_dict'].items()}
    decoder=TransformerDecoder(.1,.2,agent.action_head.config).eval()
    decoder.load_state_dict({n.removeprefix('action_head.trajectory_decoder.'):v for n,v in state.items() if n.startswith('action_head.trajectory_decoder.')},strict=True)
    head=MLP(256,1024,24).eval()
    head.load_state_dict({n.removeprefix('action_head.traj_head.4.'):v for n,v in state.items() if n.startswith('action_head.traj_head.4.')},strict=True)
    hist=torch.nn.Linear(11,256)
    hist.load_state_dict({n.removeprefix('action_head.hist_encoding.'):v for n,v in state.items() if n.startswith('action_head.hist_encoding.')},strict=True)
    torch.manual_seed(409)
    queries=torch.randn(2,64,256);memory=torch.randn(2,64,256)
    status=torch.tensor([[1.,0.,0.,0.,8.,-2.,1.,-.5],[0.,1.,0.,0.,3.,1.,-.2,.8]])
    agent.action_head.eval()
    with torch.no_grad():
        old_hidden=decoder(queries,memory);new_hidden=agent.action_head.attention(queries,memory)
        old_output=head(old_hidden).reshape(4,2,64,8,3)
        new_output=agent.action_head.normalizer.inverse(agent.action_head.trajectory_head(new_hidden).reshape(4,2,64,8,3))
        difference=new_output-old_output;difference[...,2]=wrap_angle(difference[...,2])
        old_ego=hist(torch.cat((torch.zeros(2,3),status),-1))
        new_ego=agent.action_head.hist_encoding(torch.cat((torch.zeros(2,3),agent.action_head.ego_normalizer(status)),-1))
        torch.testing.assert_close(old_hidden,new_hidden,atol=0,rtol=0)
        torch.testing.assert_close(difference,torch.zeros_like(difference),atol=3e-5,rtol=0)
        torch.testing.assert_close(new_ego,old_ego,atol=2e-6,rtol=1e-5)
    report.update(real_checkpoint_replay=dict(status='PASS',source=args.input,decoder_max_abs_diff=float((old_hidden-new_hidden).abs().max()),
        physical_head_max_abs_diff=float(difference.abs().max()),ego_max_abs_diff=float((new_ego-old_ego).abs().max())),
        source_fingerprint=source_fingerprint())
    torch.save(dict(schema='planreg_v2_declared_warm_start_v2',model=agent.state_dict(),config=cfg,audit=report),args.output)
    Path(args.output+'.audit.json').write_text(json.dumps(report,indent=2))
