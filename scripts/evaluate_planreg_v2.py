"""Explicit student-only selected/oracle PDM entry; never launched automatically."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from navsim.agents.EpisodeDrive.planreg_v2.checkpoint import load_student
from navsim.agents.EpisodeDrive.planreg_v2.data import InputOnlyV2Dataset,v2_collate
from navsim.agents.EpisodeDrive.planreg_v2.agent import file_sha256


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--allow-full-navtest',action='store_true');p.add_argument('--limit',type=int,default=32)
    args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError('Do not overwrite any previous evaluation')
    if args.limit>32 and not args.allow_full_navtest:raise ValueError('Explicit full-navtest authorization flag required')
    agent=load_student(args.checkpoint,'cuda')
    data=InputOnlyV2Dataset(args.manifest,agent.config['vlm_path'],world_model=False)
    loader=DataLoader(data,batch_size=1,collate_fn=v2_collate,num_workers=2)
    rows=[]
    with torch.no_grad():
        for index,(features,targets) in enumerate(loader):
            if index>=args.limit:break
            pred=agent(features)
            scores=agent.compute_metric_targets(targets,pred['proposals'])
            selected=int(pred['selected_indices'][0]);truth=scores[0,:,-1]
            rows.append(dict(token=targets['token'][0],selected_pdms=float(truth[selected]),oracle64=float(truth.max()),
                candidate_mean=float(truth.mean()),candidate_p10=float(torch.quantile(truth,.1)),
                candidate_p25=float(torch.quantile(truth,.25)),selected_components=scores[0,selected].tolist()))
    result=dict(checkpoint_sha256=file_sha256(args.checkpoint),manifest_sha256=file_sha256(args.manifest),scenes=len(rows),rows=rows,
        selected_pdms=float(np.mean([r['selected_pdms'] for r in rows])),oracle64=float(np.mean([r['oracle64'] for r in rows])))
    result['regret']=result['oracle64']-result['selected_pdms']
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if agent._score_pool is not None:agent._score_pool.shutdown()


if __name__=='__main__':main()
