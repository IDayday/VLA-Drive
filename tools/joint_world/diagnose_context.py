"""Privileged-future imputation diagnostic only; never supplies a deployed planner."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from starVLA.model.modules.joint_world.flow import JointTrajectoryFlow
from tools.joint_world.train_graph import load_samples, current_batch
from tools.structured_world_v1p1.budget import start, record
from tools.structured_world_v1p1.reaudit_metrics import write_csv


@torch.no_grad()
def impute(model, noise, current, known_xy, known_mask):
    # This analysis function is deliberately separate from the current-only
    # deployment sample API. Unknown values never initialize the hidden trajectory.
    clean = model.encode_trajectories(known_xy, current['current_xy'])
    x = torch.where(known_mask[...,None], clean, noise)
    for i in range(10):
        velocity, _ = model(x, x.new_full((len(x),), i/10), **current,
                            known_xy=known_xy, known_mask=known_mask)
        x = torch.where(known_mask[...,None], clean, x+velocity/10)
    return model.decode_trajectories(x, current['current_xy'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','cache','targets','data-root','output','ledger','run-id']:
        p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['identity']['config']
    identity={'arguments':vars(a),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
              'scope':'Privileged other-actor future context used only for imputation analysis; NOT deployment score.',
              'selection':'Ego slot0 and one hash-selected neighbor slot; no target-validity filtering.'}
    start(a.ledger,a.run_id,0,identity)
    try:
        samples,_,_=load_samples(a.cache,a.targets,a.data_root,8)
        if not 2<=len(samples)<=64:raise ValueError('Diagnostic bounded to2..64 explicit scenes')
        model=JointTrajectoryFlow(samples[0]['cache']['context'].shape[-1],**{k:v for k,v in cfg.items()
            if k in ['dim','heads','layers','scale_m','trajectory_mode','agent_scale_m']}).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['model'],strict=True);rows=[]
        for i,sample in enumerate(samples):
            if not record(a.ledger,a.run_id,0):raise RuntimeError('Budget reached')
            current=current_batch([sample]);xy=sample['xy'].cuda();valid=sample['valid'].cuda();token=sample['cache']['token']
            value=int.from_bytes(hashlib.sha256(token.encode()).digest()[:4],'little')
            donor=samples[(i+1)%len(samples)]
            for role,slot in [('ego',0),('neighbor',1+value%(xy.shape[1]-1))]:
                generator=torch.Generator(device='cuda').manual_seed(value)
                noise=torch.randn(xy.shape,device='cuda',generator=generator)
                reference=None
                for mode in ['true_context','blank_context','cross_scene_context']:
                    known=xy if mode!='cross_scene_context' else donor['xy'].cuda()
                    mask=(valid if mode!='cross_scene_context' else donor['valid'].cuda()).clone();mask[:,slot]=False
                    if mode=='blank_context':mask.zero_()
                    pred=impute(model,noise,current,known,mask)
                    if reference is None:reference=pred
                    error=(pred[:,slot]-xy[:,slot]).norm(dim=-1);selected=valid[:,slot]
                    if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite conditional rollout')
                    rows.append({'token':token,'status':'ok','role':role,'slot':slot,'mode':mode,
                                 'valid_points':int(selected.sum()),'error_sum_m':float(error[selected].sum()),
                                 'final_valid':bool(selected[0,-1]),'final_error_m':float(error[0,-1]) if selected[0,-1] else None,
                                 'visible_context_points':int(mask.sum()),
                                 'selected_xy_delta_from_true_max_m':float((pred[:,slot]-reference[:,slot]).abs().max())})
        results=[]
        for role in ['ego','neighbor']:
            for mode in ['true_context','blank_context','cross_scene_context']:
                chosen=[r for r in rows if r['role']==role and r['mode']==mode];points=sum(r['valid_points'] for r in chosen)
                finals=[r['final_error_m'] for r in chosen if r['final_valid']]
                results.append({'role':role,'mode':mode,'scenes':len(chosen),'zero_valid_scenes':sum(r['valid_points']==0 for r in chosen),
                                'valid_points':points,'ADE_m':sum(r['error_sum_m'] for r in chosen)/points if points else None,
                                'FDE_m':sum(finals)/len(finals) if finals else None,
                                'mean_selected_xy_delta_from_true_max_m':sum(r['selected_xy_delta_from_true_max_m'] for r in chosen)/len(chosen)})
        write_csv(out/'scenes.csv',rows)
        report={'identity':identity,'results':results,'failed':0,
                'limitations':['Uses training Hungarian association, not end-to-end geometric detection coverage.',
                    'Blank/mismatched contexts may be out of distribution; sensitivity does not prove causal interaction understanding.',
                    'Privileged future context is never used in the reported original-DiT PDMS.']}
        (out/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        record(a.ledger,a.run_id,0,'complete')
    except BaseException:record(a.ledger,a.run_id,0,'failed');raise


if __name__=='__main__':main()
