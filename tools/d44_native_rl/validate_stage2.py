"""Real saved TRAIN feedback + original GT endpoint credit, no resampling."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.safety_credit import raw_credit,normalize
from starVLA.rl.d44_native_rl.two_stage_reward import endpoint_quality
from starVLA.rl.qwen_world_compare.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--qualification',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.qualification);identity=json.loads((root/'identity.json').read_text());cfg=identity['recipe']
    if cfg['arm']!='D1'or identity['parent_model_sha256']!='9715b9c0cfbef7a7d3d94fb83e7b0e2f0838d21cf06d3277deef6733cccbc50e':raise ValueError('Requires actual D44 D1 feedback')
    stream=DataStream(cfg,cfg['global_scenes']);files=list((root/'behavior').glob('*/collection000001/rank*.pt'))
    if len(files)!=identity['world']:raise ValueError('Missing complete fresh multi-rank collection')
    rows=[r for f in files for r in torch.load(f,map_location='cpu',weights_only=False)['scenes']if not r['padding']]
    tokens=[r['token']for r in rows]
    if len(tokens)!=len(set(tokens))or not set(tokens)<=set(stream.tokens):raise ValueError('Duplicate or outside-TRAIN scene')
    factors=torch.stack([r['factors']for r in rows]);quality=[];confidence=[]
    for r in rows:
        requested=torch.as_tensor(np.stack([x['requested_trajectory']for x in r['records']]))
        q,w=endpoint_quality(stream.data.target(r['token']),requested);quality.append(q);confidence.append(w)
    q=torch.stack(quality);w=torch.stack(confidence)[:,None]
    raw,feasible=raw_credit(factors,endpoint=q,endpoint_confidence=w,stage2=True)
    advantage,std=normalize(raw,all_raw=raw)
    if not torch.isfinite(advantage).all()or float(raw.sum(1).abs().max())>1e-10:raise ValueError('Invalid stage-II credit')
    unsafe_positive=int(((raw>1e-10)&~feasible&feasible.any(1,keepdim=True)).sum())
    if unsafe_positive:raise ValueError('Endpoint credit granted positive unsafe advantage despite feasible alternatives')
    result=dict(status='PASS',schema='D44_stageII_actual_TRAIN_label_credit_v1',qualification=str(root),TRAIN_scenes=len(rows),candidates=len(rows)*cfg['group'],
        endpoint_quality_mean=float(q.mean()),downweighted_short_GT_scenes=int((w<1).sum()),global_raw_advantage_std=std,
        raw_group_sum_error=float(raw.sum(1).abs().max()),unsafe_positive_when_feasible_exists=unsafe_positive,
        all_infeasible_groups=int((~feasible.any(1)).sum()),all_feasible_groups=int(feasible.all(1).sum()),
        source_files={str(Path(module.__file__).relative_to(Path.cwd())):file_sha(module.__file__)for module in [__import__('starVLA.rl.d44_native_rl.safety_credit',fromlist=['']),__import__('starVLA.rl.d44_native_rl.two_stage_reward',fromlist=[''])]},
        new_model_rollouts=0,new_scorer_calls=0,optimizer_updates=0,old_behavior_unchanged=True,
        limits='Real TRAIN label/advantage bridge only; not stage-II model training or deployment improvement.')
    output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve prior qualification evidence')
    atomic_json(output,result)


if __name__=='__main__':main()
