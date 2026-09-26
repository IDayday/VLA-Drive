"""Current-input inference export and separate structured-target diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from runtime import load_baseline,load_dataset,load_world_batch,seed_all
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy
from starVLA.model.modules.structured_world.matching import match_current


def load_delta(policy,path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['world_config'] != policy.world_config:raise ValueError('Evaluation configuration mismatch')
    delta=saved['delta'];state=policy.state_dict()
    if sorted(delta)!=saved['delta_keys']:raise ValueError('Checkpoint key manifest mismatch')
    required={k for k in state if not k.startswith('baseline.')}
    if not required<=set(delta) or not set(delta)<=set(state):raise ValueError('Missing world or unexpected checkpoint keys')
    with torch.no_grad():
        for k,value in delta.items():
            if value.shape!=state[k].shape:raise ValueError(f'Checkpoint shape mismatch: {k}')
            state[k].copy_(value.to(state[k].device))
    return saved


def diagnostics(pred,target):
    rows,cols=match_current(pred,target)
    eligible=target.current_supervision_mask.bool()
    gt_count=int(eligible.sum())+target.overflow
    exists=pred['logits'].argmax(-1)!=pred['logits'].shape[-1]-1
    distance=torch.linalg.vector_norm(pred['boxes'][rows,:2]-target.current_boxes[cols,:2],dim=-1)
    detected=exists[rows] & (distance<2.)
    good_rows,good_cols=rows[detected],cols[detected]
    n=int(detected.sum())
    result={'gt_targets':gt_count,'matched_detection_targets':n,'predicted_objects':int(exists.sum()),
            'false_positives':int(exists.sum())-n,'matched_centre_error_sum':float(distance[detected].sum()),
            'end_to_end_motion_targets':0,'valid_motion_points':0,'ade_sum':0.,'fde_sum':0.,'fde_targets':0,'yaw_error_sum':0.}
    if n:
        pb=pred['boxes'][good_rows];tb=target.current_boxes[good_cols]
        yaw=torch.atan2(pb[:,6],pb[:,7])-torch.atan2(tb[:,6],tb[:,7])
        result['yaw_error_sum']=float(torch.atan2(yaw.sin(),yaw.cos()).abs().sum())
        mask=target.future_valid_mask[good_cols]
        errors=torch.linalg.vector_norm(pred['future_xy'][good_rows]-target.future_xy_in_ego_t0[good_cols],dim=-1)
        result['ade_sum']=float(errors[mask].sum());result['valid_motion_points']=int(mask.sum())
        result['end_to_end_motion_targets']=int(mask.any(-1).sum())
        # FDE uses the requested terminal horizon, not an earlier convenient visible point.
        result['fde_sum']=float(errors[:,-1][mask[:,-1]].sum());result['fde_targets']=int(mask[:,-1].sum())
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ['checkpoint','vlm','data-root','manifest','target-cache','config','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--delta');p.add_argument('--limit',type=int);p.add_argument('--seed',type=int,default=20260926)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();seed_all(42)
    agent=load_baseline(a.checkpoint,a.vlm);cfg=yaml.safe_load(Path(a.config).read_text())
    policy=StructuredWorldPolicy(agent.model,cfg).cuda().eval();policy.requires_grad_(False)
    if a.delta:load_delta(policy,a.delta)
    ds=load_dataset(agent,a.manifest,a.data_root,a.limit)
    out=Path(a.output);(out/'predictions').mkdir(parents=True,exist_ok=True)
    from infer import deal_action_1225
    from omegaconf import OmegaConf
    act_norm=OmegaConf.select(agent.model_config,'datasets.vla_data.act_norm',default=0)
    records=[]
    for i in range(a.shard,len(ds),a.shards):
        raw=ds[i];token=raw['token'];record={'token':token,'status':'ok'}
        try:
            # WorldTargets and ego actions are not present in the inference example.
            example={k:raw[k] for k in ['image','lang','state','token']}
            inputs,_=load_world_batch([example],a.target_cache,load_targets=False)
            seed=int.from_bytes(hashlib.sha256(f'{a.seed}:{token}'.encode()).digest()[:4],'little');seed_all(seed)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():result=policy.predict_action([example],inputs)
            torch.cuda.synchronize();record['latency_seconds']=time.perf_counter()-start
            actions=result['normalized_actions'];trajectory=deal_action_1225(actions,act_norm=act_norm)[0]
            if not np.isfinite(trajectory).all():raise ValueError('Nonfinite trajectory')
            arrays={'trajectory':trajectory,'normalized_actions':actions[0]}
            if result.get('world_prediction') is not None:
                pred={k:v[0].float() for k,v in result['world_prediction'].items()}
                arrays.update({k:v.cpu().numpy() for k,v in pred.items()})
                # Supervision is loaded only after deployment inference has completed.
                from starVLA.model.modules.structured_world.contracts import WorldTargets
                values=torch.load(Path(a.target_cache)/'targets'/f'{token}.pt',map_location='cuda',weights_only=True)
                record.update(diagnostics(pred,WorldTargets(**values)))
            np.savez(out/'predictions'/f'{token}.npz',**arrays)
        except Exception as error:
            if not any(r['status']=='failed' for r in records):
                import traceback
                traceback.print_exc()
            record.update(status='failed',error=repr(error))
        records.append(record)
        with (out/f'progress_{a.shard}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    keys=sorted(set(k for row in records for k in row))
    with (out/f'scenes_{a.shard}.csv').open('w') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(records)
    (out/f'manifest_{a.shard}.json').write_text(json.dumps({'arguments':vars(a),'samples':len(records),'failed':sum(r['status']!='ok' for r in records),'precision':'original Qwen BF16 / action FP32; matched-protocol pilot','candidate_count':1,'scorer':None},indent=2))

if __name__=='__main__':main()
