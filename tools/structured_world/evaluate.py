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
from starVLA.model.modules.structured_world.metrics import diagnostics


def load_delta(policy,path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['world_config'] != policy.world_config:raise ValueError('Evaluation configuration mismatch')
    delta=saved['delta'];state=policy.state_dict()
    if sorted(delta)!=saved['delta_keys']:raise ValueError('Checkpoint key manifest mismatch')
    required={k for k in state if not k.startswith('baseline.')}
    if policy.world_config.get('train_action',False):
        required.update('baseline.action_model.'+n for n,_ in policy.baseline.action_model.named_parameters())
    if policy.world_config.get('vision_trainable',False):
        required.update('baseline.qwen_vl_interface.model.model.visual.'+n for n,_ in policy.baseline.qwen_vl_interface.model.model.visual.named_parameters())
    if set(delta)!=required or not set(delta)<=set(state):raise ValueError('Missing critical or unexpected checkpoint keys')
    with torch.no_grad():
        for k,value in delta.items():
            if value.shape!=state[k].shape:raise ValueError(f'Checkpoint shape mismatch: {k}')
            state[k].copy_(value.to(state[k].device))
    return saved



def main():
    p=argparse.ArgumentParser()
    for name in ['checkpoint','vlm','data-root','manifest','target-cache','config','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--split',choices=['train','test','mini'],default='train');p.add_argument('--skip-world-diagnostics',action='store_true');p.add_argument('--delta');p.add_argument('--limit',type=int);p.add_argument('--seed',type=int,default=20260926)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();seed_all(42)
    agent=load_baseline(a.checkpoint,a.vlm);cfg=yaml.safe_load(Path(a.config).read_text())
    policy=StructuredWorldPolicy(agent.model,cfg).cuda().eval();policy.requires_grad_(False)
    if a.delta:load_delta(policy,a.delta)
    ds=load_dataset(agent,a.manifest,a.data_root,a.limit,split=a.split)
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
                if not a.skip_world_diagnostics:
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
