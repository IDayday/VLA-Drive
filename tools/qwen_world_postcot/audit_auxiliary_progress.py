"""Frozen matched IL auxiliary readouts on fixed TRAIN/DEV scenes; no updates."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.model.modules.qwen_world_postcot.model import QwenWorldPostCoT
from starVLA.model.modules.qwen_world_postcot.teacher_targets import TeacherCacheReader
from starVLA.model.modules.qwen_world_postcot.readouts import sample_future_queries,gather_future_targets
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.qwen_world_postcot.contracts import require_dispatch,file_sha,fingerprint
from starVLA.rl.qwen_world_postcot.loading import inspect_checkpoint
from starVLA.rl.qwen_world_postcot.data import WorldSceneData


def feature_metrics(pred,target,valid,baseline):
    p,y=pred.float()[valid],target.float()[valid]
    if not len(p):return None
    return dict(MSE=float((p-y).square().mean()),cosine=float(F.cosine_similarity(p,y,dim=-1).mean()),
        TRAIN_mean_baseline_MSE=float((baseline.float()[valid]-y).square().mean()),valid_patches=len(p))


def object_metrics(pred,truth):
    boxes=pred['boxes'][0].float().cpu();classes=pred['class_logits'][0].float().softmax(-1).cpu()
    exists=pred['existence_logits'][0].sigmoid().float().cpu();y=torch.tensor(truth['boxes']).float()
    labels=torch.tensor(truth['classes'],dtype=torch.long)
    if not len(y):return dict(visible_objects=0,precision=None,coverage_complete=truth['coverage_valid'])
    scale=torch.tensor([50.,50.,5.,10.,5.,5.,1.,1.])
    cost=torch.cdist(boxes/scale,y/scale,p=1)-classes[:,labels]
    q,g=linear_sum_assignment(cost.numpy());b,y=boxes[q],y[g];right=classes[q].argmax(-1)==labels[g]
    distance=(b[:,:2]-y[:,:2]).norm(dim=-1)
    heading=(torch.atan2(b[:,6],b[:,7])-torch.atan2(y[:,6],y[:,7])+torch.pi)%(2*torch.pi)-torch.pi
    return dict(visible_objects=len(y),matched_class_correct=int(right.sum()),
        matched_xy_distance_m_sum=float(distance.sum()),matched_heading_abs_deg_sum=float(heading.abs().sum()*180/torch.pi),
        visible_positive_recall2m_count=int(((distance<2)&right&(exists[q]>=.5)).sum()),
        precision=None,coverage_complete=truth['coverage_valid'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    require_dispatch();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');configure_numerics();torch.set_num_threads(4)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);device=torch.device('cuda:0');start=time.monotonic()
    base=Path('/mnt/project/qwen-world-postcot-20261025')
    data={role:WorldSceneData(base/'data'/role,role=role) for role in ('TRAIN','DEV')}
    readers={role:TeacherCacheReader(base/'teachers'/(role+'_resolved_v3'),expected_cohort=role,max_open_shards=1) for role in data}
    selected={role:[ds.tokens[i] for i in torch.randperm(len(ds.tokens),generator=torch.Generator().manual_seed(420028)).tolist()[:64]] for role,ds in data.items()}
    for role,reader in readers.items():
        if (reader.identity['records_sha256']!=data[role].manifest['records_sha256']
                or any(reader.identity[k]!=readers['TRAIN'].identity[k] for k in ('dino','wa'))):
            raise ValueError('Task teacher/data contracts differ')
    # A position-specific constant feature baseline is fitted ONLY on the64
    # fixed TRAIN scenes, then frozen for both role readouts and all checkpoints.
    sums={};counts={}
    for token in selected['TRAIN']:
        row=readers['TRAIN'].get(token,source_record_hash=data['TRAIN'].records[token]['source_record_hash'])
        for name in ('dino','future'):
            v=row[name].float();m=row[name+'_valid'];x=torch.where(m[...,None],v,0.)
            sums[name]=x.double() if name not in sums else sums[name]+x.double()
            counts[name]=m.double() if name not in counts else counts[name]+m.double()
    means={k:(s/counts[k].clamp_min(1)[...,None]).float() for k,s in sums.items()}
    processor=QwenObservationProcessor('/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct')
    plan=dict(schema='world_aux_progress_fixed_readouts_v1',split_tokens=selected,checkpoint_updates=[20000,40000,60000],
        source_sha256=file_sha(__file__),training_parameters_updated=False,new_policy_rollouts=0,new_scorer_calls=0,
        note='DEV auxiliary quality diagnosis only, not planning performance/method selection. Object precision undefined: unmatched queries lack complete negatives.')
    plan['sha256']=fingerprint(plan);(out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    for update in plan['checkpoint_updates']:
        checkpoint=base/'runs/IL_full_seed42/checkpoints'/f'update{update:06d}'
        info=inspect_checkpoint(checkpoint,verify_files='model');identity=info['identity']
        raw=torch.load(checkpoint/info['seal']['model_file'],map_location='cpu',weights_only=False)
        model=QwenWorldPostCoT(None,config=copy.deepcopy(identity['model_config']),checkpoint_config=copy.deepcopy(identity['qwen_config']),train_heads=True)
        model.load_state_dict(raw['module'],strict=True);del raw;model.eval().requires_grad_(False).to(device)
        rows=[]
        for role in ('TRAIN','DEV'):
            for i,token in enumerate(selected[role]):
                obs=data[role].observation(token);target=readers[role].get(token,source_record_hash=data[role].records[token]['source_record_hash'])
                inputs=processor([obs]);ego=torch.tensor([obs['proprio']],device=device)
                seed=int(token[:8],16)^420028;queries=sample_future_queries(1,128,generator=torch.Generator().manual_seed(seed))
                future,fv=gather_future_targets(target['future'][None],target['future_valid'][None],queries)
                fmean,_=gather_future_targets(means['future'][None],torch.ones_like(target['future_valid'][None]),queries)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    memory=model.encode_condition(inputs,ego);dino=model.dino_readout(memory);f=model.future_readout(memory,queries.to(device));obj=model.object_readout(memory)
                row=dict(role=role,token=token,update=update,
                    dino=feature_metrics(dino.cpu(),target['dino'][None],target['dino_valid'][None],means['dino'][None]),
                    future=feature_metrics(f.cpu(),future,fv,fmean),objects=object_metrics(obj,data[role].records[token]['objects']))
                rows.append(row)
        torch.cuda.synchronize();(out/f'update{update:06d}.json').write_text(json.dumps(dict(updates=update,model_sha256=info['model_sha256'],rows=rows),indent=2)+'\n')
        print(json.dumps(dict(update=update,scenes=len(rows),elapsed=time.monotonic()-start)),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',checkpoints=3,scene_forwards=384,new_scorer_calls=0,optimizer_updates=0,wall_seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated()))+'\n')
    for reader in readers.values():reader.close()


if __name__=='__main__':main()
