"""Raw-image native single-candidate shift5 ODE with canonical independent scoring.

Per-scene atomic results permit explicit resume. Physical invocation costs stay
separate from logical scene count; no teacher, scorer or labels enter inference.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import uuid
import numpy as np
import torch
from starVLA.rl.qwen_sourceflow.contracts import ROOT,QWEN_PATH,require_dispatch,file_sha,fingerprint,source_files,configure_native_numerics
from starVLA.rl.qwen_sourceflow.data import RawSceneData
from starVLA.rl.qwen_sourceflow.loading import load_il
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.flow_grpo.transactions import atomic_json


def json_feedback(value):
    """Lossless JSON representation of official NumPy arrays/scalars."""
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:json_feedback(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_feedback(v) for v in value]
    return value


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cohort',choices=['DEV','NAVTEST','TRAIN'],required=True);p.add_argument('--data',required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--qualification',action='store_true');p.add_argument('--limit',type=int)
    p.add_argument('--parent-selection');p.add_argument('--rl-export',help='Sealed FP32 merged native action export')
    a=p.parse_args();host=require_dispatch();configure_native_numerics();torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.limit and not a.qualification:raise ValueError('No partial population in formal evaluation')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    (root/'scenes').mkdir(exist_ok=True);(root/'invocations').mkdir(exist_ok=True)
    lock=(root/f'rank{a.rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();inv=uuid.uuid4().hex;device=torch.device('cuda',0)
    data=RawSceneData(a.data,cohort=a.cohort);tokens=data.tokens[:a.limit] if a.limit else data.tokens
    if not 0<=a.rank<a.world:raise ValueError('Bad evaluation shard')
    model,parent=load_il(a.checkpoint,device=device,qualification=a.qualification)
    if not a.qualification and a.cohort=='NAVTEST':
        from starVLA.rl.qwen_sourceflow.selection import validate_selection
        validate_selection(a.parent_selection,parent)
    export_identity=None
    if a.rl_export:
        from starVLA.rl.qwen_sourceflow.export import load_export
        export_identity=load_export(a.rl_export,model,parent)
        if not a.qualification and export_identity['qualification']:
            raise ValueError('Qualification RL weights cannot become a formal result')
    model.requires_grad_(False).eval();loaded=time.monotonic()-start
    processor=QwenObservationProcessor(QWEN_PATH)
    service=RewardService(Path.cwd(),workers=1,cache_roots=[
        '/mnt/project/flow-safe-improvement-20261013/metric_cache_v1',
        '/mnt/project/constrained-flow-noise-20261014/metric_cache_v1',
        '/mnt/project/DriveDreamer-Policy-paired/runs/paired_dev_pdms_v1/metric_cache'])
    identity=dict(schema='native_qwen_flow_evaluation_v1',checkpoint=str(Path(a.checkpoint).resolve()),
        parent_model_sha256=parent['files']['model.pt'],rl_export=export_identity,cohort=a.cohort,
        data_sha256=data.sha,token_sha256=fingerprint(tokens),count=len(tokens),full_population=len(tokens)==len(data.tokens),
        metric='canonical_NAVSIM_v1.1',protocol=service.protocol,seed=2101,
        inference='single_candidate_source_shift5_ODE_K10',qualification=a.qualification,
        precision='FP32 storage/action; BF16 Qwen/provider prefill; FP32 ODE latent',source_files=source_files())
    identity['sha256']=fingerprint(identity)
    # One common file under a filesystem lock, independent of worker ordering.
    with (root/'identity.lock').open('a') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX)
        if (root/'identity.json').exists():
            if json.loads((root/'identity.json').read_text())!=identity:raise ValueError('Changed evaluation identity')
        else:atomic_json(root/'identity.json',identity)
        layout=dict(world=a.world,scene_partition='tokens[rank::world]')
        if (root/'layout.json').exists():
            if json.loads((root/'layout.json').read_text())!=layout:raise ValueError('Active shard layout changed; require explicit drained migration')
        else:atomic_json(root/'layout.json',layout)
    record=dict(invocation=inv,rank=a.rank,world=a.world,host=host,start_unix=time.time(),identity=identity['sha256'],
        model_loading_seconds=loaded,forward_seconds=0.,scoring_wait_seconds=0.,physical_inference=0,new=0,reused=0,status='RUNNING')
    invfile=root/'invocations'/f'{inv}.json';atomic_json(invfile,record)
    try:
        for token in tokens[a.rank::a.world]:
            require_dispatch();dest=root/'scenes'/f'{token}.json'
            if dest.exists():
                old=json.loads(dest.read_text())
                if old['identity']!=identity['sha256'] or old['status']!='OK':raise ValueError('Invalid reused scene')
                seal=old.pop('sha256')
                if seal!=fingerprint(old):raise ValueError('Changed reused scene payload')
                record['reused']+=1;continue
            obs=data.observation(token);batch=processor([obs]);state=torch.tensor([obs['proprio']],device=device,dtype=torch.float32)
            noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,token)),dtype=torch.float32).to(device)
            tick=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):condition=model.encode_condition(batch,state)
            raw=model.policy.sample_ode(condition,noise)
            requested=ActionCodec.denorm_odo(raw).cpu().numpy();torch.cuda.synchronize()
            record['forward_seconds']+=time.monotonic()-tick;record['physical_inference']+=1
            atomic_json(invfile,record)
            tick=time.monotonic();scores=service.score((token,data.records[token]['log'],requested))
            record['scoring_wait_seconds']+=time.monotonic()-tick
            if len(scores)!=1 or scores[0]['status']!='OK':raise ValueError('Invalid official feedback; denominator retained')
            row=dict(identity=identity['sha256'],token=token,log=data.records[token]['log'],status='OK',
                invocation=inv,normalized_action=raw[0].cpu().tolist(),requested_trajectory=requested[0].tolist(),feedback=json_feedback(scores[0]))
            row['sha256']=fingerprint(row);atomic_json(dest,row)
            record['new']+=1;record['wall_seconds']=time.monotonic()-start;atomic_json(invfile,record)
        record['status']='COMPLETE'
    except BaseException:
        record.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        record['wall_seconds']=time.monotonic()-start;record['end_unix']=time.time()
        atomic_json(invfile,record);service.close()
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
