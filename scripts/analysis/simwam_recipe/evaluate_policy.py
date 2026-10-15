"""Original raw-observation DDP, native and shift5 single-ODE readouts.

Both trajectories are independently scored against the fixed official PDM
reference. There is no candidate choice, LoRA wrapper, teacher, or cache in
the deployment actor. Fixed seed2101; Navtest requires a prior explicit lock.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from omegaconf import OmegaConf
import torch
from starVLA.rl.simwam_recipe.contracts import *
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json

STOP=False
def stop(*_):
    global STOP
    STOP=True


def shifted_ode(head,condition,root):
    scheduler=WanContinuousFlowMatchScheduler(1000,shift=5)
    times,deltas=scheduler.build_inference_schedule(10,root.device,torch.float32)
    # Same source RL numerical adaptation (FP32 latent), with zero innovations.
    action=root.float();buckets=[]
    with torch.autocast('cuda',dtype=torch.float32):
        for t,delta in zip(times,deltas):
            bucket=((1-t/1000)*head.num_timestep_buckets).long().reshape(1).expand(root.shape[0])
            buckets.append(int(bucket[0]));velocity=head.predict_velocity(action,bucket,condition)
            action=scheduler.step(-velocity,delta,action)
    return action,buckets


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--export',required=True);p.add_argument('--export-sha256',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cohort',choices=['DEV','TRAIN_DIAGNOSTIC','NAVTEST'],default='DEV')
    p.add_argument('--shard',type=int);p.add_argument('--shards',type=int,default=8)
    p.add_argument('--max-scenes',type=int);p.add_argument('--final-lock')
    a=p.parse_args();host=require_dispatch();configure_numerics()
    rank=a.shard if a.shard is not None else int(os.environ.get('RANK',0));torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if not 0<=rank<a.shards:raise ValueError('Invalid shard')
    for s in (signal.SIGUSR1,signal.SIGTERM):signal.signal(s,stop)
    export=Path(a.export);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if file_sha(export/'pytorch_model.pt')!=a.export_sha256:raise ValueError('Export hash changed')
    if a.export_sha256!=SFT_SHA256 and not (export/'COMPLETE').exists():raise ValueError('Unsealed export')
    pop=json.loads(Path(POPULATION).read_text());selection=None
    if a.cohort=='DEV':tokens=pop['dev_tokens']
    elif a.cohort=='TRAIN_DIAGNOSTIC':
        previous=json.loads(Path('/mnt/project/flow-safe-improvement-20261013/evaluations/original_export/SFT_TRAIN_D5_samples/manifest.json').read_text())
        tokens=previous['explicit_tokens']
        if len(tokens)!=512 or not set(tokens)<=set(pop['train_tokens']):raise ValueError('TRAIN diagnostic changed')
    else:
        if not a.final_lock:raise ValueError('NAVTEST remains closed until endpoint/selection lock')
        selection=json.loads(Path(a.final_lock).read_text());sha=selection.pop('sha256')
        if fingerprint(selection)!=sha or selection['status']!='LOCKED_BEFORE_FINAL_NAVTEST':raise ValueError('Invalid final lock')
        if a.export_sha256 not in selection['weight_sha256'] or selection['population_identity']!=pop['identity_sha256']:
            raise ValueError('Model/population not locked')
        if selection['sampler_sha256']!=file_sha(__file__) or selection['inference_seeds']!=[2101]:raise ValueError('Evaluation protocol changed after lock')
        tokens=pop['navtest_tokens']
    cfg=OmegaConf.load(export/'config.yaml')
    os.environ['BASE_VLM']='/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-WorldAction'
    os.environ['VLM_ATTN_IMPLEMENTATION']=cfg.framework.qwenvl.attn_implementation
    from infer import VLAAgent
    agent=VLAAgent(str(export),device='cuda',qwen_forward_mode='optimized');model=agent.model.eval().requires_grad_(False)
    profile=dict(qwen=str(next(model.qwen_vl_interface.parameters()).dtype),history=str(next(model.action_input_model.parameters()).dtype),
                 action=str(next(model.action_model.parameters()).dtype),attention=cfg.framework.qwenvl.attn_implementation)
    if (profile['qwen'],profile['history'],profile['action'])!=('torch.bfloat16','torch.float32','torch.float32'):raise ValueError('Original deployment dtype profile changed')
    if model.action_model.num_inference_timesteps!=10:raise ValueError('Changed native ODE')
    evaluation_caches=CACHE_ROOTS+['/mnt/project/DriveDreamer-Policy-paired/runs/paired_dev_pdms_v1/metric_cache']
    if selection:evaluation_caches=[selection['v1_cache']]+evaluation_caches
    service=RewardService(Path.cwd(),workers=1,cache_roots=evaluation_caches)
    identity=dict(schema='simwam_dual_grid_single_ODE_v1',weight_sha256=a.export_sha256,config_sha256=file_sha(export/'config.yaml'),
                  population_identity=pop['identity_sha256'],tokens_sha256=fingerprint(tokens),scenes=len(tokens),cohort=a.cohort,
                  seed=2101,modes=['E_NATIVE','E_SHIFT5'],profile=profile,feature_cache=False,actor='unchanged VLAAgent raw observation',
                  scoring=service.protocol['sha256'],source_sha256=file_sha(__file__),shards=a.shards,final_lock=selection,
                  qualification_only=a.max_scenes is not None)
    identity['sha256']=fingerprint(identity)
    with (out/'manifest.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=identity:raise ValueError('Evaluation identity changed')
        atomic_json(out/'manifest.json',identity)
    guard=(out/f'rank{rank}.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data_cfg=copy.deepcopy(agent.model_config)
    data_cfg.datasets.vla_data.data_root=selection['data_root'] if selection else '/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/dataset'
    data_cfg.datasets.vla_data.datalist_path=selection['navtest_datalist'] if selection else '/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/navtrain_tokens.json'
    data_cfg.enable_image_aug=0;data_cfg.w_depth=0
    dataset=KeyedDataset(data_cfg,split='test' if selection else 'train')
    assigned=tokens[rank::a.shards]
    if a.max_scenes:assigned=assigned[:a.max_scenes]
    status=dict(status='RUNNING',host=host,pid=os.getpid(),new=0,reused=0,scene_encoder_calls=0,new_candidates=0,
                velocity_forward_calls=0,generation_seconds=0.,scoring_seconds=0.,direct_original_check=False)
    begin=time.monotonic()
    try:
        for token in assigned:
            if STOP:break
            require_dispatch();path=out/'scenes'/token[:2]/(token+'.pt')
            if path.with_suffix('.json').exists():
                seal=json.loads(path.with_suffix('.json').read_text())
                if seal['identity_sha256']!=identity['sha256'] or seal['payload_sha256']!=file_sha(path):raise ValueError('Changed evaluation record')
                status['reused']+=1;continue
            started=time.monotonic();key=seed_for('original_ode_initial',2101,token)
            obs=prepare_policy_observation([dataset[(token,key)]])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                condition,raw_dtype=model.encode_policy_condition(obs,return_raw_dtype=True,feature_output='normalized')
                root=torch.randn((1,8,4),device='cuda',dtype=raw_dtype,generator=torch.Generator(device='cuda').manual_seed(key))
                with torch.autocast('cuda',dtype=torch.float32):native=model.action_model._euler_sample(root,condition)
                shifted,buckets=shifted_ode(model.action_model,condition,root)
                if not status['direct_original_check']:
                    direct=model.predict_action_infer_1d(obs.examples(),initial_noise=root)['normalized_actions']
                    if not np.array_equal(direct,native.float().cpu().numpy()):raise ValueError('Native evaluation differs from original deployment')
                    status['direct_original_check']=True
            actions=torch.cat([native,shifted]).float().cpu()
            if not torch.isfinite(actions).all():raise ValueError('Nonfinite policy output; retain scene in denominator and abort')
            requested=decode_request(actions.numpy(),act_norm=int(data_cfg.datasets.vla_data.act_norm))
            torch.cuda.synchronize();status['generation_seconds']+=time.monotonic()-started
            started=time.monotonic();records=service.score((token,pop['token_logs'][token],requested));status['scoring_seconds']+=time.monotonic()-started
            payload=dict(token=token,log=pop['token_logs'][token],identity_sha256=identity['sha256'],initial_noise=root.cpu(),
                         noise_key=key,normalized_actions=actions,records=dict(zip(identity['modes'],records)),shift5_buckets=buckets)
            atomic_torch(path,payload)
            atomic_json(path.with_suffix('.json'),dict(token=token,log=payload['log'],identity_sha256=identity['sha256'],payload_sha256=file_sha(path),
                        metrics={k:v['metrics'] for k,v in payload['records'].items()}))
            status['new']+=1;status['new_candidates']+=2;status['scene_encoder_calls']+=1;status['velocity_forward_calls']+=20
            status['elapsed']=time.monotonic()-begin
            if status['new']%8==0:atomic_json(out/f'rank{rank}.json',status)
        status.update(status='STOPPED_AT_SCENE_BOUNDARY' if STOP else ('QUALIFICATION_COMPLETE' if a.max_scenes else 'ASSIGNED_COMPLETE'),elapsed=time.monotonic()-begin,
                      peak_GPU_bytes=torch.cuda.max_memory_allocated())
        atomic_json(out/f'rank{rank}.json',status)
    finally:service.close()


if __name__=='__main__':main()
