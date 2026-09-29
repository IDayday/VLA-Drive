"""FP32-master deployment timing through the exact formal export loading/prediction path."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
from tools.foresight.checkpoints import checkpoint_identity, load_student, scene_noise
from tools.foresight.deployment_precision import describe
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.ddpolicy_vehicle.campaign import charged_gpu_hours


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('training-run','checkpoint-tag','current-root','output','campaign-root','run-id'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--scenes',type=int,default=8);p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--sampling-seed',type=int,default=42)
    p.add_argument('--campaign-gpu-hours',type=float,default=192);p.add_argument('--max-seconds',type=float,default=1800)
    a=p.parse_args()
    if min(a.scenes,a.repeats,a.campaign_gpu_hours,a.max_seconds)<=0 or a.warmup<0:raise ValueError('Invalid timing bounds')
    out=Path(a.output)
    if out.exists():raise FileExistsError('Do not overwrite deployment timing evidence')
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Lock timing source first')
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'FP32_deployment_profile','real_optimizer_updates':0}) as (meter,_,save):
        if charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:raise RuntimeError('Budget exhausted')
        training,checkpoint=checkpoint_identity(a.training_run,a.checkpoint_tag)
        data=ForesightCurrentDataset(a.current_root)
        if data.identity['split']=='navtest':raise ValueError('Use fixed train/dev inputs for cost measurement')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        started=time.perf_counter();model=load_student(a.training_run,a.checkpoint_tag,training,precision='fp32',strip=True)
        torch.cuda.synchronize();loading=time.perf_counter()-started
        observations=[data[i] for i in range(min(a.scenes,len(data)))];rows=[]
        for iteration in range(a.warmup+a.repeats*len(observations)):
            if time.time()-meter['start_unix']>=a.max_seconds or charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:
                meter['status']='PAUSED';break
            i=(iteration-a.warmup)%len(observations);token=data.index[i]['token']
            noise=scene_noise(token,a.sampling_seed,'cuda');torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():trajectory=model.predict_action([observations[i]],initial_noise=noise)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if trajectory.shape!=(1,8,3) or trajectory.dtype!=torch.float32 or not torch.isfinite(trajectory).all():
                raise FloatingPointError('Invalid FP32 deployment output')
            if iteration>=a.warmup:rows.append({'scene_index':i,'seconds':elapsed})
        times=[r['seconds'] for r in rows]
        result={'schema':'ddp_full_fp32_deployment_v1','checkpoint':checkpoint,'training_source':training['source_sha'],
            'evaluation_source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'protocol':describe(model,'FP32 optimizer master tensors','tools.foresight.checkpoints.load_student(strip=True, precision=fp32)'),
            'current_identity':data.identity,'scenes':len(observations),'warmup':a.warmup,'repeats':a.repeats,
            'complete':len(rows)==a.repeats*len(observations),'batch':1,'sampling_seed':a.sampling_seed,
            'inference_steps':model.action_model.num_inference_timesteps,'RGB_loading_included':False,
            'loading_seconds':loading,'rows':rows,'p50_s':float(np.median(times)) if times else None,
            'p95_s':float(np.percentile(times,95)) if times else None,
            'quality_result':'NOT_EVALUATED; this is a protocol-matched latency measurement, not a PDMS claim'}
        atomic_json(out,result);meter['inference_scenes']=len(rows);save()


if __name__=='__main__':main()
