"""Current-only ego export with removed auxiliary heads and strict final Navtest lock."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
from starVLA.model.modules.vehicle_joint.initialization import file_sha256,identity_hash
from .checkpoints import checkpoint_identity,scene_noise,load_student


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('training-run','checkpoint-tag','current-root','output','campaign-root','run-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--sampling-seed',type=int,required=True);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--final-lock')
    p.add_argument('--campaign-gpu-hours',type=float,default=6000)
    a=p.parse_args()
    if not 0<=a.rank<a.world_size or a.limit<0 or min(a.max_seconds,a.campaign_gpu_hours)<=0:raise ValueError('Invalid shard/budget')
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'foresight_current_camera_export','real_optimizer_updates':0}) as (record,_,save):
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        from tools.ddpolicy_vehicle.campaign import charged_gpu_hours
        if charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:raise RuntimeError('Campaign budget exhausted before export')
        if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Lock evaluation source first')
        data=ForesightCurrentDataset(a.current_root);training,checkpoint=checkpoint_identity(a.training_run,a.checkpoint_tag)
        protocol={'precision':'FP32','tf32':False,'scorer':None,'candidates_per_scene':1,'sampling_seed':a.sampling_seed,
                  'solver':'original Euler','steps':training['config']['framework']['action_model']['num_inference_timesteps'],
                  'current_views':['CAM_F0','CAM_L0','CAM_R0'],'history_images':0,'future_conditioning':False,
                  'W_retained':training['config']['foresight']['num_queries'],'auxiliary_heads_removed':True,
                  'trajectory':'ego only, original DDP postprocessing; 8x0.5s; xy/yaw in ego(t0)',
                  'noise':'SHA256 foresight-action-v1:seed:token, CPU torch.randn(1,8,4)'}
        if data.identity['split']=='navtest':
            if not a.final_lock or a.limit or training['scope']!='formal':raise ValueError('Navtest requires a full locked formal model')
            lock=json.loads(Path(a.final_lock).read_text())
            from .lock_navtest import validate_lock
            validate_lock(lock,checkpoint,source,data.identity,len(data),a.sampling_seed,protocol['steps'])
        identity={'checkpoint':checkpoint,'current_identity':data.identity,'evaluation_source':source,'protocol':protocol,
                  'world_size':a.world_size,'limit':a.limit}
        out=Path(a.output);(out/'predictions').mkdir(parents=True,exist_ok=True)
        if (out/'identity.json').exists():
            if json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Export cache identity mismatch')
        else:atomic_json(out/'identity.json',identity)
        signature=identity_hash(identity);record.update(checkpoint=checkpoint['sha256'],protocol=protocol);save()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        model=load_student(a.training_run,a.checkpoint_tag,training)
        ids=list(range(min(a.limit or len(data),len(data))))[a.rank::a.world_size];done=failed=0
        for index in ids:
            if time.time()-record['start_unix']>=a.max_seconds or charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:
                record['status']='PAUSED';break
            scene=data.index[index];token=scene['token'];dest=out/'predictions'/(token+'.npz');meta=dest.with_suffix('.json')
            if meta.exists():
                row=json.loads(meta.read_text())
                if row['identity_sha256']!=signature:raise ValueError('Scene export identity changed')
                if row['status']=='ok' and file_sha256(dest)!=row['proposal_sha256']:raise ValueError('Scene artifact changed')
                failed+=row['status']!='ok';done+=1;continue
            row={'token':token,'log':scene['log'],'identity_sha256':signature,'status':'ok'}
            try:
                observation=data[index]  # only current/*.json and its image files; no ego/ or target caches
                noise=scene_noise(token,a.sampling_seed,'cuda');torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode():trajectory=model.predict_action([observation],initial_noise=noise)[0]
                torch.cuda.synchronize();row['inference_seconds']=time.perf_counter()-start
                if trajectory.shape!=(8,3) or not torch.isfinite(trajectory).all():raise FloatingPointError('Invalid ego trajectory')
                temp=dest.with_suffix(f'.{os.getpid()}.tmp')
                with temp.open('wb') as stream:np.savez_compressed(stream,trajectory=trajectory.cpu().numpy())
                temp.replace(dest);row['proposal_sha256']=file_sha256(dest)
            except Exception as error:row.update(status='failed',error=repr(error));failed+=1
            atomic_json(meta,row);done+=1;record.update(inference_scenes=done,failed=failed);save()
        atomic_json(out/f'shard_{a.rank}.json',{'status':'complete' if done==len(ids) else 'paused','requested':len(ids),
                    'completed':done,'failed':failed,'identity_sha256':signature})
        record.update(inference_scenes=done,failed=failed)
        if failed:raise RuntimeError('Failed scene rows preserved; export is incomplete')


if __name__=='__main__':main()
