"""Timestamp-aligned deterministic FLUX targets; missing frames retain ego scenes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from tools.ddpolicy_vehicle.prepare_data import load_trusted,atomic_json,CAMERAS,camera_path
from tools.ddpolicy_vehicle.run_meter import metered_run
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256
from .prepare_vehicle_trajectories import timed_frames
from .flux_targets import FluxTargetEncoder


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('split-manifest','raw-log-root','sensor-root','vae-root','vae-identity','output','campaign-root','run-id'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--fallback-sensor-root',action='append',default=[])
    p.add_argument('--split',choices=('train','dev'),required=True);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--short-side',type=int,default=256)
    p.add_argument('--max-seconds',type=float,default=14400);p.add_argument('--resume',action='store_true')
    p.add_argument('--campaign-gpu-hours',type=float,default=6000)
    a=p.parse_args()
    if not 0<=a.shard<a.shards or a.short_side<128 or min(a.max_seconds,a.campaign_gpu_hours)<=0:raise ValueError('Invalid target encoder allocation')
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'future_visual_target_cache','real_optimizer_updates':0}) as (meter,_,save):
        from tools.ddpolicy_vehicle.campaign import charged_gpu_hours
        if charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:raise RuntimeError('Campaign budget exhausted before target encoding')
        if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze target encoding source first')
        split=json.loads(Path(a.split_manifest).read_text());all_tokens=split[a.split+'_tokens'];logs=split['token_logs']
        if {logs[t] for t in split['train_tokens']}&{logs[t] for t in split['dev_tokens']}:raise ValueError('Cross-log split leakage')
        tokens=all_tokens[a.shard::a.shards]
        vae_identity=json.loads(Path(a.vae_identity).read_text());vae=FluxTargetEncoder(a.vae_root,vae_identity)
        identity={'schema':'foresight_flux_targets_v1','vae':vae_identity,'split':a.split,'partition_hash':file_sha256(a.split_manifest),
                  'horizons_s':[1.,2.,4.],'timestamp_tolerance_s':.05,'views':CAMERAS,'short_side':a.short_side,
                  'preprocessing':'same DDP16:9 current/future crop; aspect-preserving LANCZOS, right/bottom stride padding, range[-1,1]',
                  'scaling_factor':vae.scale,'shift_factor':vae.shift,'stride':vae.stride,'channels':vae.channels,
                  'posterior':'mode','dtype':'float32_encoder_float16_cache','shards':a.shards,
                  'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
        identity['identity']=identity_hash(identity);out=Path(a.output)/('shard_'+str(a.shard).zfill(3))
        if out.exists():
            if not a.resume or json.loads((out/'identity.json').read_text())!=json.loads(json.dumps(identity)):raise ValueError('Future cache identity mismatch')
        else:
            (out/'targets').mkdir(parents=True);atomic_json(out/'identity.json',identity)
        groups=defaultdict(list)
        for token in tokens:groups[logs[token]].append(token)
        counters=np.zeros((3,3),dtype=np.int64);errors=np.zeros((3,3),dtype=np.float64);elements=np.zeros((3,3),dtype=np.int64)
        failed=[];completed=0
        for log,scene_tokens in groups.items():
            frames=load_trusted(Path(a.raw_log_root)/(log+'.pkl'));where={f['token']:i for i,f in enumerate(frames)}
            for token in scene_tokens:
                if time.time()-meter['start_unix']>=a.max_seconds or charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:
                    meter['status']='PAUSED';meter['inference_scenes']=completed;save();return
                destination=out/'targets'/(token+'.pt')
                if destination.exists():
                    payload=torch.load(destination,weights_only=True,map_location='cpu')
                    if payload['identity']!=identity['identity']:raise ValueError('Existing future record identity mismatch')
                else:
                    at=where[token];future=timed_frames(frames,at,[1.,2.,4.],.05)
                    encoded={};transforms={};paths={};timestamps={};missing=[]
                    for horizon,frame in [(-1,frames[at]),*enumerate(future)]:
                        for view,cam in enumerate(CAMERAS):
                            key=(horizon,view)
                            if frame is None:missing.append([horizon,view,'timestamp_missing']);continue
                            try:path=camera_path(frame['cams'][cam]['data_path'],vars(a))
                            except FileNotFoundError:missing.append([horizon,view,'image_missing']);continue
                            image,transform=vae.preprocess(path,a.short_side)
                            z=vae.encode_images(image[None])[0].cpu()
                            encoded[key]=z;transforms[str(key)]=transform;paths[str(key)]=str(path);timestamps[str(key)]=int(frame['timestamp'])
                    current=[encoded.get((-1,v)) for v in range(3)]
                    if any(z is None for z in current):raise ValueError('Missing current image; must separately repair valid ego input')
                    shapes={tuple(z.shape) for z in encoded.values()}
                    if len(shapes)!=1:raise ValueError('Different natural view shapes require an explicitly registered ragged-grid adapter')
                    shape=next(iter(shapes));latent=torch.zeros((3,3,*shape));valid=torch.zeros(3,3,dtype=torch.bool)
                    copy_error=torch.zeros(3,3,dtype=torch.float64);copy_count=torch.zeros(3,3,dtype=torch.int64)
                    for h in range(3):
                        for v in range(3):
                            if (h,v) in encoded:
                                latent[h,v]=encoded[h,v];valid[h,v]=True
                                delta=(encoded[h,v]-current[v]).double().square()
                                copy_error[h,v]=delta.sum();copy_count[h,v]=delta.numel()
                    payload={'identity':identity['identity'],'token':token,'log':log,'latent':latent.half(),
                             'valid':valid,'current_latent':torch.stack(current).half(),'timestamps':timestamps,
                             'preprocessing':transforms,'source_paths':paths,'missing':missing,
                             'copy_current_squared_error_sum':copy_error,'copy_current_elements':copy_count}
                    tmp=destination.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(destination)
                counters+=payload['valid'].numpy();errors+=payload['copy_current_squared_error_sum'].numpy();elements+=payload['copy_current_elements'].numpy()
                completed+=1;meter['inference_scenes']=completed;save()
            atomic_json(out/'progress.json',{'completed':completed,'requested':len(tokens),'failed':failed})
        atomic_json(out/'COMPLETE.json',{'identity':identity['identity'],'scenes':completed,'failed':failed,
            'valid_horizon_view_counts':counters.tolist(),'copy_current_squared_error_sum':errors.tolist(),
            'copy_current_element_counts':elements.tolist(),'copy_current_mse':np.divide(errors,elements,out=np.zeros_like(errors),where=elements>0).tolist()})


if __name__=='__main__':main()
