"""Atomic shared image-feature chunks; current/future are separate scene indices."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from safetensors.torch import save_file
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.ddpolicy_vehicle.campaign import charged_gpu_hours
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256
from starVLA.model.modules.foresight.dinov3_target_encoder import DINOv3TargetEncoder, preprocess_image


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('index','model-root','output','campaign-root','run-id'):p.add_argument('--'+k,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--chunk-size',type=int,default=64);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--max-seconds',type=float,default=21600);p.add_argument('--campaign-gpu-hours',type=float,default=6000)
    p.add_argument('--limit-chunks',type=int,default=0);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not 0<=a.shard<a.shards or min(a.chunk_size,a.batch,a.max_seconds,a.campaign_gpu_hours)<=0:raise ValueError('Invalid allocation')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);index=Path(a.index)
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'dinov3_current_future_cache','real_optimizer_updates':0}) as (meter,_,save):
        if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze extraction source first')
        lock=(root/f'worker_{a.shard:03d}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        model=DINOv3TargetEncoder(a.model_root,'cuda');images=json.loads((index/'images.json').read_text())
        idx=json.loads((index/'identity.json').read_text())
        if not (index/'COMPLETE.json').exists() or file_sha256(index/'images.json')!=idx['files']['images']:raise ValueError('Invalid image index')
        identity={'schema':'dinov3_dense_cache_v1','index':idx['identity'],'recipe':model.recipe,
                  'chunk_size':a.chunk_size,'image_count':len(images),'dtype':'float16','grid_hw':[16,29],
                  'writer_sha256':file_sha256(__file__)}
        identity['identity']=identity_hash(identity)
        guard=(root/'IDENTITY.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX)
        if (root/'identity.json').exists():
            if json.loads((root/'identity.json').read_text())!=identity:raise ValueError('Existing cache identity mismatch')
        else:atomic_json(root/'identity.json',identity)
        fcntl.flock(guard,fcntl.LOCK_UN);guard.close()
        chunks=(len(images)+a.chunk_size-1)//a.chunk_size
        # Prioritize fixed train64 diagnostic chunks by DATA ORDER, never model output.
        scenes=json.loads((index/'train_scenes.json').read_text())[:64]
        priority={i//a.chunk_size for r in scenes for h in r['images'] for i in h if i>=0}
        order=sorted(priority)+[i for i in range(chunks) if i not in priority]
        assigned=[i for i in order if i%a.shards==a.shard]
        if a.limit_chunks:assigned=assigned[:a.limit_chunks]
        completed=0;quant_sum=quant_count=0;quant_max=0.
        def read(entry):
            path=Path(entry['path']);st=path.stat()
            if st.st_size!=entry['bytes'] or st.st_mtime_ns!=entry['mtime_ns']:raise ValueError('Source image changed')
            pixels,mask,transform=preprocess_image(path,model.patch_size)
            return pixels,mask,transform,file_sha256(path)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for chunk in assigned:
                if time.time()-meter['start_unix']>=a.max_seconds or charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:
                    meter['status']='PAUSED';save();return
                dest=root/f'chunk_{chunk:06d}.safetensors';meta=root/f'chunk_{chunk:06d}.json'
                rows=images[chunk*a.chunk_size:(chunk+1)*a.chunk_size]
                if meta.exists():
                    record=json.loads(meta.read_text())
                    if record['identity']!=identity['identity'] or record['keys']!=[r['key'] for r in rows] or not dest.exists() or dest.stat().st_size!=record['bytes']:
                        raise ValueError('Incomplete or foreign chunk')
                    if file_sha256(dest)!=record['sha256']:raise ValueError('Corrupt cached chunk')
                else:
                    features=[];masks=[];transforms=[];hashes=[];chunk_error=chunk_elements=0;chunk_max=0.
                    for start in range(0,len(rows),a.batch):
                        batch=list(pool.map(read,rows[start:start+a.batch]))
                        pixels=torch.stack([r[0] for r in batch]).cuda();mask=torch.stack([r[1] for r in batch])
                        z=model(pixels).cpu();stored=z.half()
                        if z.shape[1:]!=(model.feature_dim,16,29):raise ValueError('Register different rectangular grid explicitly')
                        if not torch.isfinite(stored).all():raise ValueError('FP16 target overflow')
                        error=(stored.float()-z).square();valid=mask[:,None].expand_as(z)
                        chunk_error+=float(error[valid].sum());chunk_elements+=int(valid.sum())
                        chunk_max=max(chunk_max,float((stored.float()-z)[valid].abs().max()))
                        features.append(stored);masks.append(mask);transforms.extend(r[2] for r in batch);hashes.extend(r[3] for r in batch)
                    tmp=dest.with_suffix(f'.{os.getpid()}.tmp')
                    save_file({'features':torch.cat(features),'valid':torch.cat(masks)},str(tmp),metadata={'identity':identity['identity'],'chunk':str(chunk)})
                    tmp.replace(dest)
                    record={'identity':identity['identity'],'chunk':chunk,'keys':[r['key'] for r in rows],
                        'image_sha256':hashes,'transforms':transforms,'sha256':file_sha256(dest),'bytes':dest.stat().st_size,
                        'quantization_squared_error_sum':chunk_error,'quantization_elements':chunk_elements,'quantization_max_abs':chunk_max}
                    atomic_json(meta,record)
                completed+=len(rows);quant_sum+=record['quantization_squared_error_sum'];quant_count+=record['quantization_elements'];quant_max=max(quant_max,record['quantization_max_abs'])
                meter.update(inference_images=completed,completed_chunks=completed//a.chunk_size);save()
                atomic_json(root/f'worker_{a.shard:03d}_progress.json',{'identity':identity['identity'],'images':completed,
                    'assigned_chunks':len(assigned),'quantization_mse':quant_sum/max(1,quant_count),'quantization_max_abs':quant_max})
        atomic_json(root/f'worker_{a.shard:03d}_COMPLETE.json',{'identity':identity['identity'],'chunks':assigned,'images':completed,'limited':bool(a.limit_chunks)})
        if all((root/f'chunk_{i:06d}.json').exists() for i in range(chunks)):
            atomic_json(root/'COMPLETE.json',{'identity':identity['identity'],'images':len(images),'chunks':chunks,'failures':0})

if __name__=='__main__':main()
