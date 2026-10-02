"""Offline native video or independent-frame targets, atomic sharded resume."""
import argparse,fcntl,json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
from safetensors.torch import save_file
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from starVLA.model.modules.vehicle_joint.initialization import file_sha256,identity_hash
from starVLA.model.modules.foresight.video_target_encoder import VideoTargetEncoder
from starVLA.model.modules.foresight.dinov3_target_encoder import DINOv3TargetEncoder
from starVLA.model.modules.foresight.tradeoff import preprocess_current


def load_real_view(view):
    frames=[];hashes=[]
    for r in view:
        if r is None:raise ValueError('Never fill a missing clip')
        p=Path(r['path']);st=p.stat()
        if (st.st_size,st.st_mtime_ns)!=(r['bytes'],r['mtime_ns']):raise ValueError('Source image changed')
        with Image.open(p) as img:frames.append(img.convert('RGB'))
        hashes.append(file_sha256(p))
    return frames,hashes


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('index','output','campaign-root','run-id'):p.add_argument('--'+k,required=True)
    p.add_argument('--target-type',choices=('video_clip','dino_sequence'),required=True)
    p.add_argument('--split',choices=('train','dev'),required=True)
    for k in ('source-root','source-sha','checkpoint','weight-sha256','dino-model-root'):p.add_argument('--'+k)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--chunk-size',type=int,default=16);p.add_argument('--clip-batch',type=int,default=2)
    p.add_argument('--limit-scenes',type=int,default=0);a=p.parse_args()
    if not 0<=a.shard<a.shards or min(a.chunk_size,a.clip_batch)<1 or a.limit_scenes<0:raise ValueError('Invalid finite extraction bounds')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze extraction source')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    idx=Path(a.index);index=json.loads((idx/'identity.json').read_text())
    if index['schema']!='action_video_clip_index_v1' or not (idx/'COMPLETE.json').exists():raise ValueError('Complete real frame index required')
    scene_path=idx/(a.split+'_scenes.json')
    if file_sha256(scene_path)!=index['files'][a.split]:raise ValueError('Changed clip population')
    scenes=json.loads(scene_path.read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with metered_run(a.campaign_root,a.run_id,1,{'kind':a.target_type+'_target_extraction','real_optimizer_updates':0}) as (meter,_,save):
        if a.target_type=='video_clip':
            model=VideoTargetEncoder(a.source_root,a.checkpoint,source_sha=a.source_sha,weight_sha256=a.weight_sha256).cuda()
            recipe=model.identity;t=4;spans=[[.5,1.],[1.5,2.],[2.5,3.],[3.5,4.]]
        else:
            model=DINOv3TargetEncoder(a.dino_model_root).cuda();recipe={**model.recipe,'input_width_height':[384,288],'pool':2,
                'native_grid_hw':[18,24],'output_grid_hw':[9,12],'preprocess':'full source RGB, PIL bicubic384x288, ImageNet once, no crop',
                'precision':'FP32, TF32 off','temporal_path':'eight independent images; no video attention'}
            t=8;spans=[[x,x] for x in index['times_s']]
        ident={'schema':'action_video_clip_targets_v1','future_target_type':a.target_type,'clip_index':index['identity'],
            'scene_index_hash':index['index_hashes'][a.split],'partition_sha256':index['partition_sha256'],'split':a.split,
            'scene_count':len(scenes),'target_shape':[3,t,9,12,1024],'time_intervals_s':spans,'times_s':index['times_s'],
            'recipe':recipe,'dtype':'float16','normalization':'raw teacher features; training nonaffine channel LayerNorm',
            'chunk_size':a.chunk_size,'source':source,'writer_sha256':file_sha256(__file__),
            'missing_policy':'invalid whole view if any of eight real frames unavailable; identical eligibility for both teachers'}
        ident['identity']=identity_hash(ident)
        with (out/'IDENTITY.lock').open('a+') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=ident:raise ValueError('Existing cache is incompatible')
            atomic_json(out/'identity.json',ident)
        count=min(a.limit_scenes,len(scenes)) if a.limit_scenes else len(scenes);chunks=(count+a.chunk_size-1)//a.chunk_size
        with ThreadPoolExecutor(max_workers=8) as pool:
            for chunk in range(a.shard,chunks,a.shards):
                rows=scenes[chunk*a.chunk_size:(chunk+1)*a.chunk_size];dest=out/f'chunk_{chunk:06d}.safetensors';meta=dest.with_suffix('.json')
                if meta.exists():
                    old=json.loads(meta.read_text())
                    if old['identity']!=ident['identity'] or old['tokens']!=[r['token'] for r in rows] or file_sha256(dest)!=old['sha256']:raise ValueError('Foreign/corrupt existing target chunk')
                    continue
                features=torch.zeros(len(rows),3,t,9,12,1024,dtype=torch.float16);valid=torch.zeros(features.shape[:-1],dtype=torch.bool)
                jobs=[(i,v,refs) for i,row in enumerate(rows) for v,refs in enumerate(row['frames_by_view']) if row['clip_view_valid'][v]]
                quant=[0.,0,0.];hashes={}
                for start in range(0,len(jobs),a.clip_batch):
                    batch=jobs[start:start+a.clip_batch];loaded=list(pool.map(load_real_view,[j[2] for j in batch]))
                    if a.target_type=='video_clip':
                        x=torch.stack([model.preprocess(frames) for frames,_ in loaded]).cuda();z=model(x).cpu()
                    else:
                        # Explicit independent frame encoding, same exact RGB source/ROI as video.
                        values=[]
                        for frames,_ in loaded:
                            for frame in frames:
                                import numpy as np
                                rgb=torch.from_numpy(np.asarray(frame.resize((384,288),Image.Resampling.BICUBIC)).copy()).permute(2,0,1).float()/255.
                                values.append((rgb-rgb.new_tensor([.485,.456,.406])[:,None,None])/rgb.new_tensor([.229,.224,.225])[:,None,None])
                        with torch.inference_mode():native=model(torch.stack(values).cuda())
                        pooled=torch.nn.functional.avg_pool2d(native,2)
                        z=pooled.reshape(len(batch),8,1024,9,12).permute(0,1,3,4,2).cpu()
                    stored=z.half()
                    if not torch.isfinite(stored).all():raise FloatingPointError('FP16 clip overflow')
                    error=stored.float()-z;quant[0]+=float(error.square().sum());quant[1]+=z.numel();quant[2]=max(quant[2],float(error.abs().max()))
                    for j,(scene,view,_) in enumerate(batch):
                        features[scene,view]=stored[j];valid[scene,view]=True;hashes[f'{scene}:{view}']=loaded[j][1]
                    meter['encoded_clips']=meter.get('encoded_clips',0)+len(batch);save()
                temp=dest.with_suffix(f'.{os.getpid()}.tmp');save_file({'features':features,'valid':valid},str(temp),metadata={'identity':ident['identity']});temp.replace(dest)
                atomic_json(meta,{'identity':ident['identity'],'tokens':[r['token'] for r in rows],'image_sha256':hashes,'sha256':file_sha256(dest),
                    'bytes':dest.stat().st_size,'quantization_mse':quant[0]/quant[1] if quant[1] else None,'quantization_max_abs':quant[2],
                    'valid_views':int(valid.flatten(2).any(-1).sum())})
                meter['last_chunk']=chunk;save()
        total=(len(scenes)+a.chunk_size-1)//a.chunk_size
        if all((out/f'chunk_{i:06d}.json').exists() for i in range(total)):
            atomic_json(out/'COMPLETE.json',{'identity':ident['identity'],'scenes':len(scenes),'chunks':total,
                'cache_bytes':sum((out/f'chunk_{i:06d}.safetensors').stat().st_size for i in range(total))})

if __name__=='__main__':main()
