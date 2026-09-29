"""One teacher encode per resolution; independent per-view post-encoder pooling."""
import argparse,fcntl,json,os,time,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from safetensors.torch import save_file
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.ddpolicy_vehicle.campaign import charged_gpu_hours
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256
from starVLA.model.modules.foresight.dinov3_target_encoder import DINOv3TargetEncoder
from starVLA.model.modules.foresight.tradeoff import CANDIDATES,preprocess_current,pool_patches


def main(full_method=False):
 p=argparse.ArgumentParser(__doc__)
 for key in ('index','model-root','output','campaign-root','run-id'):p.add_argument('--'+key,required=True)
 p.add_argument('--width',type=int,required=True);p.add_argument('--chunk-size',type=int,default=96);p.add_argument('--batch',type=int,default=16);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit-images',type=int,default=0)
 p.add_argument('--reuse-current-root',help='Full-method only: byte-verified compatible per-image cache')
 p.add_argument('--max-seconds',type=float,default=21600);p.add_argument('--campaign-gpu-hours',type=float,required=True);a=p.parse_args()
 if a.reuse_current_root and not full_method:raise ValueError('Reuse adapter is explicit to the new full-method schema')
 candidates=[c for c in CANDIDATES.values() if c.width==a.width]
 if not candidates or min(a.chunk_size,a.batch,a.max_seconds,a.campaign_gpu_hours)<=0 or not 0<=a.shard<a.shards:raise ValueError('Invalid extraction config')
 root=Path(a.output);root.mkdir(parents=True,exist_ok=True);idx=Path(a.index)
 with metered_run(a.campaign_root,a.run_id,1,{'kind':'full_current_future_dino_cache' if full_method else 'current_dino_teacher_cache','width':a.width,'shared_candidates':[c.name for c in candidates]}) as (meter,_,save):
  if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Lock clean extraction source first')
  lock=(root/f'width{a.width}_shard{a.shard}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  model=DINOv3TargetEncoder(a.model_root,'cuda')
  if model.patch_size!=(16,16) or model.feature_dim!=1024:raise ValueError('Wrong teacher')
  index=json.loads((idx/'identity.json').read_text());images=json.loads((idx/'images.json').read_text())
  expected_schema='ddp_full_dino_index_v1' if full_method else 'dino_tradeoff_current_index_v1'
  if index['schema']!=expected_schema or file_sha256(idx/'images.json')!=index['files']['images']:raise ValueError('Invalid image/time index')
  recipe={k:v for k,v in model.recipe.items() if k not in ('source_rgb_range','short_side','padding','pooling','implementation_sha256','original_user_recipe')}
  recipe.update(schema='dino_tradeoff_target_recipe_v1',source_rgb_range='full original16:9 image resized anisotropically to4:3; no crop',padding='none',actual_tensor_chw=[3,candidates[0].height,a.width],normalization='ImageNet RGB once; last post-norm patches; no extra L2',encoder_implementation=file_sha256(__import__('starVLA.model.modules.foresight.dinov3_target_encoder',fromlist=['x']).__file__),preprocess_implementation=file_sha256(__import__('starVLA.model.modules.foresight.tradeoff',fromlist=['x']).__file__))
  reuse=None
  if a.reuse_current_root:
   from tools.full_foresight.cache_reuse import CurrentTargetReuse
   reuse=CurrentTargetReuse(a.reuse_current_root,candidates,recipe)
  identities={}
  for c in candidates:
   dest=root/c.name;dest.mkdir(exist_ok=True)
   ident={'schema':'ddp_full_dino_cache_v1' if full_method else 'dino_tradeoff_cache_v1','index':index['identity'],'recipe':recipe,'candidate':c.record(),'chunk_size':a.chunk_size,'image_count':len(images),'dtype':'float16','writer':file_sha256(__file__)}
   if full_method:ident.update(grid_hw=list(c.grid_hw),horizons_s=[0.,1.,2.,4.])
   ident['identity']=identity_hash(ident)
   guard=(dest/'IDENTITY.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX)
   if (dest/'identity.json').exists() and json.loads((dest/'identity.json').read_text())!=ident:raise ValueError('Foreign cache')
   atomic_json(dest/'identity.json',ident);guard.close();identities[c.name]=ident
  count=min(len(images),a.limit_images) if a.limit_images else len(images);chunks=(count+a.chunk_size-1)//a.chunk_size
  def read(r):
   path=Path(r['path']);st=path.stat()
   if (st.st_size,st.st_mtime_ns)!=(r['bytes'],r['mtime_ns']):raise ValueError('Source RGB changed')
   pixels,transform=preprocess_current(path,a.width,candidates[0].height)
   return pixels,transform,file_sha256(path)
  with ThreadPoolExecutor(max_workers=8) as pool:
   for chunk in range(a.shard,chunks,a.shards):
    # Always write whole chunks even for a bounded preflight prefix.
    rows=images[chunk*a.chunk_size:(chunk+1)*a.chunk_size]
    if time.time()-meter['start_unix']>=a.max_seconds or charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:meter['status']='PAUSED';save();return
    missing=[]
    for c in candidates:
     folder=root/c.name;meta=folder/f'chunk_{chunk:06d}.json';dest=folder/f'chunk_{chunk:06d}.safetensors'
     if meta.exists():
      m=json.loads(meta.read_text())
      if m['identity']!=identities[c.name]['identity'] or m['keys']!=[r['key'] for r in rows] or file_sha256(dest)!=m['sha256']:raise ValueError('Corrupt/foreign existing chunk')
     else:missing.append(c)
    if not missing:continue
    tensors={c.name:[] for c in missing};transforms=[];hashes=[];origins=[];errors={c.name:[0.,0,0.] for c in missing};norms={c.name:[0.,0.] for c in missing}
    for start in range(0,len(rows),a.batch):
     batch_rows=rows[start:start+a.batch];reused=[reuse.get(row) if reuse else None for row in batch_rows]
     fresh_indices=[i for i,x in enumerate(reused) if x is None]
     fresh=list(pool.map(read,[batch_rows[i] for i in fresh_indices]));fresh_map=dict(zip(fresh_indices,range(len(fresh_indices))))
     native=model(torch.stack([r[0] for r in fresh]).cuda()) if fresh else None
     for c in missing:
      newly_encoded=pool_patches(native,c).cpu() if native is not None else None
      z=torch.stack([newly_encoded[fresh_map[i]] if reused[i] is None else reused[i][0][c.name].float() for i in range(len(batch_rows))]);stored=z.half()
      if not torch.isfinite(stored).all():raise ValueError('FP16 overflow')
      if fresh_indices:
       error=stored[fresh_indices].float()-newly_encoded;e=errors[c.name];e[0]+=float(error.square().sum());e[1]+=newly_encoded.numel();e[2]=max(e[2],float(error.abs().max()))
      norms[c.name][0]+=float(z.sum());norms[c.name][1]+=float(z.square().sum());tensors[c.name].append(stored)
     for i in range(len(batch_rows)):
      item=reused[i]
      if item is None:
       f=fresh[fresh_map[i]];transforms.append(f[1]);hashes.append(f[2]);origins.append(None)
      else:
       # Matching key alone is insufficient if original bytes changed on disk.
       path=Path(batch_rows[i]['path'])
       if file_sha256(path)!=item[2]:raise ValueError('Reused target source RGB changed')
       transforms.append(item[1]);hashes.append(item[2]);origins.append(item[3])
    for c in missing:
     folder=root/c.name;dest=folder/f'chunk_{chunk:06d}.safetensors';temp=dest.with_suffix(f'.{os.getpid()}.tmp');z=torch.cat(tensors[c.name]);identity=identities[c.name]['identity']
     save_file({'features':z,'valid':torch.ones(len(z),*c.grid_hw,dtype=torch.bool)},str(temp),metadata={'identity':identity});temp.replace(dest)
     e=errors[c.name];n=norms[c.name]
     atomic_json(folder/f'chunk_{chunk:06d}.json',{'identity':identity,'keys':[r['key'] for r in rows],'image_sha256':hashes,'transforms':transforms,'bytes':dest.stat().st_size,'sha256':file_sha256(dest),'quantization_mse':e[0]/e[1] if e[1] else None,'quantization_max_abs':e[2] if e[1] else None,'quantization_scope':'newly_encoded_only; reused quantization retained in reuse_sources','reuse_sources':[x[c.name] if x else None for x in origins],'target_mean':n[0]/z.numel(),'target_second_moment':n[1]/z.numel()})
    meter['reused_images']=meter.get('reused_images',0)+sum(x is not None for x in origins)
    meter['encoded_images']=meter.get('encoded_images',0)+sum(x is None for x in origins)
    meter['inference_images']=meter.get('inference_images',0)+len(rows);meter['last_chunk']=chunk;save()
  for c in candidates:
   total=(len(images)+a.chunk_size-1)//a.chunk_size
   if all((root/c.name/f'chunk_{i:06d}.json').exists() for i in range(total)):atomic_json(root/c.name/'COMPLETE.json',{'identity':identities[c.name]['identity'],'images':len(images),'chunks':total})
if __name__=='__main__':main()
