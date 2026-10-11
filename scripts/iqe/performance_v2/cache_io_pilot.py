"""Eight-rank random IO/H2D measurement over an already complete64-scene pilot.
No model training, new cache generation, compression, global cache eviction or GT.
"""
import argparse,json,os,random,time
from pathlib import Path
import torch
import torch.distributed as dist
from iqe.resources import qualify_distributed
from iqe.training.trainer import init_distributed
from iqe.contracts import require
from iqe.io import read_json,file_hash,atomic_json
from iqe.performance.vision_cache import FrozenVisionPayload,combine_payloads

p=argparse.ArgumentParser(__doc__);p.add_argument('--cache',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
qualify_distributed('cuda');rank,world=init_distributed('cuda');require(world==8,'eight-rank pilot')
device=f'cuda:{os.environ["LOCAL_RANK"]}';torch.cuda.set_device(device);torch.set_num_threads(2)
receipt=read_json(a.cache/'COMPLETE.json');index={}
for name,sha in receipt['rank_indexes'].items():
 require(file_hash(a.cache/name)==sha,'index checksum')
 index.update(read_json(a.cache/name)['index'])
require(len(index)==64,'this entrypoint is limited to exactly64 unique pilot scenes')
keys=sorted(index);random.Random(42).shuffle(keys);mine=keys[rank::world]
shards={v['shard'] for v in index.values()}
if rank==0:
 for name in shards:
  with (a.cache/name).open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
dist.barrier();loaded={};rows=[]
for repetition in range(4):
 for at in range(0,len(mine),4):
  dist.barrier();start=time.perf_counter();verify=0.;payloads=[]
  for key in mine[at:at+4]:
   item=index[key];name=item['shard']
   if name not in loaded:
    t=time.perf_counter();require(file_hash(a.cache/name)==item['checksum'],'shard checksum');verify+=time.perf_counter()-t
    loaded[name]=torch.load(a.cache/name,mmap=True,weights_only=True,map_location='cpu')
   payloads.append(FrozenVisionPayload(**loaded[name][item['index']]))
  read=time.perf_counter()-start
  event0,event1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);event0.record()
  value=combine_payloads(payloads,0,device,payloads[0].slots)
  event1.record();event1.synchronize();seconds=time.perf_counter()-start
  bytes_=sum(v.numel()*v.element_size() for q in payloads for v in (q.image_embeddings,*q.deepstack))
  rows.append({'repetition':repetition,'scenes':len(payloads),'payload_bytes':bytes_,'total_seconds':seconds,'checksum_seconds':verify,'mmap_metadata_seconds':read-verify,'combine_H2D_cuda_ms':event0.elapsed_time(event1)})
  del value
all_rows=[None]*world;dist.all_gather_object(all_rows,rows)
if rank==0:
 atomic_json(a.output,{'status':'COMPLETE','unique_scenes':64,'world_size':world,'random_seed':42,'bytes':sum((a.cache/n).stat().st_size for n in shards),'shards':len(shards),'rows_by_rank':all_rows,'scope':'random payload IO/combine/H2D only, not training throughput; first pass file-specific DONTNEED advisory, later passes same process warm; checksum verifies each shard once per rank; no decompression'})
dist.destroy_process_group()
