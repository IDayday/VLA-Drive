"""Public frozen current-vision extraction. Launch independent shards with torchrun."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from starVLA.model.modules.joint_world.public_baseline import PublicQwenBaseline
from starVLA.model.modules.joint_world.visual_cache import FrozenVisualCache,visual_identity
from tools.local_interaction_mask_v2.data import current_metadata_from_training_pickle,current_example


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('public-qwen','config','provenance','datasets','cache','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    if world>1:dist.init_process_group('gloo')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    base=PublicQwenBaseline(a.public_qwen,OmegaConf.load(a.config),json.loads(Path(a.provenance).read_text()),initialize_head=False).eval()
    code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    identity=visual_identity(base,code)
    if rank==0:FrozenVisualCache(a.cache,identity,write=True)
    if world>1:dist.barrier()
    cache=FrozenVisualCache(a.cache,identity,write=True)
    records=[];seen=set()
    for spec in json.loads(Path(a.datasets).read_text()):
        for token in json.loads(Path(spec['tokens']).read_text()):
            if token in seen:raise ValueError('Repeated scene in visual extraction manifest')
            seen.add(token);records.append((token,spec))
    started=time.monotonic();completed=0;parity=[]
    for i in range(rank,len(records),world):
        if (out/'STOP_REQUESTED').exists():break
        token,spec=records[i]
        record=current_metadata_from_training_pickle(Path(spec['meta_root'])/(token+'.pkl'),token)
        example,observation=current_example(Path(spec['observations'])/(token+'.npz'),record)
        q=base.qwen_vl_interface.build_qwenvl_inputs(images=[example['image']],instructions=[example['lang']])
        with torch.autocast('cuda',dtype=torch.bfloat16),torch.no_grad():
            parts,deep=cache.get(example,q,base.qwen_vl_interface.model)
            if completed==0:
                p2,d2=cache.get(example,q,base.qwen_vl_interface.model)
                parity.append(max(float((x-y).abs().max()) for x,y in zip(list(parts)+list(deep),list(p2)+list(d2))))
                if parity[-1]!=0:raise AssertionError('Frozen visual disk roundtrip not exact')
        completed+=1
        if completed%20==0:print(json.dumps({'rank':rank,'completed':completed,'seconds':time.monotonic()-started,'hits':cache.hits,'misses':cache.misses}),flush=True)
    (out/f'shard_{rank}.json').write_text(json.dumps({'rank':rank,'world':world,'completed':completed,'total_assigned':len(records[rank::world]),
        'seconds':time.monotonic()-started,'disk_roundtrip_max_error':parity,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'identity':identity},indent=2)+'\n')
    if world>1:dist.barrier()
    if rank==0:
        rows=[json.loads((out/f'shard_{r}.json').read_text()) for r in range(world)]
        (out/'status.json').write_text(json.dumps({'status':'complete' if all(r['completed']==r['total_assigned'] for r in rows) else 'paused',
            'completed':sum(r['completed'] for r in rows),'expected':len(records)})+'\n')
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
