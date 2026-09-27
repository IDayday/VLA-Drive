"""Frozen trained public foundation -> current-only local graph and conditioning cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tools.local_interaction_mask_v2.foundation import create_world,restore_modules
from tools.local_interaction_mask_v2.data import current_metadata_from_training_pickle,current_example
from starVLA.model.modules.joint_world.public_baseline import sha256
from starVLA.model.modules.joint_world.local_cache import cache_identity,build_payload,pack_payload,signature
from starVLA.model.modules.joint_world.current_refinement import restore_current_head


def load_foundation(path,public_qwen,visual_cache=None,perception_checkpoint=None):
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    identity=saved['identity']
    if identity.get('private_driving_weights_loaded',True):raise ValueError('Private foundation forbidden')
    cfg=identity.get('model_config')
    if cfg is None:
        # Legacy within-campaign trainer records exact source commit/config path.
        cfg=OmegaConf.create(subprocess.check_output(['git','show',identity['code_sha']+':'+identity['arguments']['config']],text=True))
    else:cfg=OmegaConf.create(cfg)
    world=create_world(public_qwen,cfg,identity['public_provenance'],identity['arguments']['seed'],visual_cache)
    if world.baseline.public_origin!=saved['public_origin']:raise ValueError('Public foundation reconstruction differs')
    restore_modules(world,saved['modules'])
    world.foundation_sha256=sha256(path)
    world.perception_refinement=(restore_current_head(world.heads,perception_checkpoint,world.foundation_sha256,saved['public_origin'])
                                if perception_checkpoint else None)
    world.eval().requires_grad_(False)
    metadata={'training_identity':identity,'step':saved['step'],'presentations':saved['presentations'],'epoch':saved['epoch'],
              'public_origin':saved['public_origin'],'foundation_sha256':world.foundation_sha256,
              'perception_refinement':world.perception_refinement}
    return world,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','public-qwen','dataset','selector','graph-config','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--visual-cache');p.add_argument('--perception-checkpoint');a=p.parse_args()
    rank=int(os.environ.get('RANK',0));count=int(os.environ.get('WORLD_SIZE',1));torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if count>1:dist.init_process_group('gloo')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    world,metadata=load_foundation(a.checkpoint,a.public_qwen,a.visual_cache,a.perception_checkpoint)
    checkpoint_hash=metadata['foundation_sha256']
    selector=json.loads(Path(a.selector).read_text());graph_config=json.loads(Path(a.graph_config).read_text())
    identity=cache_identity(checkpoint_hash,metadata['public_origin'],selector,graph_config)
    identity['foundation_training']=metadata['training_identity'];identity['foundation_step']=metadata['step']
    if metadata['perception_refinement'] is not None:identity['perception_override']=metadata['perception_refinement']
    spec=json.loads(Path(a.dataset).read_text());tokens=json.loads(Path(spec['tokens']).read_text())
    records=[];parity=[]
    for token in tokens[rank::count]:
        if (out/'STOP_REQUESTED').exists():break
        record=(json.loads((Path(spec['current_records'])/(token+'.json')).read_text()) if 'current_records' in spec else
                current_metadata_from_training_pickle(Path(spec['meta_root'])/(token+'.pkl'),token))
        example,observation=current_example(Path(spec['observations'])/(token+'.npz'),record)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):native,pred=world.encode_conditions([example])
        payload=build_payload(native,pred,observation,identity)
        path=out/(token+'.pt')
        if path.exists():raise FileExistsError('Current cache rebuild requires a fresh directory/shard')
        torch.save(payload,path)
        if not records:
            loaded=torch.load(path,weights_only=True);online=pack_payload(payload,identity);cached=pack_payload(loaded,identity)
            error=max(float((online[k]-cached[k]).abs().max()) for k in ('actor_features','context','current_xy','existence'))
            parity.append(error)
            if error!=0:raise AssertionError('Local current disk roundtrip changed condition')
        records.append({'token':token,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        if len(records)%50==0:print(json.dumps({'rank':rank,'completed':len(records),'assigned':len(tokens[rank::count])}),flush=True)
    (out/f'shard_{rank}.json').write_text(json.dumps({'records':records,'parity_max_error':parity,'expected':len(tokens[rank::count]),'peak_gpu_bytes':torch.cuda.max_memory_allocated()},indent=2)+'\n')
    if count>1:dist.barrier()
    if rank==0:
        shards=[json.loads((out/f'shard_{r}.json').read_text()) for r in range(count)];records=sum([s['records'] for s in shards],[])
        records.sort(key=lambda r:tokens.index(r['token']))
        complete=len(records)==len(tokens)
        (out/'manifest.json').write_text(json.dumps({'identity':identity,'identity_sha256':signature(identity),'records':records,'expected':len(tokens),'failed':0,'complete':complete},indent=2)+'\n')
        (out/'status.json').write_text(json.dumps({'status':'complete' if complete else 'paused','completed':len(records),'expected':len(tokens)})+'\n')
    if count>1:dist.destroy_process_group()


if __name__=='__main__':main()
