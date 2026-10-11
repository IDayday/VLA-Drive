"""Explicit metadata-only fork to the proved FP32 online path.

Never launches or stops a job. The caller must pause the cached run first.
The parent checkpoint is immutable; a new checkpoint keeps every training-state
tensor, RNG and consumed cursor. The new execution identity records the source
switch. This is not an exact-resume claim across arbitrary backend changes.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
from pathlib import Path
import torch
from iqe.contracts import require
from iqe.io import atomic_json, atomic_torch, digest, file_hash, read_json
from iqe.training.resume import inspect_checkpoint
from iqe.vision_cache.qualification import validate_cache_qualification
from iqe.vision_cache.runtime import ProductionOptions, ProductionRuntime


PRESERVED = ('model', 'optimizer', 'scheduler', 'optimizer_step', 'epoch', 'offset',
             'dependencies', 'rng_by_rank', 'fit_scene_ids', 'seed', 'mode',
             'resume_boundary', 'learned_driving_weights_loaded')


def state_fingerprint(state):
    """Complete tensors, including RNG/Adam, without a second resident checkpoint."""
    import numpy as np
    h=hashlib.sha256()
    def add(value):
        if isinstance(value, torch.Tensor):
            t=value.detach().cpu().contiguous()
            h.update(digest({'dtype':str(t.dtype),'shape':list(t.shape)}).encode())
            b=memoryview(t.reshape(-1).view(torch.uint8).numpy())
            for offset in range(0,len(b),8*1024**2):h.update(b[offset:offset+8*1024**2])
        elif isinstance(value,np.ndarray):add(torch.from_numpy(value.copy()))
        elif isinstance(value,dict):
            for key in sorted(value,key=repr):h.update(repr(key).encode());add(value[key])
        elif isinstance(value,(list,tuple)):
            h.update(type(value).__name__.encode())
            for v in value:add(v)
        else:h.update(repr(value).encode())
    add({k:state[k] for k in PRESERVED})
    return h.hexdigest()


def convert_metadata(state, certificate, online_runtime_identity, parent, checksum):
    old=deepcopy(state['performance_identity']);expected=deepcopy(certificate['runtime_identity'])
    # A full cache has a different index from the qualified representative set.
    for value in (old,expected):
        value['production_cache_runtime'].pop('cache_root',None)
        value['production_cache_runtime'].pop('cache_index_sha256',None)
    require(old==expected,'unqualified code/backend/environment or optimization state')
    require(state['training_visual_source']=='cached_fp32' and not state.get('terminal',False),
            'fallback requires a nonterminal cached run')
    require(state['mode'] in {'full','performance_v2'},'invalid parent execution mode')
    online=deepcopy(online_runtime_identity)
    require(online['training_visual_source']=='online_fp32' and online['cache_root'] is None,
            'fallback must use the original online FP32 visual')
    normalized=deepcopy(online);normalized.pop('cache_root',None)
    normalized['training_visual_source']='cached_fp32'
    require(normalized==old['production_cache_runtime'],'fallback changed more than data execution source')
    converted=dict(state)
    converted['performance_identity']=deepcopy(state['performance_identity'])
    converted['performance_identity']['production_cache_runtime']=online
    converted.update(training_visual_source='online_fp32',cache_execution_dependency=online,
        parent_checkpoint=parent,execution_migration={'kind':'QUALIFIED_FP32_CACHE_TO_ONLINE',
            'parent_checkpoint':parent,'parent_sha256':checksum,'training_state_preserved':list(PRESERVED),
            'same_backend_required':True,'requires_parent_paused':True,'new_optimizer_steps':0})
    return converted


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--parent-checkpoint',required=True);p.add_argument('--production-config',required=True)
    p.add_argument('--qualification',required=True);p.add_argument('--output',required=True)
    p.add_argument('--diagnostic',action='store_true',help='retain performance_v2 identity; never promote to formal')
    a=p.parse_args();parent=Path(a.parent_checkpoint).resolve();root=Path(a.output).resolve()
    require(root!=parent.parent and not root.exists(),'fallback needs a new, unused output identity')
    # The normal runner holds this exact lock throughout checkpoint writes.
    with (parent.parent/'run.lock').open('a+b') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        info=inspect_checkpoint(parent)
        newer=[x for x in parent.parent.glob('step_*.pt') if int(x.stem[5:])>info['optimizer_step']]
        require(not newer,'fallback must preserve latest progress, including incomplete newer state')
        result=read_json(parent.parent/'RESULT.json')
        require(result['final_step']==info['optimizer_step'] and not result['training_terminal'] and
                result['status'] in {'COMPLETE','OPERATOR_PAUSED'},'parent has not finished its bounded/pause boundary')
        state=torch.load(parent,map_location='cpu',mmap=True,weights_only=False)
        require((state['mode']=='performance_v2')==a.diagnostic,'diagnostic/formal source identity must be explicit')
        certificate=validate_cache_qualification(a.qualification,fit_hash=digest(state['fit_scene_ids']),for_build=True)
        ProductionOptions.load(a.production_config)
        runtime=ProductionRuntime(a.production_config).identity()
        converted=convert_metadata(state,certificate,runtime,str(parent),info['checksum'])
        before=state_fingerprint(state);require(before==state_fingerprint(converted),'training state changed during conversion')
        from iqe.vision_cache.plan import capacity_check
        root.parent.mkdir(parents=True,exist_ok=True)
        capacity=capacity_check(root.parent,parent.stat().st_size,reserve_bytes=200*1024**3,
                                temporary_bytes=parent.stat().st_size)
        require(capacity['sufficient'],'fallback copy exceeds shared-storage reserve')
        root.mkdir();path=root/parent.name
        atomic_torch(path,converted,immutable=True)
        after=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        require(before==state_fingerprint(after),'serialized fallback state differs')
        atomic_json(str(path)+'.COMPLETE.json',{'checksum':file_hash(path),'optimizer_step':info['optimizer_step'],
                    'dependencies':state['dependencies']},immutable=True)
        report={'status':'PASS','kind':'metadata_only_execution_migration','parent':info,'output':str(path),
                'training_state_sha256':before,'parent_unchanged':file_hash(parent)==info['checksum'],
                'requires_host':state['performance_identity']['execution_environment']['host'],
                'GPU_continuation_executed':False,'training_steps_added':0,'script_hash':file_hash(__file__)}
        atomic_json(root/'MIGRATION.json',report,immutable=True)
        derived=deepcopy(certificate);derived['runtime_identity']=converted['performance_identity']
        derived['evidence']['execution_migration']={'path':str(root/'MIGRATION.json'),'sha256':file_hash(root/'MIGRATION.json')}
        atomic_json(root/'ONLINE_QUALIFICATION.json',derived,immutable=True)
        validate_cache_qualification(root/'ONLINE_QUALIFICATION.json',runtime_identity=converted['performance_identity'])
        print({'status':'PASS','checkpoint':str(path),'qualification':str(root/'ONLINE_QUALIFICATION.json'),
               'next_action':'explicit online train-base-cached launch with new output/lineage; do not restart cached parent'})


if __name__=='__main__':main()
