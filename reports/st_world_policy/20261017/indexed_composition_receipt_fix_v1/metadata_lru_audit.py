import os, time, resource, json, sys, io, builtins
from pathlib import Path
from collections import OrderedDict
from unittest.mock import patch
from starVLA.dataloader.st_world_metadata_index import MetadataIndex
from starVLA.dataloader.st_world_schema import TeacherFeatureContract
from starVLA.dataloader.st_world_targets import payload_key
import torch

torch.set_num_threads(2)
started=time.monotonic()
def rchar():return int(dict(line.split(':',1) for line in Path('/proc/self/io').read_text().splitlines())['rchar'])
read_start=rchar()
def guard():
    if time.monotonic()-started>180:raise TimeoutError('Metadata audit exceeds 180s')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2*(1<<30):raise MemoryError('Metadata audit RSS exceeds 2GiB')
    if rchar()-read_start>1<<30:raise RuntimeError('Metadata audit read budget exceeds 1GiB')
original_io,original_open=io.open,builtins.open
forbidden=('.safetensors','.jpg','.jpeg','.png','.pt','.ckpt','.pkl','.pickle','.npz','.npy')
def reject_payload(original):
    def checked(file,*args,**kwargs):
        if not isinstance(file,int) and str(file).lower().endswith(forbidden):raise AssertionError('Payload open forbidden: '+str(file))
        return original(file,*args,**kwargs)
    return checked
root=Path('/mnt/project/st-world-policy-20261016/caches')
indices=[]
with patch('io.open',reject_payload(original_io)),patch('builtins.open',reject_payload(original_open)):
    try:
        for name in ('future32_request_metadata_index_v1','full_current_metadata_index_v1'):
            p=root/name; receipt=json.loads((p/'COMPLETE.json').read_text());guard()
            indices.append(MetadataIndex(p,expected_request_identity=receipt['request_identity'],expected_index_identity=receipt['identity']))
            guard()
        req,src=indices
        assert req.receipt['scene_count']==32 and req.receipt['record_count']==294
        c=TeacherFeatureContract.from_dict(req.receipt['contract'])
        scenes=list(req.manifest['scenes']);guard()
        scene_keys={s['token']:list(dict.fromkeys(payload_key(v,c) for f in s['frames'] for v in f['views'] if v is not None)) for s in scenes}
        keys=set(k for row in scene_keys.values() for k in row)
        hits={k for k in keys if k in src.locations};misses=keys-hits
        resident=OrderedDict();rows=[]
        def remember(key):
            if key in resident:resident.move_to_end(key);return 0
            n=0
            while len(resident)>=8:resident.popitem(last=False);n+=1
            resident[key]=None;return n
        def simulate(token,case):
            stats=dict(ram_hits=0,immutable_hits=0,computed_images=0,evictions=0)
            missing=[]
            for key in scene_keys[token]:
                if key in resident:
                    stats['ram_hits']+=1;resident.move_to_end(key);stats['evictions']+=remember(key)
                elif key in hits:
                    stats['immutable_hits']+=1;stats['evictions']+=remember(key)
                else:missing.append(key)
            # Provider collects all misses before executor batches; delivery follows requested order.
            for key in missing:stats['computed_images']+=1;stats['evictions']+=remember(key)
            rows.append({'scene':token,'case':case,'unique_keys':len(scene_keys[token]),'source_misses':sum(k in misses for k in scene_keys[token]),**stats})
            guard()
        for s in scenes:
            resident.clear();simulate(s['token'],'cold');simulate(s['token'],'partial_lru')
            resident.clear();simulate(s['token'],'cold_restored')
        resident.clear()
        for s in reversed(scenes):simulate(s['token'],'reordered')
        out={'status':'READ_ONLY_METADATA_SIMULATION_COMPLETE','scope':'Not a teacher numerical qualification',
            'request_index_identity':req.receipt['identity'],'source_index_identity':src.receipt['identity'],
            'scenes':len(scenes),'unique_keys':len(keys),'hit_keys':len(hits),'miss_keys':len(misses),
            'source_misses_per_scene':[sum(k in misses for k in scene_keys[s['token']]) for s in scenes],
            'partial_read_count':sum(r['case']=='partial_lru' and r['ram_hits']>0 and r['computed_images']>0 for r in rows),
            'mixed_read_count':sum(r['immutable_hits']>0 and r['computed_images']>0 for r in rows),'rows':rows,
            'tensor_image_model_payload_opens':0,'gpu_discovery_or_execution':False,
            'elapsed_seconds':time.monotonic()-started,'logical_read_rchar_bytes':rchar()-read_start,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        guard()
    finally:
        for index in indices:index.close()
p=Path(sys.argv[1]);p.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
