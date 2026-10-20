"""Bounded CPU shard reader and strict deployment-context whitelist."""
from collections import OrderedDict
from pathlib import Path
import torch
from .common import verify_artifact,verify_identity,read_json
from .collect_base_trajectories import validate_context_manifest


class ContextReader:
    def __init__(self,manifest,binding_path,max_shards=2):
        if max_shards<1:raise ValueError('Explicit bounded LRU required')
        self.binding=verify_identity(read_json(binding_path));self.manifest=verify_identity(read_json(manifest))
        cohort=verify_identity(read_json(verify_artifact(self.binding['cohort'])))
        expected=cohort['entries'][:self.manifest['max_samples']]
        validate_context_manifest(self.manifest,self.binding,expected)
        self.rows={r['scene_id']:r for r in self.manifest['entries']};self.cache=OrderedDict();self.max_shards=max_shards
        self.checked={};self.loads=0

    def __getitem__(self,scene_id):
        r=self.rows[scene_id];p=Path(r['shard']['path']);st=p.stat();stamp=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        if self.checked.get(str(p))!=stamp:
            verify_artifact(r['shard']);self.checked[str(p)]=stamp;self.cache.pop(str(p),None)
        key=str(p)
        if key not in self.cache:
            self.cache[key]=torch.load(p,map_location='cpu',weights_only=False);self.loads+=1
            while len(self.cache)>self.max_shards:self.cache.popitem(last=False)
        self.cache.move_to_end(key);x=self.cache[key][r['shard_offset']]
        if any(x[k]!=r[k] for k in ('scene_id','source_log','role','observation_hash','raw_hash','physical_hash')):
            raise ValueError('Payload/manifest identity mismatch')
        if x['base_binding_hash']!=self.binding['base_checkpoint_binding_hash']:raise ValueError('Payload Base mismatch')
        for k in ('memory','ego','condition','raw','physical'):
            if not x[k].is_floating_point() or not torch.isfinite(x[k]).all() or x[k].requires_grad:
                raise ValueError('Invalid frozen payload '+k)
        return x


def batch_context(records,device):
    # Never forward target_raw, trajectories, feedback or metric-cache references.
    memory=torch.stack([x['memory'] for x in records]).to(device).detach()
    ego=torch.stack([x['ego'] for x in records]).to(device).detach()
    return {'memory':memory,'ego':ego,'mask':torch.ones(memory.shape[:2],dtype=torch.bool,device=device)}
