"""Small, immutable, sharded frozen-vision payloads. No trainable embeddings or GT.

The entrypoint is the exact whitelist-only _build_qwen_batch boundary. Cached
ids/positions/visual tensors precede embedding lookup/history/query substitution.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,time
import torch
from ..contracts import require
from ..io import atomic_json,atomic_torch,read_json,file_hash,digest

@dataclass
class FrozenVisionPayload:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    slots: dict
    image_embeddings: torch.Tensor
    deepstack: tuple
    image_grid_thw: torch.Tensor

    def validate(self,slot_names):
        require(self.input_ids.ndim==1 and self.input_ids.dtype==torch.long,'payload token schema')
        require(self.position_ids.shape==(3,len(self.input_ids)),'payload MRoPE shape')
        require(set(self.slots)==set(slot_names) and 'foresight' in self.slots,'complete token schema including foresight required')
        require(self.image_grid_thw.shape==(3,3),'payload three-view grid required')
        require(self.image_embeddings.ndim==2 and len(self.deepstack)==3,'locked Qwen payload requires all three DeepStack outputs')
        for v in (self.image_embeddings,*self.deepstack):
            require(v.shape==self.image_embeddings.shape and not v.requires_grad and not torch.is_inference(v),'frozen regular tensors required')
        for v in self.slots.values():require(v.dtype==torch.long and bool(((v>=0)&(v<len(self.input_ids))).all()),'payload slot range')
        return self


def observation_key(observation,instruction,identity):
    require(set(observation)=={'image','lang','state'},'cache boundary retains deployment whitelist')
    require(len(observation['image'])==3,'three-view identity')
    h=hashlib.sha256(digest(identity).encode());h.update(instruction.encode())
    for im in observation['image']:
        h.update(str((im.mode,im.size)).encode());h.update(im.tobytes())
    # Ego state is intentionally not cached: its trainable embedding is computed every step.
    return h.hexdigest()


def combine_payloads(payloads,pad_token_id,device,slot_names):
    for p in payloads:p.validate(slot_names)
    size=max(len(p.input_ids) for p in payloads);batch=len(payloads)
    ids=torch.full((batch,size),pad_token_id,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids);positions=torch.ones((3,batch,size),dtype=torch.long,device=device)
    slots={name:[] for name in slot_names}
    for i,p in enumerate(payloads):
        offset=size-len(p.input_ids);ids[i,offset:]=p.input_ids.to(device);mask[i,offset:]=1
        positions[:,i,offset:]=p.position_ids.to(device)
        for n,v in p.slots.items():slots[n].append(v.to(device)+offset)
    return ids,mask,positions,{n:torch.stack(v) for n,v in slots.items()},torch.cat([p.image_embeddings for p in payloads]).to(device),[torch.cat([p.deepstack[i] for p in payloads]).to(device) for i in range(len(payloads[0].deepstack))]


class FrozenVisionCache:
    def __init__(self,model,root,identity,rank,*,build=False,max_scenes=64):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.identity=identity;self.rank=rank;self.build=build;self.max_scenes=max_scenes
        self.original=model._build_qwen_batch;self.model=model;self.index={};self.loaded={};self.timings=[]
        self.keys=set();self.bytes=0
        if build:
            require(not (self.root/'COMPLETE.json').exists(),'immutable cache already committed')
            self.grid=None
            visual=model.qwen_vl_interface.model.model.visual
            def capture_grid(module,args,kwargs):self.grid=(kwargs.get('grid_thw',args[1] if len(args)>1 else None)).detach().cpu().clone()
            self.hook=visual.register_forward_pre_hook(capture_grid,with_kwargs=True)
        else:
            receipt=read_json(self.root/'COMPLETE.json');require(receipt['identity']==identity,'visual cache execution identity mismatch')
            for name,checksum in receipt['rank_indexes'].items():
                require(file_hash(self.root/name)==checksum,'cache index checksum mismatch')
                for key,value in read_json(self.root/name)['index'].items():
                    require(key not in self.index or self.index[key]==value,'duplicate conflicting observation cache')
                    self.index[key]=value
        model._build_qwen_batch=self.consume

    def consume(self,observations,instructions):
        begin=time.perf_counter();keys=[observation_key(o,t,self.identity) for o,t in zip(observations,instructions)]
        if self.build:
            value=self.original(observations,instructions)
            ids,mask,positions,slots,visual,deep=value
            require(self.grid is not None and self.grid.shape==(len(observations)*3,3),'visual grid capture unavailable')
            payloads=[];offset=0
            for i,key in enumerate(keys):
                length=int(mask[i].sum());start=ids.shape[1]-length
                count=int(ids[i].eq(self.model.qwen_vl_interface.model.config.image_token_id).sum())
                payload=FrozenVisionPayload(ids[i,start:].detach().cpu().clone(),positions[:,i,start:].detach().cpu().clone(),
                    {n:(v[i]-start).detach().cpu().clone() for n,v in slots.items()},visual[offset:offset+count].detach().cpu().clone(),
                    tuple(v[offset:offset+count].detach().cpu().clone() for v in deep),self.grid[i*3:i*3+3].clone()).validate(slots)
                require(all(bool(torch.isfinite(v).all()) for v in (payload.image_embeddings,*payload.deepstack)), 'nonfinite frozen visual payload')
                offset+=count;payloads.append(payload)
            require(offset==len(visual),'visual packing count mismatch')
            name=f'rank{self.rank}_'+digest(keys)+'.pt';path=self.root/name
            atomic_torch(path,[p.__dict__ for p in payloads],immutable=True);checksum=file_hash(path)
            for i,key in enumerate(keys):self.index[key]={'shard':name,'index':i,'checksum':checksum}
            self.keys.update(keys);require(len(self.keys)<=self.max_scenes,'pilot cache budget exceeded')
            self.bytes+=path.stat().st_size
            self.timings.append({'kind':'build','seconds':time.perf_counter()-begin,'scenes':len(keys),'bytes':path.stat().st_size})
            return value
        payloads=[]
        for key in keys:
            require(key in self.index,'CACHE_MISS: pilot has no permission to silently build a full cache')
            item=self.index[key];name=item['shard']
            if name not in self.loaded:
                require(file_hash(self.root/name)==item['checksum'],'cache shard checksum mismatch')
                self.loaded[name]=torch.load(self.root/name,map_location='cpu',mmap=True,weights_only=True)
            payloads.append(FrozenVisionPayload(**self.loaded[name][item['index']]))
        value=combine_payloads(payloads,self.model.qwen_vl_interface.processor.tokenizer.pad_token_id,
            next(self.model.parameters()).device,self.model._special_token_ids)
        self.timings.append({'kind':'read','seconds':time.perf_counter()-begin,'scenes':len(keys)})
        return value

    def finish(self):
        self.model._build_qwen_batch=self.original
        if self.build:
            self.hook.remove()
            atomic_json(self.root/f'rank_{self.rank}.json',{'identity':self.identity,'index':self.index,'timings':self.timings,'bytes':self.bytes},immutable=True)
