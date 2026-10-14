"""Capture final merged visual tokens from the SAME unmodified VLA forward.

The capture wraps a return boundary, returns the original objects unchanged, and
never adds a VLM token or invokes an auxiliary future head. Spatial tensors are
an optional scorer-only side channel; the planner path is unchanged.
"""
from pathlib import Path
from collections import OrderedDict
import time
import torch
from iqe.contracts import require
from iqe.io import read_json, atomic_json, atomic_torch, file_hash, digest
from iqe.data.sources import load_observation
from iqe.internal_baseline.features import encode_joint
from .binding import scenes_by_role
from .data import ReusedContext


class SharedCapture:
    def __init__(self,adapter):
        self.adapter=adapter;self.model=adapter.framework;self.original=self.model._build_qwen_batch
        self.grid=None;self.payload=None;self.count={'visual':0,'language':0}
        visual=self.model.qwen_vl_interface.model.model.visual
        def pre(module,args,kwargs):
            self.grid=kwargs.get('grid_thw',args[1] if len(args)>1 else None).detach().clone();self.count['visual']+=1
        def language(*_):self.count['language']+=1
        self.handles=[visual.register_forward_pre_hook(pre,with_kwargs=True),
             self.model.qwen_vl_interface.model.model.language_model.register_forward_hook(language)]
        def capture(*args,**kwargs):
            value=self.original(*args,**kwargs)
            self.payload=value[4].detach()
            return value
        self.model._build_qwen_batch=capture
        self.merge=int(visual.spatial_merge_size)
        self.views=tuple(adapter.contract['input']['views']);self.view_count=len(self.views)
    def close(self):
        self.model._build_qwen_batch=self.original
        for h in self.handles:h.remove()
    @torch.no_grad()
    def encode(self,observations):
        self.grid=None;self.payload=None;self.count={'visual':0,'language':0}
        planner,context=encode_joint(self.adapter,observations)
        require(self.count=={'visual':1,'language':1},'shared VLA was called more/less than once')
        require(self.grid.shape==(len(observations)*self.view_count,3),'actual view layout changed')
        tokens=[];offset=0
        for grid in self.grid:
            t,h,w=map(int,grid.tolist());require(t==1 and h%self.merge==0 and w%self.merge==0,'current per-view spatial grid')
            h//=self.merge;w//=self.merge;n=h*w
            y,x=torch.meshgrid(torch.linspace(-1,1,h,device=self.payload.device),torch.linspace(-1,1,w,device=self.payload.device),indexing='ij')
            tokens.append((self.payload[offset:offset+n],torch.stack((x.flatten(),y.flatten()),-1)));offset+=n
        require(offset==len(self.payload),'visual token/grid packing mismatch')
        maxn=max(len(x[0]) for x in tokens);b=len(observations);d=self.payload.shape[-1]
        v=self.payload.new_zeros((b,self.view_count,maxn,d));xy=self.payload.new_zeros((b,self.view_count,maxn,2));mask=torch.zeros((b,self.view_count,maxn),device=v.device,dtype=torch.bool)
        for i,(x,p) in enumerate(tokens):
            row,view=divmod(i,self.view_count);v[row,view,:len(x)]=x;xy[row,view,:len(x)]=p;mask[row,view,:len(x)]=True
        values=context.tensors()|{'planner_scene':planner.scene,'planner_ego':planner.ego,'planner_mask':planner.valid_tokens,
                                 'visual':v,'visual_valid':mask,'visual_xy':xy,'visual_grid':self.grid.reshape(b,self.view_count,3)}
        return planner,values


def spatial_key(config,scene):
    b=read_json(config.root/'BASE_BINDING.json')
    return digest({'schema':'same_forward_final_merged_visual_v1','base':b['checkpoint_hash'],'runtime':b['inference_runtime'],
        'observation':scene.observation_hash,'input':b['contract_hash'],'code':file_hash(__file__),'views':read_json(config.root/'OFFLINE_BASE_CONTRACT.json')['input']['views']})


class SpatialCache:
    def __init__(self,config,capacity=32):self.config=config;self.capacity=capacity;self.loaded=OrderedDict()
    def path(self,scene):return self.config.root/'spatial_context'/(scene.scene_id+'.pt')
    def valid(self,scene):
        p=self.path(scene);r=Path(str(p)+'.COMPLETE.json')
        if not r.exists():return False
        x=read_json(r);require(x['key']==spatial_key(self.config,scene) and x['checksum']==file_hash(p),'spatial feature mismatch/corruption')
        return True
    def write(self,scene,values):
        p=self.path(scene)
        atomic_torch(p,{k:v.detach().cpu().contiguous().clone() for k,v in values.items()},immutable=True)
        atomic_json(str(p)+'.COMPLETE.json',{'key':spatial_key(self.config,scene),'checksum':file_hash(p),
                    'scene_id':scene.scene_id,'shapes':{k:list(v.shape) for k,v in values.items()},'label_inputs':False},immutable=True)
    def read(self,scene):
        if scene.scene_id in self.loaded:self.loaded.move_to_end(scene.scene_id);return self.loaded[scene.scene_id]
        require(self.valid(scene),'missing spatial context '+scene.scene_id)
        x=torch.load(self.path(scene),map_location='cpu',weights_only=True);self.loaded[scene.scene_id]=x
        while len(self.loaded)>self.capacity:self.loaded.popitem(last=False)
        return x
    def batch(self,scenes,device):
        rows=[self.read(s) for s in scenes];maxn=max(x['visual'].shape[1] for x in rows)
        padded=[]
        for x in rows:
            n=x['visual'].shape[1];row={}
            for key in ('visual','visual_valid','visual_xy'):
                shape=(x[key].shape[0],maxn,*x[key].shape[2:]);value=x[key].new_zeros(shape);value[:,:n]=x[key];row[key]=value
            padded.append(row)
        return {k:torch.stack([r[k] for r in padded]).to(device) for k in padded[0]}


def build(config,device='cuda:0',max_samples=None,shard=0,shards=1):
    require(0<=shard<shards<=3,'authorized bounded spatial cache shards')
    roles=scenes_by_role(config)
    scenes=roles['scorer_fit']+roles['stage_val']+roles['selector_cal']+roles['dev_report']
    scenes=sorted(scenes,key=lambda s:digest([config.selection_seed,'spatial',s.scene_id]))[shard::shards]
    if max_samples is not None:require(0<max_samples<=256,'spatial pilot max256');scenes=scenes[:max_samples]
    cache=SpatialCache(config);pending=[s for s in scenes if not cache.valid(s)]
    if not pending:return {'status':'COMPLETE','scenes':len(scenes),'reused':True}
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from iqe.s0_adapter import S0Adapter
    adapter=S0Adapter.load_and_validate_s0(read_json(config.root/'OFFLINE_BASE_CONTRACT.json'),device)
    require(not any(p.requires_grad for p in adapter.parameters()),'shared VLA not frozen')
    capture=SharedCapture(adapter);old=ReusedContext(config);start=time.perf_counter();count=0;errors=[]
    try:
        for at in range(0,len(pending),config.cache_batch):
            sub=pending[at:at+config.cache_batch];obs=[load_observation(s,adapter.contract) for s in sub]
            planner,values=capture.encode(obs)
            pred=adapter.predict_s0(planner)
            if count==0:
                # Labels cannot enter encode_joint's explicit observation whitelist.
                dirty=[dict(o,GT='shuffled_label',future_actors='excluded',metric_cache='excluded') for o in obs]
                p2,v2=capture.encode(dirty)
                require(all(torch.equal(values[k],v2[k]) for k in ('world','action','planner_scene','visual')), 'label changed score/planning context')
            for i,s in enumerate(sub):
                prior=old.read(s)
                err={'scene_id':s.scene_id,'raw_max_abs':float((pred.raw[i].cpu()-prior['base_raw']).abs().max()),
                     'world_max_abs':float((values['world'][i].cpu()-prior['world']).abs().max()),
                     'action_max_abs':float((values['action'][i].cpu()-prior['action']).abs().max())}
                # Batch regrouping is reported; canonical planner/W/A training cache is unchanged.
                errors.append(err)
                cache.write(s,{k:values[k][i] for k in ('visual','visual_valid','visual_xy','visual_grid')})
            count+=len(sub)
            atomic_json(config.root/f'spatial_worker_{shard}.json',{'status':'RUNNING','new_records':count,'pending':len(pending),
                    'wall_seconds':time.perf_counter()-start,'VLM_calls_per_batch':1,'visual_calls_per_batch':1})
    finally:capture.close()
    result={'status':'PILOT_COMPLETE' if max_samples else 'COMPLETE','new_records':count,'requested':len(scenes),
            'wall_seconds':time.perf_counter()-start,'numeric_comparison':errors,'labels_do_not_affect_context':True,
            'runtime':'unchanged locked Base','same_forward_visual_and_language':True}
    atomic_json(config.root/f'SPATIAL_CONTEXT_AUDIT_{shard}{"_pilot" if max_samples else ""}.json',result,immutable=True)
    return {k:v for k,v in result.items() if k!='numeric_comparison'}
