"""Optional profiler scopes and CUDA event intervals. Disabled in throughput runs."""
from collections import defaultdict
from contextlib import contextmanager
import functools
import threading
import time
import torch
from ..io import atomic_json


def shapes(value, depth=0):
    if torch.is_tensor(value):
        return {'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device),'requires_grad':value.requires_grad}
    if depth > 2: return type(value).__name__
    if isinstance(value,dict): return {k:shapes(v,depth+1) for k,v in value.items() if k not in {'past_key_values'}}
    if isinstance(value,(list,tuple)): return [shapes(v,depth+1) for v in value[:8]]
    if hasattr(value,'last_hidden_state'): return shapes(value.last_hidden_state,depth+1)
    return type(value).__name__

class Instrumentation:
    def __init__(self, enabled, device):
        self.enabled,self.device=enabled,device
        self.calls=defaultdict(list);self.intervals=[];self.restores=[]
        self.start=torch.cuda.Event(enable_timing=True) if enabled else None
        if enabled: self.start.record();self.start.synchronize()

    @contextmanager
    def region(self,name):
        if not self.enabled:
            yield;return
        cuda=torch.cuda.current_stream(self.device)
        first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin=time.perf_counter(); first.record(cuda)
        with torch.profiler.record_function('iqe/'+name):
            try: yield
            finally:
                last.record(cuda)
                self.intervals.append((name,first,last,time.perf_counter()-begin,threading.get_ident()))

    def wrap(self,obj,attribute,name):
        original=getattr(obj,attribute)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            if not self.enabled:return original(*args,**kwargs)
            inp=shapes((args,kwargs))
            with self.region(name): out=original(*args,**kwargs)
            self.calls[name].append({'input':inp,'output':shapes(out),'grad_enabled':torch.is_grad_enabled()})
            return out
        setattr(obj,attribute,wrapped);self.restores.append((obj,attribute,original))

    def attach(self,model):
        if not self.enabled:return
        q=model.qwen_vl_interface.model.model
        for obj,attr,name in [(model,'_build_qwen_batch','build_qwen_batch'),(model,'_qwen_language_forward','deepstack_injection_and_language'),
          (q.visual,'forward','frozen_visual'),(q.language_model,'forward','language_forward'),
          (model.action_model,'forward','query_IL_branch'),(model.action_model.q_former,'forward','query_q_former'),
          (model.action_model.expert,'forward','query_decoder_and_heads'),
          (model,'compute_clip_loss','future_clip_loss_and_H2D'),
          (model.qwen_vl_interface,'build_qwenvl_inputs','processor_tokenizer_H2D')]: self.wrap(obj,attr,name)
        for name in ['dino_head','spatiotemporal_head','interaction_head']:
            if hasattr(model,name): self.wrap(getattr(model,name),'forward',name)
        # Layer scopes distinguish the original forward from recomputation under backward.
        for i,layer in enumerate(q.language_model.layers): self.wrap(layer,'forward',f'language_layer_{i}')
        for i,merger in enumerate(getattr(q.visual,'deepstack_merger_list',[])): self.wrap(merger,'forward',f'visual_deepstack_merger_{i}')
        self.wrap(q.visual.merger,'forward','visual_merger')
        self.inventory={}
        for name,module in [('visual',q.visual),('language',q.language_model),('query',model.action_model),('current',model.dino_head),('future',model.spatiotemporal_head),('interaction',model.interaction_head)]:
            counts=defaultdict(int)
            for p in module.parameters():counts[f'{"trainable" if p.requires_grad else "frozen"}/{p.dtype}']+=p.numel()
            self.inventory[name]=dict(counts)
        self.inventory['attention_implementation']=str(getattr(q.language_model.config,'_attn_implementation',None))

    def finish(self,path,model,optimizer):
        if not self.enabled:return
        torch.cuda.synchronize(self.device)
        intervals=[{'name':n,'start_ms':self.start.elapsed_time(s),'end_ms':self.start.elapsed_time(e),'inclusive_cuda_ms':s.elapsed_time(e),'host_seconds':h,'thread':t} for n,s,e,h,t in self.intervals]
        kinds=defaultdict(int)
        for name,p in model.named_parameters():
            if p.requires_grad:
                kinds[f'parameter/{p.dtype}']+=p.numel()
                if p.grad is not None:kinds[f'gradient/{p.grad.dtype}']+=p.grad.numel()
        for state in optimizer.state.values():
            for key,v in state.items():
                if torch.is_tensor(v): kinds[f'optimizer/{key}/{v.dtype}']+=v.numel()
        atomic_json(path,{'modules':getattr(self,'inventory',{}),'dtypes':dict(kinds),'calls':dict(self.calls),'intervals':intervals,
            'warning':'inclusive CUDA intervals across concurrent streams and nested scopes are not additive; use trace critical path',
            'optimizer_defaults':{k:v for k,v in optimizer.defaults.items() if isinstance(v,(str,int,float,bool,type(None),tuple))}})
