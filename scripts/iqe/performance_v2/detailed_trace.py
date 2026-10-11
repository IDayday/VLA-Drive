"""Trace-only extra CPU/image/tokenizer and reduction scopes; never used for throughput."""
import functools,importlib,os,sys,time
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image,ImageFile
from iqe.performance import run as runtime
from iqe.performance.instrumentation import Instrumentation
from iqe.io import atomic_json,file_hash
assert '--trace' in sys.argv,'this wrapper is only for explicitly profiled <=8 updates'
root=Path(sys.argv[sys.argv.index('--output')+1]);cpu=defaultdict(list)

def cpu_wrap(obj,attr,label):
 original=getattr(obj,attr)
 @functools.wraps(original)
 def call(*args,**kwargs):
  start=time.perf_counter()
  with torch.profiler.record_function('iqe_cpu/'+label):result=original(*args,**kwargs)
  cpu[label].append(time.perf_counter()-start)
  return result
 setattr(obj,attr,call)

cpu_wrap(Image,'open','image_open');cpu_wrap(ImageFile.ImageFile,'load','image_decode');cpu_wrap(Image.Image,'resize','image_resize')
attach=Instrumentation.attach

def detailed(self,model):
 attach(self,model)
 language=model.qwen_vl_interface.model.model.language_model
 self.wrap(language,'_deepstack_process','deepstack_injection')
 tokenizer=model.qwen_vl_interface.processor.tokenizer
 cpu_wrap(type(tokenizer),'__call__','tokenizer_call')
 self.wrap(torch.nn.utils,'clip_grad_norm_','gradient_clip_only')
 self.wrap(runtime,'finite_gradients','finite_gradient_check_only')
 self.wrap(importlib.import_module('iqe.query_base'),'expert_il_terms','query_IL_reduction')
 source=importlib.import_module('starVLA.model.framework.DDPForesight')
 self.wrap(source,'masked_regression','current_DINO_reduction')
 self.wrap(source,'interaction_loss','interaction_reduction')
 source=importlib.import_module('starVLA.model.framework.ddp_action_video_foresight')
 self.wrap(source,'normalized_clip_loss','future_clip_reduction')
Instrumentation.attach=detailed
runtime.main()
atomic_json(root/f'CPU_DETAIL_rank{os.environ["RANK"]}.json',{'wrapper_sha256':file_hash(__file__),'cpu_region_seconds':dict(cpu),'note':'nested/parallel CPU scopes are not additive; trace only, not throughput'})
