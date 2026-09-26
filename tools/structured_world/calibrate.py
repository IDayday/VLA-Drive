"""One training-only shared-query gradient calibration, with no optimizer steps."""
import argparse,json
from pathlib import Path
import numpy as np
import torch,yaml
from runtime import load_baseline,load_dataset,load_world_batch,seed_all
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy
from starVLA.model.modules.structured_world.losses import world_losses


def main():
 p=argparse.ArgumentParser()
 for n in ['checkpoint','vlm','data-root','manifest','target-cache','config','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();seed_all(42);agent=load_baseline(a.checkpoint,a.vlm);agent.model.requires_grad_(False)
 policy=StructuredWorldPolicy(agent.model,yaml.safe_load(Path(a.config).read_text())).cuda().eval();ds=load_dataset(agent,a.manifest,a.data_root,2);records=[]
 for i in range(2):
  e=ds[i];inputs,targets=load_world_batch([e],a.target_cache)
  with torch.autocast('cuda',dtype=torch.bfloat16):condition,pred=policy.encode_conditions([e],inputs)
  gt=torch.as_tensor(np.array([e['action']]),device='cuda',dtype=torch.float32)
  repeats=agent.model.config.framework.action_model.get('repeated_diffusion_steps',1)
  with torch.autocast('cuda',dtype=torch.float32):ego=agent.model.action_model(condition.repeat(repeats,1,1),gt.repeat(repeats,1,1),None)
  terms,_=world_losses(pred,targets);terms['ego']=ego;grads={}
  for name,value in terms.items():
   gradient=torch.autograd.grad(value,policy.reader.queries,retain_graph=True,allow_unused=True)[0]
   grads[name]=float(gradient.float().norm()) if gradient is not None else 0.
  records.append({'token':e['token'],'losses':{k:float(v.detach()) for k,v in terms.items()},'shared_query_gradient_norms':grads})
 med={k:float(np.median([r['shared_query_gradient_norms'][k] for r in records])) for k in ['ego','cls','box','motion']}
 weights={k:float(np.clip(.1*med['ego']/max(med[k],1e-8),.1,10.)) for k in ['cls','box','motion']}
 Path(a.output).write_text(json.dumps({'records':records,'median_gradient_norms':med,'weights':weights,'rule':'Each auxiliary aims at 0.1x median ego shared-query gradient, clipped [0.1,10]; one training-only calibration; no test selection','optimizer_steps':0},indent=2));print(weights)
if __name__=='__main__':main()
