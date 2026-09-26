"""Real FP32 intermediate parity; override legacy hard-coded AMP in this test process only."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from runtime import load_baseline,load_dataset,seed_all
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy


def main():
 p=argparse.ArgumentParser()
 for n in ['checkpoint','vlm','data-root','manifest','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();seed_all(42);agent=load_baseline(a.checkpoint,a.vlm);agent.model.float().requires_grad_(False)
 ds=load_dataset(agent,a.manifest,a.data_root,1);example=ds[0];policy=StructuredWorldPolicy(agent.model,{'enabled':False,'agent_tokens':64}).cuda().eval()
 native_autocast=torch.autocast
 def full_precision(device_type,*args,**kwargs):
  if device_type=='cuda':kwargs['enabled']=False
  return native_autocast(device_type,*args,**kwargs)
 torch.autocast=full_precision
 original=agent.model.action_model.predict_action;captured=[]
 def capture(x,*args,**kwargs):captured.append(x.detach().clone());return original(x,*args,**kwargs)
 agent.model.action_model.predict_action=capture
 with torch.inference_mode():
  seed_all(7);baseline=agent.predict([example])['normalized_actions'];legacy=captured[-1]
  condition,_=policy.encode_conditions([example],include_world=False)
  assert condition.dtype==legacy.dtype==torch.float32
  torch.testing.assert_close(condition,legacy,rtol=1e-5,atol=1e-5)
  seed_all(7);disabled=policy.predict_action([example])['normalized_actions'];np.testing.assert_array_equal(disabled,baseline)
 torch.autocast=native_autocast
 report={'status':'PASS','token':example['token'],'compute':'FP32, nested legacy CUDA autocast disabled only in isolated test process','intermediate_max_abs':float((condition-legacy).abs().max()),'intermediate_atol':1e-5,'intermediate_rtol':1e-5,'world_disabled_prediction_exact':True}
 Path(a.output).write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
