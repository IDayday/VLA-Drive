"""Online verified external prior → Reader → Qwen → gated original action path."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from tools.structured_world.runtime import load_baseline,load_dataset,load_world_batch,seed_all
from tools.structured_world_v1p1.budget import start,record
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy


def main():
 p=argparse.ArgumentParser()
 for k in ['probe','baseline','vlm','data-root','manifest','observations','config','output','ledger','run-id']:p.add_argument('--'+k,required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 start(a.ledger,a.run_id,0,{'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'arguments':vars(a)})
 try:
  seed_all(42);agent=load_baseline(a.baseline,a.vlm);agent.model.requires_grad_(False);cfg=json.loads(Path(a.config).read_text());policy=StructuredWorldPolicy(agent.model,cfg).cuda().eval().requires_grad_(False)
  trained=torch.load(a.probe,map_location='cpu',weights_only=False)
  for name in ['reader','heads']:getattr(policy,name).load_state_dict({k[len(name)+1:]:v for k,v in trained['model'].items() if k.startswith(name+'.')},strict=True)
  ds=load_dataset(agent,a.manifest,a.data_root,1);raw=ds[0];e={k:raw[k] for k in ['image','lang','state','token']};inputs,_=load_world_batch([e],a.observations,load_targets=False)
  hidden=[]
  h=policy.heads.register_forward_pre_hook(lambda m,args:hidden.append(args[0].detach().clone()))
  with torch.no_grad():
   seed_all(9);native=agent.model.predict_action_infer_1d([e])['normalized_actions']
   torch.cuda.synchronize();begin=time.perf_counter();seed_all(9);closed=policy.predict_action([e],inputs);torch.cuda.synchronize();latency=time.perf_counter()-begin
   real=hidden[-1]
   def blank(m,args,result):
    features,coordinates,support,metadata=result
    return torch.zeros_like(features),coordinates,support,metadata
   handle=policy.provider.register_forward_hook(blank)
   seed_all(9);zero=policy.predict_action([e],inputs);blank_hidden=hidden[-1];handle.remove()
   policy.adapter.gate.fill_(.1);seed_all(9);opened=policy.predict_action([e],inputs)
  h.remove()
  result={'scene':e['token'],'provider_metadata':policy.provider.metadata(),'all_provider_parameters_frozen':all(not p.requires_grad for p in policy.provider.parameters()),'gate0_native_exact':bool(np.array_equal(native,closed['normalized_actions'])),'post_qwen_agent_hidden_change_on_zero_features':float((real-blank_hidden).abs().max()),'open_gate_action_change':float(np.abs(opened['normalized_actions']-closed['normalized_actions']).max()),'full_online_policy_seconds':latency,'target_loading':False,'provider_probe_steps':trained['steps'],'optimizer_updates':0,'interpretation':'Connectivity and native-policy fidelity only. Feature blanking/opening gate are OOD diagnostics, not planning effectiveness or causal proof.'}
  result['status']='PASS' if result['gate0_native_exact'] and result['post_qwen_agent_hidden_change_on_zero_features']>0 and result['open_gate_action_change']>0 else 'FAIL'
  (out/'PRETRAINED_INJECTION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);record(a.ledger,a.run_id,0,'complete')
 except BaseException:
  record(a.ledger,a.run_id,0,'failed');raise
if __name__=='__main__':main()
