"""Real trained-reference-head roundtrip, target independence and gate-zero parity."""
import argparse,io,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from tools.structured_world.runtime import load_baseline,load_dataset,seed_all
from tools.structured_world_v1p1.budget import start,record
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy


def main():
 p=argparse.ArgumentParser()
 for k in ['trained','output','ledger','run-id']:p.add_argument('--'+k,required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 payload=Path(a.trained).read_bytes();saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);args=saved['identity']['arguments'];cfg=dict(saved['identity']['config'],action_adapter=True)
 code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();start(a.ledger,a.run_id,0,{'code_sha':code,'trained_sha256':hashlib.sha256(payload).hexdigest(),'kind':'no_update_regression'})
 try:
  seed_all(42);agent=load_baseline(args['checkpoint'],args['vlm']);agent.model.requires_grad_(False);policy=StructuredWorldPolicy(agent.model,cfg).cuda().eval().requires_grad_(False)
  for name in ['reader','heads']:getattr(policy,name).load_state_dict(saved[name],strict=True)
  ds=load_dataset(agent,args['manifest'],args['data_root'],1);raw=ds[0];e={k:raw[k] for k in ['image','lang','state','token']}
  with torch.no_grad():
   seed_all(51);native=agent.model.predict_action_infer_1d([e])['normalized_actions']
   seed_all(51);before=policy.predict_action([e])
   dirty=dict(e,action=np.full((8,4),1e9),WorldTargets={'current_boxes':torch.full((100,8),float('nan')),'future_file':'/nonexistent/future.png'})
   seed_all(51);poisoned=policy.predict_action([dirty])
  delta={name:getattr(policy,name).state_dict() for name in ['reader','heads','adapter']}
  torch.save({'config':cfg,'delta':delta},out/'world_roundtrip.pt')
  loaded=torch.load(out/'world_roundtrip.pt',map_location='cpu',weights_only=True)
  seed_all(19);restored=StructuredWorldPolicy(agent.model,loaded['config']).cuda().eval().requires_grad_(False)
  for name,state in loaded['delta'].items():getattr(restored,name).load_state_dict(state,strict=True)
  with torch.no_grad():seed_all(51);after=restored.predict_action([e])
  result={'code_sha':code,'trained_sha256':hashlib.sha256(payload).hexdigest(),'trained_step':saved['step'],'token':e['token'],'optimizer_updates':0,'gate0_native_exact':bool(np.array_equal(native,before['normalized_actions'])),'action_roundtrip_exact':bool(np.array_equal(before['normalized_actions'],after['normalized_actions'])),'target_independence_exact':bool(np.array_equal(before['normalized_actions'],poisoned['normalized_actions'])),'world_prediction_roundtrip_exact':all(torch.equal(v,after['world_prediction'][k]) for k,v in before['world_prediction'].items()),'world_target_independence_exact':all(torch.equal(v,poisoned['world_prediction'][k]) for k,v in before['world_prediction'].items()),'critical_load':'strict=True separately for reader, heads including reference buffers, and new adapter','adapter_initialization':'Explicit new untrained zero gate added after loading trained world modules; original frozen policy reloaded from base'}
  result['status']='PASS' if all(result[k] for k in ['gate0_native_exact','action_roundtrip_exact','target_independence_exact','world_prediction_roundtrip_exact','world_target_independence_exact']) else 'FAIL'
  (out/'CHECKPOINT_REGRESSION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);record(a.ledger,a.run_id,0,'complete')
 except BaseException:
  record(a.ledger,a.run_id,0,'failed');raise
if __name__=='__main__':main()
