"""Real frozen Qwen + trained BEV checkpoint: deployment/cache/adapter contracts."""
import argparse,copy,hashlib,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch,yaml
from runtime import load_baseline,load_dataset,load_world_batch,seed_all
from evaluate import load_delta
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy
from starVLA.model.modules.structured_world.providers import ExternalBEVFeatures,calibration_fingerprint
from starVLA.model.modules.structured_world.action_adapter import WorldToActionAdapter


def main():
 p=argparse.ArgumentParser()
 for n in ['checkpoint','vlm','data-root','manifest','target-cache','delta','config','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();seed_all(42);agent=load_baseline(a.checkpoint,a.vlm);agent.model.requires_grad_(False)
 policy=StructuredWorldPolicy(agent.model,yaml.safe_load(Path(a.config).read_text())).cuda().eval();saved=load_delta(policy,a.delta);policy.requires_grad_(False)
 ds=load_dataset(agent,a.manifest,a.data_root,2);raw=[ds[i] for i in range(2)];examples=[{k:e[k] for k in ['image','lang','state','token']} for e in raw]
 # Build a deployment observation directory containing no target/future files.
 import tempfile,shutil
 with tempfile.TemporaryDirectory() as directory:
  observations=Path(directory)/'observations';observations.mkdir()
  token=examples[0]['token'];shutil.copy2(Path(a.target_cache)/'observations'/(token+'.npz'),observations/(token+'.npz'))
  inputs,_=load_world_batch(examples[:1],directory,load_targets=False)
 provider=policy.provider;provider.bind_cache_identity()
 from unittest.mock import patch
 bad=dict(saved);key=next(k for k in saved['delta'] if k.startswith('heads.'))
 bad['delta']={k:v for k,v in saved['delta'].items() if k!=key};bad['delta_keys']=sorted(bad['delta'])
 with patch('torch.load',return_value=bad):
  try:load_delta(policy,'unused.pt')
  except ValueError:pass
  else:raise AssertionError('Missing world-head weight accepted')
 bad=dict(saved);bad['delta']=dict(saved['delta'],unexpected_world_weight=torch.zeros(1));bad['delta_keys']=sorted(bad['delta'])
 with patch('torch.load',return_value=bad):
  try:load_delta(policy,'unused.pt')
  except ValueError:pass
  else:raise AssertionError('Unexpected weight accepted')
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):f,xyz,support,meta=provider(inputs)
 import starVLA.model.modules.structured_world.providers as module
 meta=dict(meta,scene_token=inputs.scene_tokens[0],decision_time=int(inputs.decision_time[0]),provider_weights_sha256=provider.cache_weight_identity,provider_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),input_tensor_sha256=hashlib.sha256(inputs.current_images.cpu().numpy().tobytes()).hexdigest(),image_sha256=[],image_transforms=inputs.image_transforms.cpu().tolist(),dtype=str(f.dtype),calibration_sha256=calibration_fingerprint(inputs))
 payload={'metadata':meta,'features':f.cpu(),'coordinates':xyz.cpu(),'observation_support':support.cpu()};cached=replace(inputs,optional_current_feature_cache=payload)
 seed_all(7);online=policy.predict_action(examples[:1],inputs)
 seed_all(7);offline=policy.predict_action(examples[:1],cached)
 np.testing.assert_array_equal(online['normalized_actions'],offline['normalized_actions'])
 dynamic={'scene_token','decision_time','input_tensor_sha256','image_sha256','image_transforms','calibration_sha256'}
 policy.provider=ExternalBEVFeatures({k:v for k,v in meta.items() if k not in dynamic})
 seed_all(7);external=policy.predict_action(examples[:1],cached)
 np.testing.assert_array_equal(online['normalized_actions'],external['normalized_actions'])
 policy.provider=provider
 rejected=[]
 for name,changed in [('future_time',replace(cached,timestamps=cached.timestamps+1)),('calibration',replace(cached,camera_intrinsics=cached.camera_intrinsics+1))]:
  try:
   with torch.autocast('cuda',dtype=torch.bfloat16):provider(changed)
  except ValueError:rejected.append(name)
  else:raise AssertionError(name)
 for key,value in [('schema_version',999),('sensor_contract',{}),('scene_token','wrong')]:
  bad=copy.deepcopy(payload);bad['metadata'][key]=value
  try:
   with torch.autocast('cuda',dtype=torch.bfloat16):provider(replace(inputs,optional_current_feature_cache=bad))
  except ValueError:rejected.append(key)
  else:raise AssertionError(key)
 # Repeat alignment uses two different real scenes and the actual policy forward.
 batch_inputs,_=load_world_batch(examples,a.target_cache,load_targets=False)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):conditions,_=policy.encode_conditions(examples,batch_inputs)
 captured=[]
 original=policy.baseline.action_model.forward
 def capture(c,acts,*args,**kwargs):captured.append((c.detach().clone(),acts.detach().clone()));return c.sum()*0
 policy.baseline.action_model.forward=capture
 policy(raw,batch_inputs,None)
 policy.baseline.action_model.forward=original
 repeats=policy.baseline.config.framework.action_model.get('repeated_diffusion_steps',1)
 torch.testing.assert_close(captured[0][0],conditions.repeat(repeats,1,1),atol=0,rtol=0)
 torch.testing.assert_close(captured[0][1],torch.tensor(np.array([e['action'] for e in raw]),device='cuda',dtype=torch.float32).repeat(repeats,1,1),atol=0,rtol=0)
 # No optimizer step: verify gate-first then opened-branch gradients with real Qwen conditions.
 adapter=WorldToActionAdapter(conditions.shape[-1]).cuda();world=conditions.detach().float();action=world.clone()
 loss=adapter(action,world).square().mean();loss.backward();gate=float(adapter.gate.grad.abs());assert gate>0
 adapter.zero_grad(set_to_none=True)
 with torch.no_grad():adapter.gate.fill_(.01)
 adapter(action,world).square().mean().backward();branch=sum(float(p.grad.square().sum()) for n,p in adapter.named_parameters() if n!='gate' and p.grad is not None)**.5;assert branch>0
 import time
 timings=[]
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for i in range(23):
   torch.cuda.synchronize();start=time.perf_counter();provider(inputs);torch.cuda.synchronize()
   if i>=3:timings.append(time.perf_counter()-start)
 report={'provider_latency_seconds':{'mean':float(np.mean(timings)),'median':float(np.median(timings)),'measurements':20,'scope':'online provider including CNN/calibration/projection/fusion, current RGB tensors already on GPU; BF16'},'status':'PASS','online_cached_external_action_max_abs':0,'current_only_without_target_loading':True,'target_directory_physically_absent':True,'missing_unexpected_weights_rejected':True,'rejected':rejected,'real_two_scene_repeat_alignment':True,'repeats':repeats,'zero_gate_gradient':gate,'opened_adapter_branch_gradient':branch,'provider_metadata':meta}
 Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
