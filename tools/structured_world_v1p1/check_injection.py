"""P1 real-model, no-update comparison with original sampling precision/noise."""
import argparse,csv,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from tools.structured_world.runtime import load_baseline,load_dataset,seed_all
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy


def diff(x,y):
 x=x.float();y=y.float()
 return {'max_abs':float((x-y).abs().max()),'mean_abs':float((x-y).abs().mean()),'allclose':bool(torch.allclose(x,y,atol=.02,rtol=.02)), 'exact':bool(torch.equal(x,y))}


def main():
 p=argparse.ArgumentParser()
 for name in ['checkpoint','vlm','data-root','manifest','output']:p.add_argument('--'+name,required=True)
 p.add_argument('--limit',type=int,default=64);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();start=time.time();seed_all(42)
 agent=load_baseline(a.checkpoint,a.vlm);agent.model.requires_grad_(False);base=agent.model
 policies={}
 for name,layout,adapter in [('legacy_pre_action','legacy_pre_action',False),('append_tail','append_tail',False),('append_tail_gate0','append_tail',True)]:
  seed_all(42);policies[name]=StructuredWorldPolicy(base,{'enabled':True,'agent_tokens':64,'scene_tokens':64,'token_layout':layout,'action_adapter':adapter}).cuda().eval().requires_grad_(False)
 ds=load_dataset(agent,a.manifest,a.data_root,a.limit)
 captured={}
 def capture_norm(m,args):captured['hidden']=args[0].detach().cpu().clone()
 def capture_lm(m,args,kwargs):
  for key in ['inputs_embeds','position_ids','attention_mask','visual_pos_masks']:
   value=kwargs.get(key)
   if torch.is_tensor(value):captured[key]=value.detach().cpu().clone()
 def capture_action(m,args):pass
 handles=[base.qwen_vl_interface.model.model.language_model.norm.register_forward_pre_hook(capture_norm),base.qwen_vl_interface.model.model.language_model.register_forward_pre_hook(capture_lm,with_kwargs=True)]
 original=base.action_model.predict_action
 def action(condition,*args,**kw):captured['action']=condition.detach().cpu().clone();return original(condition,*args,**kw)
 base.action_model.predict_action=action
 calls=[]
 handles.append(policies['append_tail_gate0'].adapter.register_forward_hook(lambda *args:calls.append(1)))
 rows=[]
 for i in range(len(ds)):
  raw=ds[i];e={k:raw[k] for k in ['image','lang','state','token']};token=e['token']
  noise=int.from_bytes(hashlib.sha256(f'20260926:{token}'.encode()).digest()[:4],'little')
  seed_all(noise);captured.clear()
  with torch.inference_mode():a0=base.predict_action_infer_1d([e])['normalized_actions']
  native=dict(captured);length=native['hidden'].shape[1]
  arrays={'A0':a0}
  for name,policy in policies.items():
   seed_all(noise);captured.clear()
   with torch.inference_mode():result=policy.predict_action([e])['normalized_actions']
   row={'token':token,'variant':name}
   for field in ['action']:
    row.update({field+'_'+k:v for k,v in diff(captured[field],native[field]).items()})
   row.update({'trajectory_'+k:v for k,v in diff(torch.from_numpy(result),torch.from_numpy(a0)).items()})
   if name.startswith('append_tail'):
    for key in ['hidden','inputs_embeds','position_ids','attention_mask','visual_pos_masks']:
     x=captured[key];y=native[key];x=x[:,:,:length] if key=='position_ids' else x[:,:length]
     row.update({f'native_prefix_{key}_{k}':v for k,v in diff(x,y).items()})
   arrays[name]=result;rows.append(row)
  np.savez(out/(token+'.npz'),**arrays)
  if (i+1)%8==0:print(json.dumps({'scenes':i+1,'last':rows[-1]}),flush=True)
 for h in handles:h.remove()
 base.action_model.predict_action=original
 keys=sorted({k for r in rows for k in r})
 with (out/'scenes.csv').open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
 summary={}
 for name in policies:
  group=[r for r in rows if r['variant']==name]
  summary[name]={k:all(r[k] for r in group) if k.endswith(('allclose','exact')) else max(r[k] for r in group) for k in group[0] if k not in ['token','variant']}
 report={'code_sha':code,'arguments':vars(a),'scenes':len(ds),'optimizer_updates':0,'gpu_hours':(time.time()-start)/3600,'tolerance':{'atol':.02,'rtol':.02,'source':'predeclared original BF16 regression tolerance; unchanged'},'precision':'original Qwen BF16/action FP32','gate0_adapter_actual_calls':len(calls),'summary':summary}
 report['status']='PASS' if all(summary[n]['action_allclose'] and summary[n]['trajectory_allclose'] and all(summary[n]['native_prefix_'+k+'_exact'] for k in ['inputs_embeds','position_ids','attention_mask','visual_pos_masks']) for n in ['append_tail','append_tail_gate0']) else 'FAIL'
 (out/'INJECTION_SHIFT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
