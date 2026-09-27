"""Independent perception probe of a verified frozen prior, before Qwen injection."""
import argparse,hashlib,json,random,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tools.structured_world.provider_cli import current_observation
from tools.structured_world_v1p1.budget import start,record
from tools.structured_world_v1p1.reaudit_metrics import summarize,write_csv
from starVLA.model.modules.structured_world.providers import calibration_fingerprint
from starVLA.model.modules.structured_world.scene_agent_reader import SceneAgentReader
from starVLA.model.modules.structured_world.rehab import ReferenceAgentHeads,reference_loss_sums
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.metrics import diagnostics


def main():
 p=argparse.ArgumentParser()
 for k in ['train-manifest','holdout-manifest','train-features','holdout-features','targets','vlm-config','source-audit','output','ledger','run-id']:p.add_argument('--'+k,required=True)
 p.add_argument('--steps',type=int,default=600);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 if a.steps>1000:raise ValueError('Provider probe cap')
 torch.manual_seed(42);random.seed(42);np.random.seed(42)
 code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();identity={'code_sha':code,'arguments':vars(a),'kind':'pretrained provider frozen; independent trainable Reader/current heads; NO Qwen forward','seed':42,'global_batch':8}
 start(a.ledger,a.run_id,a.steps,identity);step=0
 try:
  source=json.loads(Path(a.source_audit).read_text());banks={};fingerprint=hashlib.sha256()
  for split,manifest,features in [('train',a.train_manifest,a.train_features),('holdout',a.holdout_manifest,a.holdout_features)]:
   items=[];tokens=json.loads(Path(manifest).read_text())
   for token in tokens:
    path=Path(features)/(token+'.pt');data=path.read_bytes();fingerprint.update(token.encode()+hashlib.sha256(data).digest());payload=torch.load(path,map_location='cuda',weights_only=True)
    if set(payload)!={'metadata','features','coordinates','observation_support'}:raise ValueError('Unexpected feature fields')
    meta=payload['metadata'];inputs,image_hashes=current_observation(Path(a.targets)/'observations'/(token+'.npz'),'/', 'cpu')
    if meta['scene_token']!=token or meta['decision_time']!=int(inputs.decision_time[0]) or meta['image_sha256']!=image_hashes or meta['calibration_sha256']!=calibration_fingerprint(inputs):raise ValueError('Current observation/cache identity mismatch')
    if meta['backbone_weights_sha256']!=source['weights_sha256'] or meta['sensor_contract']!={'cameras':['CAM_F0','CAM_L0','CAM_R0'],'time':'current_only'} or meta['pretrained_bev'] is not False:raise ValueError('External provider signature mismatch')
    if meta['feature_dimension']!=1024 or payload['features'].shape!=(1,1960,1024):raise ValueError('Invalid feature grid')
    for k in ['features','coordinates','observation_support']:
     if not torch.isfinite(payload[k]).all():raise ValueError('Invalid feature values')
    tp=Path(a.targets)/'targets'/(token+'.pt');fingerprint.update(hashlib.sha256(tp.read_bytes()).digest());target=WorldTargets(**torch.load(tp,map_location='cuda',weights_only=True))
    if target.overflow:raise ValueError('Full GT required')
    items.append((token,payload,target))
   banks[split]=items
  if set(t for t,_,_ in banks['train'])&set(t for t,_,_ in banks['holdout']):raise ValueError('Overlapping holdout')
  dim=json.loads(Path(a.vlm_config).read_text())['text_config']['hidden_size']
  model=nn.ModuleDict({'reader':SceneAgentReader(1024,dim,scene_tokens=64,agent_tokens=64),'heads':ReferenceAgentHeads(dim,slots=64)}).cuda()
  model['heads'].motion.requires_grad_(False)
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
  def predict(payload):
   with torch.autocast('cuda',dtype=torch.bfloat16):
    memory=model['reader'](payload['features'],payload['coordinates'],payload['observation_support'],payload['metadata'],agent_references=model['heads'].references)
    return model['heads'](memory.agent_memory)
  def evaluate(split,tag):
   model.eval();rows=[]
   for token,payload,target in banks[split]:
    with torch.no_grad():pred=predict(payload)
    row={'token':token,'status':'ok'};row.update(diagnostics({k:v[0] for k,v in pred.items()},target,include_legacy=False));rows.append(row)
    if tag=='final':
     folder=out/(split+'_predictions');folder.mkdir(exist_ok=True);np.savez(folder/(token+'.npz'),**{k:v[0].detach().cpu().numpy() for k,v in pred.items()})
   write_csv(out/(split+'_'+tag+'.csv'),rows);s=summarize(rows);s['motion_status']='NOT_TRAINED; do not interpret random displacement errors as adapted motion';(out/(split+'_'+tag+'.json')).write_text(json.dumps(s,indent=2));model.train();print(json.dumps({'split':split,'step':tag,'recall':s['filtered']['recall_2m'],'precision':s['filtered']['precision_2m']}),flush=True)
  identity['feature_target_sha256']=fingerprint.hexdigest();identity['train_scenes']=len(banks['train']);identity['holdout_scenes']=len(banks['holdout']);(out/'manifest.json').write_text(json.dumps(identity,indent=2))
  evaluate('train','0');order=list(range(len(banks['train'])));random.shuffle(order);position=0
  for step in range(1,a.steps+1):
   if not record(a.ledger,a.run_id,step-1):raise RuntimeError('Campaign cap')
   opt.zero_grad(set_to_none=True);sums=[];counts=[]
   # No Qwen graph exists here. Eight small Reader graphs fit in bounded memory.
   for _ in range(8):
    if position==len(order):random.shuffle(order);position=0
    token,payload,target=banks['train'][order[position]];position+=1
    s,c,_=reference_loss_sums(predict(payload),[target]);sums.append(s);counts.append(c)
   loss=sum(sum(s[k] for s in sums)/max(sum(c[k] for c in counts),1.) for k in ['cls','box'])
   loss.backward();g=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();record(a.ledger,a.run_id,step)
   with (out/'train.jsonl').open('a') as f:f.write(json.dumps({'step':step,'loss':float(loss.detach()),'grad_before_clip':g,'sample_presentations':step*8,'effective_epochs':step*8/len(banks['train']),'unique_scenes_seen':min(step*8,len(banks['train'])),'gpu_peak_bytes':torch.cuda.max_memory_allocated()})+'\n')
   if step%100==0:evaluate('train',str(step))
  torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'steps':step,'identity':identity,'backbone_updated':False},out/'checkpoint.pt');evaluate('train','final');evaluate('holdout','final');record(a.ledger,a.run_id,step,'complete')
 except BaseException:
  record(a.ledger,a.run_id,step,'failed');raise
if __name__=='__main__':main()
