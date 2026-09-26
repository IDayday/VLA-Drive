"""Training-only dense geometric-provider perception isolation and adaptation."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from provider_cli import current_observation
from budget import reserve,record
from starVLA.model.modules.structured_world.providers import GeometricBEVProvider


class DenseHead(nn.Module):
 def __init__(self,channels=64):
  super().__init__();self.trunk=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.GroupNorm(8,channels),nn.GELU());self.cls=nn.Conv2d(channels,7,1);self.box=nn.Conv2d(channels,8,1);nn.init.constant_(self.cls.bias,-4.595)
 def forward(self,features,shape):
  x=self.trunk(features.transpose(1,2).reshape(features.shape[0],-1,*shape))
  return self.cls(x).flatten(2).transpose(1,2),self.box(x).flatten(2).transpose(1,2)


def dense_targets(target,provider):
 boxes=target['current_boxes'];classes=target['current_classes'];valid=target['current_supervision_mask']
 xy=boxes[:,:2];lo=xy.new_tensor(provider.bounds[:2]);cell=((xy-lo)/provider.resolution).floor().long();ny=provider.grid_shape[1]
 indices=cell[:,0]*ny+cell[:,1];heat=boxes.new_zeros(len(provider.coordinates),7);reg=boxes.new_zeros(len(provider.coordinates),8);occupied=torch.zeros(len(provider.coordinates),device=boxes.device,dtype=torch.bool);collisions=0
 for j in range(len(boxes)):
  if not valid[j]:continue
  idx=int(indices[j])
  if idx<0 or idx>=len(heat):raise ValueError('Target outside provider grid')
  if occupied[idx]:collisions+=1;continue
  occupied[idx]=True;heat[idx,classes[j]]=1.;reg[idx]=boxes[j]
  reg[idx,:2]-=provider.coordinates[idx,:2]
 return heat,reg,occupied,collisions


def objective(logits,raw,heat,reg,occupied,support):
 probability=logits.sigmoid();ce=F.binary_cross_entropy_with_logits(logits,heat,reduction='none')
 pt=probability*heat+(1-probability)*(1-heat);alpha=.25*heat+.75*(1-heat)
 cls=(ce*(1-pt).square()*alpha*support[...,None]).sum()/occupied.sum().clamp_min(1)
 decoded=torch.cat([raw[...,:3],F.softplus(raw[...,3:6])+.01,raw[...,6:]],-1)
 scale=raw.new_tensor([1.,1.,3.,5.,5.,5.,1.,1.])
 box=F.smooth_l1_loss(decoded[occupied]/scale,reg[occupied]/scale,reduction='sum')/occupied.sum().clamp_min(1)
 return cls+box,cls,box,decoded


@torch.no_grad()
def evaluate(provider,head,tokens,root,sensors,device="cuda"):
 results=[]
 for token in tokens:
  inputs,_=current_observation(root/'observations'/f'{token}.npz',sensors,device);target=torch.load(root/'targets'/f'{token}.pt',map_location=device,weights_only=True)
  f,coords,support,_=provider(inputs);logits,raw=head(f,provider.grid_shape);scores,classes=logits[0].sigmoid().max(-1)
  maxima=F.max_pool2d(scores.reshape(1,1,*provider.grid_shape),3,stride=1,padding=1).flatten();scores=scores.masked_fill((scores<maxima)|~support[0],0.)
  selected=torch.argsort(scores,descending=True)[:64];selected=selected[scores[selected]>.2]
  centres=raw[0,selected,:2]+coords[0,selected,:2]
  gt=target['current_boxes'][target['current_supervision_mask']];gc=target['current_classes'][target['current_supervision_mask']]
  n=0;error=0.
  if len(centres) and len(gt):
   distances=torch.cdist(centres,gt[:,:2]);cost=distances+5*(classes[selected,None]!=gc[None])
   rows,cols=linear_sum_assignment(cost.cpu().numpy());good=(distances[rows,cols]<2)&(classes[selected[rows]]==gc[cols]);n=int(good.sum());error=float(distances[rows,cols][good].sum())
  results.append({'token':token,'gt':len(gt)+target['overflow'],'detections':len(selected),'matched':n,'centre_error_sum':error,'false_positives':len(selected)-n})
 total=sum(r['gt'] for r in results);matched=sum(r['matched'] for r in results)
 return {'scenes':len(results),'gt':total,'matched':matched,'recall_2m_class_correct':matched/max(total,1),'false_positives':sum(r['false_positives'] for r in results),'matched_centre_error':sum(r['centre_error_sum'] for r in results)/max(matched,1),'threshold':.2,'records':results}


def main():
 p=argparse.ArgumentParser()
 for n in ['manifest','target-cache','sensor-root','output','ledger']:p.add_argument('--'+n,required=True)
 p.add_argument('--steps',type=int,default=600);p.add_argument('--run-id',default='provider_dense_isolation');p.add_argument('--resume');p.add_argument('--stop-after',type=int);p.add_argument('--save-every',type=int,default=100);p.add_argument('--deterministic',action='store_true')
 a=p.parse_args()
 if a.steps>1000:raise ValueError('Provider adaptation per-run budget exceeded')
 if not a.resume:reserve(a.ledger,a.run_id,a.steps,vars(a))
 torch.manual_seed(42);random.seed(42);np.random.seed(42)
 if a.deterministic:torch.use_deterministic_algorithms(True)
 tokens=json.loads(Path(a.manifest).read_text());root=Path(a.target_cache);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 provider=GeometricBEVProvider().cuda();head=DenseHead().cuda();optimizer=torch.optim.AdamW(list(provider.parameters())+list(head.parameters()),lr=3e-4)
 order=list(tokens);random.shuffle(order);first=0
 def save(step):
  payload={'delta':{'provider.'+k:v.detach().cpu() for k,v in provider.state_dict().items()},'world_config':{'bev_channels':64},'dense_head':head.state_dict(),'optimizer':optimizer.state_dict(),'steps':step,'planned_steps':a.steps,'order':order,'arguments':vars(a),'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},'scope':'Current box/class training only; dense auxiliary head excluded from deployment','provider_metadata':provider.metadata(),'resume_boundary':'optimizer boundary; exact continuation requires --deterministic (CPU geometry resampler during provider training), identical software/topology' ,'manifest_sha256':__import__('hashlib').sha256(Path(a.manifest).read_bytes()).hexdigest(),'target_audit_sha256':__import__('hashlib').sha256((root/'audit.json').read_bytes()).hexdigest()}
  tmp=out/'checkpoint.tmp';torch.save(payload,tmp);tmp.replace(out/'checkpoint.pt')
 if a.resume:
  saved=torch.load(a.resume,map_location='cpu',weights_only=False)
  for key in ['steps','manifest','target_cache','run_id','deterministic']:
   if saved['arguments'][key]!=getattr(a,key):raise ValueError('Provider resume contract mismatch: '+key)
  for key,path in [('manifest_sha256',Path(a.manifest)),('target_audit_sha256',root/'audit.json')]:
   if saved[key]!=__import__('hashlib').sha256(path.read_bytes()).hexdigest():raise ValueError('Provider resume data changed')
  provider.load_state_dict({k.removeprefix('provider.'):v for k,v in saved['delta'].items()},strict=True);head.load_state_dict(saved['dense_head'],strict=True);optimizer.load_state_dict(saved['optimizer'])
  first=saved['steps'];order=saved['order'];random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy']);torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
 else:
  before=evaluate(provider,head,tokens,root,a.sensor_root);(out/'before.json').write_text(json.dumps(before,indent=2))
 start=time.perf_counter()
 for step in range(first,a.steps):
  if step and step%len(order)==0:random.shuffle(order)
  token=order[step%len(order)];inputs,_=current_observation(root/'observations'/f'{token}.npz',a.sensor_root,'cuda');target=torch.load(root/'targets'/f'{token}.pt',map_location='cuda',weights_only=True)
  f,_,support,_=provider(inputs);logits,raw=head(f,provider.grid_shape);heat,reg,occupied,collisions=dense_targets(target,provider)
  loss,cls,box,_=objective(logits[0],raw[0],heat,reg,occupied,support[0]);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(provider.parameters())+list(head.parameters()),5.,error_if_nonfinite=True);optimizer.step();record(a.ledger,a.run_id,step+1)
  if (step+1)%a.save_every==0 or step+1==a.steps or step+1==a.stop_after:save(step+1)
  if step+1==a.stop_after:return
  if (step+1)%20==0:
   row={'step':step+1,'loss':float(loss.detach()),'classification':float(cls.detach()),'box':float(box.detach()),'cell_collisions':collisions,'elapsed_seconds':time.perf_counter()-start}
   print(json.dumps(row),flush=True)
   with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
 after=evaluate(provider,head,tokens,root,a.sensor_root);(out/'after.json').write_text(json.dumps(after,indent=2))
 record(a.ledger,a.run_id,a.steps,'complete')
 print(json.dumps({'after_recall':after['recall_2m_class_correct']}))

if __name__=='__main__':main()
