"""Matched W_PRE/W_POST with frozen driving policy and graph-bounded accumulation."""
import argparse,hashlib,json,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
from tools.structured_world.runtime import load_baseline,load_dataset,load_world_batch,seed_all
from tools.structured_world_v1p1.budget import start,record
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy
from starVLA.model.modules.structured_world.rehab import reference_loss_sums
from starVLA.model.modules.structured_world.metrics import diagnostics
from tools.structured_world_v1p1.reaudit_metrics import summarize,write_csv


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_visual_memo(model):
    original=model.model.get_image_features;memo={};keys=set()
    if any(p.requires_grad for p in model.model.visual.parameters()):raise ValueError('Cannot memoize trainable vision')
    def cached(pixels,grid):
        key=hashlib.sha256(pixels.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()+grid.cpu().numpy().tobytes()).hexdigest()
        keys.add(key)
        if key not in memo:
            with torch.no_grad():memo[key]=original(pixels,grid)
        return memo[key]
    model.model.get_image_features=cached
    return keys


def main():
 p=argparse.ArgumentParser()
 for k in ['checkpoint','vlm','data-root','manifest','targets','output','ledger','run-id','config']:p.add_argument('--'+k,required=True)
 p.add_argument('--steps',type=int,default=1000);p.add_argument('--limit',type=int,default=16);p.add_argument('--batch',type=int,default=8)
 p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=42);p.add_argument('--stop-after',type=int);p.add_argument('--resume');p.add_argument('--eval-every',type=int,default=100)
 a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.output)
 if a.limit not in [16,64] or a.steps>1000:raise ValueError('Small-set bounds')
 if a.resume:out.mkdir(exist_ok=True)
 else:out.mkdir(parents=True,exist_ok=False)
 code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 manifest=json.loads(Path(a.manifest).read_text())[:a.limit]
 fingerprint=hashlib.sha256(''.join(t+digest(Path(a.targets)/'targets'/f'{t}.pt') for t in manifest).encode()).hexdigest()
 identity={'code_sha':code,'config':cfg,'manifest_sha256':digest(a.manifest),'target_sha256':fingerprint,'arguments':vars(a),'original_checkpoint_sha256':digest(Path(a.checkpoint)/'pytorch_model.pt')}
 start(a.ledger,a.run_id,a.steps,identity,resume=bool(a.resume));step=0
 try:
  seed_all(a.seed);agent=load_baseline(a.checkpoint,a.vlm);agent.model.requires_grad_(False)
  seed_all(a.seed);policy=StructuredWorldPolicy(agent.model,cfg).cuda().train();policy.baseline.eval()
  keys=install_visual_memo(policy.baseline.qwen_vl_interface.model)
  ds=load_dataset(agent,a.manifest,a.data_root,a.limit)
  samples=[]
  for i in range(len(ds)):
   raw=ds[i];e={k:raw[k] for k in ['image','lang','state','token']};_,ts=load_world_batch([e],a.targets)
   samples.append((e,ts[0]))
  params=[p for p in policy.parameters() if p.requires_grad];names=[n for n,p in policy.named_parameters() if p.requires_grad]
  if any(n.startswith('baseline.') for n in names):raise AssertionError('Driving policy accidentally trainable')
  opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
  order=list(range(len(samples)));random.shuffle(order);position=0;seen=set();presentations=0
  if a.resume:
   saved=torch.load(a.resume,map_location='cpu',weights_only=False)
   for k in ['config','manifest_sha256','target_sha256','code_sha']:
    if saved['identity'][k]!=identity[k]:raise ValueError('Resume identity mismatch '+k)
   for name,module in [('reader',policy.reader),('heads',policy.heads)]:module.load_state_dict(saved[name],strict=True)
   opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);step=saved['step'];order=saved['order'];position=saved['position'];seen=set(saved['seen']);presentations=saved['presentations']
   random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy']);torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
  (out/'manifest.json').write_text(json.dumps(identity,indent=2));(out/'trainable.json').write_text(json.dumps({'names':names,'parameters':sum(p.numel() for p in params)},indent=2))
  def predict(e):
   with torch.autocast('cuda',dtype=torch.bfloat16):return policy.encode_conditions([e],world_only=True)[1]
  def evaluate(tag):
   rows=[];policy.eval()
   for e,t in samples:
    with torch.no_grad():pred=predict(e)
    row={'token':e['token'],'status':'ok'}
    xy=pred['boxes'][0,:,:2].float();pair=torch.cdist(xy,xy);pair.fill_diagonal_(float('inf'))
    row['slot_nearest_distance_mean']=float(pair.min(-1).values.mean())
    hist=torch.bincount(pred['logits'][0].argmax(-1),minlength=pred['logits'].shape[-1])
    for k,value in enumerate(hist):row['predicted_class_'+str(k)]=int(value)
    row.update(diagnostics({k:v[0] for k,v in pred.items()},t,include_legacy=False));rows.append(row)
   write_csv(out/f'eval_{tag}.csv',rows);summary=summarize(rows);(out/f'eval_{tag}.json').write_text(json.dumps(summary,indent=2));policy.train();policy.baseline.eval()
   print(json.dumps({'evaluation_step':tag,'recall':summary['filtered']['recall_2m'],'precision':summary['filtered']['precision_2m'],'class_recall':summary['class_filtered']['recall_2m']}),flush=True)
  def save():
   saved={'identity':identity,'reader':policy.reader.state_dict(),'heads':policy.heads.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'step':step,'order':order,'position':position,'seen':sorted(seen),'presentations':presentations,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},'visual_memo_keys':sorted(keys),'resume_boundary':'optimizer-step, same code/topology/precision; no pending microbatches'}
   torch.save(saved,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
  if not a.resume:evaluate('0')
  for next_step in range(step+1,a.steps+1):
   if not record(a.ledger,a.run_id,step):break
   ids=[]
   for _ in range(a.batch):
    if position==len(order):random.shuffle(order);position=0
    ids.append(order[position]);position+=1
   # Deterministic count pass. No Qwen autograd graph retained between scenes.
   counts={k:0. for k in ['cls','box','motion']};individual=[]
   with torch.no_grad():
    for i in ids:
     e,t=samples[i];_,c,_=reference_loss_sums(predict(e),[t]);individual.append(c)
     for k in counts:counts[k]+=c[k]
   task_grad={};opt.zero_grad(set_to_none=True);totals={k:0. for k in counts};before={n:p.detach().flatten()[:16].clone() for n,p in policy.named_parameters() if p.requires_grad}
   for i,c0 in zip(ids,individual):
    e,t=samples[i];pred=predict(e);s,c,matches=reference_loss_sums(pred,[t])
    if any(abs(c[k]-c0[k])>1e-5 for k in c):raise AssertionError('Count pass changed under grad')
    loss=sum(cfg.get('lambda_'+k,1.)*s[k]/max(counts[k],1.) for k in s)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite world loss')
    if not task_grad and (next_step==1 or next_step%100==0):
     for k in s:
      g=torch.autograd.grad(s[k]/max(counts[k],1.),policy.reader.queries,retain_graph=True,allow_unused=True)[0]
      task_grad[k]=None if g is None else float(g.float().norm())
    loss.backward()
    for k in s:totals[k]+=float(s[k].detach())/max(counts[k],1.)
    seen.add(e['token']);presentations+=1
   groups={}
   for n,p0 in policy.named_parameters():
    if p0.requires_grad and p0.grad is not None:
     group='queries' if n in ['reader.queries','reader.types'] else n.split('.')[0]+'.'+n.split('.')[1]
     groups[group]=groups.get(group,0.)+float(p0.grad.float().square().sum())
   grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step();scheduler.step();step=next_step
   changes={}
   for n,p0 in policy.named_parameters():
    if p0.requires_grad:
     changes[n]=float((p0.detach().flatten()[:16]-before[n]).abs().max())
   r={'step':step,'loss':totals,'grad_norm_before_clip':grad,'grad_norm_after_clip':min(grad,1.),'task_query_gradient_norm':task_grad,'group_grad_norm':{k:v**.5 for k,v in groups.items()},'updated_parameter_probes':sum(v>0 for v in changes.values()),'query_update_max':changes.get('reader.queries'),'unique_scenes_seen':len(seen),'sample_presentations':presentations,'effective_epochs':presentations/len(samples),'global_batch':a.batch,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
   with (out/'train.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
   if step%10==0:print(json.dumps(r),flush=True)
   record(a.ledger,a.run_id,step)
   if step%a.eval_every==0 or step==a.steps or step==a.stop_after:evaluate(str(step));save()
   if a.stop_after and step>=a.stop_after:break
  save();record(a.ledger,a.run_id,step,'complete' if step==a.steps else 'paused')
 except BaseException:
  record(a.ledger,a.run_id,step,'failed');raise

if __name__=='__main__':main()
