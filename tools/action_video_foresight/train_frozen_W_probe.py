"""Train only lightweight future readouts on frozen100k W; never a planner update."""
import argparse,json,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors import safe_open
from starVLA.model.modules.foresight.future_spatiotemporal_head import FutureSpatiotemporalHead,normalized_clip_loss
from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256,initialization_seed
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run


class ProbeData:
    def __init__(self, representations, current, targets, split):
        self.rroot=Path(representations)/'representations';self.current=Path(current);self.targets=Path(targets)
        self.index=json.loads((Path(representations)/(split+'_representation_queries.json')).read_text())
        all_index=json.loads((self.current/'index.json').read_text());self.mapping={r['token']:i for i,r in enumerate(all_index)}
        self.identity=json.loads((self.targets/'identity.json').read_text())
        current_id=json.loads((self.current/'identity.json').read_text())
        if self.identity['split']!=split or self.identity['scene_index_hash']!=current_id['index_sha256']:raise ValueError('Probe source population mismatch')
        if not (self.targets/'COMPLETE.json').exists():raise ValueError('Full matching target population required for scientific probe')
    def batch(self,ids,device):
        w=[];action=[];features=[];valid=[]
        for i in ids:
            token=self.index[i]['token'];r=torch.load(self.rroot/(token+'.pt'),weights_only=True);w.append(r['W'])
            label=torch.load(self.current/'ego'/(token+'.pt'),weights_only=True);gt=decode_ego(label['ego'])
            action.append(torch.cat((gt[...,:2],gt[...,2:3].sin(),gt[...,2:3].cos()),-1))
            chunk,at=divmod(self.mapping[token],self.identity['chunk_size'])
            with safe_open(str(self.targets/f'chunk_{chunk:06d}.safetensors'),framework='pt',device='cpu') as f:
                if f.metadata()['identity']!=self.identity['identity']:raise ValueError('Probe cache shard mismatch')
                features.append(f.get_slice('features')[at]);valid.append(f.get_slice('valid')[at])
        return [torch.stack(x).to(device) for x in (w,action,features,valid)]


def evaluate(model,data,device,batch,permutation_seed=812):
    model.eval();rows=[];swapped=[]
    generator=torch.Generator().manual_seed(permutation_seed);order=torch.randperm(len(data.index),generator=generator).tolist()
    # Fixed, whole-population partner manifest, no error-based selection.
    mapping={order[i]:order[(i+max(1,len(order)//2))%len(order)] for i in range(len(order))}
    with torch.inference_mode():
        for start in range(0,len(data.index),batch):
            ids=list(range(start,min(start+batch,len(data.index))));w,a,t,v=data.batch(ids,device)
            wp,ap,_,_=data.batch([mapping[i] for i in ids],device)
            inputs=[('normal',w,a),('permuted_W',wp,a),('permuted_action',w,ap)]
            scores={}
            for name,ww,aa in inputs:
                with torch.autocast('cuda',dtype=torch.bfloat16):p=model(ww if model.use_world else None,(9,12),gt_action=aa if model.action_condition=='gt_ego' else None)
                for j,i in enumerate(ids):
                    loss,n=normalized_clip_loss(p[j:j+1],t[j:j+1],v[j:j+1]);scores.setdefault(i,{})[name]=float(loss) if n else None
            for j,i in enumerate(ids):rows.append({**data.index[i],'valid_clip_views':int(v[j].flatten(1).any(-1).sum()),'swap_token':data.index[mapping[i]]['token'],**scores[i],'failure':None})
    result={'requested':len(data.index),'scenes':len(rows),'failed':0}
    for name in ('normal','permuted_W','permuted_action'):
        values=[r[name] for r in rows if r[name] is not None];result[name+'_normalized_mse']=sum(values)/len(values) if values else None;result[name+'_valid_scenes']=len(values)
    model.train();return rows,result


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('representations','train-data','dev-data','train-targets','dev-targets','output','campaign-root','run-id'):p.add_argument('--'+k,required=True)
    p.add_argument('--condition',choices=('none','gt_ego'),required=True);p.add_argument('--action-only',action='store_true')
    p.add_argument('--action-injection',choices=('memory_only','memory_and_query'),default='memory_only')
    p.add_argument('--action-query-scale',type=float,default=1.)
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--batch',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if min(a.updates,a.batch)<1:raise ValueError('Fixed finite probe budget')
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze probe source')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    train=ProbeData(a.representations,a.train_data,a.train_targets,'train');dev=ProbeData(a.representations,a.dev_data,a.dev_targets,'dev')
    if {r['log'] for r in train.index}&{r['log'] for r in dev.index}:raise ValueError('Probe log leakage')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=a.resume)
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'frozen_W_lightweight_probe','main_model_optimizer_updates':0}) as (meter,_,save):
        first=torch.load(train.rroot/(train.index[0]['token']+'.pt'),weights_only=True)
        with initialization_seed(a.seed+2500):
            model=FutureSpatiotemporalHead(first['W'].shape[-1],train.identity['target_shape'][-1],
                train.identity['time_intervals_s'],action_condition=a.condition,use_world=not a.action_only,
                action_injection=a.action_injection,action_query_scale=a.action_query_scale).cuda()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.001)
        rng=torch.Generator().manual_seed(a.seed+13000);completed=epoch=offset=exposure=0;order=torch.randperm(len(train.index),generator=rng)
        identity={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'frozen_checkpoint':first['checkpoint'],
            'train_queries':identity_hash(train.index),'dev_queries':identity_hash(dev.index),'train_targets':train.identity['identity'],
            'dev_targets':dev.identity['identity'],'condition':a.condition,'use_world':not a.action_only,'updates':a.updates,'batch':a.batch,'seed':a.seed,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        if a.action_injection != 'memory_only' or a.action_query_scale != 1.:
            identity.update(action_injection=a.action_injection,action_query_scale=a.action_query_scale)
        if a.resume:
            s=torch.load(out/'latest.pt',weights_only=False)
            if s['identity']!=identity:raise ValueError('Probe resume identity')
            model.load_state_dict(s['model'],strict=True);optimizer.load_state_dict(s['optimizer']);rng.set_state(s['shuffle']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
            completed,epoch,offset,exposure,order=[s[k] for k in ('completed','epoch','offset','exposure','order')]
        else:atomic_json(out/'identity.json',identity)
        def checkpoint():
            payload={'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'shuffle':rng.get_state(),'torch':torch.get_rng_state(),
                'cuda':torch.cuda.get_rng_state_all(),'completed':completed,'epoch':epoch,'offset':offset,'exposure':exposure,'order':order}
            torch.save(payload,out/'latest.tmp');(out/'latest.tmp').replace(out/'latest.pt')
        while completed<a.updates:
            if (out/'STOP_REQUESTED').exists():meter['status']='PAUSED';checkpoint();save();return
            if offset>=len(order):epoch+=1;offset=0;order=torch.randperm(len(train.index),generator=rng)
            ids=order[offset:offset+a.batch].tolist();w,action,target,valid=train.batch(ids,'cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(w if model.use_world else None,(9,12),gt_action=action if a.condition=='gt_ego' else None)
            loss,n=normalized_clip_loss(prediction,target,valid);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('Invalid probe gradient')
            optimizer.step();completed+=1;offset+=len(ids);exposure+=len(ids)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps({'update':completed,'exposure':exposure,'loss':float(loss),'valid_scenes':n,'epoch':epoch,'offset':offset})+'\n')
            if completed%200==0:checkpoint()
            meter.update(real_optimizer_updates=completed,main_model_optimizer_updates=0,scene_exposure=exposure);save()
        checkpoint();rows,result=evaluate(model,dev,'cuda',a.batch)
        atomic_json(out/'SUMMARY.json',result)
        with (out/'dev_queries.jsonl').open('w') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        save()

if __name__=='__main__':main()
