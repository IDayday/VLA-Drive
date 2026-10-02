"""Matched frozen H_A/W/H_A+W ego readouts; diagnostic, never deployment PDMS."""
import argparse,json,subprocess
from pathlib import Path
import torch
from torch import nn
from starVLA.model.modules.foresight.future_latent_head import CrossReadout
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed,identity_hash
from starVLA.dataloader.foresight_dataset import decode_ego
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run


class EgoReadout(nn.Module):
    def __init__(self,hidden):
        super().__init__();self.project=nn.Linear(hidden,256);self.query=nn.Parameter(torch.randn(8,256)*.02)
        self.blocks=nn.ModuleList([CrossReadout(256,8),CrossReadout(256,8)]);self.output=nn.Linear(256,4)
    def forward(self,x):
        memory=self.project(x);q=self.query[None].expand(len(x),-1,-1)
        for block in self.blocks:q=block(q,memory)
        return self.output(q)


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('representations','train-data','dev-data','output','campaign-root','run-id'):p.add_argument('--'+k,required=True)
    p.add_argument('--mode',choices=('H_A','W','H_A+W'),required=True);p.add_argument('--updates',type=int,default=2000);p.add_argument('--batch',type=int,default=32);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze probe source')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);root=Path(a.representations);cache=root/'representations'
    tr=json.loads((root/'train_representation_queries.json').read_text());dev=json.loads((root/'dev_representation_queries.json').read_text())
    if {r['log'] for r in tr}&{r['log'] for r in dev}:raise ValueError('Probe log overlap')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    first=torch.load(cache/(tr[0]['token']+'.pt'),weights_only=True)
    def batch(rows,data):
        x=[];gt=[]
        for r in rows:
            z=torch.load(cache/(r['token']+'.pt'),weights_only=True)
            if z['checkpoint']!=first['checkpoint']:raise ValueError('Representation cache checkpoint mismatch')
            x.append(torch.cat((z['H_A'],z['W'])) if a.mode=='H_A+W' else z[a.mode])
            gt.append(torch.load(Path(data)/'ego'/(r['token']+'.pt'),weights_only=True)['ego'])
        return torch.stack(x).cuda(),torch.stack(gt).cuda()
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'frozen_representation_ego_readout','main_model_optimizer_updates':0}) as (meter,_,save):
        with initialization_seed(4242):head=EgoReadout(first['W'].shape[-1]).cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.001);rng=torch.Generator().manual_seed(17042)
        identity={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'frozen_checkpoint':first['checkpoint'],
            'mode':a.mode,'train_queries':identity_hash(tr),'dev_queries':identity_hash(dev),'updates':a.updates,'batch':a.batch,
            'trainable_parameters':sum(p.numel() for p in head.parameters()),'scope':'frozen-representation diagnostic, not planner replacement'}
        atomic_json(out/'identity.json',identity);order=torch.randperm(len(tr),generator=rng);offset=0;exposure=0
        for step in range(a.updates):
            if offset>=len(order):offset=0;order=torch.randperm(len(tr),generator=rng)
            ids=order[offset:offset+a.batch].tolist();x,gt=batch([tr[i] for i in ids],a.train_data)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=head(x)
            loss=(pred.float()-gt.float()).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            if not torch.isfinite(loss):raise FloatingPointError('Invalid matched probe')
            opt.step();offset+=len(ids);exposure+=len(ids)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps({'step':step+1,'loss':float(loss),'exposure':exposure})+'\n')
            meter.update(real_optimizer_updates=step+1,scene_exposure=exposure);save()
        head.eval();rows=[]
        with torch.inference_mode():
            for start in range(0,len(dev),a.batch):
                queries=dev[start:start+a.batch];x,gt=batch(queries,a.dev_data)
                with torch.autocast('cuda',dtype=torch.bfloat16):encoded=head(x)
                prediction=decode_ego(encoded.float());truth=decode_ego(gt);distance=(prediction[...,:2]-truth[...,:2]).norm(dim=-1)
                if not torch.isfinite(distance).all():raise FloatingPointError('Invalid ego readout')
                for i,r in enumerate(queries):rows.append({**r,'ADE':float(distance[i].mean()),'FDE':float(distance[i,-1]),'failure':None})
        atomic_json(out/'SUMMARY.json',{'scenes':len(rows),'failures':0,'ADE':sum(r['ADE'] for r in rows)/len(rows),
            'FDE':sum(r['FDE'] for r in rows)/len(rows),'exposure':exposure,'main_model_optimizer_updates':0})
        with (out/'dev_queries.jsonl').open('w') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        torch.save({'identity':identity,'model':head.state_dict(),'optimizer':opt.state_dict(),'shuffle':rng.get_state(),'offset':offset,'order':order},out/'completed.pt')

if __name__=='__main__':main()
