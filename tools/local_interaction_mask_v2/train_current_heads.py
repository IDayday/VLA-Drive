"""Finite current-head refinement on frozen public features; no action or future learning."""
import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from starVLA.model.modules.joint_world.local_cache import load_payload
from starVLA.model.modules.joint_world.public_baseline import sha256
from starVLA.model.modules.joint_world.current_refinement import head_code_digest
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.rehab import ReferenceAgentHeads,reference_loss_sums
from starVLA.model.modules.structured_world.metrics import diagnostics
from tools.local_interaction_mask_v2.foundation import to_device,rng_state,restore_rng,collate
from tools.local_interaction_mask_v2.graph_runtime import cached_worker_init
from tools.local_interaction_mask_v2.train_foundation import atomic_json


class CurrentHeadDataset(torch.utils.data.Dataset):
    def __init__(self,cache,targets):
        self.root=Path(cache);self.targets=Path(targets);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if not self.manifest['complete'] or self.manifest['failed']:raise ValueError('Incomplete frozen current features')
        self.records=self.manifest['records'];self.memo=OrderedDict()
        if len({r['token'] for r in self.records})!=len(self.records):raise ValueError('Duplicate current scenes')
        digest=hashlib.sha256()
        for row in self.records:digest.update(row['token'].encode()+bytes.fromhex(sha256(self.targets/'targets'/(row['token']+'.pt'))))
        self.target_fingerprint=digest.hexdigest()
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        if i not in self.memo:
            row=self.records[i];payload=load_payload(self.root,row,self.manifest)
            target=WorldTargets(**torch.load(self.targets/'targets'/(row['token']+'.pt'),weights_only=True))
            target.future_valid_mask.zero_();target.future_xy_in_ego_t0.zero_()
            self.memo[i]={'token':row['token'],'features':payload['full_current']['actor_features'][:,1:], 'target':target}
            if len(self.memo)>64:self.memo.popitem(last=False)
        self.memo.move_to_end(i);return self.memo[i]


@torch.no_grad()
def evaluate(head,dataset):
    head.eval();rows=[]
    for i in range(len(dataset)):
        sample=dataset[i];target=to_device(sample['target'],'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=head(sample['features'].cuda().bfloat16())
        values=diagnostics({k:v[0] for k,v in prediction.items()},target,include_legacy=False)
        rows.append({'token':sample['token'],'annotation_valid':bool(target.annotation_valid_mask),
            **{k:values.get(k,0) for k in ('gt_targets','class_filtered_tp','class_filtered_fp','class_filtered_centre_error_sum')}})
    valid=[r for r in rows if r['annotation_valid']];tp=sum(r['class_filtered_tp'] for r in valid);fp=sum(r['class_filtered_fp'] for r in valid);gt=sum(r['gt_targets'] for r in valid)
    head.train()
    return {'scenes':len(rows),'missing_current_annotations':len(rows)-len(valid),'tp':tp,'fp':fp,'gt':gt,
        'recall':tp/max(gt,1),'precision':tp/max(tp+fp,1),'F1':2*tp/max(tp+fp+gt,1),
        'centre_error_m':sum(r['class_filtered_centre_error_sum'] for r in valid)/max(tp,1),'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','targets','foundation','holdout-cache','holdout-targets','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--epochs',type=int,default=16);p.add_argument('--schedule-epochs',type=int,default=16)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--workers',type=int,default=2);p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    if not 1<=a.epochs<=a.schedule_epochs or not 1<=a.batch<=256:raise ValueError('Invalid bounded head refinement plan')
    cached_worker_init(0);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists() and not a.resume:raise FileExistsError('Use a fresh refinement output')
    train=CurrentHeadDataset(a.cache,a.targets);holdout=CurrentHeadDataset(a.holdout_cache,a.holdout_targets)
    if train.manifest['identity_sha256']!=holdout.manifest['identity_sha256']:raise ValueError('Current train/holdout features differ')
    if {r['token'] for r in train.records}&{r['token'] for r in holdout.records}:raise ValueError('Refinement holdout overlaps training tokens')
    parent_sha=sha256(a.foundation)
    if parent_sha!=train.manifest['identity']['foundation_sha256']:raise ValueError('Current features come from another foundation')
    foundation=torch.load(a.foundation,map_location='cpu',weights_only=False,mmap=True)
    if foundation['identity'].get('private_driving_weights_loaded',True):raise ValueError('Private foundation forbidden')
    state=foundation['modules']['heads']
    head=ReferenceAgentHeads(state['shared.1.weight'].shape[1],slots=len(state['references']),classes=state['classifier.weight'].shape[0]-1,
        steps=state['motion.weight'].shape[0]//2,dim=state['shared.1.weight'].shape[0]).cuda()
    head.load_state_dict(state,strict=True);head.motion.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=a.lr,weight_decay=.01)
    horizon=math.ceil(len(train)/a.batch)*a.schedule_epochs
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/50,1)*(.1+.9*.5*(1+math.cos(math.pi*min(s,horizon)/horizon))))
    identity={'kind':'public_frozen_feature_current_refinement_v1','code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'arguments':{k:v for k,v in vars(a).items() if k not in ('output','resume','stop_after','epochs')},
        'foundation_sha256':parent_sha,'public_origin':foundation['public_origin'],'current_identity':train.manifest['identity_sha256'],
        'current_feature_manifest_sha256':sha256(Path(a.cache)/'manifest.json'),'labels':train.target_fingerprint,'holdout_labels':holdout.target_fingerprint,
        'holdout_feature_manifest_sha256':sha256(Path(a.holdout_cache)/'manifest.json'),'head_source_files':head_code_digest(),
        'upstream_language_numerics':train.manifest['identity'].get('language_numerics'),
        'trainable':'existing current shared MLP/classifier/box only','frozen':'Qwen/LoRA/driving tokens/Reader/DiT/history/unused motion head',
        'seed':42,'future_labels_erased':True,'selector_tuning':False,'Navtest_consulted':False,
        'head_input_dtype':'native_Qwen_BF16_recovered_exactly_from_float32_current_feature_storage',
        'selection':'best current class-filtered2m F1 on fixed training-domain64scene59log holdout, including epoch0'}
    step=epoch=offset=presentations=0;best=-1.
    if a.resume:
        saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Head refinement resume identity differs')
        head.load_state_dict(saved['head'],strict=True);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        step,epoch,offset,presentations=[saved[k] for k in ('step','epoch','offset','presentations')];best=saved['best_F1'];restore_rng(saved['rng'])
    atomic_json(out/'manifest.json',identity)
    def payload():return {'identity':identity,'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
        'step':step,'epoch':epoch,'offset':offset,'presentations':presentations,'best_F1':best,'rng':rng_state()}
    def assess():
        nonlocal best
        result=evaluate(head,holdout);atomic_json(out/f'holdout_{epoch}.json',result)
        if result['F1']>best:
            best=result['F1'];torch.save(payload(),out/'selected.tmp');(out/'selected.tmp').replace(out/'selected.pt')
            atomic_json(out/'selection.json',{'epoch':epoch,'step':step,'holdout_F1':best,'criterion':identity['selection'],'Navtest_consulted':False})
    if not a.resume:assess()
    paused=False
    while epoch<a.epochs:
        order=np.random.default_rng(np.random.SeedSequence([42,epoch])).permutation(len(train)).tolist()
        batches=[order[i:i+a.batch] for i in range(offset,len(order),a.batch)]
        loader=torch.utils.data.DataLoader(train,batch_sampler=batches,num_workers=a.workers,collate_fn=collate,
            worker_init_fn=cached_worker_init,prefetch_factor=1 if a.workers else None,
            multiprocessing_context='spawn' if a.workers else None,generator=torch.Generator().manual_seed(42+epoch))
        for samples in loader:
            opt.zero_grad(set_to_none=True);features=torch.cat([s['features'] for s in samples]).cuda().bfloat16()
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction=head(features)
            sums,counts,_=reference_loss_sums(prediction,[to_device(s['target'],'cuda') for s in samples])
            loss=sum(sums[k]/max(counts[k],1) for k in ('cls','box'))
            if not torch.isfinite(loss):raise FloatingPointError('Current refinement nonfinite')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True));opt.step();scheduler.step()
            step+=1;offset+=len(samples);presentations+=len(samples)
            row={'step':step,'epoch':epoch,'presentations':presentations,'loss':float(loss.detach()),'loss_components':{k:float(sums[k].detach())/max(counts[k],1) for k in ('cls','box')},
                'supervised_coordinates':counts,'gradient_before_clip':norm,'lr':opt.param_groups[0]['lr'],'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            end_epoch=offset==len(train)
            if end_epoch:epoch+=1;offset=0
            atomic_json(out/'progress.json',{'step':step,'epoch':epoch,'offset':offset,'presentations':presentations})
            paused=((a.stop_after is not None and step>=a.stop_after) or (out/'STOP_REQUESTED').exists()) and epoch<a.epochs
            if end_epoch:assess()
            if end_epoch or paused:
                torch.save(payload(),out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt');break
        if paused:break
    atomic_json(out/'status.json',{'status':'paused' if paused else 'complete','step':step,'epochs':epoch,'presentations':presentations,'best_holdout_F1':best})


if __name__=='__main__':main()
