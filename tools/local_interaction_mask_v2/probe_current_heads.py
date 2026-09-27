"""Bounded fixed-feature perception fitting diagnostic, never a planning result."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.model.modules.joint_world.local_cache import load_payload
from starVLA.model.modules.joint_world.public_baseline import sha256
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.rehab import ReferenceAgentHeads,reference_loss_sums
from starVLA.model.modules.structured_world.metrics import diagnostics
from tools.local_interaction_mask_v2.foundation import to_device
from tools.local_interaction_mask_v2.train_foundation import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','foundation','targets','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--batch',type=int,default=16);a=p.parse_args()
    if not 1<=a.steps<=1000 or not 1<=a.batch<=64:raise ValueError('Bounded engineering probe only')
    root=Path(a.cache);manifest=json.loads((root/'manifest.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if not 1<=len(manifest['records'])<=64:raise ValueError('Fixed-feature overfit probe is limited to64training scenes')
    if sha256(a.foundation)!=manifest['identity']['foundation_sha256']:raise ValueError('Wrong public foundation')
    saved=torch.load(a.foundation,map_location='cpu',weights_only=False,mmap=True);state=saved['modules']['heads']
    if saved['identity'].get('private_driving_weights_loaded',True):raise ValueError('Private features forbidden')
    torch.manual_seed(42);head=ReferenceAgentHeads(state['shared.1.weight'].shape[1],slots=len(state['references']),
        classes=state['classifier.weight'].shape[0]-1,dim=state['shared.1.weight'].shape[0]).cuda()
    head.load_state_dict(state,strict=True);head.motion.requires_grad_(False)
    features=[];targets=[]
    for record in manifest['records']:
        payload=load_payload(root,record,manifest)
        features.append(payload['full_current']['actor_features'][:,1:].cuda())
        target=WorldTargets(**torch.load(Path(a.targets)/'targets'/(record['token']+'.pt'),weights_only=True))
        # Only current perception is fitted; erase future labels even on loss side.
        target.future_valid_mask.zero_();target.future_xy_in_ego_t0.zero_()
        targets.append(to_device(target,'cuda'))
    features=torch.cat(features);opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=1e-3,weight_decay=.01)
    generator=torch.Generator().manual_seed(42);order=torch.randperm(len(targets),generator=generator).tolist();offset=0
    atomic_json(out/'manifest.json',{'scope':'P1 fixed-feature current-head fitting only, no planning claim',
        'hypothesis':'Current supervised head optimization may be incomplete despite weak early holdout stabilization; isolate it from frozen feature quality',
        'cache_identity':manifest['identity_sha256'],'foundation_sha256':manifest['identity']['foundation_sha256'],
        'trainable':'existing shared head MLP, classifier and box regressor','frozen':'all current feature producers and unused motion head',
        'steps':a.steps,'batch':a.batch,'seed':42,'lr':.001,'future_labels_erased':True,'scenes':len(targets)})
    @torch.no_grad()
    def evaluate(step):
        head.eval();rows=[]
        for i,target in enumerate(targets):
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction=head(features[i:i+1])
            stats=diagnostics({k:v[0] for k,v in prediction.items()},target,include_legacy=False)
            rows.append({'token':manifest['records'][i]['token'],**{k:stats[k] for k in ('gt_targets','class_filtered_tp','class_filtered_fp','class_filtered_centre_error_sum')}})
        tp=sum(r['class_filtered_tp'] for r in rows);fp=sum(r['class_filtered_fp'] for r in rows);gt=sum(r['gt_targets'] for r in rows)
        atomic_json(out/f'eval_{step}.json',{'rows':rows,'scenes':len(rows),'tp':tp,'fp':fp,'gt':gt,'recall':tp/max(gt,1),'precision':tp/max(tp+fp,1),
            'centre_error_m':sum(r['class_filtered_centre_error_sum'] for r in rows)/max(tp,1),'scope':'fitting scenes only, no generalization claim'})
        head.train()
    evaluate(0)
    for step in range(1,a.steps+1):
        ids=[]
        for _ in range(a.batch):
            if offset==len(order):order=torch.randperm(len(targets),generator=generator).tolist();offset=0
            ids.append(order[offset]);offset+=1
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=head(features[ids])
        sums,counts,_=reference_loss_sums(prediction,[targets[i] for i in ids])
        loss=sum(sums[k]/max(counts[k],1) for k in ('cls','box'))
        if not torch.isfinite(loss):raise FloatingPointError('Current-head probe nonfinite')
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
        row={'step':step,'loss':float(loss.detach()),'loss_cls_box':{k:float(sums[k].detach())/max(counts[k],1) for k in ('cls','box')},
            'presentations':step*a.batch,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        atomic_json(out/'progress.json',{'step':step,'presentations':step*a.batch})
        stop=(out/'STOP_REQUESTED').exists()
        if step in (100,250,a.steps) or stop:evaluate(step)
        if stop:break
    torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'step':step,'sampler_order':order,'sampler_offset':offset,'sampler_rng':generator.get_state(),
        'purpose':'diagnostic only; not a complete/resumable public foundation or graph checkpoint'},out/'diagnostic_head.pt')
    atomic_json(out/'status.json',{'status':'complete' if step==a.steps else 'paused','step':step})


if __name__=='__main__':main()
