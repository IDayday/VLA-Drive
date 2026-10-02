"""Train-only mean/static references for the matched frozen-W future probes.

Means live in the actual channel-LayerNorm loss space. The arithmetic mean is
not normalized again; a separate unit-normalized template is also reported.
Different teacher raw losses are never compared as a common quality scale.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from starVLA.dataloader.full_foresight_dataset import FullForesightDataset
from starVLA.model.modules.foresight.future_spatiotemporal_head import FutureSpatiotemporalHead
from starVLA.model.modules.vehicle_joint.initialization import identity_hash
from tools.action_video_foresight.train_frozen_W_probe import ProbeData
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def normalized_values(values, valid):
    if valid.dtype!=torch.bool or valid.shape!=values.shape[:-1]:
        raise ValueError('Boolean spatial/temporal mask required')
    if not torch.isfinite(values[valid]).all():
        raise ValueError('Illegal valid reference value')
    return F.layer_norm(torch.where(valid[...,None],values.float(),0.),(values.shape[-1],),eps=1e-5)


def errors_in_normalized_space(prediction,target,valid):
    error=(prediction-target).square().mean(-1)
    count=valid.sum((2,3,4));views=count>0
    by_view=(error*valid).sum((2,3,4))/count.clamp_min(1)
    by_scene=(by_view*views).sum(-1)/views.sum(-1).clamp_min(1)
    return [float(by_scene[i]) if views[i].any() else None for i in range(len(target))]


def fit_mean(data, path):
    identity={'train_queries':identity_hash(data.index),'train_targets':data.identity['identity'],
              'space':'mean of channel-LayerNorm target, no second norm on arithmetic mean',
              'split':'train','scenes':len(data.index)}
    if path.exists():
        saved=torch.load(path,weights_only=True)
        if saved['identity']!=identity:
            raise ValueError('Train mean identity mismatch')
        return saved['mean']
    sums=torch.zeros(data.identity['target_shape'],dtype=torch.float64)
    counts=torch.zeros(data.identity['target_shape'][:-1],dtype=torch.int64)
    for start in range(0,len(data.index),8):
        _,_,target,valid=data.batch(list(range(start,min(start+8,len(data.index)))),'cpu')
        normalized=normalized_values(target,valid)
        sums+=(normalized*valid[...,None]).double().sum(0)
        counts+=valid.sum(0)
    if (counts==0).any():
        raise ValueError('No training population for some reference positions')
    mean=(sums/counts[...,None]).float()
    temporary=path.with_suffix('.tmp')
    torch.save({'identity':identity,'mean':mean,'counts':counts},temporary);temporary.replace(path)
    return mean


class DinoCurrentReference:
    def __init__(self,current,cache,index,future_identity):
        ident=json.loads((Path(cache)/'identity.json').read_text())
        old,new=ident['recipe'],future_identity['recipe']
        for key in ('repository','revision','weight_sha256','config_sha256','forward_source_sha256',
                    'feature','extra_norm','rope','prefix_tokens_excluded','patch_size','feature_dim',
                    'mean','std','interpolation'):
            if old[key]!=new[key]:
                raise ValueError('Current/static DINO encoding differs from new sequence')
        if old['encoder_implementation']!=new['implementation_sha256'] or ident['candidate']['width']!=384 or ident['candidate']['height']!=288 or ident['candidate']['pool']!=2:
            raise ValueError('Wrong spatial or encoder recipe for static reference')
        self.data=FullForesightDataset(current,candidate='C3',current=True,future=False,
            dino_root=cache,dino_index=index,expected_dino=ident['identity'])
        self.rows={r['token']:r for r in self.data.dino_scenes}

    def get(self,token,steps):
        pairs=[self.data.read_image(i) for i in self.rows[token]['images'][0]]
        features=torch.stack([z for z,_ in pairs]).permute(0,2,3,1)
        valid=torch.stack([v for _,v in pairs])
        return features[:,None].expand(-1,steps,-1,-1,-1),valid[:,None].expand(-1,steps,-1,-1)


def main():
    parser=argparse.ArgumentParser(__doc__)
    for key in ('representations','train-data','dev-data','train-targets','dev-targets',
                'probe','mean-output','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--static-video-reference')
    parser.add_argument('--dino-current-cache')
    parser.add_argument('--dino-index')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    train=ProbeData(args.representations,args.train_data,args.train_targets,'train')
    dev=ProbeData(args.representations,args.dev_data,args.dev_targets,'dev')
    if {r['log'] for r in train.index}&{r['log'] for r in dev.index}:
        raise ValueError('Reference/probe log leakage')
    mean=fit_mean(train,Path(args.mean_output)).cuda()
    checkpoint=torch.load(Path(args.probe)/'latest.pt',map_location='cpu',weights_only=False)
    identity=checkpoint['identity']
    if (identity['train_queries'],identity['dev_queries'],identity['train_targets'],identity['dev_targets'])!=(
            identity_hash(train.index),identity_hash(dev.index),train.identity['identity'],dev.identity['identity']) or checkpoint['completed']!=identity['updates']:
        raise ValueError('Probe is not the complete registered comparison')
    # The actual saved projection size, rather than a backbone name, is authoritative.
    hidden=checkpoint['model']['project.weight'].shape[1]
    model=FutureSpatiotemporalHead(hidden,dev.identity['target_shape'][-1],dev.identity['time_intervals_s'],
        action_condition=identity['condition'],use_world=identity['use_world'])
    model.load_state_dict(checkpoint['model'],strict=True);model.cuda().eval()
    static=None
    if dev.identity['future_target_type']=='dino_sequence':
        static=DinoCurrentReference(args.dev_data,args.dino_current_cache,args.dino_index,dev.identity)
    else:
        static_root=Path(args.static_video_reference)
        reference=json.loads((static_root/'identity.json').read_text())
        complete=json.loads((static_root/'COMPLETE.json').read_text())
        if reference['schema']!='action_video_static_reference_v1' or reference['future_target_identity']!=dev.identity['identity'] or complete['identity']!=reference['identity'] or complete['scenes']!=len(dev.index):
            raise ValueError('Genuine same-encoder static video reference required')
    rows=[]
    with torch.inference_mode():
        for start in range(0,len(dev.index),8):
            ids=list(range(start,min(start+8,len(dev.index))));w,action,target,valid=dev.batch(ids,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(w if model.use_world else None,(9,12),gt_action=action if model.action_condition=='gt_ego' else None)
            normalized_target=normalized_values(target,valid)
            scores={'model':errors_in_normalized_space(normalized_values(pred,valid),normalized_target,valid),
                'training_mean':errors_in_normalized_space(mean[None].expand_as(target),normalized_target,valid),
                'unit_normalized_training_mean':errors_in_normalized_space(normalized_values(mean[None].expand_as(target),valid),normalized_target,valid)}
            statics=[];masks=[]
            for i in ids:
                if static is not None:
                    z,m=static.get(dev.index[i]['token'],target.shape[2])
                else:
                    value=torch.load(static_root/'targets'/(dev.index[i]['token']+'.pt'),weights_only=True)
                    if value['identity']!=reference['identity']:
                        raise ValueError('Static video scene identity changed')
                    z=value['features'];m=torch.ones(z.shape[:-1],dtype=torch.bool)
                statics.append(z);masks.append(m)
            statics=torch.stack(statics).cuda();mask=torch.stack(masks).cuda()&valid
            if not torch.equal(mask,valid):
                raise ValueError('Reference invalidity must not shrink scientific denominator')
            scores['static_current_reference']=errors_in_normalized_space(normalized_values(statics,mask),normalized_target,mask)
            for j,i in enumerate(ids):
                rows.append({**dev.index[i],**{k:v[j] for k,v in scores.items()},'failure':None})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    with (out/'scenes.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    summary={'requested':len(dev.index),'scenes':len(rows),'failed':0,'probe_identity':identity,
             'scope':'frozen readout references, not deployment/PDMS; no target-type raw-MSE ranking'}
    for key in scores:
        values=[r[key] for r in rows if r[key] is not None]
        summary[key]={'normalized_mse':sum(values)/len(values),'valid_scenes':len(values)}
    atomic_json(out/'SUMMARY.json',summary)


if __name__=='__main__':
    main()
