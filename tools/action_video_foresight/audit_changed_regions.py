"""Fixed train-derived DINO-change groups; frozen old models, zero updates.

Feature change may reflect camera motion or appearance as well as object motion.
This is not a motion segmentation label. The upper training quartile is fixed
before evaluating development errors and is never tuned to an improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from starVLA.dataloader.full_foresight_dataset import FullForesightDataset
from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
from starVLA.model.modules.vehicle_joint.initialization import identity_hash
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.foresight.checkpoints import checkpoint_identity,load_student
from tools.full_foresight.evaluate_auxiliary import masked_feature_error

H=(0,1,2,4)


def change_values(current,future,valid):
    if current.shape!=future.shape or valid.shape!=current.shape[-2:] or valid.dtype!=torch.bool:
        raise ValueError('Change group shape/mask mismatch')
    mask=valid[None].expand_as(current)
    if not torch.isfinite(current[mask]).all() or not torch.isfinite(future[mask]).all():
        raise ValueError('Illegal valid change reference')
    c=torch.where(mask,current.float(),0.);f=torch.where(mask,future.float(),0.)
    return (f-c).square().mean(0)


def label_features(ds,i):
    pairs=[[ds.read_image(j) for j in row] for row in ds.dino_scenes[i]['images']]
    return torch.stack([torch.stack([p[0] for p in row]) for row in pairs]),torch.stack(
        [torch.stack([p[1] for p in row]) for row in pairs])


def data(args,split):
    root=Path(args.dino_root);identity=json.loads((root/'identity.json').read_text())
    return FullForesightDataset(getattr(args,split+'_data'),candidate=args.candidate,current=True,future=True,
        dino_root=root,dino_index=args.dino_index,expected_dino=identity['identity'],allow_partial=False)


def thresholds(train,queries):
    mapping={r['token']:i for i,r in enumerate(train.index)};values={}
    if not queries or any(train.index[mapping[r['token']]]!=r for r in queries):
        raise ValueError('Change threshold population must be fixed training queries')
    for row in queries:
        target,valid=label_features(train,mapping[row['token']])
        for h in range(1,4):
            for v in range(3):
                m=valid[0,v]&valid[h,v]
                values.setdefault((h,v),[]).append(change_values(target[0,v],target[h,v],m)[m])
    result={}
    for key,tensors in values.items():
        all_values=torch.cat(tensors)
        if not len(all_values):raise ValueError('No training reference population for a change group')
        result[f'h{H[key[0]]}_v{key[1]}']=float(torch.quantile(all_values,.75))
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    for key in ('training-run','checkpoint-tag','train-data','dev-data','train-queries',
                'dino-root','dino-index','output','campaign-root','run-id'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--candidate',choices=('C0','C1'),required=True)
    parser.add_argument('--representations')
    parser.add_argument('--dev-queries')
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze diagnostic source')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    with metered_run(args.campaign_root,args.run_id,1,{'kind':'fixed_train_change_group_audit','real_optimizer_updates':0}) as (meter,_,save):
        train=data(args,'train');dev=data(args,'dev')
        if {r['log'] for r in train.index}&{r['log'] for r in dev.index}:raise ValueError('Threshold/development log overlap')
        train_queries=json.loads(Path(args.train_queries).read_text())
        cutoff=thresholds(train,train_queries)
        selected=json.loads(Path(args.dev_queries).read_text()) if args.dev_queries else dev.index
        mapping={r['token']:i for i,r in enumerate(dev.index)}
        if len({r['token'] for r in selected})!=len(selected) or any(dev.index[mapping[r['token']]]!=r for r in selected):
            raise ValueError('Fixed development query identity changed')
        training,cp=checkpoint_identity(args.training_run,args.checkpoint_tag)
        if cp['completed']!=100000 or cp['arm']!=args.candidate:raise ValueError('Matched old100k endpoint required')
        atomic_json(out/'identity.json',{'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'checkpoint':cp,'train_queries':identity_hash(train_queries),'dev_queries':identity_hash(selected),
            'train_dino_identity':train.dino_identity['identity'],'rule':'upper training quartile per horizon/view of raw feature squared change',
            'thresholds':cutoff,'precision':'FP32 master/FP32 readout, TF32 off',
            'scope':'feature-change stratification; camera/appearance changes included; not semantic motion labels or PDMS'})
        model=load_student(args.training_run,args.checkpoint_tag,training,strip=False)
        rep_root=Path(args.representations)/'representations' if args.representations else None
        current_data=ForesightCurrentDataset(args.dev_data) if rep_root is None else None
        world={}
        def get_w(row):
            if row['token'] not in world:
                if rep_root:
                    saved=torch.load(rep_root/(row['token']+'.pt'),weights_only=True)
                    if saved['checkpoint']!=cp['sha256']:raise ValueError('Wrong frozen W checkpoint')
                    value=saved['W'][None].float()
                else:
                    with torch.inference_mode():value=model.encode_current([current_data[mapping[row['token']]]])['W'].cpu()
                world[row['token']]=value
            return world[row['token']].cuda()
        order=sorted(range(len(selected)),key=lambda i:hashlib.sha256(('fixed-change-W-v1:'+selected[i]['token']).encode()).hexdigest())
        swap={order[i]:order[(i+len(order)//2)%len(order)] for i in range(len(order))}
        rows=[];sums={}
        for i,row in enumerate(selected):
            result={**row,'failure':None,'swap_token':selected[swap[i]]['token']}
            try:
                target,valid=label_features(dev,mapping[row['token']]);target=target.cuda();valid=valid.cuda()
                w=get_w(row);sw=get_w(selected[swap[i]])
                with torch.inference_mode():
                    pred=torch.stack([model.dino_head(w,torch.tensor([float(h)],device='cuda'),target.shape[-2:])[0] for h in H])
                    shuffled=torch.stack([model.dino_head(sw,torch.tensor([float(h)],device='cuda'),target.shape[-2:])[0] for h in H])
                for h in range(4):
                    for v in range(3):
                        key=f'h{H[h]}_v{v}'
                        active=valid[h,v] & valid[0,v]
                        groups={'all_joint_valid':active}
                        if h:
                            delta=change_values(target[0,v],target[h,v],active)
                            groups.update(changed_upper_train_quartile=active&(delta>cutoff[key]),
                                other_change=active&(delta<=cutoff[key]))
                        for group,mask in groups.items():
                            for name,value in [('model',pred[h,v]),('copy_actual_current',target[0,v]),
                                               ('copy_predicted_current',pred[0,v]),('shuffled_W',shuffled[h,v])]:
                                score=masked_feature_error(value,target[h,v],mask[None]);tag=key+'_'+group+'_'+name
                                result[tag]=score
                                total=sums.setdefault(tag,{'elements':0,'squared_error':0.,'valid_scenes':0})
                                total['elements']+=score['elements'];total['squared_error']+=score['squared_error'];total['valid_scenes']+=score['elements']>0
            except Exception as error:result['failure']=repr(error)
            rows.append(result)
            with (out/'scenes.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
            meter['inference_scenes']=len(rows);save()
        failed=sum(row['failure'] is not None for row in rows)
        atomic_json(out/'SUMMARY.json',{'requested':len(selected),'scenes':len(rows),'logs':len({r['log'] for r in selected}),
            'failed':failed,'candidate':args.candidate,'thresholds':cutoff,'real_optimizer_updates':0,
            'scope':'full C1 or preselected limited C0 dev diagnostic, not a new method planning result',
            'groups':{k:{**v,'mse':v['squared_error']/v['elements'] if v['elements'] and not failed else None} for k,v in sums.items()}})
        if failed:raise RuntimeError('Diagnostic failures preserved; no complete scientific result')


if __name__=='__main__':main()
