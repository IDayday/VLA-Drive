"""All-time DINO and frozen GT-MAE diagnostics, separate from official planning."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
import torch
from torch.nn import functional as F
from starVLA.dataloader.full_foresight_dataset import FullForesightDataset
from tools.foresight.checkpoints import checkpoint_identity, load_student
from tools.foresight.evaluate_auxiliary_tasks import interaction_statistics
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.ddpolicy_vehicle.campaign import charged_gpu_hours


class FixedPositionMoments:
    """Scene variance at fixed coordinates; a fixed rich template has zero variance."""
    def __init__(self):
        self.count = self.total = self.square = None

    def add(self, value, valid):
        active = torch.broadcast_to(valid, value.shape)
        if valid.dtype != torch.bool or not torch.isfinite(value[active]).all():
            raise ValueError('Invalid fixed-coordinate variance input')
        clean = torch.where(active, value.double(), 0.).cpu()
        if self.total is None:
            self.total = torch.zeros_like(clean)
            self.square = torch.zeros_like(clean)
            self.count = torch.zeros_like(clean)
        self.total += clean
        self.square += clean.square()
        self.count += active.cpu()

    def result(self):
        if self.count is None:
            return None
        n = self.count.clamp_min(1)
        variance = (self.square/n-(self.total/n).square()).clamp_min(0)
        selected = self.count > 1
        return float(variance[selected].mean()) if selected.any() else None


def masked_feature_error(prediction, target, valid):
    if prediction.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('Reference/target shape mismatch')
    mask = torch.broadcast_to(valid, target.shape)
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError('Nonfinite valid reference/target')
    p = torch.where(mask, prediction.float(), 0.)
    t = torch.where(mask, target.float(), 0.)
    return {'squared_error': float((p-t).square().sum()), 'elements': int(mask.sum())}


def frozen_mae_decode(teacher, latent, current_anchor):
    """The exact raw-Z teacher interface: reconstruct already contains LayerNorm."""
    return teacher.reconstruct(latent.float()) * teacher.xy_scale + current_anchor[..., None, :]


def spatial_statistics(prediction, target, current, valid):
    if prediction.shape!=target.shape or current.shape!=target.shape or prediction.ndim!=4:
        raise ValueError('Expected three separate C,H,W view tensors')
    if valid.shape!=(prediction.shape[0],*prediction.shape[-2:]) or valid.dtype!=torch.bool:
        raise ValueError('Invalid spatial mask')
    active=valid[:,None].expand_as(target)
    if any(not torch.isfinite(x[active]).all() for x in (prediction,target,current)):
        raise ValueError('Nonfinite valid spatial feature')
    values=[torch.where(active,x.float(),0.) for x in (prediction,target,current)]
    p,t,c=values;rows=[]
    for i in range(len(p)):
        mask=valid[i];count=int(mask.sum());channels=p.shape[1]
        norms_p=p[i].norm(dim=0);norms_t=t[i].norm(dim=0)
        cosine=F.cosine_similarity(p[i],t[i],dim=0,eps=1e-8)
        rows.append({'elements':count*channels,'patches':count,
            'squared_error':float((p[i]-t[i]).square().sum()),
            'copy_current_squared_error':float((c[i]-t[i]).square().sum()),
            'cosine_sum':float(cosine[mask].sum()),'prediction_norm_sum':float(norms_p[mask].sum()),
            'target_norm_sum':float(norms_t[mask].sum()),
            'prediction_sum':float(p[i].sum()),'prediction_square_sum':float(p[i].square().sum()),
            'target_sum':float(t[i].sum()),'target_square_sum':float(t[i].square().sum())})
    return rows


def summarize(rows, requested):
    result={'scenes':len(rows),'requested':requested,'failed':sum(x['failure'] is not None for x in rows),
            'complete':len(rows)==requested,'visual':{},'interaction':{},'scope':'label-side auxiliary diagnostics, not PDMS'}
    for h in (0,1,2,4):
        for view in range(3):
            key=f'h{h}_v{view}';items=[x[key] for x in rows if key in x]
            sums={k:sum(x[k] for x in items) for k in items[0]} if items else {}
            n=sums.get('elements',0);patches=sums.get('patches',0)
            record={'elements':n,'patches':patches,'valid_scenes':sum(x['elements']>0 for x in items)}
            if n:
                record.update(mse=sums['squared_error']/n,cosine=sums['cosine_sum']/patches,
                    prediction_norm=sums['prediction_norm_sum']/patches,target_norm=sums['target_norm_sum']/patches,
                    pooled_prediction_variance=sums['prediction_square_sum']/n-(sums['prediction_sum']/n)**2,
                    pooled_target_variance=sums['target_square_sum']/n-(sums['target_sum']/n)**2)
                if h:
                    copy=sums['copy_current_squared_error'];record.update(copy_current_mse=copy/n,
                        relative_improvement_over_copy=1-sums['squared_error']/copy if copy>0 else None)
            result['visual'][key]=record
    n=sum(x.get('interaction',{}).get('elements',0) for x in rows)
    error=sum(x.get('interaction',{}).get('squared_error',0.) for x in rows)
    result['interaction']={'elements':n,'valid_scenes':sum(x.get('interaction',{}).get('elements',0)>0 for x in rows),
                           'mse':error/n if n else None}
    result['valid']=result['complete'] and result['failed']==0
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('training-run','checkpoint-tag','data','dino-root','dino-index','output','campaign-root','run-id'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--interaction-root');p.add_argument('--local-image-root');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--campaign-gpu-hours',type=float,required=True);a=p.parse_args()
    if min(a.max_seconds,a.campaign_gpu_hours)<=0 or a.limit<0:raise ValueError('Invalid evaluation bound')
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'full_method_auxiliary_evaluation','real_optimizer_updates':0}) as (meter,_,save):
        if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze evaluation source')
        training,checkpoint=checkpoint_identity(a.training_run,a.checkpoint_tag)
        if training['schema']!='ddp_full_foresight_student_v1':raise ValueError('Full-method checkpoint required')
        cfg=training['config']['foresight'];di=json.loads((Path(a.dino_root)/'identity.json').read_text())
        ii=json.loads((Path(a.interaction_root)/'identity.json').read_text()) if a.interaction_root else None
        if bool(ii)!=cfg['enable_interaction']:raise ValueError('Interaction evaluation inventory changed')
        data=FullForesightDataset(a.data,candidate=cfg['candidate'],current=True,future=True,
            dino_root=a.dino_root,dino_index=a.dino_index,expected_dino=di['identity'],allow_partial=bool(a.limit),
            interaction_root=a.interaction_root,expected_interaction=ii['identity'] if ii else None,image_root=a.local_image_root)
        if data.identity['split'] not in ('train','dev'):raise ValueError('No auxiliary Navtest model selection')
        identity={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  'checkpoint':checkpoint,'dino':di['identity'],'interaction':ii,'data':data.identity,'limit':a.limit}
        out=Path(a.output)
        if out.exists():
            if not a.resume or json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Evaluation resume identity mismatch')
        else:out.mkdir(parents=True);atomic_json(out/'identity.json',identity)
        path=out/'scenes.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        if [x['token'] for x in rows]!=[x['token'] for x in data.index[:len(rows)]]:raise ValueError('Partial population mismatch')
        requested=min(a.limit or len(data),len(data))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_student(a.training_run,a.checkpoint_tag,training,strip=False)
        for i in range(len(rows),requested):
            if time.time()-meter['start_unix']>=a.max_seconds or charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:
                meter['status']='PAUSED';break
            row={**data.index[i],'failure':None}
            try:
                observation,targets=data[i]
                with torch.inference_mode():
                    world=model.encode_current([observation])['W']
                    if hasattr(model,'dino_head'):
                        cur=targets['current_dino'].cuda()
                        for hidx,h in enumerate((0,1,2,4)):
                            target=cur if not h else targets['future_dino'][hidx-1].cuda()
                            valid=targets['current_dino_valid'].cuda() if not h else targets['future_dino_valid'][hidx-1].cuda()
                            predicted=model.dino_head(world,torch.tensor([float(h)],device='cuda'),target.shape[-2:])[0]
                            for view,stats in enumerate(spatial_statistics(predicted,target,cur,valid)):
                                row[f'h{h}_v{view}']=stats
                    if hasattr(model,'interaction_head'):
                        row['interaction']=interaction_statistics(model.interaction_head(world)[0],
                            targets['interaction_latent'].cuda(),bool(targets['interaction_valid']),model.foresight_config.normalization_eps)
            except Exception as error:row={**data.index[i],'failure':repr(error)}
            with path.open('a') as stream:stream.write(json.dumps(row)+'\n')
            rows.append(row);meter['inference_scenes']=len(rows);save()
        flat=[]
        for row in rows:
            item={}
            for k,v in row.items():
                if isinstance(v,dict):item.update({k+'_'+sub:value for sub,value in v.items()})
                else:item[k]=v
            flat.append(item)
        with (out/'scenes.csv').open('w') as stream:
            writer=csv.DictWriter(stream,sorted(set().union(*(x.keys() for x in flat))) if flat else ['token','log','failure'])
            writer.writeheader();writer.writerows(flat)
        result=summarize(rows,requested);atomic_json(out/'summary.json',result)
        if result['failed']:raise RuntimeError('Failed auxiliary rows retained; no complete result claimed')

if __name__=='__main__':main()
