"""Label-side latent diagnostics. These are NOT planning/PDMS results.

Encode the current cameras once, then read the same W for every horizon/view
and the teacher's fixed ego-time latent. Missing labels preserve scene rows.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from starVLA.dataloader.foresight_dataset import ForesightTrainingDataset
from starVLA.model.modules.vehicle_joint.initialization import identity_hash
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.ddpolicy_vehicle.campaign import charged_gpu_hours
from .checkpoints import checkpoint_identity,load_student


def visual_statistics(prediction,future,current,valid):
    if prediction.shape!=future.shape or current.shape!=future.shape or valid.shape!=(len(future),) or valid.dtype!=torch.bool:
        raise ValueError('Expected per-view C,H,W latent arrays and boolean availability')
    active=valid[:,None,None,None]
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(future[valid]).all() or not torch.isfinite(current[valid]).all():
        raise ValueError('Nonfinite valid visual latent')
    p=torch.where(active,prediction.float(),0.);f=torch.where(active,future.float(),0.);c=torch.where(active,current.float(),0.)
    n=prediction[0].numel()
    return [{'elements':n if bool(v) else 0,'prediction_squared_error':float((p[i]-f[i]).square().sum()),
             'copy_current_squared_error':float((c[i]-f[i]).square().sum())} for i,v in enumerate(valid)]


def interaction_statistics(prediction,target,valid,eps):
    if prediction.shape!=target.shape or prediction.shape!=(8,512):raise ValueError('Fixed ego-time latent contract')
    if not valid:return {'elements':0,'squared_error':0.}
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():raise ValueError('Nonfinite interaction latent')
    error=(F.layer_norm(prediction.float(),(512,),eps=eps)-F.layer_norm(target.float(),(512,),eps=eps)).square()
    return {'elements':error.numel(),'squared_error':float(error.sum())}


def summarize(rows):
    summary={'scenes':len(rows),'failed':sum(r['failure'] is not None for r in rows),'visual':{},'interaction':{},
             'scope':'auxiliary regression only; no planning claim; all horizons evaluated'}
    for h in (1,2,4):
        for v in range(3):
            key=f'visual_{h}s_v{v}';count=sum(r.get(key+'_elements',0) for r in rows)
            pred=sum(r.get(key+'_squared_error',0.) for r in rows);copy=sum(r.get(key+'_copy_squared_error',0.) for r in rows)
            summary['visual'][key]={'elements':count,'valid_scenes':sum(r.get(key+'_elements',0)>0 for r in rows),
                'mse':pred/count if count else None,'copy_current_mse':copy/count if count else None,
                'relative_improvement_over_copy':1-pred/copy if copy>0 else None}
    count=sum(r.get('interaction_elements',0) for r in rows);error=sum(r.get('interaction_squared_error',0.) for r in rows)
    summary['interaction']={'elements':count,'valid_scenes':sum(r.get('interaction_elements',0)>0 for r in rows),'mse':error/count if count else None}
    summary['valid_complete_result']=bool(rows) and not summary['failed']
    return summary


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('training-run','checkpoint-tag','data','output','campaign-root','run-id'):p.add_argument('--'+key,required=True)
    for key in ('future-root','future-identity','interaction-root','interaction-identity'):p.add_argument('--'+key)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--campaign-gpu-hours',type=float,default=6000)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if min(a.max_seconds,a.campaign_gpu_hours)<=0 or a.limit<0:raise ValueError('Invalid budget/limit')
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'auxiliary_evaluation','real_optimizer_updates':0}) as (meter,_,save):
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze evaluation source')
        if charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:raise RuntimeError('Campaign budget exhausted')
        training,checkpoint=checkpoint_identity(a.training_run,a.checkpoint_tag);arm=training['config']['foresight']['arm']
        if arm not in ('B','C','D') or bool(a.future_root)!=(arm in ('B','D')) or bool(a.interaction_root)!=(arm in ('C','D')):
            raise ValueError('Auxiliary cache/arm inventory mismatch')
        data=ForesightTrainingDataset(a.data,future_root=a.future_root,expected_future=a.future_identity,
                                     interaction_root=a.interaction_root,expected_interaction=a.interaction_identity)
        if data.identity['split'] not in ('train','dev'):raise ValueError('Auxiliary diagnostics must not use Navtest for model selection')
        identity={'checkpoint':checkpoint,'data':data.identity,'future':data.future_identity,'interaction':data.interaction_identity,
                  'source_sha':source,'limit':a.limit,'copy_reference':'same encoded current, FP16 cache converted to FP32 errors'}
        out=Path(a.output)
        if out.exists():
            if not a.resume or json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Auxiliary evaluation identity mismatch')
        else:out.mkdir(parents=True);atomic_json(out/'identity.json',identity)
        model=load_student(a.training_run,a.checkpoint_tag,training,strip=False);rows=[];requested=min(a.limit or len(data),len(data))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        record_path=out/'scenes.jsonl'
        if record_path.exists():rows=[json.loads(line) for line in record_path.read_text().splitlines()]
        if [r['token'] for r in rows]!=[r['token'] for r in data.index[:len(rows)]]:raise ValueError('Partial scene population changed')
        for i in range(len(rows),requested):
            if time.time()-meter['start_unix']>=a.max_seconds or charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:
                meter['status']='PAUSED';break
            scene=data.index[i];row={'token':scene['token'],'log':scene['log'],'failure':None}
            try:
                observation,targets=data[i]
                with torch.inference_mode():
                    W=model.encode_current([observation])['W']
                    if a.future_root:
                        folder=data.future_shards[i%data.future_identity['shards']]
                        raw=torch.load(folder/'targets'/(scene['token']+'.pt'),map_location='cpu',weights_only=True)
                        current=raw['current_latent'].cuda();future=targets['future_latent'].cuda();valid=targets['future_valid'].cuda()
                        for h,horizon in enumerate((1,2,4)):
                            prediction=model.future_head(W,torch.tensor([h],device='cuda'),future.shape[-2:])[0]
                            for v,stats in enumerate(visual_statistics(prediction,future[h],current,valid[h])):
                                key=f'visual_{horizon}s_v{v}';row[key+'_elements']=stats['elements']
                                row[key+'_squared_error']=stats['prediction_squared_error'];row[key+'_copy_squared_error']=stats['copy_current_squared_error']
                    if a.interaction_root:
                        prediction=model.interaction_head(W)[0]
                        stats=interaction_statistics(prediction,targets['interaction_latent'].cuda(),bool(targets['interaction_valid']),model.foresight_config.normalization_eps)
                        row['interaction_elements']=stats['elements'];row['interaction_squared_error']=stats['squared_error']
            except Exception as error:row={'token':scene['token'],'log':scene['log'],'failure':repr(error)}
            with record_path.open('a') as stream:stream.write(json.dumps(row)+'\n')
            rows.append(row);meter.update(inference_scenes=len(rows),failed=sum(r['failure'] is not None for r in rows));save()
        fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['token','log','failure']
        with (out/'scenes.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
        summary=summarize(rows);summary.update(requested=requested,complete=len(rows)==requested)
        summary['valid_complete_result'] &= summary['complete'];atomic_json(out/'summary.json',summary)
        if summary['failed']:raise RuntimeError('Auxiliary failures preserved; aggregate is not a complete result')


if __name__=='__main__':main()
