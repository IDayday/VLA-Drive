"""One TRAIN-only bounded residual/time and output-layer gradient calibration."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.model.modules.qwen_world_postcot.flow import shift_sigma
from starVLA.rl.qwen_world_postcot.contracts import require_dispatch,file_sha,fingerprint
from starVLA.rl.qwen_world_postcot.loss_designs import LossDesign
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_postcot.loading import load_deployment_checkpoint
from starVLA.rl.flow_grpo.reproducibility import configure_numerics


def residual_time_weights(data):
    records=data.records;tokens=data.tokens
    gt=np.array([records[t]['requested_GT'] for t in tokens],dtype=np.float64)
    ego=np.array([records[t]['proprio'] for t in tokens],dtype=np.float64)
    dt=np.array([records[t]['raw_future_dt_seconds'] for t in tokens]);times=np.arange(1,9)*.5
    valid=np.abs(np.cumsum(dt,axis=1)-times)<.06
    ca=ego[:,None,:2]*times[None,:,None]+.5*ego[:,None,2:4]*times[None,:,None]**2
    residual=gt[:,:,:2]-ca;mad=[]
    for t in range(8):
        selected=residual[valid[:,t],t]
        if not len(selected) or not np.isfinite(selected).all():raise ValueError('Invalid TRAIN residual statistics')
        median=np.median(selected,axis=0);mad.append((1.4826*np.median(np.abs(selected-median),axis=0)).tolist())
    risk=np.sqrt(np.mean((np.array(mad)/np.array([.5,.25]))**2,axis=1))
    raw=np.clip(1/np.maximum(1,risk),.5,2.)
    return dict(weights=(raw/raw.mean()).tolist(),CA_residual_scaled_MAD_xy_m=mad,scenes=len(tokens),
        physical_floor_xy_m=[.5,.25],pre_normalization_bounds=[.5,2.],data_sha256=data.sha,
        interpretation='Robust TRAIN current-CA residual proxy; not irreducible conditional uncertainty; no GT/noise normalization change.')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--parent',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    require_dispatch();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');configure_numerics();torch.set_num_threads(4);torch.manual_seed(42)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();device=torch.device('cuda:0')
    model,info=load_deployment_checkpoint(a.parent,device);model.actor.requires_grad_(True)
    data=WorldSceneData(info['identity']['config']['training_data']);time_stats=residual_time_weights(data)
    tokens=[data.tokens[data.index(info['seal']['cursor']+i)]for i in range(32)]
    processor=QwenObservationProcessor('/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct')
    params=(model.actor.action_output.weight,model.actor.action_output.bias,model.actor.output_norm.weight,model.actor.output_norm.bias)
    rng=torch.Generator().manual_seed(4263000);noise=torch.randn(32,8,3,generator=rng);sigma=shift_sigma(torch.rand(32,generator=rng))
    objectives=dict(CONTROL=LossDesign('CONTROL'),TIME_RESIDUAL=LossDesign('TIME_RESIDUAL',horizon_weights=time_stats['weights']),
        DELTA_UNIT=LossDesign('DELTA',delta_coefficient=1.),HUBER=LossDesign('HUBER'))
    denominator=sum(int(((torch.tensor(data.records[t]['raw_future_dt_seconds']).cumsum(0)-torch.arange(1,9)*.5).abs()<.06).sum())for t in tokens)*3
    gradients={name:None for name in objectives};loss_sums={name:0. for name in objectives}
    for offset in range(0,32,2):
        selected=tokens[offset:offset+2];obs=[data.observation(t)for t in selected]
        ego=torch.tensor([o['proprio']for o in obs],device=device)
        target=ActionCodec.norm_odo(torch.stack([data.target(t)for t in selected])).to(device)
        valid=torch.stack([(torch.tensor(data.records[t]['raw_future_dt_seconds']).cumsum(0)-torch.arange(1,9)*.5).abs()<.06 for t in selected]).to(device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):memory=model.encode_condition(processor(obs),ego)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            context=model.actor.prepare_context(memory.actor_memory,memory.actor_valid,ego)
            s=sigma[offset:offset+2].to(device);eps=noise[offset:offset+2].to(device)
            prediction=model.actor((1-s[:,None,None])*target+s[:,None,None]*eps,s,context)
        for index,(name,objective)in enumerate(objectives.items()):
            _,stats=objective(prediction,target,valid);loss=stats['numerator']/denominator
            grad=torch.autograd.grad(loss,params,retain_graph=index<len(objectives)-1)
            vector=torch.cat([v.detach().float().cpu().flatten()for v in grad]);loss_sums[name]+=float(loss)
            gradients[name]=vector if gradients[name]is None else gradients[name]+vector
        del memory,context,prediction
    baseline=gradients['CONTROL'];delta=gradients['DELTA_UNIT']-baseline
    coefficient=float(.1*baseline.norm()/delta.norm())
    raw=dict(TIME_RESIDUAL=gradients['TIME_RESIDUAL'],DELTA=baseline+coefficient*delta,HUBER=gradients['HUBER'])
    scales={name:float(baseline.norm()/g.norm())for name,g in raw.items()}
    if any(not math.isfinite(v)or not .25<=v<=4 for v in scales.values())or not math.isfinite(coefficient)or coefficient<=0:
        raise ValueError('Calibration outside bounded preregistered gradient scales')
    arms=dict(CONTROL=dict(kind='CONTROL',scale=1.),TIME_RESIDUAL=dict(kind='TIME_RESIDUAL',scale=scales['TIME_RESIDUAL'],horizon_weights=time_stats['weights']),
        DELTA=dict(kind='DELTA',scale=scales['DELTA'],delta_coefficient=coefficient),HUBER=dict(kind='HUBER',scale=scales['HUBER']))
    row=dict(status='PASS',schema='world_IL_loss_designs_TRAIN_calibration_v1',arms=arms,time_statistics=time_stats,
        parent_model_sha256=info['model_sha256'],parent_identity_sha256=fingerprint(info['identity']),split='TRAIN',scenes=32,token_sha256=fingerprint(tokens),
        gradient_scope=['action_output.weight','action_output.bias','output_norm.weight','output_norm.bias'],
        baseline_output_gradient_norm=float(baseline.norm()),delta_unit_output_gradient_norm=float(delta.norm()),delta_target_fraction=.1,
        gradient_cosines={name:float(torch.dot(baseline,g)/(baseline.norm()*g.norm()))for name,g in raw.items()},
        raw_loss_values=loss_sums,new_model_forwards=16,optimizer_updates=0,scorer_calls=0,
        warning='Output-layer gradient calibration only, not full-model gradient matching; no DEV/Test calibration.',
        wall_seconds=time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),code_sha256=file_sha(__file__))
    row['sha256']=fingerprint(row);(out/'CALIBRATION.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',arms=arms,wall_seconds=row['wall_seconds'])),flush=True)


if __name__=='__main__':main()
