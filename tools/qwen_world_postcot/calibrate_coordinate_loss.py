"""Finite TRAIN-only physical-error and output-layer gradient calibration."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.model.modules.qwen_world_postcot.flow import shift_sigma
from starVLA.rl.qwen_world_postcot.contracts import require_dispatch, file_sha, fingerprint
from starVLA.rl.qwen_world_postcot.coordinate_objective import tolerance_weights
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_postcot.loading import load_deployment_checkpoint
from starVLA.rl.flow_grpo.reproducibility import configure_numerics


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();require_dispatch();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    configure_numerics();torch.set_num_threads(4);torch.manual_seed(42)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);device=torch.device('cuda:0');started=time.monotonic()
    model,info=load_deployment_checkpoint(a.parent,device);model.actor.requires_grad_(True)
    data=WorldSceneData(info['identity']['config']['training_data'])
    cursor=info['seal']['cursor'];tokens=[data.tokens[data.index(cursor+i)] for i in range(32)]
    processor=QwenObservationProcessor('/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct')
    parameters=(model.actor.action_output.weight,model.actor.action_output.bias,
                model.actor.output_norm.weight,model.actor.output_norm.bias)
    streams=torch.Generator(device='cpu').manual_seed(4263000)
    noise=torch.randn((32,8,3),generator=streams);sigma=shift_sigma(torch.rand(32,generator=streams))
    denominator=sum(int(((torch.tensor(data.records[t]['raw_future_dt_seconds']).cumsum(0)-torch.arange(1,9)*.5).abs()<.06).sum()) for t in tokens)*3
    grads=[None]*3;errors=[];valids=[];memory_grad_norms=[]
    for offset in range(0,32,2):
        selected=tokens[offset:offset+2];obs=[data.observation(t) for t in selected]
        inputs=processor(obs);ego=torch.tensor([o['proprio'] for o in obs],device=device)
        target=ActionCodec.norm_odo(torch.stack([data.target(t) for t in selected])).to(device)
        valid=torch.stack([(torch.tensor(data.records[t]['raw_future_dt_seconds']).cumsum(0)-torch.arange(1,9)*.5).abs()<.06 for t in selected]).to(device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            memory=model.encode_condition(inputs,ego)
        # Only the diagnostic memory leaf is enabled; no encoder parameters
        # update or cache enters either formal full-model continuation.
        leaf=memory.actor_memory.detach().requires_grad_(True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            context=model.actor.prepare_context(leaf,memory.actor_valid,ego)
            s=sigma[offset:offset+2].to(device);eps=noise[offset:offset+2].to(device)
            prediction=model.actor((1-s[:,None,None])*target+s[:,None,None]*eps,s,context)
        error=(prediction.float()-target).abs()*valid[:,:,None]
        errors.append(error.detach().cpu());valids.append(valid.cpu())
        # Add vectors before measuring norm: this matches the actual gradient
        # of a global scene mean, rather than averaging microbatch norms.
        for channel in range(3):
            gradient=torch.autograd.grad(error[:,:,channel].sum()/denominator,parameters+(leaf,),retain_graph=channel<2)
            vector=torch.cat([v.detach().float().cpu().flatten() for v in gradient[:-1]])
            grads[channel]=vector if grads[channel] is None else grads[channel]+vector
            memory_grad_norms.append(dict(offset=offset,channel=channel,norm=float(gradient[-1].float().norm())))
        del memory,leaf,context,prediction,error
    errors=torch.cat(errors);valid=torch.cat(valids);count=valid.sum(0)[:,None].expand(8,3).float()
    weights=torch.tensor(tolerance_weights());baseline=sum(grads);weighted=sum(w*g for w,g in zip(weights,grads))
    scale=float(baseline.norm()/weighted.norm())
    if not math.isfinite(scale) or not .25<=scale<=4:
        raise ValueError('Calibration outside preregistered numerical scale bounds')
    gram=torch.stack([torch.stack([torch.dot(x,y) for y in grads]) for x in grads])
    norms=gram.diag().sqrt();cos=gram/(norms[:,None]*norms[None,:]).clamp_min(1e-30)
    physical=errors.sum(0)/count.clamp_min(1)*torch.tensor([3337.,2100.,1.765*180/math.pi])
    row=dict(schema='world_coordinate_TRAIN_calibration_v1',status='PASS',scenes=32,seed=42,
        parent_model_sha256=info['model_sha256'],parent_identity_sha256=fingerprint(info['identity']),
        data_sha256=data.sha,token_sha256=fingerprint(tokens),split='TRAIN',
        tokens=tokens,physical_MAE_by_time_channel=physical.tolist(),units=['cm','cm','degree_unwrapped'],
        physical_tolerances=[.5,.25,math.radians(2)],normalized_weights=weights.tolist(),
        baseline_output_layer_gradient_norm=float(baseline.norm()),
        physical_output_layer_gradient_norm_before_scale=float(weighted.norm()),fixed_scale=scale,
        component_gradient_gram=gram.tolist(),component_gradient_cosine=cos.tolist(),
        gradient_scope=['actor.action_output.weight','actor.action_output.bias','actor.output_norm.weight','actor.output_norm.bias'],
        memory_input_gradient_norms=memory_grad_norms,sigma=sigma.tolist(),
        warning='Output-layer norm calibration, not equality of full-model gradients. No DEV/Test inputs. Tolerances fixed before calibration; no per-batch whitening.',
        new_model_forwards=16,optimizer_updates=0,scorer_calls=0,wall_seconds=time.monotonic()-started,
        peak_allocated=torch.cuda.max_memory_allocated(),code_sha256=file_sha(__file__))
    row['sha256']=fingerprint(row);(out/'CALIBRATION.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps({k:row[k] for k in ('status','scenes','fixed_scale','baseline_output_layer_gradient_norm','physical_output_layer_gradient_norm_before_scale','wall_seconds','peak_allocated')}),flush=True)


if __name__=='__main__':main()
