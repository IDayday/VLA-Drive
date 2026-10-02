"""Actual RGB/GT-action/video four-loss gradients and pure-current deploy parity."""
import argparse,json,subprocess
from pathlib import Path
import torch
from omegaconf import OmegaConf
from starVLA.model.framework import build_framework
from starVLA.dataloader.action_video_foresight_dataset import ActionVideoForesightDataset
from starVLA.dataloader.foresight_dataset import collate_training,decode_ego
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,module_manifest,tensor_hash
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('config','data','dino-root','dino-index','interaction-root','clip-root','campaign-root','run-id','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--samples',type=int,default=4);a=p.parse_args()
    if not 1<=a.samples<=8:raise ValueError('Bounded fixed training-data calibration')
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze real validation source')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'action_video_real_four_loss_validation','real_optimizer_updates':0}) as (meter,_,save):
        torch.cuda.init()
        cfg=OmegaConf.load(a.config);cfg.foresight.lambda_fut=1.
        di=json.loads((Path(a.dino_root)/'identity.json').read_text());ii=json.loads((Path(a.interaction_root)/'identity.json').read_text());ci=json.loads((Path(a.clip_root)/'identity.json').read_text())
        ds=ActionVideoForesightDataset(a.data,candidate='C1',current=True,future=True,dino_root=a.dino_root,
            dino_index=a.dino_index,expected_dino=di['identity'],interaction_root=a.interaction_root,expected_interaction=ii['identity'],
            clip_root=a.clip_root,expected_clip=ci['identity'],future_type=cfg.foresight.future_target_type,
            expected_teacher=cfg.foresight.video_teacher_identity,allow_partial=True)
        if ds.identity['split']!='train':raise ValueError('Calibration uses training only')
        chosen=[]
        for i in range(64):
            row=ds[i]
            if row[1]['interaction_valid'] and row[1]['future_clip_valid'].all():chosen.append((i,collate_training([row])))
            if len(chosen)==a.samples:break
        if len(chosen)!=a.samples:raise ValueError('Missing real complete calibration labels')
        model=build_framework(cfg).cuda().eval()
        params=dict(model.named_parameters());q=next(p for n,p in params.items() if 'language_model.layers.0.self_attn.q_proj.weight' in n)
        selected={'W':model.foresight_queries,'Qwen_layer0':q}
        gradients=[];qwen_calls=[0]
        hook=model.qwen_vl_interface.model.model.language_model.register_forward_hook(lambda *_:qwen_calls.__setitem__(0,qwen_calls[0]+1))
        action_param=model.action_model.action_decoder.layer2.weight
        for i,(observation,targets) in chosen:
            count=qwen_calls[0]
            output=model.forward_train(observation,targets,completed_updates=1000,
                noise_generator=torch.Generator(device='cuda').manual_seed(9042),time_generator=torch.Generator(device='cuda').manual_seed(10042))
            if qwen_calls[0]-count!=1 or set(output['losses'])!={'ego_fm','current_dino','future_clip','interaction'}:raise AssertionError('One Qwen forward/four real losses required')
            record={}
            for task,loss in output['losses'].items():
                grad=torch.autograd.grad(loss,tuple(selected.values()),retain_graph=True,allow_unused=True)
                norms={k:float(g.float().norm()) if g is not None else 0. for k,g in zip(selected,grad)}
                if any(v<=0 for v in norms.values()) or any(not torch.isfinite(g).all() for g in grad if g is not None):raise AssertionError('Missing shared gradient: '+task)
                record[task]={'weighted_loss_at_unit_future':float(loss.detach()),**norms}
            fut=output['losses']['future_clip']
            forbidden,condition=torch.autograd.grad(fut,(action_param,model.spatiotemporal_head.action_encoder.point[0].weight),retain_graph=True,allow_unused=True)
            if forbidden is not None or condition is None or not condition.abs().sum():raise AssertionError('GT auxiliary gradient split is wrong')
            output['loss'].backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('Nonfinite real backward')
            gradients.append(record);model.zero_grad(set_to_none=True);del output
        hook.remove()
        obs,targets=chosen[0][1]
        with torch.inference_mode():
            encoded=model.encode_current(obs);gt=decode_ego(targets['ego'].cuda())
            physical=torch.cat((gt[...,:2],gt[...,2:3].sin(),gt[...,2:3].cos()),-1)
            with model.amp():
                f0=model.spatiotemporal_head(encoded['W'],(9,12),gt_action=physical)
                f1=model.spatiotemporal_head(encoded['W'],(9,12),gt_action=physical+1.)
            if torch.equal(f0,f1):raise AssertionError('GT condition not actually read')
            noise=torch.zeros(1,8,4,device='cuda');before=model.predict_action(obs,initial_noise=noise)
            poison={**obs[0],'gt_ego':physical+100.,'teacher_latent':float('nan'),'future_clip':None,'future_valid':False}
            changed=model.encode_current([poison]);other=model.predict_action([poison],initial_noise=noise)
            for k in encoded:torch.testing.assert_close(encoded[k],changed[k],rtol=0,atol=0)
            torch.testing.assert_close(before,other,rtol=0,atol=0)
            model.strip_auxiliary_heads();after=model.predict_action(obs,initial_noise=noise)
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        means={task:{k:sum(r[task][k] for r in gradients)/len(gradients) for k in selected} for task in gradients[0]}
        weight=((means['current_dino']['W']/means['future_clip']['W'])*(means['current_dino']['Qwen_layer0']/means['future_clip']['Qwen_layer0']))**.5
        atomic_json(a.output,{'passed':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'initialization':'genericQwen/randomdriving only; no learned driving checkpoint loaded','action_initialization':module_manifest(model.action_model),
            'W_initialization':tensor_hash(model.foresight_queries),'four_loss_gradients':gradients,'mean_shared_gradient_norms':means,
            'lambda_cur':1.,'lambda_fut':weight,'lambda_int':float(cfg.foresight.lambda_int),'warmup':1000,
            'calibration':'one training-only geometric-mean current/new-future shared-gradient scale calibration',
            'real_training_scenes':[ds.index[i]['token'] for i,_ in chosen],'raw_future_condition_difference':float((f0-f1).abs().max()),
            'GT_changes_future_only':True,'pure_current_encoding_and_ego_exact':True,'deployment_stripping_exact':True,
            'one_Qwen_forward':True,'future_loss_action_model_gradient':None,'real_optimizer_updates':0,
            'future_cache':ci['identity'],'teacher':ci['recipe'],'current_dino':di['identity'],'MAE':ii['identity'],
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        meter['inference_scenes']=len(chosen);save()

if __name__=='__main__':main()
