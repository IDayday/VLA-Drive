"""Actual generic Qwen/DDP GPU preflight, no optimizer update and no planning claim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import torch
from omegaconf import OmegaConf
from starVLA.model.framework.DDPForesight import DDPForesight
from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset,encode_ego
from starVLA.model.modules.foresight.losses import masked_regression,interaction_loss
from starVLA.model.modules.vehicle_joint.initialization import tensor_hash,module_manifest
from tools.ddpolicy_vehicle.prepare_data import atomic_json,load_trusted
from tools.ddpolicy_vehicle.evaluate_ego import relative_ego_target
from tools.ddpolicy_vehicle.run_meter import metered_run


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('config','current-root','processed-root','output','campaign-root','run-id'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'real_camera_gradient_preflight','real_optimizer_updates':0}) as (record,_,save):
        if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Lock source first')
        config=OmegaConf.load(a.config);data=ForesightCurrentDataset(a.current_root)
        model=DDPForesight(config).cuda().eval()
        observation=data[0];observations=[observation]
        def norms():
            result={}
            for name,p in model.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient '+name)
                    if name=='foresight_queries':result['W']=float(p.grad.float().norm())
                    if 'language_model.layers.0.self_attn.q_proj.weight' in name:result['qwen_layer0']=float(p.grad.float().norm())
                    if name.startswith('future_head.') or name.startswith('interaction_head.'):
                        prefix=name.split('.')[0];result[prefix]=result.get(prefix,0.)+float(p.grad.float().norm())
            return result
        # Real current camera input, SYNTHETIC auxiliary targets solely to test the
        # backward path before the real teacher/VAE targets exist. Never learning evidence.
        encoded=model.encode_current(observations)
        with model.amp():pred=model.future_head(encoded['W'],torch.tensor([0],device='cuda'),(4,6))
        loss,_=masked_regression(pred,torch.ones_like(pred),torch.ones(1,3,1,1,1,device='cuda',dtype=torch.bool))
        loss.backward();visual=norms();model.zero_grad(set_to_none=True);del encoded,pred,loss
        encoded=model.encode_current(observations)
        with model.amp():pred=model.predict_interaction(encoded)
        target=torch.linspace(-1,1,pred.numel(),device='cuda').reshape_as(pred)
        loss,_=interaction_loss(pred,target,torch.ones(1,device='cuda',dtype=torch.bool))
        loss.backward();interaction=norms();model.zero_grad(set_to_none=True);del encoded,pred,loss
        encoded=model.encode_current(observations)
        raw=load_trusted(Path(a.processed_root)/(observation['token']+'.pkl'))
        ego=torch.from_numpy(encode_ego(relative_ego_target(raw['glo_status']['global_poses'][:12])))[None].cuda()
        generator=torch.Generator(device='cuda').manual_seed(42);noise=torch.randn(1,8,4,device='cuda',generator=generator)
        with model.amp():loss=model.action_model(encoded['action_queries'],ego,noise=noise,times=torch.tensor([.5],device='cuda'))
        loss.backward();main=norms();model.zero_grad(set_to_none=True);del encoded,loss
        for values in (visual,interaction,main):
            if values.get('W',0)<=0 or values.get('qwen_layer0',0)<=0:raise AssertionError('Missing real shared gradient')
        if any(p.grad is not None for p in model.qwen_vl_interface.model.model.visual.parameters()):raise AssertionError('Frozen visual received grad')
        with torch.inference_mode():
            normal=model.encode_current(observations)
            poisoned=dict(observation,teacher_latent=torch.full((8,512),float('nan')),future_images=None,future_valid=False,gt_vehicles=999)
            changed=model.encode_current([poisoned])
            for key in normal:
                if not torch.equal(normal[key],changed[key]):raise AssertionError('Target contamination '+key)
            before=model.predict_action(observations,initial_noise=noise)
            model.strip_auxiliary_heads()
            after=model.predict_action(observations,initial_noise=noise)
            if not torch.equal(before,after):raise AssertionError('Deployment crop changed ego')
        report={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'real_current_images':True,'auxiliary_targets':'SYNTHETIC gradient plumbing ONLY; not auxiliary learning',
                'real_optimizer_updates':0,'parameters':sum(p.numel() for p in model.parameters()),
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'gradients':{'visual':visual,'interaction':interaction,'ego_FM':main},
                'label_poison_forward_equal':True,'deployment_strip_bitwise_equal':True,
                'W_kept_in_deployment':hasattr(model,'foresight_queries'),'frozen_native_vision':True,
                'peak_memory_bytes':torch.cuda.max_memory_allocated(),
                'driving_initialization':{'action':module_manifest(model.action_model),'state':module_manifest(model.action_input_model),
                                           'W':tensor_hash(model.foresight_queries)}}
        atomic_json(out/'RESULTS.json',report);record['inference_scenes']=1;save()


if __name__=='__main__':main()
