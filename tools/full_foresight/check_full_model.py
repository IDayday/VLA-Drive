"""Real RGB, real four-task targets, actual Qwen/DDP shared-gradient check.

No optimizer updates. Calibrates future/interaction weights on a fixed training
prefix using the geometric mean of W and first-Qwen-layer gradient ratios to
current DINO (weight1). Formal training must use the resulting common weights.
"""
import argparse
import json
from pathlib import Path
import subprocess
import torch
from omegaconf import OmegaConf
from starVLA.model.framework import build_framework
from starVLA.dataloader.full_foresight_dataset import FullForesightDataset
from starVLA.dataloader.foresight_dataset import collate_training
from starVLA.model.modules.vehicle_joint.initialization import module_manifest, tensor_hash
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('config','data','dino-root','dino-index','interaction-root','teacher-verification','campaign-root','run-id','output'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--local-image-root');p.add_argument('--samples',type=int,default=4)
    a=p.parse_args()
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'full_four_loss_gradient_calibration','real_optimizer_updates':0}) as (meter,_,save):
        if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze source before GPU checks')
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        verification=json.loads(Path(a.teacher_verification).read_text())
        di=json.loads((Path(a.dino_root)/'identity.json').read_text());ii=json.loads((Path(a.interaction_root)/'identity.json').read_text())
        if not verification['passed'] or ii['identity']!=verification['export_identities']['train']:
            raise ValueError('A verified real full-training MAE export is required')
        config=OmegaConf.load(a.config)
        # Unit raw weights are solely for calibration, never a formal run.
        config.foresight.lambda_fut=1.;config.foresight.lambda_int=1.
        data=FullForesightDataset(a.data,candidate=config.foresight.candidate,current=True,future=True,
            dino_root=a.dino_root,dino_index=a.dino_index,expected_dino=di['identity'],allow_partial=True,
            interaction_root=a.interaction_root,expected_interaction=ii['identity'],image_root=a.local_image_root)
        if data.identity['split']!='train' or config.foresight.candidate!='C3':
            raise ValueError('Weight calibration fixed to C3 and training data')
        # Fixed data-only rule, selected before model predictions: first eligible
        # training scenes within the 64-scene diagnostic prefix, all labels real.
        chosen=[];batches=[]
        for i in range(min(64,len(data))):
            row=data[i]
            if bool(row[1]['interaction_valid']) and bool(row[1]['future_dino_valid'].all()):
                chosen.append(i);batches.append(collate_training([row]))
            if len(chosen)==a.samples:break
        if len(chosen)!=a.samples:raise ValueError('Insufficient complete real diagnostic targets')
        model=build_framework(config).cuda().eval();params=dict(model.named_parameters())
        qname=next(n for n in params if 'language_model.layers.0.self_attn.q_proj.weight' in n)
        selected={'W':model.foresight_queries,'query_view':model.query_geometry.view.weight,
                  'query_position':model.query_geometry.position.weight,'qwen_layer0_q':params[qname]}
        gradients=[];calls=[0]
        def count(*_):calls[0]+=1
        hook=model.qwen_vl_interface.model.model.language_model.register_forward_hook(count)
        for observation,targets in batches:
            before=calls[0]
            output=model.forward_train(observation,targets,completed_updates=1000,
                noise_generator=torch.Generator(device='cuda').manual_seed(9042),
                time_generator=torch.Generator(device='cuda').manual_seed(10042),
                horizon_generator=torch.Generator().manual_seed(11042))
            if calls[0]-before!=1:raise AssertionError('More than one Qwen forward for four tasks')
            if set(output['losses'])!={'ego_fm','current_dino','future_dino','interaction'}:
                raise AssertionError('Missing full-model objective')
            record={}
            for task,loss in output['losses'].items():
                grad=torch.autograd.grad(loss,tuple(selected.values()),retain_graph=True,allow_unused=True)
                norms={k:float(g.float().norm()) if g is not None else 0. for k,g in zip(selected,grad)}
                if any(not torch.isfinite(g).all() for g in grad if g is not None) or any(v<=0 for v in norms.values()):
                    raise AssertionError('Missing/nonfinite shared gradient '+task+str(norms))
                record[task]={'raw_loss':float(loss.detach()),**norms}
            output['loss'].backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('Nonfinite backward parameter gradient')
            if any(p.grad is not None for p in model.qwen_vl_interface.model.model.visual.parameters()):
                raise AssertionError('Frozen visual encoder received gradients')
            gradients.append(record);model.zero_grad(set_to_none=True);del output
        hook.remove()
        means={task:{k:sum(r[task][k] for r in gradients)/len(gradients) for k in ('W','qwen_layer0_q')}
               for task in ('ego_fm','current_dino','future_dino','interaction')}
        weights={task:((means['current_dino']['W']/means[task]['W'])*
                       (means['current_dino']['qwen_layer0_q']/means[task]['qwen_layer0_q']))**.5
                 for task in ('future_dino','interaction')}
        observation,targets=batches[0];noise=torch.zeros(1,8,4,device='cuda')
        with torch.inference_mode():
            encoded=model.encode_current(observation)
            poison={**observation[0],'current_dino':float('nan'),'future_dino':None,
                    'teacher_latent':float('nan'),'gt_vehicles':999,'interaction_valid':False}
            changed=model.encode_current([poison])
            for key in encoded:torch.testing.assert_close(encoded[key],changed[key],rtol=0,atol=0)
            saved=model.foresight_queries.clone();model.foresight_queries.add_(.5)
            perturbed=model.encode_current(observation);model.foresight_queries.copy_(saved)
            delta=float((encoded['action_queries']-perturbed['action_queries']).float().abs().max())
            if delta==0:raise AssertionError('Action queries cannot read W')
            before=model.predict_action(observation,initial_noise=noise)
            model.strip_auxiliary_heads();after=model.predict_action(observation,initial_noise=noise)
            torch.testing.assert_close(before,after,rtol=0,atol=0)
            if not torch.isfinite(after).all():raise FloatingPointError('Invalid deployment ego')
        report={'passed':True,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'candidate':'C3','real_current_RGB':True,'all_four_targets_real':True,'real_optimizer_updates':0,
            'selected_training_indices':chosen,'selected_scene_tokens':[data.index[i]['token'] for i in chosen],
            'gradients':gradients,'mean_shared_gradient_norms':means,
            'calibration_rule':'geometric mean of current/future-or-interaction W and Qwen-layer0 gradient ratios; training prefix only',
            'lambda_cur':1.,'lambda_fut':weights['future_dino'],'lambda_int':weights['interaction'],
            'auxiliary_warmup_updates':1000,'current_weight_warmup':False,
            'one_Qwen_forward_per_four_loss_batch':True,'labels_do_not_change_current_encoding':True,
            'deployment_head_removal_exact':True,'W_retained':True,'action_query_W_perturbation_max_abs':delta,
            'target_identities':{'dino':di['identity'],'interaction':ii['identity']},
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'action_initialization':module_manifest(model.action_model),'W_initialization':tensor_hash(model.foresight_queries)}
        atomic_json(a.output,report);meter['inference_scenes']=len(chosen);save()

if __name__=='__main__':main()
