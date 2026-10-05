"""Real current/GT/cache four-task gradient and information-flow validation."""
import argparse,json,subprocess
from pathlib import Path
import torch
from omegaconf import OmegaConf
from starVLA.model.framework import build_framework
from starVLA.dataloader.action_video_foresight_dataset import ActionVideoForesightDataset
from starVLA.dataloader.foresight_dataset import collate_training,decode_ego
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('config','data','dino-root','dino-index','interaction-root','clip-root','campaign-root','run-id','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--samples',type=int,default=4);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze source')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    cfg=OmegaConf.load(a.config)
    def ident(path):return json.loads((Path(path)/'identity.json').read_text())['identity']
    ds=ActionVideoForesightDataset(a.data,candidate='C1',current=True,future=True,
       dino_root=a.dino_root,dino_index=a.dino_index,expected_dino=ident(a.dino_root),
       interaction_root=a.interaction_root,expected_interaction=ident(a.interaction_root),
       clip_root=a.clip_root,expected_clip=ident(a.clip_root),future_type=cfg.foresight.future_target_type,
       expected_teacher=cfg.foresight.video_teacher_identity,allow_partial=False)
    if ds.identity['split']!='train':raise ValueError('Validation uses training data only')
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'real_interface_gradient_check','real_optimizer_updates':0}) as (meter,_,save):
        model=build_framework(cfg).cuda().eval();fc=model.foresight_config;records=[]
        original=model.encode_current;captured=[]
        def encode(obs):
            z=original(obs);captured.append(z);return z
        model.encode_current=encode
        parameters=dict(model.named_parameters())
        q=next(v for k,v in parameters.items() if 'language_model.layers.0.self_attn.q_proj.weight' in k)
        named={'W_input':model.foresight_queries,'Qwen_layer0':q,
          'action_decoder':model.action_model.action_decoder.layer2.weight,
          'future_output':model.spatiotemporal_head.output.weight,'interaction_output':model.interaction_head.output.weight}
        for i in range(64):
            observation,target=ds[i]
            if not target['interaction_valid'] or not target['future_clip_valid'].all():continue
            obs,t=collate_training([(observation,target)])
            result=model.forward_train(obs,t,completed_updates=1000,
                noise_generator=torch.Generator(device='cuda').manual_seed(9042),
                time_generator=torch.Generator(device='cuda').manual_seed(10042))
            if set(result['losses'])!={'ego_fm','current_dino','future_clip','interaction'}:raise AssertionError('Four losses absent')
            groups={**named,'final_W':captured[-1]['W'],'final_H_A':captured[-1]['action_queries']};vectors={};row={'index':i,'gradients':{},'angles':{}}
            for task,loss in result['losses'].items():
                gs=torch.autograd.grad(loss,tuple(groups.values()),retain_graph=True,allow_unused=True)
                norms={name:float(g.float().norm()) if g is not None else None for name,g in zip(groups,gs)}
                row['gradients'][task]={'weighted_loss':float(loss),'norms':norms}
                vectors[task]={name:g.detach().flatten().float() for name,g in zip(groups,gs) if g is not None}
                if norms['W_input'] is None or norms['W_input']<=0 or norms['Qwen_layer0'] is None or norms['Qwen_layer0']<=0:raise AssertionError('Shared path detached: '+task)
                if task!='ego_fm' and norms['action_decoder'] is not None:raise AssertionError('Auxiliary directly trains execution head')
                if task=='interaction':
                    key='final_H_A' if fc.interaction_readout_source=='action' else 'final_W'
                    if norms[key] is None or norms[key]<=0:raise AssertionError('Interaction reads wrong source')
            tasks=list(vectors)
            for j,left in enumerate(tasks):
                for right in tasks[j+1:]:
                    row['angles'][left+'__'+right]={name:float(torch.nn.functional.cosine_similarity(vectors[left][name],vectors[right][name],dim=0)) for name in ('W_input','Qwen_layer0')}
            result['loss'].backward()
            if any(not torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):raise FloatingPointError('Nonfinite real gradient')
            records.append(row);model.zero_grad(set_to_none=True);captured.clear();del result,gs,vectors
            if len(records)==a.samples:break
        model.encode_current=original
        if len(records)!=a.samples:raise ValueError('Not enough legal real targets')
        obs,t=collate_training([ds[records[0]['index']]])
        with torch.inference_mode(), model.amp():
            encoded=model.encode_current(obs);gt=decode_ego(t['ego'].cuda());action=torch.cat((gt[...,:2],gt[...,2:3].sin(),gt[...,2:3].cos()),-1)
            z=model.predict_interaction(encoded);cur=model.dino_head(encoded['W'],torch.zeros(1,device='cuda'),(6,8))
            if fc.future_action_condition=='gt_ego':
                model.spatiotemporal_head.capture_conditioning=True
                first=model.spatiotemporal_head(encoded['W'],(9,12),gt_action=action)
                attention=model.spatiotemporal_head.conditioning_diagnostics
                other=model.spatiotemporal_head(encoded['W'],(9,12),gt_action=action+1.)
                if torch.equal(first,other):raise AssertionError('GT condition not read')
            else:attention=None
            noise=torch.zeros(1,8,4,device='cuda');before=model.predict_action(obs,initial_noise=noise)
            altered=[{**obs[0],'future_clip':float('nan'),'GT_action':action+100.,'interaction_target_valid':False}]
            again=model.encode_current(altered)
            for key in encoded:torch.testing.assert_close(encoded[key],again[key],rtol=0,atol=0)
            torch.testing.assert_close(z,model.predict_interaction(again),rtol=0,atol=0)
            torch.testing.assert_close(cur,model.dino_head(again['W'],torch.zeros(1,device='cuda'),(6,8)),rtol=0,atol=0)
            torch.testing.assert_close(before,model.predict_action(altered,initial_noise=noise),rtol=0,atol=0)
            model.strip_auxiliary_heads();torch.testing.assert_close(before,model.predict_action(obs,initial_noise=noise),rtol=0,atol=0)
        atomic_json(a.output,{'passed':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'configuration':dict(cfg.foresight),'real_optimizer_updates':0,'training_samples':len(records),
          'gradient_samples':records,'conditioning_attention':attention,'no_GT_leak':True,'deployment_stripping_exact':True,
          'precision':'existing BF16 forward/FP32 gradient diagnostic, TF32 off','peak_allocated':torch.cuda.max_memory_allocated()})
        save()

if __name__=='__main__':main()
