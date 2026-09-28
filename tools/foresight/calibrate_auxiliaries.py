"""One training-only gradient calibration, never based on dev or Navtest scores.

The fixed rule sets each auxiliary's median shared-gradient norm to25% of main
FM across W and the first/last Qwen q projections. Reports all missing-label
scenes and probe norms; requires nonzero W AND language gradients. No update.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from omegaconf import OmegaConf
from starVLA.model.framework.DDPForesight import DDPForesight
from starVLA.dataloader.foresight_dataset import ForesightTrainingDataset,collate_training
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from starVLA.model.modules.vehicle_joint.initialization import identity_hash


def weight_from_gradients(rows,task,target_ratio=.25):
    eligible=[row for row in rows if row[task]['norm']>0 and row['ego_fm']['norm']>0]
    if not eligible:raise ValueError('No actual auxiliary gradient for '+task)
    if any(row[task]['W']<=0 or row[task]['language']<=0 for row in eligible):raise ValueError('Auxiliary bypasses shared W/language')
    value=target_ratio*float(np.median([row['ego_fm']['norm']/row[task]['norm'] for row in eligible]))
    if not np.isfinite(value) or value<=0:raise ValueError('Invalid calibrated weight')
    return {'weight':value,'valid_scenes':len(eligible),'missing_or_zero_scenes':len(rows)-len(eligible),
            'weighted_shared_gradient_ratio_median':float(np.median([value*row[task]['norm']/row['ego_fm']['norm'] for row in eligible]))}


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('config','data','future-root','future-identity','interaction-root','interaction-identity','output','campaign-root','run-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--scenes',type=int,default=64);a=p.parse_args()
    if a.scenes<1 or a.scenes>64:raise ValueError('Fixed calibration is at most64training scenes')
    out=Path(a.output)
    if out.exists():raise FileExistsError('Calibration cannot be overwritten')
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'real_auxiliary_gradient_calibration','real_optimizer_updates':0}) as (meter,_,save):
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Lock source first')
        data=ForesightTrainingDataset(a.data,future_root=a.future_root,expected_future=a.future_identity,
                                     interaction_root=a.interaction_root,expected_interaction=a.interaction_identity)
        if data.identity['split']!='train':raise ValueError('Only training data may calibrate weights')
        config=OmegaConf.load(a.config)
        if config.foresight.arm!='D':raise ValueError('Calibrate both real heads in D')
        config.foresight.lambda_vis=1.;config.foresight.lambda_int=1.;config.foresight.auxiliary_warmup=1
        first=data[0][1]['future_latent'];channels,height,width=first.shape[-3:]
        if channels!=data.future_identity['channels']:raise ValueError('VAE channel contract changed')
        for key,value in (('latent_channels',channels),('latent_height',height),('latent_width',width)):
            config.foresight[key]=int(value)
        torch.manual_seed(int(config.seed));model=DDPForesight(config).cuda().train()
        language=model.qwen_vl_interface.model.model.language_model
        probes={'W':model.foresight_queries,'first_q':language.layers[0].self_attn.q_proj.weight,
                'last_q':language.layers[-1].self_attn.q_proj.weight}
        generators={key:torch.Generator(device=device).manual_seed(int(config.seed)+offset)
                    for key,device,offset in (('noise','cuda',9000),('time','cuda',10000),('horizon','cpu',11000))}
        rows=[]
        for i in range(min(a.scenes,len(data))):
            observations,targets=collate_training([data[i]])
            output=model(observations,targets,noise_generator=generators['noise'],time_generator=generators['time'],horizon_generator=generators['horizon'])
            row={'scene_index':i,'losses':{k:float(v.detach()) for k,v in output['losses'].items()},
                 'visual_valid_elements':int(output['metrics']['visual_global_elements']),
                 'interaction_valid_elements':int(output['metrics']['interaction_global_elements'])}
            for at,key in enumerate(('ego_fm','visual','interaction')):
                model.zero_grad(set_to_none=True);output['losses'][key].backward(retain_graph=at<2)
                norms={name:float(parameter.grad.float().norm()) if parameter.grad is not None else 0. for name,parameter in probes.items()}
                if not all(np.isfinite(v) for v in norms.values()):raise FloatingPointError('Nonfinite shared gradient')
                row[key]={'W':norms['W'],'language':float(np.hypot(norms['first_q'],norms['last_q'])),
                          'norm':float(np.linalg.norm(list(norms.values()))),'probes':norms}
            model.zero_grad(set_to_none=True);del output;rows.append(row)
            meter['inference_scenes']=len(rows);save()
        visual=weight_from_gradients(rows,'visual');interaction=weight_from_gradients(rows,'interaction')
        report={'source_sha':source,'data_identity':data.identity,'calibration_index_sha256':identity_hash(data.index[:len(rows)]),
                'future_identity':data.future_identity,'interaction_identity':data.interaction_identity,
                'rule':'median main/aux shared-gradient ratio times0.25; W and first/last qproj; training-only; one calibration',
                'real_optimizer_updates':0,'visual':visual,'interaction':interaction,'rows':rows,
                'foresight_overrides':{'lambda_vis':visual['weight'],'lambda_int':interaction['weight'],
                    'latent_channels':int(channels),'latent_height':int(height),'latent_width':int(width)}}
        atomic_json(out,report)


if __name__=='__main__':main()
