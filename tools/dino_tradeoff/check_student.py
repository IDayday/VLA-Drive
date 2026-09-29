"""Real RGB/current-DINO checks on the actual Qwen+DDP student; zero updates."""
import argparse,json,subprocess,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from starVLA.model.framework.DDPForesight import DDPForesight
from starVLA.dataloader.tradeoff_dataset import TradeoffDataset
from starVLA.dataloader.foresight_dataset import collate_training
from starVLA.model.modules.foresight.losses import masked_regression
from starVLA.model.modules.vehicle_joint.initialization import module_manifest,tensor_hash
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run


def main():
 p=argparse.ArgumentParser(__doc__)
 for k in ('config','data','dino-root','dino-index','output','campaign-root','run-id'):p.add_argument('--'+k,required=True)
 p.add_argument('--local-image-root');a=p.parse_args()
 with metered_run(a.campaign_root,a.run_id,1,{'kind':'tradeoff_real_gradient_and_input_contract','real_optimizer_updates':0}) as (meter,_,save):
  if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze source')
  torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
  config=OmegaConf.load(a.config);identity=json.loads((Path(a.dino_root)/'identity.json').read_text())
  data=TradeoffDataset(a.data,dino_root=a.dino_root,dino_index=a.dino_index,expected_dino=identity['identity'],candidate=config.foresight.arm,allow_partial=True,image_root=a.local_image_root)
  observations,targets=collate_training([data[0]])
  model=DDPForesight(config).cuda().eval();cfg=model.foresight_config;gradients={}
  for task in ('ego','current'):
   encoded=model.encode_current(observations)
   if task=='ego':
    with model.amp():loss=model.action_model(encoded['action_queries'],targets['ego'].cuda(),noise=torch.zeros(1,8,4,device='cuda'),times=torch.tensor([.5],device='cuda'))
   else:
    with model.amp():prediction=model.dino_head(encoded['W'],torch.zeros(1,device='cuda'),(cfg.dino_height,cfg.dino_width))
    loss,_=masked_regression(prediction,targets['current_dino'].cuda(),targets['current_dino_valid'].cuda()[:,:,None])
   loss.backward();norms={}
   for name,param in model.named_parameters():
    if param.grad is not None:
     if not torch.isfinite(param.grad).all():raise FloatingPointError('Nonfinite gradient '+name)
     if name=='foresight_queries':norms['reasoning_queries']=float(param.grad.float().norm())
     if 'language_model.layers.0.self_attn.q_proj.weight' in name:norms['qwen_layer0_q']=float(param.grad.float().norm())
   if any(norms.get(k,0)<=0 for k in ('reasoning_queries','qwen_layer0_q')):raise AssertionError('Missing shared gradient '+task+str(norms))
   gradients[task]={'loss':float(loss.detach()),**norms};model.zero_grad(set_to_none=True);del encoded,loss
  noise=torch.zeros(1,8,4,device='cuda')
  with torch.inference_mode():
   encoded=model.encode_current(observations)
   poison=dict(observations[0],current_dino=torch.full((3,1024,6,8),float('nan')),future_dino=None,teacher_latent=float('nan'),gt_vehicles=999)
   changed=model.encode_current([poison])
   for key in encoded:torch.testing.assert_close(encoded[key],changed[key],rtol=0,atol=0)
   query=model.foresight_queries.clone();model.foresight_queries.add_(.5)
   perturbed=model.encode_current(observations)
   delta=float((perturbed['action_queries']-encoded['action_queries']).float().abs().max());model.foresight_queries.copy_(query)
   if delta==0:raise AssertionError('Action cannot read reasoning queries')
   before=model.predict_action(observations,initial_noise=noise);model.strip_auxiliary_heads();after=model.predict_action(observations,initial_noise=noise)
   torch.testing.assert_close(before,after,rtol=0,atol=0)
   if not torch.isfinite(after).all():raise FloatingPointError('Invalid ego')
  report={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'arm':cfg.arm,'real_RGB_and_DINO_targets':True,'real_optimizer_updates':0,'gradients':gradients,'labels_do_not_change_encoded_current':True,'strip_projection_preserves_ego_exactly':True,'action_hidden_query_perturbation_max_abs':delta,'reasoning_retained':True,'sequence_lengths':model.last_sequence_lengths.cpu().tolist(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'qwen_processor':model.qwen_vl_interface.processor.image_processor.to_dict(),'state_initialization':module_manifest(model.action_input_model),'action_initialization':module_manifest(model.action_model),'reasoning_initialization_hash':tensor_hash(model.foresight_queries),'passed':True}
  atomic_json(a.output,report);meter['inference_scenes']=1;save()
if __name__=='__main__':main()
