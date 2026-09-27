"""GPU forward/backward and real-data inference only; zero optimizer updates."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import types
import numpy as np
import torch
from starVLA.model.modules.joint_scene.flow import JointSceneFlow,flow_loss_sums,normalized_loss
from starVLA.model.modules.joint_scene.graphs import GraphConfig,build_supervision_graph
from tools.joint_local_scene_v3.budget import BudgetRun,atomic_json
from tools.joint_local_scene_v3.data import AnnotatedCorpus
from tools.joint_local_scene_v3.runtime import evaluate,evaluation_noise
from tools.joint_local_scene_v3.train_mechanism import code_identity,deterministic_settings


def visualize(scene,prediction,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,7));g=scene.graph;active=g.active_actor_mask[0].numpy();boxes=g.boxes[0].numpy();future=scene.future[0].numpy();valid=scene.feature_valid[0,:,:,:2].all(-1).numpy()
    for i in np.where(active)[0]:
        for j in np.where(active)[0]:
            if j>i:ax.plot(boxes[[i,j],0],boxes[[i,j],1],color='gray',alpha=.18,linewidth=.6)
        color='black' if i==0 else plt.get_cmap('tab20')(i%20)
        ax.scatter(*boxes[i,:2],c=[color],s=40);ax.text(*boxes[i,:2],str(i))
        ax.plot(future[i,valid[i],0],future[i,valid[i],1],':',color=color,linewidth=2)
        ax.plot(prediction[i,:,0],prediction[i,:,1],'-o',color=color,markersize=2,linewidth=2 if i==0 else 1)
    context=g.context_boxes[0,g.context_mask[0]].numpy()
    ax.scatter(context[:,0],context[:,1],marker='x',c='orange',alpha=.5,label='current context (capacity limited)')
    ax.set_title('Untrained single joint sample; solid prediction / dotted GT\nNo neighbor yaw prediction; geometric support is not visibility')
    ax.set_xlabel('ego(t0) x [m]');ax.set_ylabel('ego(t0) y [m]');ax.axis('equal');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(path);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('ledger','output','real-data','config'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('New check output required')
    out.mkdir(parents=True,exist_ok=True)
    identity={'source':code_identity(),'output':str(out.resolve()),'kind':'GPU checks, no optimizer'}
    checks={}
    with BudgetRun(a.ledger,'v3_pretrain_gpu_checks',identity,gpu_count=1) as budget:
        deterministic_settings();torch.manual_seed(12);torch.cuda.manual_seed_all(12)
        boxes=torch.tensor([[8.,1.,0.,4.,2.,1.5,0.,1.],[18.,-1.,0.,4.,2.,1.5,0.,1.]])
        g=build_supervision_graph(boxes,torch.zeros(2,dtype=torch.long),torch.ones(2,dtype=torch.bool),torch.ones(2),torch.tensor([4.,4.,0.,0.,1.,0.,0.]),GraphConfig(max_neighbors=3,primary_neighbors=2,max_context=4)).to('cuda')
        model=JointSceneFlow(dim=32,heads=4,layers=2,steps=3,condition_dim=12).cuda();hook=model.register_forward_pre_hook(lambda *unused:budget.note('synthetic',forwards=1))
        noise=torch.randn(1,4,3,4,device='cuda');time=torch.tensor([.4],device='cuda')
        actor=torch.randn(1,4,12,device='cuda',requires_grad=True);scene=torch.randn(1,3,12,device='cuda',requires_grad=True);sm=torch.tensor([[True,False,True]],device='cuda')
        dirty=copy.deepcopy(g);dirty.boxes[~g.active_actor_mask]=float('nan');dirty.classes[~g.active_actor_mask]=999
        dirty.context_boxes[~g.context_mask]=float('nan');dirty.context_classes[~g.context_mask]=-1;dirty.context_geometric_support[~g.context_mask]=float('nan')
        dirty.geometric_support[~g.active_actor_mask]=float('nan');dirty.edge_features[~g.edge_mask]=float('nan')
        gradients=[];outputs=[]
        for graph in (g,dirty):
            model.zero_grad(set_to_none=True)
            ap=torch.where(g.active_actor_mask[...,None],actor,float('nan'));sp=torch.where(sm[...,None],scene,float('nan'))
            output=model(noise,time,graph,actor_features=ap,scene_features=sp,scene_mask=sm);output.square().sum().backward();budget.note('synthetic',backwards=1)
            outputs.append(output.detach());gradients.append({n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None})
        checks['padding_forward_identical']=torch.equal(*outputs);checks['padding_parameter_gradients_identical']=all(torch.equal(gradients[0][k],gradients[1][k]) for k in gradients[0])
        checks['padding_gradients_finite']=all(torch.isfinite(x).all().item() for x in gradients[1].values())
        y=torch.zeros_like(noise);y[...,:2]=g.boxes[:,:,:2,None].transpose(-1,-2);y[:,0,:,0]=torch.tensor([2.,4.,6.],device='cuda');y[:,0,:,3]=1
        valid=g.modeled_state_mask[:,:,None].expand_as(y).clone();only_ego=valid.clone();only_ego[:,1:]=False
        model.zero_grad(set_to_none=True);actor.grad=None;scene.grad=None
        sums,counts=flow_loss_sums(model,y,only_ego,g.active_actor_mask,noise,time,g,actor_features=actor,scene_features=scene,scene_mask=sm)
        normalized_loss(sums,counts).backward();budget.note('synthetic',backwards=1)
        checks['ego_loss_interaction_gradient']=float(model.blocks[0].actor.in_proj_weight.grad.abs().sum())>0
        checks['ego_yaw_gradient']=float(model.velocity[-1].weight.grad[2:].abs().sum())>0
        checks['condition_tensor_gradient_not_VLM']=float(actor.grad.abs().sum()+scene.grad.abs().sum())>0
        with torch.no_grad():
            first=model.sample(noise,g,sampling_steps=3);changed=noise.clone();changed[:,1:,:,2:]=float('nan');second=model.sample(changed,g,sampling_steps=3)
            checks['unmodeled_yaw_noise_identical']=torch.equal(first,second)
            original=model.forward
            def changed_velocity(self,*args,**kwargs):
                value=original(*args,**kwargs).clone();value[:,1:,:,2:]=12345.;return value
            model.forward=types.MethodType(changed_velocity,model)
            third=model.sample(noise,g,sampling_steps=3);model.forward=original
            checks['unmodeled_yaw_velocity_identical']=torch.equal(first,third)
            known=valid.clone();known[:,0]=False
            result,path=model.sample_conditional(noise,g,y,known,sampling_steps=3,return_path=True)
            poison=torch.where(known,y,float('nan'))
            checks['known_clamped_every_step']=all(torch.equal(point[known],y[known]) for point in path)
            checks['hidden_poison_identical']=torch.equal(result,model.sample_conditional(noise,g,poison,known,sampling_steps=3))
            action=model.predict_action(noise,replace(g,origin='predicted_inference'),sampling_steps=3)
            checks['ego_is_joint_slot0']=torch.equal(action['executed_ego_xyyaw'][...,:2],action['joint_trajectories'][:,0,:,:2])
        hook.remove();del model
        cfg=json.loads(Path(a.config).read_text());real=AnnotatedCorpus(a.real_data,limit=4)
        full=JointSceneFlow(**cfg['model']).cuda();full.eval();hook=full.register_forward_pre_hook(lambda *unused:budget.note('real',forwards=1))
        report=evaluate(full,real,seed=cfg['sampling_seed'],sampling_steps=cfg['sampling_steps'],device='cuda',output=out/'real_untrained_queries')
        checks['real_complete_query_forward']=report['summary']['aggregate_valid'];checks['real_scene_count']=len(report['scene_rows'])==4
        for i in range(len(real)):
            budget.check();s=real[i]
            with torch.no_grad():prediction=full.sample(evaluation_noise([s],cfg['sampling_seed']),s.graph.to('cuda'),sampling_steps=cfg['sampling_steps'])[0].cpu().numpy()
            visualize(s,prediction,out/f'private_scene_{i}.png')
        hook.remove()
        summary={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'synthetic_optimizer_updates':0,'real_optimizer_updates':0,'real_scenes':4,
            'real_evaluated_queries':len(report['query_rows']),'scope':'GPU numerical/gradient correctness; random models, annotated current input. No learned result, no camera VLM validation, no PDMS.','device':torch.cuda.get_device_name(0),'source':identity['source']}
        atomic_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))
        if not all(checks.values()):raise AssertionError('GPU check failed')


if __name__=='__main__':main()
