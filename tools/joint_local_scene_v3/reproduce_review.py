"""Read-only CPU counterexamples; run against the original review commit."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import types
import torch
from starVLA.model.modules.joint_scene.contracts import AnnotatedLocalScene
from starVLA.model.modules.joint_scene.graphs import GraphConfig,build_supervision_graph
from starVLA.model.modules.joint_scene.flow import JointSceneFlow
from starVLA.model.modules.joint_scene.masks import role_completion_mask
from tools.joint_local_scene_v3 import train_mechanism
from tools.joint_local_scene_v3.runtime import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(12)
    boxes=torch.tensor([[8.,1.,0.,4.,2.,1.5,0.,1.],[20.,-2.,0.,4.,2.,1.5,0.,1.]])
    state=torch.tensor([4.,4.,0.,0.,1.,0.,0.]);cfg=GraphConfig(max_neighbors=2,primary_neighbors=1,max_context=3)
    graph=build_supervision_graph(boxes,torch.zeros(2,dtype=torch.long),torch.ones(2,dtype=torch.bool),torch.ones(2),state,cfg)
    model=JointSceneFlow(dim=32,heads=4,layers=2,steps=3,condition_dim=12).eval();noise=torch.randn(1,3,3,4)
    base=model.sample(noise,graph,sampling_steps=3)
    changed=noise.clone();changed[:,1:,:,2:]=100.
    result={'scope':'CPU no optimizer updates; original b4f08d0 behavior','cases':{}}
    diff=float((base[...,:2]-model.sample(changed,graph,sampling_steps=3)[...,:2]).abs().max())
    result['cases']['unmodeled_yaw_noise']={'expected_invariant':True,'failed_before':diff>0.,'max_xy_difference_m':diff}
    patched=copy.deepcopy(model);original=patched.forward
    def altered(self,*args,**kwargs):
        output=original(*args,**kwargs);output=output.clone();output[:,1:,:,2:]+=100.
        return output
    patched.forward=types.MethodType(altered,patched)
    diff=float((base[...,:2]-patched.sample(noise,graph,sampling_steps=3)[...,:2]).abs().max())
    result['cases']['unmodeled_yaw_velocity']={'failed_before':diff>0.,'max_xy_difference_m':diff}
    dirty=copy.deepcopy(graph);dirty.context_boxes[~dirty.context_mask]=float('nan')
    output=model(noise,torch.tensor([.3]),dirty);output.square().sum().backward()
    result['cases']['padding_context_nan']={'failed_before':not bool(torch.isfinite(output).all()),'forward_finite':bool(torch.isfinite(output).all()),'parameter_gradients_finite':all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)}
    bad=copy.deepcopy(graph);bad.classes[0,1]=19
    try:bad.validate();rejected=False
    except ValueError:rejected=True
    result['cases']['active_invalid_class_contract']={'failed_before':not rejected}
    zero=build_supervision_graph(boxes,torch.zeros(2,dtype=torch.long),torch.ones(2,dtype=torch.bool),torch.zeros(2),state,cfg)
    result['cases']['zero_support_trajectory']={'failed_before':int(zero.active_actor_mask[:,1:].sum())>0,'active_neighbors':int(zero.active_actor_mask[:,1:].sum())}
    cone=build_supervision_graph(boxes,torch.tensor([3,0]),torch.ones(2,dtype=torch.bool),torch.ones(2),state,GraphConfig(max_neighbors=1,primary_neighbors=1,max_context=3))
    result['cases']['static_obstacle_occupies_role_slot']={'failed_before':int(cone.classes[0,1])==3,'selected_class':int(cone.classes[0,1])}
    valid=torch.ones(1,3,3,4,dtype=torch.bool);valid[:,1:,:,2:]=False
    rng=torch.Generator().manual_seed(42);chosen=[]
    for _ in range(20):chosen.append(int(role_completion_mask(valid,graph.active_actor_mask,rng)[1]['chosen_actor'][0]))
    result['cases']['batch1_role_balance']={'failed_before':not any(chosen),'ego_tasks':sum(x==0 for x in chosen),'neighbor_tasks':sum(x>0 for x in chosen)}
    future=torch.zeros(1,3,3,4);future[...,3]=1;future[:,:,:,:2]=graph.boxes[:,:,:2,None].transpose(-1,-2)
    scene=AnnotatedLocalScene('synthetic','synthetic',graph,future,valid,('ego','a','b'),{})
    rows=evaluate(model,[scene],sampling_steps=2,batch=1,device='cpu')
    row=rows[0];n=row['all_hidden_neighbor_all_points'];c=row['neighbor_hidden_neighbor_all_points']
    result['cases']['conditional_target_cohort']={'failed_before':n!=c,'all_hidden_neighbor_points':n,'conditional_neighbor_points':c}
    source=inspect.getsource(train_mechanism)
    result['cases']['role_weight_ignored']={'failed_before':"cfg['role_loss_weight']" not in source,'evidence':'Trainer does not read role_loss_weight'}
    result['real_optimizer_updates']=0;result['synthetic_optimizer_updates']=0
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
