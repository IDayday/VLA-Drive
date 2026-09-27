"""Explicit synthetic fixtures for bounded trainer checks, never real-scene training."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from starVLA.model.modules.joint_scene.contracts import SCHEMA_VERSION,AnnotatedLocalScene
from starVLA.model.modules.joint_scene.graphs import GraphConfig,build_supervision_graph
from tools.joint_local_scene_v3.data import fingerprint


def write_synthetic(root,config,scenes=3):
    root=Path(root)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Synthetic output must be new')
    for name in ('current','labels'):(root/name).mkdir(parents=True,exist_ok=True)
    records=[];steps=config['model']['steps']
    for i in range(scenes):
        token=f'synthetic-{i}';n=i%3
        boxes=torch.tensor([[8.,1.,0.,4.,2.,1.5,0.,1.],[18.,-1.,0.,4.,2.,1.5,0.,1.]])[:n]
        g=build_supervision_graph(boxes,torch.zeros(n,dtype=torch.long),torch.ones(n,dtype=torch.bool),torch.ones(n),torch.tensor([4.,4.,0.,0.,1.,0.,0.]),GraphConfig(**config['graph']))
        y=torch.zeros(1,g.boxes.shape[1],steps,4);t=torch.arange(1,steps+1)*.5
        y[...,:2]=g.boxes[:,:,:2,None].transpose(-1,-2);y[:,0,:,0]=4*t;y[:,0,:,3]=1
        if n:y[:,1:1+n,:,0]+=2*t
        v=g.modeled_state_mask[:,:,None].expand_as(y).clone()
        if n>1:v[:,2,-1,:2]=False
        tracks=tuple(['ego']+[f'synthetic-track-{j}' if j<n else '' for j in range(g.boxes.shape[1]-1)])
        meta={'schema_version':SCHEMA_VERSION,'source_current_objects':n,'raw_log_population_available':False,'synthetic':True}
        AnnotatedLocalScene(token,'synthetic-log',g,y,v,tracks,meta).validate()
        cp=root/'current'/(token+'.pt');lp=root/'labels'/(token+'.pt')
        torch.save({'schema_version':SCHEMA_VERSION,'token':token,'graph':asdict(g),'current_record':{'synthetic':True},'observation_path':''},cp)
        torch.save({'schema_version':SCHEMA_VERSION,'token':token,'future':y,'feature_valid':v,'track_ids':tracks,'metadata':meta},lp)
        records.append({'token':token,'log':'synthetic-log','current_sha256':fingerprint(cp),'labels_sha256':fingerprint(lp)})
    manifest={'schema_version':SCHEMA_VERSION,'complete':True,'failed':0,'data_kind':'synthetic','horizon_steps':steps,'time_step_s':.5,
        'graph_config':config['graph'],'records':records,'generator_sha256':fingerprint(__file__),'generator':'explicit analytic current positions + constant future velocities; no dataset scenes'}
    manifest['identity_sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
