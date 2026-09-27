"""Build reproducible annotated scenes; current graph files never contain futures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.joint_world.observability import box_corners,project_points
from starVLA.model.modules.joint_scene.contracts import LocalSceneGraph,AnnotatedLocalScene
from starVLA.model.modules.joint_scene.graphs import GraphConfig,build_supervision_graph
from tools.local_interaction_mask_v2.data import current_metadata_from_training_pickle,observation_from_files
from tools.local_interaction_mask_v2.foundation import ego_label

SCHEMA=4


def fingerprint(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_ego_vector(record):
    poses=np.asarray(record['ego_history_poses'],dtype=np.float64)
    angle=poses[3,2];rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    velocity=(poses[3,:2]-poses[2,:2])@rotation/.5
    nav=np.zeros(4);nav[['left','straight','right','unknown'].index(record['navigation'])]=1
    return torch.tensor([record['ego_speed_mps'],*velocity,*nav],dtype=torch.float32)


def geometric_support(boxes,observation):
    if not len(boxes):return torch.zeros(0)
    points=box_corners(boxes.detach().cpu().numpy());w,h=observation.image_size;supports=[]
    for v in range(3):
        uv,valid=project_points(points,observation.intrinsics[v],observation.camera_to_ego[v],observation.distortion[v])
        lo=np.where(valid[...,None],uv,np.inf).min(1);hi=np.where(valid[...,None],uv,-np.inf).max(1)
        overlap=(valid.sum(-1)>=4)&(np.minimum(hi,[w,h])>np.maximum(lo,[0,0])).all(-1)
        supports.append(overlap)
    return torch.tensor(np.stack(supports,1).mean(-1),dtype=torch.float32)


def annotate(token,log,target,ego,graph):
    a=graph.boxes.shape[1];t=len(ego)
    future=torch.zeros(1,a,t,4);valid=torch.zeros_like(future,dtype=torch.bool)
    future[0,0]=ego;valid[0,0]=torch.isfinite(ego)
    tracks=['ego'];selected=[]
    for slot in range(1,a):
        source=int(graph.source_indices[0,slot])
        if source<0:tracks.append('');continue
        selected.append(source);tracks.append(target.track_ids[source])
        mask=target.future_valid_mask[source].bool()
        future[0,slot,:,:2]=torch.where(mask[:,None],target.future_xy_in_ego_t0[source],0.)
        valid[0,slot,:,:2]=mask[:,None]
        # This verified target cache has no future neighbor yaw; do not invent it.
    metadata={'schema_version':SCHEMA,'input_mode':'annotated_current_state_mechanism',
        'neighbor_yaw_labels_available':False,'source_current_objects':len(target.current_boxes),
        'current_supervised_objects':int(target.current_supervision_mask.sum()),'selected_neighbors':len(selected),
        'selected_with_future':int(valid[0,1:,:,:2].any((-1,-2)).sum()),
        'valid_neighbor_xy_coordinates':int(valid[0,1:,:,:2].sum()),
        'all_current_future_xy_coordinates':int((target.future_valid_mask&target.current_supervision_mask[:,None]).sum())*2,
        'context_objects':int(graph.context_mask.sum()),
        'context_overflow':graph.selection_metadata[0]['context_overflow'],
        'trajectory_candidates':graph.selection_metadata[0]['trajectory_candidates'],
        'raw_log_population_available':False,
        'source_population_filtered':True,
        'missing_risk_context':'Source cache excludes objects outside its original ROI/FOV and other prior filters; absent current objects cannot be reconstructed here',
        'current_unobserved_selected':int(((graph.geometric_support[0,1:]==0)&graph.active_actor_mask[0,1:]).sum()),
        'geometric_support_is_visibility':False,'selection_reads_future':False,'track_ids_embedded':False}
    metadata['groups']={}
    for record in graph.selection_metadata[0]['objects']:
        if not record['current_valid']:continue
        source=record['source_index'];distance=record['distance_m']
        names=[f"class_{record['class_id']}",'distance_'+('0_10' if distance<=10 else '10_20' if distance<=20 else '20_50' if distance<=50 else 'over50'), 'support_'+('zero' if record['geometric_support']==0 else 'positive')]
        for name in names:
            group=metadata['groups'].setdefault(name,dict(source=0,candidates=0,selected=0,context=0,selected_with_future=0))
            group['source']+=1;group['candidates']+=int(record['trajectory_candidate']);group['selected']+=int(record['selected']);group['context']+=int(record['context_retained'])
            group['selected_with_future']+=int(record['selected'] and bool(target.future_valid_mask[source].any()))
    return AnnotatedLocalScene(token,log,graph,future,valid,tuple(tracks),metadata).validate()


def build_one(args):
    row,paths,cfg=args;token=row['token'];output=Path(paths['output'])
    target_path=Path(paths['targets'])/'targets'/(token+'.pt');meta=Path(paths['meta_root'])/(token+'.pkl')
    obs_path=Path(paths['observations'])/(token+'.npz')
    target=WorldTargets(**torch.load(target_path,map_location='cpu',weights_only=True))
    if target.overflow:raise ValueError('Need complete original target cache')
    record=current_metadata_from_training_pickle(meta,token)
    observation,_=observation_from_files(obs_path,record,verify_transform=False)
    boxes=target.current_boxes
    # Only currently labeled valid geometry, never the future-valid mask, controls candidates.
    current_valid=target.current_supervision_mask.bool() & bool(target.annotation_valid_mask)
    current_valid &= target.box_valid_mask.bool().all(-1)
    support=geometric_support(boxes,observation)
    graph=build_supervision_graph(boxes,target.current_classes,current_valid,support,current_ego_vector(record),GraphConfig(**cfg))
    ego=ego_label(meta);ego[:,:2]=ego[:,:2]*ego.new_tensor([8.805105,2.277741])+ego.new_tensor([10.172484,.360762])
    scene=annotate(token,row['log'],target,ego,graph)
    current={'schema_version':SCHEMA,'token':token,'graph':asdict(graph),'current_record':record,'observation_path':str(obs_path)}
    labels={'schema_version':SCHEMA,'token':token,'future':scene.future,'feature_valid':scene.feature_valid,'track_ids':scene.track_ids,'metadata':scene.metadata}
    cp=output/'current'/(token+'.pt');lp=output/'labels'/(token+'.pt')
    if cp.exists() or lp.exists():raise FileExistsError('Use new data output, never overwrite')
    torch.save(current,cp);torch.save(labels,lp)
    return {'token':token,'log':row['log'],'current_sha256':fingerprint(cp),'labels_sha256':fingerprint(lp),
        'source_targets_sha256':fingerprint(target_path),'source_meta_sha256':fingerprint(meta),'source_observation_sha256':fingerprint(obs_path),**scene.metadata}


class AnnotatedCorpus(torch.utils.data.Dataset):
    def __init__(self,root,limit=None):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest['schema_version']!=SCHEMA or not self.manifest['complete'] or self.manifest['failed']:raise ValueError('Incomplete/mismatched annotated corpus')
        identity=dict(self.manifest);expected=identity.pop('identity_sha256')
        if hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()!=expected:raise ValueError('Corpus manifest identity changed')
        self.records=self.manifest['records'][:limit];self.memo={}
        if not self.records or len({r['token'] for r in self.records})!=len(self.records):raise ValueError('Empty/duplicate corpus')
        if limit is not None and limit<1:raise ValueError('Positive corpus limit required')
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        if i not in self.memo:
            r=self.records[i];token=r['token'];cp=self.root/'current'/(token+'.pt');lp=self.root/'labels'/(token+'.pt')
            if fingerprint(cp)!=r['current_sha256'] or fingerprint(lp)!=r['labels_sha256']:raise ValueError('Scene artifact bytes changed')
            c=torch.load(cp,map_location='cpu',weights_only=True);l=torch.load(lp,map_location='cpu',weights_only=True)
            if c['schema_version']!=SCHEMA or l['schema_version']!=SCHEMA or c['token']!=token or l['token']!=token or set(c)!={'schema_version','token','graph','current_record','observation_path'}:raise ValueError('Unexpected current fields')
            graph=LocalSceneGraph(**c['graph'])
            expected_config=asdict(GraphConfig(**self.manifest['graph_config']).validate())
            if len(graph.selection_metadata)!=1 or graph.selection_metadata[0].get('graph_config')!=expected_config:
                raise ValueError('Scene graph rules differ from corpus manifest')
            scene=AnnotatedLocalScene(token,r['log'],graph,l['future'],l['feature_valid'],l['track_ids'],l['metadata']).validate()
            if scene.future.shape[2]!=self.manifest['horizon_steps']:
                raise ValueError('Scene label horizon differs from corpus manifest')
            self.memo[i]=scene
        return self.memo[i]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('index','targets','meta-root','observations','config','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--limit',type=int)
    a=parser.parse_args();out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('New versioned output required')
    for name in ('current','labels'):(out/name).mkdir(parents=True,exist_ok=True)
    index=json.loads(Path(a.index).read_text())[:a.limit];config=json.loads(Path(a.config).read_text())['graph']
    paths=vars(a);records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(build_one,((row,paths,config) for row in index),chunksize=16):
            records.append(r)
            if len(records)%500==0:print(json.dumps({'prepared':len(records),'total':len(index)}),flush=True)
    summary={k:sum(r[k] for r in records) for k in ('source_current_objects','current_supervised_objects','selected_neighbors','selected_with_future','valid_neighbor_xy_coordinates','all_current_future_xy_coordinates','current_unobserved_selected','context_objects','context_overflow','trajectory_candidates')}
    summary['groups']={}
    for r in records:
        for name,values in r['groups'].items():
            group=summary['groups'].setdefault(name,{k:0 for k in values})
            for key,value in values.items():group[key]+=value
    summary.update(raw_log_population_available=False,source_population_filtered=True,missing_risk_context='Outside original target-cache ROI/FOV or removed before this stage',scenes=len(records),ego_only_scenes=sum(r['selected_neighbors']==0 for r in records),neighbor_task_available_scenes=sum(r['selected_with_future']>0 for r in records),failed=0)
    code_files=['tools/joint_local_scene_v3/data.py','starVLA/model/modules/joint_scene/contracts.py','starVLA/model/modules/joint_scene/graphs.py','starVLA/model/modules/structured_world/targets.py','starVLA/model/modules/joint_world/observability.py']
    manifest={'data_kind':'real','horizon_steps':8,'time_step_s':.5,'code_files_sha256':{p:fingerprint(p) for p in code_files},'schema_version':SCHEMA,'complete':True,'failed':0,'source':paths,'graph_config':config,'records':records,'summary':summary,'input_label_files_separate':True}
    manifest['identity_sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
