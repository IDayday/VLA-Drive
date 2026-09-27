"""Current-prediction graph audit, with label-side full raw target denominators."""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import subprocess
import numpy as np
import torch
from starVLA.model.modules.joint_world.local_graph import build_local_graph,corridor_relations
from starVLA.model.modules.joint_world.local_targets import local_targets
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.geometry import geometric_fov
from starVLA.model.modules.structured_world.targets import CLASSES
from tools.local_interaction_mask_v2.data import load_current_cache,current_metadata_from_training_pickle,observation_from_files


@lru_cache(maxsize=2)
def raw_log(path):
    with open(path,'rb') as f:frames=pickle.load(f)
    return {frame['token']:frame for frame in frames}


def worker(payload):
    record,a,manifest,config=payload;token=record['token']
    try:
        cache=load_current_cache(a['cache'],record,manifest,purpose=a['purpose'])
        current=current_metadata_from_training_pickle(Path(a['meta_root'])/(token+'.pkl'),token)
        observation,_=observation_from_files(Path(a['observations'])/(token+'.npz'),current,verify_transform=True)
        boxes=cache['current_boxes'][0].float();logits=cache['current_logits'][0].float()
        graph=build_local_graph(boxes.numpy(),logits.numpy(),observation,config)
        # Graph is FIXED and persisted before opening any target/annotation file.
        out=Path(a['output']);torch.save(graph.tensor_state(),out/'graphs'/(token+'.pt'))
        target=WorldTargets(**torch.load(Path(a['targets'])/'targets'/(token+'.pt'),weights_only=True))
        prediction={'boxes':boxes[None],'logits':logits[None]}
        _,valid,associations=local_targets(prediction,graph,[target],torch.zeros(1,8,2))
        association=associations[0];nodes=graph.node_records[0];provenance=graph.graph_provenance[0]
        with np.load(Path(a['observations'])/(token+'.npz')) as z:log=Path(str(z['image_paths'][0])).parts[-3]
        frame=raw_log(str(Path(a['raw_log_root'])/(log+'.pkl')))[token]
        anns=frame['anns'];raw=np.asarray(anns['gt_boxes'],dtype=np.float64);tracks=list(anns['track_tokens'])
        if not np.allclose(frame['lidar2ego'],np.eye(4),atol=1e-6):raise ValueError('Need explicit annotation coordinate adapter')
        classes=np.array([CLASSES.index(name) for name in anns['gt_names']])
        encoded=np.concatenate([raw[:,:6],np.sin(raw[:,6:7]),np.cos(raw[:,6:7])],-1)
        support=geometric_fov(raw[:,:3],observation.intrinsics,observation.camera_to_ego,observation.distortion)
        relevant,_,_,_=corridor_relations(encoded,classes,observation,config)
        chosen={x['track_id'] for x in association['assignments'] if x['used_for_local_loss']}
        selected=np.array([t in chosen for t in tracks])
        if int(selected.sum())!=len(chosen):raise ValueError('GT track association not found in raw current annotations')
        old_valid=sum(x['future_valid_points']>0 for x in association['assignments'])
        row={'token':token,'log':log,'status':'ok','scope':a['purpose'],'navigation':observation.navigation,
            'original_slots':len(boxes),'existence_mean':float(np.mean([n['existence_score'] for n in nodes])),
            'duplicates':sum(n['duplicate_of'] is not None for n in nodes),'reliable_proxy_slots':sum(n['reliable_proxy'] for n in nodes),
            'supported_slots':sum(n['support_views']>0 for n in nodes),'group_A':sum(n['group']=='A' for n in nodes),
            'group_B':sum(n['group']=='B' for n in nodes),'group_C':sum(n['group']=='C' for n in nodes),'group_D':sum(n['group']=='D' for n in nodes),
            'active_nodes':int(graph.active_actor_mask.sum()),'communication_edges_excluding_self':int(graph.edge_mask.sum()-graph.active_actor_mask.sum()),
            'direct_neighbors':provenance['direct_count'],'second_hop':provenance['second_hop_count'],
            'old_mask_probability_any_future':old_valid/len(boxes),'local_matched':len(chosen),'local_valid_future_points':int(valid[0,1:].sum()),
            'filtered_associations':association['filtered_current_associations'],'assignment_distance_sum':sum(x['distance_m'] for x in association['assignments']),
            'assignments':len(association['assignments']),'raw_all_targets':len(raw),'roi_cached_targets':association['full_current_gt'],
            'raw_geometric_supported_targets':int(support.sum()),'raw_current_relevance_proxy_targets':int(relevant.sum()),
            'raw_supported_relevant_targets':int((support&relevant).sum()),'retained_supported_relevant_targets':int((selected&support&relevant).sum()),
            'excluded_supported_relevant_targets':int((~selected&support&relevant).sum())}
        (out/'nodes'/(token+'.json')).write_text(json.dumps({'nodes':nodes,'graph_provenance':provenance,'association_LABEL_SIDE_ONLY':association,
            'raw_gt_after_graph_only':{'boxes':encoded.tolist(),'classes':classes.tolist(),'tracks':tracks,'supported':support.tolist(),'current_relevance_proxy':relevant.tolist(),
            'retained':selected.tolist()},'row':row},indent=2)+'\n')
        return row
    except Exception as e:return {'token':token,'status':'failed','scope':a['purpose'],'error':repr(e)}


def visualize(token,a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from starVLA.model.modules.joint_world.local_graph import LocalInteractionGraph
    out=Path(a['output']);data=json.loads((out/'nodes'/(token+'.json')).read_text())
    graph=LocalInteractionGraph(**torch.load(out/'graphs'/(token+'.pt'),weights_only=True))
    cache=torch.load(Path(a['cache'])/(token+'.pt'),weights_only=True)
    boxes=cache['current_boxes'][0].float().numpy()
    current=current_metadata_from_training_pickle(Path(a['meta_root'])/(token+'.pkl'),token)
    obs,images=observation_from_files(Path(a['observations'])/(token+'.npz'),current,load_images=True)
    fig=plt.figure(figsize=(20,11));grid=fig.add_gridspec(3,4);palette={'A':'limegreen','B':'orange','C':'deepskyblue','D':'gray'}
    for v in range(3):
        ax=fig.add_subplot(grid[v,:2]);ax.imshow(images[v]);ax.set_title(obs.camera_names[v]);ax.set_xlim(0,1024);ax.set_ylim(576,0);ax.axis('off')
        for r in data['nodes']:
            if not r['observation_support'][v]:continue
            x1,y1,x2,y2=r['projected_boxes'][v];color=palette[r['group']]
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=1,alpha=.8 if r['group']!='D' else .35))
            ax.text(x1,max(12,y1),str(r['source_slot_id'])+r['group'],color=color,fontsize=7)
    ax=fig.add_subplot(grid[:2,2]);ax.scatter([0],[0],marker='*',c='black',s=150,label='ego')
    for r in data['nodes']:
        i=r['source_slot_id'];x,y=boxes[i,:2]
        ax.scatter([y],[x],color=palette[r['group']],s=25,alpha=.7);ax.text(y,x,str(i),fontsize=6)
    ids=graph.source_slot_ids[0].tolist();edges=graph.edge_mask[0].numpy()
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            if edges[i,j]:
                p=np.zeros(2) if ids[i]==-1 else boxes[ids[i],:2];q=np.zeros(2) if ids[j]==-1 else boxes[ids[j],:2]
                ax.plot([p[1],q[1]],[p[0],q[0]],c='purple',lw=1.5 if i else .8)
    gt=np.array(data['raw_gt_after_graph_only']['boxes'])
    if len(gt):ax.scatter(gt[:,1],gt[:,0],marker='x',s=10,c='red',alpha=.35,label='raw GT audit only')
    ax.set(xlim=(25,-25),ylim=(-8,60),xlabel='ego y (left)',ylabel='ego x (forward)',title='Fixed current graph; GT added afterwards');ax.grid();ax.legend(fontsize=7)
    ax=fig.add_subplot(grid[:,3]);ax.axis('off');lines=[]
    for r in data['nodes']:
        if r['group'] in ('A','B'):
            lines.append(f"{r['source_slot_id']:2d} {r['group']} {r['entity_type']} {r['reason']}\n  exist={r['existence_score']:.2f} support={r['support_views']} reliable={r['visual_reliability']:.2f}")
    ax.text(0,1,'A=trajectory; B=uncertain risk (NOT free)\nC=original scene path; D=invalid\n\n'+'\n'.join(lines[:23]),va='top',fontsize=7)
    ax=fig.add_subplot(grid[2,2]);ax.axis('off');row=data['row'];ax.text(0,1,'\n'.join(f'{k}: {row[k]}' for k in ('navigation','direct_neighbors','second_hop','group_B','local_matched','raw_all_targets','raw_supported_relevant_targets','excluded_supported_relevant_targets')),va='top',fontsize=9)
    fig.suptitle(token+' | '+a['purpose']+' | current projection proxy is NOT visibility',fontsize=12)
    fig.tight_layout();fig.savefig(out/'visualizations'/(token+'.png'),dpi=100);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','meta-root','observations','targets','raw-log-root','selector','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--purpose',choices=['historical_engineering_audit','formal_public_origin'],default='formal_public_origin')
    p.add_argument('--workers',type=int,default=6);p.add_argument('--limit',type=int);a=vars(p.parse_args());out=Path(a['output']);out.mkdir(parents=True,exist_ok=False)
    for name in ('graphs','nodes','visualizations'):(out/name).mkdir()
    manifest=json.loads((Path(a['cache'])/'manifest.json').read_text());config=json.loads(Path(a['selector']).read_text());records=manifest['records'][:a['limit']]
    with get_context('spawn').Pool(a['workers']) as pool:rows=list(pool.imap(worker,((r,a,manifest,config) for r in records),chunksize=8))
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'scenes.csv').open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    good=[r for r in rows if r['status']=='ok'];failed=[r for r in rows if r['status']!='ok']
    summary={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'scope':a['purpose'],
        'input_manifest_sha256':hashlib.sha256((Path(a['cache'])/'manifest.json').read_bytes()).hexdigest(),'requested':len(rows),'completed':len(good),'failed':failed,
        'denominator_semantics':'raw_all includes all raw current annotations, not only ROI; support is centre geometric FOV, not occlusion visibility; relevance is current corridor proxy used only for posthoc GT audit',
        'formal_algorithm_evidence':a['purpose']=='formal_public_origin'}
    for key in ('raw_all_targets','roi_cached_targets','raw_geometric_supported_targets','raw_current_relevance_proxy_targets','raw_supported_relevant_targets','retained_supported_relevant_targets','excluded_supported_relevant_targets','local_matched','local_valid_future_points','filtered_associations'):
        summary[key]=sum(r[key] for r in good)
    for key in ('active_nodes','group_B','group_C','group_D','second_hop','old_mask_probability_any_future','reliable_proxy_slots'):
        v=[r[key] for r in good];summary[key]={'mean':float(np.mean(v)),'quantiles':np.quantile(v,[0,.25,.5,.75,1]).tolist()} if v else None
    chosen=[r['token'] for r in good[:16]]
    for predicate in (lambda r:r['active_nodes']==1,lambda r:r['second_hop']>0,lambda r:r['navigation'] in ('left','right'),lambda r:r['group_B']>0,lambda r:r['filtered_associations']>0):
        for r in [r for r in good if predicate(r)][:4]:
            if r['token'] not in chosen:chosen.append(r['token'])
    for r in good:
        if len(chosen)>=32:break
        if r['token'] not in chosen:chosen.append(r['token'])
    chosen=chosen[:32]
    for token in chosen:visualize(token,a)
    summary['visualization_tokens']=chosen;summary['visualization_selection']='First16 manifest + first current-rule boundary categories, then manifest fill; no future/score selection'
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if failed:raise RuntimeError('Graph audit failures retained in scenes.csv')


if __name__=='__main__':main()
