"""Fixed-graph privileged future-condition removal, explicitly not a causal test."""
import argparse
import csv
import json
from pathlib import Path
import torch
from tools.local_interaction_mask_v2.graph_runtime import LocalCorpus,batch_current
from tools.local_interaction_mask_v2.train_foundation import atomic_json
from starVLA.model.modules.joint_world.flow import JointTrajectoryFlow
from starVLA.model.modules.joint_world.local_masks import stable_noise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','cache','targets','meta-root','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    corpus=LocalCorpus(a.cache,a.targets,a.meta_root);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if saved['identity']['cache_identity']!=corpus.manifest['identity_sha256']:raise ValueError('Diagnostic graph/foundation mismatch')
    cfg=saved['identity']['config'];model=JointTrajectoryFlow(corpus[0]['current']['context'].shape[-1],**{k:v for k,v in cfg.items()
        if k in ('dim','heads','layers','steps','scale_m','agent_scale_m','trajectory_mode','edge_feature_dim')}).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model'],strict=True);rows=[]
    with torch.no_grad():
        for i in range(len(corpus)):
            sample=corpus[i];current=batch_current([sample]);graph=current['local_graph'];valid=sample['valid'].cuda();xy=sample['xy'].cuda()
            eligible=graph.trajectory_condition_mask[0].clone();eligible[0]=False
            related=torch.where(eligible&graph.edge_mask[0,0])[0].tolist()
            weak=torch.where(eligible&~graph.edge_mask[0,0])[0].tolist()
            related.sort(key=lambda j:(-float(graph.relevance_scores[0,j]),int(graph.source_slot_ids[0,j])))
            weak.sort(key=lambda j:(float(graph.relevance_scores[0,j]),int(graph.source_slot_ids[0,j])))
            count=min(2,len(related),len(weak))
            row={'token':sample['token'],'status':'ok' if count else 'unavailable_current_graph',
                 'removed_nodes_each_arm':count,'related_available':len(related),'weak_available':len(weak),
                 'matched_neighbors':sample['mapping']['selected_with_accepted_association']}
            if count:
                related=related[:count];weak=weak[:count]
                row['related_source_slots']=json.dumps(graph.source_slot_ids[0,related].tolist())
                row['weak_source_slots']=json.dumps(graph.source_slot_ids[0,weak].tolist())
                row['related_removed_valid_points']=int(valid[0,related].sum());row['weak_removed_valid_points']=int(valid[0,weak].sum())
                noise=stable_noise([sample['token']],graph.source_slot_ids,model.steps,cfg.get('sampling_seed',2037),device='cuda')
                for name,remove in [('full',[]),('remove_related',related),('remove_weak',weak)]:
                    known=valid&graph.trajectory_condition_mask[:,:,None];known[:,0]=False;known[:,remove]=False
                    prediction,_=model.sample_conditional(noise,**current,known_xy=xy,known_mask=known,sampling_steps=cfg.get('sampling_steps',10))
                    error=(prediction[:,0]-xy[:,0]).norm(dim=-1);mask=valid[:,0]
                    row[name+'_ego_ADE_m']=float(error[mask].mean()) if mask.any() else None
                    row[name+'_ego_FDE_m']=float(error[0,-1]) if mask[0,-1] else None
                row['related_ADE_change_m']=row['remove_related_ego_ADE_m']-row['full_ego_ADE_m']
                row['weak_ADE_change_m']=row['remove_weak_ego_ADE_m']-row['full_ego_ADE_m']
            rows.append(row)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'scenes.csv').open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    eligible=[r for r in rows if r['status']=='ok']
    summary={'scenes':len(rows),'current_graph_eligible':len(eligible),'unavailable':len(rows)-len(eligible),
        'both_arms_actually_remove_labeled_points':sum(r['related_removed_valid_points']>0 and r['weak_removed_valid_points']>0 for r in eligible),
        'removed':'ONLY privileged future conditions; current nodes, edges and visual/risk context are fixed',
        'selection':'current edge/relevance/source identity; no future-valid selection or resampling',
        'interpretation':'Potentially out-of-distribution sensitivity diagnostic, not causal or deployable planning evidence'}
    for key in ('related_ADE_change_m','weak_ADE_change_m'):
        summary[key]=sum(r[key] for r in eligible)/len(eligible) if eligible else None
    atomic_json(out/'summary.json',summary);atomic_json(out/'status.json',{'status':'complete'})


if __name__=='__main__':main()
