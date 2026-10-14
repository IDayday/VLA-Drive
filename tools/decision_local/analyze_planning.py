"""Post-execution trajectory errors, fixed physical scene groups and net failures.

GT is read only after immutable predictions have been exported and scored. These
groups never select a trajectory, training example, checkpoint or residual gate.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha
from tools.decision_local.build_consequence_labels import current_log
from tools.local_interaction_mask_v2.compare_pdms import read as read_scores,compare
from starVLA.dataloader.decision_local import FrozenConsequenceData
from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.model.modules.structured_world.queries import body_points


def stationary_near_query(frame,q0,body,occupancy_valid,grid):
    """Reliable current boxes with annotated speed<=0.1m/s, within1m of body.

    This is a current near-stationary group, not a claim they stay still later.
    Uses the same mature box conversion/raster semantics as the CV labels.
    """
    import cv2
    from shapely.geometry import Polygon,MultiPoint
    from starVLA.dataloader.structured_world.adapters import NAV_CLASSES,box_container,transform_boxes_to_ego,pose_matrix
    e2g=pose_matrix(frame['ego2global_translation'],frame['ego2global_rotation'],planar=True)
    attitude=np.linalg.inv(e2g)@pose_matrix(frame['ego2global_translation'],frame['ego2global_rotation'])
    l2e=attitude@pose_matrix(frame['lidar2ego_translation'],frame['lidar2ego_rotation'])
    anns=frame['anns'];keep=np.array([x in NAV_CLASSES for x in anns['gt_names']],bool)
    boxes=transform_boxes_to_ego(box_container(anns['gt_boxes'][keep]),l2e)
    vel=np.asarray(anns.get('gt_velocity_3d',np.full((len(keep),3),np.nan)))[keep]
    good=np.isfinite(vel).all(-1);vel=np.where(good[:,None],vel,0.)@l2e[:3,:3].T
    near_static=good&(np.linalg.norm(vel[:,:2],axis=1)<=.1)
    footprints=[MultiPoint(points).convex_hull for points in body_points(torch.from_numpy(q0),body).numpy()]
    for box in boxes.corners[near_static][:,[0,3,7,4],:2].numpy():
        cells=np.round((box-[grid.xmin,grid.ymin])/[grid.dx,grid.dy]).astype(np.int32)
        mask=np.zeros(occupancy_valid.shape,np.uint8);cv2.fillPoly(mask,[cells],1);mask=mask.astype(bool)
        if not mask.any() or not occupancy_valid[mask].all():continue
        shape=Polygon(box)
        if any(p.distance(shape)<=1. for p in footprints):return True
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('anchor','features','labels','predictions','scores','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=10000);a=p.parse_args();resource=check_host();torch.set_num_threads(2)
    from starVLA.model.modules.structured_world.grid import GridSpec
    data=FrozenConsequenceData(a.features,a.labels,reader='mmap_tensors');anchor=read(a.anchor)
    pm=read(a.predictions/'identity.json');sm=read(a.scores/'identity.json')
    if sm['prediction_identity']!=pm['identity'] or pm['features']!=data.feature_identity['identity']:
        raise ValueError('Post-hoc analysis must use the declared immutable population')
    if read(a.scores/'COMPLETE.json')['identity']!=sm['identity']:raise ValueError('Complete canonical scoring required')
    original,final=[read_scores(a.scores/v/'scenes.csv') for v in ('q0','q_final')]
    if set(original)!=set(final) or set(original)!={r['token'] for r in data.index}:raise ValueError('All original scenes required')
    rows=[];split=data.feature_identity['split'];labelroot=Path(anchor['data'][split]['label_root']);grid=GridSpec()
    for i,row in enumerate(data.index):
        token=row['token'];batch=data.batch([i],'cpu',evaluation=True)
        path=a.predictions/'predictions'/(token+'.npz');receipt=read(a.predictions/'records'/(token+'.json'))
        if digest(path)!=receipt['sha256']:raise ValueError('Predictions changed after scoring')
        with np.load(path) as f:q0=f['q0'].copy();qf=f['q_final'].copy()
        torch.testing.assert_close(torch.from_numpy(q0),batch['q0'][0],rtol=0,atol=0)
        gt=decode_ego(batch['ego'])[0].numpy();valid=batch['future_valid'][0].numpy().astype(bool)
        rec={'token':token,'log':row['log'],'q0_score':float(original[token].get('score') or 0),'final_score':float(final[token].get('score') or 0)}
        for name,q in [('q0',q0),('q_final',qf)]:
            error=np.linalg.norm(q[:,:2]-gt[:,:2],axis=-1)
            rec[name+'_ADE_m']=float(error[valid].mean()) if valid.any() else None
            rec[name+'_FDE_m']=float(error[np.flatnonzero(valid)[-1]]) if valid.any() else None
            rec[name+'_heading_error_rad']=float(np.abs(np.arctan2(np.sin(q[:,2]-gt[:,2]),np.cos(q[:,2]-gt[:,2])))[valid].mean()) if valid.any() else None
            velocity=np.diff(np.vstack((np.zeros((1,2)),q[:,:2])),axis=0)/.5
            acceleration=np.diff(velocity,axis=0)/.5;jerk=np.diff(acceleration,axis=0)/.5
            rec[name+'_max_discrete_acceleration_mps2']=float(np.linalg.norm(acceleration,axis=-1).max())
            rec[name+'_max_discrete_jerk_mps3']=float(np.linalg.norm(jerk,axis=-1).max())
        rec['mean_residual_m']=float(np.linalg.norm(qf[:,:2]-q0[:,:2],axis=-1).mean())
        rel=batch['relations'][0,:,0].numpy();mask=batch['valid'][0,:,0].numpy()
        pair=mask[0,:,2]&mask[2,:,2];current=rel[0,:,2]>.5;future=rel[2,:,2]>.5
        state=decode_ego(batch['state'])[0,:2].numpy()/.5
        rawpath=labelroot/'labels'/(token+'.npz');meta=read(labelroot/'records'/(token+'.json'))
        if digest(rawpath)!=meta['label_sha256']:raise ValueError('Original group label changed')
        with np.load(rawpath) as f:known=f['occupancy_valid'][0].copy()
        frame=current_log('/mnt/navsim/trainval_navsim_logs/trainval',row['log'])[token]
        groups={'all_scenes':True,'original_high_ge_0p9':rec['q0_score']>=.9,'original_low_le_0p5':rec['q0_score']<=.5,
            'original_progress_below_0p5':float(original[token].get('ego_progress') or 0)<=.5,
            'dynamic_enter_at_query':bool((pair&~current&future).any()),'space_release_at_query':bool((pair&current&~future).any()),
            'road_boundary_or_outside':bool((mask[0,:,0]&(rel[0,:,0]<=.5)).any()),
            'current_speed_below_0p5_mps':bool(np.linalg.norm(state)<.5),
            'future_relation_support_below_half':bool(mask[2,:,1:].mean()<.5),
            'reliable_current_stationary_obstacle_near_query':stationary_near_query(frame,q0,meta['ego_body'],known,grid)}
        rec['groups']=groups;rec['future_relation_support']=float(mask[2,:,1:].mean());rows.append(rec)
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'scenes.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    groups={}
    from tools.decision_local.summarize_round1 import transitions
    for name in rows[0]['groups']:
        selected={r['token'] for r in rows if r['groups'][name]};subset=[r for r in rows if r['token'] in selected]
        if not selected:groups[name]={'scenes':0};continue
        base={t:original[t] for t in selected};result={t:final[t] for t in selected};paired,_=compare(result,base,bootstrap=a.bootstrap)
        numeric={k:float(np.mean([r[k] for r in subset if r[k] is not None])) for k in subset[0]
            if k not in ('token','log','groups') and any(r[k] is not None for r in subset)}
        groups[name]={'scenes':len(subset),'paired':paired,'transitions':transitions(result,base),'trajectory_metrics':numeric}
    atomic_json(a.output/'summary.json',{'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),'resources':resource,
        'predictions':pm['identity'],'label_identity':data.label_identity['identity'],'groups':groups,
        'scope':'Fixed physical groups; overlapping and descriptive, not post-hoc hypothesis confirmation',
        'limitations':['Static group uses reliable current speed, not proof of future stationarity',
            'No visual observability claim from label validity','Discrete acceleration/jerk are trajectory diagnostics, not official Comfort',
            'Dynamic enter/release compares actual-time future to current at the same q0 body; fixed logged environment, not counterfactual reactions']})


if __name__=='__main__':main()
