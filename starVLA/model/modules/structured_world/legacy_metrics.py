"""Frozen V1 training-assignment diagnostics. Historical only, not geometric recall."""
import torch
from .matching import match_current

def legacy_diagnostics(pred,target):
    rows,cols=match_current(pred,target)
    eligible=target.current_supervision_mask.bool()
    gt_count=int(eligible.sum())+target.overflow
    xy=pred['boxes'][:,:2]
    lo,hi=target.supervision_bounds[:2],target.supervision_bounds[2:]
    supported=((xy>=lo)&(xy<=hi)).all(-1)
    if target.supervision_grid is not None:
        grid=target.supervision_grid;cell=((xy-lo)/target.supervision_resolution).floor().long()
        supported &= grid[cell[:,0].clamp(0,grid.shape[0]-1),cell[:,1].clamp(0,grid.shape[1]-1)]
    exists=(pred['logits'].argmax(-1)!=pred['logits'].shape[-1]-1)&supported
    distance=torch.linalg.vector_norm(pred['boxes'][rows,:2]-target.current_boxes[cols,:2],dim=-1)
    detected=exists[rows] & (distance<2.)
    good_rows,good_cols=rows[detected],cols[detected]
    n=int(detected.sum())
    result={'gt_targets':gt_count,'matched_detection_targets':n,'predicted_objects':int(exists.sum()),
            'false_positives':int(exists.sum())-n,'matched_centre_error_sum':float(distance[detected].sum()),
            'end_to_end_motion_targets':0,'valid_motion_points':0,'ade_sum':0.,'fde_sum':0.,'fde_targets':0,'yaw_error_sum':0.}
    if n:
        pb=pred['boxes'][good_rows];tb=target.current_boxes[good_cols]
        yaw=torch.atan2(pb[:,6],pb[:,7])-torch.atan2(tb[:,6],tb[:,7])
        result['yaw_error_sum']=float(torch.atan2(yaw.sin(),yaw.cos()).abs().sum())
        mask=target.future_valid_mask[good_cols]
        errors=torch.linalg.vector_norm(pred['future_xy'][good_rows]-target.future_xy_in_ego_t0[good_cols],dim=-1)
        result['ade_sum']=float(errors[mask].sum());result['valid_motion_points']=int(mask.sum())
        result['end_to_end_motion_targets']=int(mask.any(-1).sum())
        # FDE uses the requested terminal horizon, not an earlier convenient visible point.
        result['fde_sum']=float(errors[:,-1][mask[:,-1]].sum());result['fde_targets']=int(mask[:,-1].sum())
    result['gt_motion_points']=int(target.future_valid_mask[target.current_supervision_mask].sum())
    result['gt_motion_instances']=int(target.future_valid_mask[target.current_supervision_mask].any(-1).sum())
    result['class_correct_detection_targets']=int((pred['logits'][good_rows].argmax(-1)==target.current_classes[good_cols]).sum())
    result['ignored_predicted_centres']=int((~supported).sum())
    amask=target.future_valid_mask[cols]
    aerror=torch.linalg.vector_norm(pred['future_xy'][rows]-target.future_xy_in_ego_t0[cols],dim=-1)
    result['assignment_motion_ade_sum']=float(aerror[amask].sum())
    result['assignment_motion_points']=int(amask.sum())
    result['assignment_motion_fde_sum']=float(aerror[:,-1][amask[:,-1]].sum())
    result['assignment_motion_fde_targets']=int(amask[:,-1].sum())
    ranges=torch.linalg.vector_norm(target.current_boxes[:,:2],dim=-1)
    for name,low,high in [('near',0,10),('middle',10,30),('far',30,float('inf'))]:
        group=(ranges>=low)&(ranges<high)&target.current_supervision_mask
        result[name+'_gt']=int(group.sum());result[name+'_detected']=int(group[good_cols].sum())
    return result

