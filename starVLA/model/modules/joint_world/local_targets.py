"""Label-side association after the deployment graph is fixed; no node changes."""
import torch
from starVLA.model.modules.structured_world.rehab import match_reference


@torch.no_grad()
def local_targets(prediction, graph, targets, ego_xy, max_distance_m=2., require_class=True):
    b,a=graph.source_slot_ids.shape;t=ego_xy.shape[1]
    if ego_xy.shape!=(b,t,2) or len(targets)!=b:raise ValueError('Local target batch mismatch')
    xy=ego_xy.new_zeros(b,a,t,2);valid=torch.zeros(b,a,t,device=ego_xy.device,dtype=torch.bool)
    xy[:,0]=torch.nan_to_num(ego_xy);valid[:,0]=torch.isfinite(ego_xy).all(-1)
    records=[]
    for bi,target in enumerate(targets):
        pred={k:v[bi] for k,v in prediction.items()}
        # Match FULL original slots once, not a new forced assignment after selection.
        rows,cols=match_reference(pred,target)
        distances=(pred['boxes'][rows,:2]-target.current_boxes[cols,:2]).norm(dim=-1)
        classes=pred['logits'][rows].argmax(-1)==target.current_classes[cols]
        accept=(distances<max_distance_m)&(classes if require_class else torch.ones_like(classes))
        source_to_local={int(s):i for i,s in enumerate(graph.source_slot_ids[bi].tolist()) if s>=0}
        mapped=[]
        for row,col,d,good,same in zip(rows.tolist(),cols.tolist(),distances.tolist(),accept.tolist(),classes.tolist()):
            slot=source_to_local.get(row)
            used=slot is not None and good and bool(graph.predictable_actor_mask[bi,slot])
            mapped.append({'source_slot':row,'local_slot':slot,'gt_index':col,'track_id':target.track_ids[col],
                           'distance_m':d,'same_class':same,'association_accepted':good,'used_for_local_loss':used,
                           'future_valid_points':int(target.future_valid_mask[col].sum()),
                           'exclusion_reason':None if used else ('not_in_local_graph' if slot is None else 'low_quality_current_association')})
            if used:
                mask=target.future_valid_mask[col].bool()
                xy[bi,slot]=torch.where(mask[:,None],target.future_xy_in_ego_t0[col],0.)
                valid[bi,slot]=mask
        valid[bi]&=(graph.active_actor_mask[bi]&graph.predictable_actor_mask[bi])[:,None]
        records.append({'assignments':mapped,'old_fullslot_assignments':len(rows),'accepted_current_associations':int(accept.sum()),
                        'filtered_current_associations':int((~accept).sum()),'selected_with_accepted_association':sum(r['used_for_local_loss'] for r in mapped),
                        'full_current_gt':int(target.current_supervision_mask.sum()),
                        'full_valid_future_points':int(target.future_valid_mask[target.current_supervision_mask].sum())})
    return xy,valid,records
