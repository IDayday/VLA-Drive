"""Training-only scale statistics and version-difference audit; no model fitting."""
import argparse,json
from pathlib import Path
import numpy as np
import torch


def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--previous');p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.cache);audit=json.loads((root/'audit.json').read_text());boxes=[];future=[];class_counts=np.zeros(7,dtype=int);changed_ids=changed_grids=0;removed=added=0;static_errors=[]
 for row in audit['records']:
  token=row['token'];target=torch.load(root/'targets'/(token+'.pt'),weights_only=True)
  boxes.append(target['current_boxes'].numpy());future.append(target['future_xy_in_ego_t0'][target['future_valid_mask']].numpy());class_counts+=np.bincount(target['current_classes'].numpy(),minlength=7)
  if a.previous:
   old=torch.load(Path(a.previous)/'targets'/(token+'.pt'),weights_only=True)
   x,y=set(old['track_ids']),set(target['track_ids']);changed_ids+=x!=y;removed+=len(x-y);added+=len(y-x);changed_grids+=int((old['supervision_grid']!=target['supervision_grid']).sum())
 b=np.concatenate(boxes);f=np.concatenate(future)
 result={k:v for k,v in audit.items() if k!='records'}
 result.update(training_only_statistics=True,box_columns=['x','y','z','length','width','height','sin_yaw','cos_yaw'],box_quantiles={str(q):np.quantile(b,q,axis=0).tolist() for q in [.01,.5,.99]},future_xy_quantiles={str(q):np.quantile(f,q,axis=0).tolist() for q in [.01,.5,.99]},class_counts=class_counts.tolist(),empty_scenes=sum(r['targets']==0 for r in audit['records']),valid_future_points=len(f),missing_annotation_scenes=sum(not r['annotation_valid'] for r in audit['records']),previous_cache=a.previous,changed_current_track_sets=changed_ids,removed_current_tracks=removed,added_current_tracks=added,changed_supervision_grid_cells=changed_grids)
 Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['completed','changed_current_track_sets','removed_current_tracks','added_current_tracks','changed_supervision_grid_cells']}))
if __name__=='__main__':main()
