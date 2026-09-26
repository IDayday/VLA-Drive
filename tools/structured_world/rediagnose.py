"""Recompute perception metrics from immutable predictions; never rerun/select planning."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from evaluate import diagnostics
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.geometry import geometric_fov
from starVLA.model.modules.structured_world.matching import match_current


def main():
 p=argparse.ArgumentParser()
 for name in ['predictions','targets','original-scenes','output']:p.add_argument('--'+name,required=True)
 a=p.parse_args();old=list(csv.DictReader(open(a.original_scenes)));records=[];root=Path(a.targets)
 for row in old:
  token=row['token'];record={k:row[k] for k in ['token','status','latency_seconds'] if k in row}
  if row['status']!='ok':records.append(dict(row));continue
  arrays=dict(np.load(Path(a.predictions)/(token+'.npz')))
  if 'boxes' not in arrays:records.append(record);continue
  pred={k:torch.from_numpy(arrays[k]) for k in ['boxes','logits','future_xy']}
  target=WorldTargets(**torch.load(root/'targets'/(token+'.pt'),weights_only=True))
  record.update(diagnostics(pred,target))
  obs=dict(np.load(root/'observations'/(token+'.npz')))
  count=np.zeros(len(target.track_ids),dtype=int)
  for v in range(3):count+=geometric_fov(target.current_boxes[:,:3].numpy(),obs['intrinsics'][v:v+1],obs['extrinsics'][v:v+1],obs['distortion'][v:v+1])
  rows,cols=match_current(pred,target);exists=pred['logits'][rows].argmax(-1)!=7
  detected=exists & ((pred['boxes'][rows,:2]-target.current_boxes[cols,:2]).norm(dim=-1)<2.)
  for n in [1,2,3]:
   group=count==n;record[f'cameras_{n}_gt']=int(group.sum());record[f'cameras_{n}_detected']=int(group[cols[detected].numpy()].sum())
  records.append(record)
 keys=sorted({k for r in records for k in r})
 with open(a.output,'w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(records)
 print(json.dumps({'scenes':len(records),'target_version':a.targets,'failed':sum(r['status']!='ok' for r in records)}))
if __name__=='__main__':main()
