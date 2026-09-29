"""Current-only pooled targets, validated separately from pure-current inputs."""
import json
from pathlib import Path
from .dino_foresight_dataset import DINOTrainingDataset
from .foresight_dataset import ForesightTrainingDataset
from starVLA.model.modules.vehicle_joint.initialization import file_sha256,identity_hash
from starVLA.model.modules.foresight.tradeoff import CANDIDATES
import torch

class TradeoffDataset(DINOTrainingDataset):
 def __init__(self,root,*,dino_root,dino_index,expected_dino,candidate,allow_partial=False,image_root=None,**kwargs):
  ForesightTrainingDataset.__init__(self,root,**kwargs)
  self.local_image_root=image_root
  if image_root:
   staged=json.loads((Path(root)/'local_stage.json').read_text())
   if staged['source_identity']!=self.identity['identity'] or Path(staged['image_root'])!=Path(image_root) or (not allow_partial and staged['scenes']!=len(self.index)):raise ValueError('Incomplete/foreign local image staging')
  self.dino_root=Path(dino_root);idx=Path(dino_index)
  ident=json.loads((self.dino_root/'identity.json').read_text());index=json.loads((idx/'identity.json').read_text())
  if ident['schema']!='dino_tradeoff_cache_v1' or ident['identity']!=expected_dino or identity_hash({k:v for k,v in ident.items() if k!='identity'})!=expected_dino:raise ValueError('Cache identity mismatch')
  c=CANDIDATES[candidate]
  if ident['candidate']!=c.record() or ident['index']!=index['identity']:raise ValueError('Resolution/pooling/index mismatch')
  split=self.identity['split'];path=idx/(split+'_scenes.json')
  if index['index_hashes'][split]!=self.identity['index_sha256'] or index['partition_sha256']!=self.identity['partition_sha256'] or file_sha256(path)!=index['files'][split]:raise ValueError('Split identity mismatch')
  if not allow_partial and not (self.dino_root/'COMPLETE.json').exists():raise ValueError('Incomplete target cache')
  self.dino_scenes=json.loads(path.read_text());self.dino_identity=ident;self._checked_chunks=set()
  if [r['token'] for r in self.dino_scenes]!=[r['token'] for r in self.index]:raise ValueError('Scene order mismatch')
  # Parent read_image uses explicit grid/feature metadata, without a teacher import.
  self.dino_identity={**ident,'grid_hw':c.grid_hw}
 def __getitem__(self,i):
  observation,targets=ForesightTrainingDataset.__getitem__(self,i)
  pairs=[self.read_image(v) for v in self.dino_scenes[i]['images']]
  targets.update(current_dino=torch.stack([v[0] for v in pairs]),current_dino_valid=torch.stack([v[1] for v in pairs]))
  return observation,targets
