"""Read-only DINO labels added AFTER whitelisted current observations."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from .foresight_dataset import ForesightTrainingDataset
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


class DINOTrainingDataset(ForesightTrainingDataset):
    def __init__(self,root,*,dino_root=None,dino_index=None,expected_dino=None,
                 current=False,future=False,allow_partial=False,**kwargs):
        super().__init__(root,**kwargs)
        self.dino_identity=None;self.dino_current=current;self.dino_future=future
        if not (current or future):
            if dino_root:raise ValueError('Unexpected DINO labels for disabled tasks')
            return
        self.dino_root=Path(dino_root);index=Path(dino_index)
        ident=json.loads((self.dino_root/'identity.json').read_text());idx=json.loads((index/'identity.json').read_text())
        if ident['schema']!='dinov3_dense_cache_v1' or ident['identity']!=expected_dino or ident['index']!=idx['identity']:
            raise ValueError('Wrong DINO encoder/preprocessing/cache identity')
        split=self.identity['split'];path=index/(split+'_scenes.json')
        if idx['index_hashes'][split]!=self.identity['index_sha256'] or idx['partition_sha256']!=self.identity['partition_sha256'] or file_sha256(path)!=idx['files'][split]:
            raise ValueError('DINO time/population/split mismatch')
        if not allow_partial and not (self.dino_root/'COMPLETE.json').exists():raise ValueError('DINO cache incomplete')
        self.dino_identity=ident;self.dino_scenes=json.loads(path.read_text());self._checked_chunks=set()
        if [r['token'] for r in self.dino_scenes]!=[r['token'] for r in self.index]:raise ValueError('DINO scene order changed')

    def read_image(self,index):
        channels=self.dino_identity['recipe']['feature_dim'];h,w=self.dino_identity['grid_hw']
        if index<0:return torch.zeros(channels,h,w,dtype=torch.float16),torch.zeros(h,w,dtype=torch.bool)
        chunk,at=divmod(index,self.dino_identity['chunk_size']);path=self.dino_root/f'chunk_{chunk:06d}.safetensors'
        if chunk not in self._checked_chunks:
            meta=json.loads((self.dino_root/f'chunk_{chunk:06d}.json').read_text())
            if meta['identity']!=self.dino_identity['identity'] or path.stat().st_size!=meta['bytes']:raise ValueError('DINO chunk identity/size mismatch')
            self._checked_chunks.add(chunk)
        with safe_open(str(path),framework='pt',device='cpu') as f:
            if f.metadata()['identity']!=self.dino_identity['identity']:raise ValueError('DINO tensor identity mismatch')
            z=f.get_slice('features')[at];valid=f.get_slice('valid')[at]
        if z.shape!=(channels,h,w) or valid.shape!=(h,w) or valid.dtype!=torch.bool:raise ValueError('Patch geometry mismatch')
        return z,valid

    def __getitem__(self,i):
        observation,targets=super().__getitem__(i)
        if self.dino_identity is None:return observation,targets
        row=self.dino_scenes[i]
        for key,enabled,indices in [('current_dino',self.dino_current,[0]),('future_dino',self.dino_future,[1,2,3])]:
            if not enabled:continue
            hs=[];ms=[]
            for h in indices:
                pairs=[self.read_image(v) for v in row['images'][h]]
                if h==0 and any(v<0 for v in row['images'][h]):raise ValueError('Current encoding failure cannot silently skip')
                hs.append(torch.stack([v[0] for v in pairs]));ms.append(torch.stack([v[1] for v in pairs]))
            targets[key]=hs[0] if key=='current_dino' else torch.stack(hs)
            targets[key+'_valid']=ms[0] if key=='current_dino' else torch.stack(ms)
        return observation,targets
