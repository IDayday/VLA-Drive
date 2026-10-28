"""An explicit planning-label overlay; original observation data stay intact."""
import json
from pathlib import Path
import numpy as np
import torch
from .contracts import file_sha,fingerprint


class OptimizedTargets:
    def __init__(self,root,data):
        root=Path(root);m=json.loads((root/'MANIFEST.json').read_text());body=dict(m);seal=body.pop('sha256',None)
        complete=json.loads((root/'COMPLETE.json').read_text())
        if (m['schema']!='world_optimized_planning_targets_v1' or fingerprint(body)!=seal
                or complete['manifest_sha256']!=file_sha(root/'MANIFEST.json')
                or m['current_observation_manifest_sha256']!=data.sha or m['DEV_or_Test_labels_used'] is not False):
            raise ValueError('Invalid optimized planning supervision identity')
        for name,digest in m['files'].items():
            if file_sha(root/name)!=digest:raise ValueError('Changed optimized target file')
        tokens=json.loads((root/'tokens.json').read_text())
        if tokens!=data.tokens or fingerprint(tokens)!=m['token_sha256']:raise ValueError('Optimized target order/population mismatch')
        self.index={t:i for i,t in enumerate(tokens)};self.targets=np.load(root/'targets.npy',mmap_mode='r',allow_pickle=False)
        if self.targets.shape!=(len(tokens),8,3) or not np.isfinite(self.targets).all():raise ValueError('Invalid native target tensor')
        self.identity=m;self.root=str(root.resolve())

    def target(self,token):return torch.from_numpy(self.targets[self.index[token]].copy())


from .data import WorldSceneData


class OptimizedWorldSceneData(WorldSceneData):
    def __init__(self,root,supervision):
        super().__init__(root,role='TRAIN')
        self.planning_targets=OptimizedTargets(supervision,self)

    def target(self,token):return self.planning_targets.target(token)
