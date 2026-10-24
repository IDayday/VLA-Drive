"""Locked C1 current targets only; explicit old F/L/R -> native L/F/R mapping."""
import io
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from .temporal_contracts import TEACHER_ID


class CurrentDINOTargets:
    def __init__(self,root):
        self.root=Path(root)
        self.identity=json.loads((self.root/'identity.json').read_text())
        complete=json.loads((self.root/'COMPLETE.json').read_text())
        if (self.identity['identity']!=TEACHER_ID or complete['identity']!=TEACHER_ID or
            not complete['current_only'] or self.identity['grid_hw']!=[6,8] or self.identity['cameras']!=3 or
            self.identity['feature_dim']!=1024 or self.identity['future_teacher_required_by_training']):
            raise ValueError('Wrong/incomplete current-only C1 teacher')

    def read(self,token,expected_sha=None):
        record=json.loads((self.root/'records'/(token+'.json')).read_text())
        if expected_sha is not None and record['sha256']!=expected_sha:raise ValueError('Teacher manifest target changed')
        payload=(self.root/'targets'/(token+'.npz')).read_bytes()
        if record['token']!=token or record['identity']!=TEACHER_ID or hashlib.sha256(payload).hexdigest()!=record['sha256']:
            raise ValueError('Changed current teacher target')
        with np.load(io.BytesIO(payload),allow_pickle=False) as values:
            y=torch.from_numpy(values['current_dino'].copy())
            valid=torch.from_numpy(values['valid'].copy())
        if y.shape!=(3,1024,6,8) or valid.shape!=(3,6,8) or valid.dtype!=torch.bool or not torch.isfinite(y).all():
            raise ValueError('Invalid current-only DINO shape/values')
        # Historical index and its writer use CAM_F0,CAM_L0,CAM_R0.
        return y[[1,0,2]].float(),valid[[1,0,2]]
