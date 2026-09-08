"""Persistent FP32 EMA accumulation, independent of forward-copy precision."""
import copy
import math
import torch
from torch import nn


def visual_parameters(backbone):
    return dict([('vision.'+n,p) for n,p in backbone.model.vision_model.named_parameters()] +
                [('adapter.'+n,p) for n,p in backbone.planning_register_adapter.named_parameters()])


class FP32MasterEMA(nn.Module):
    SCHEMA = 'planreg_v2_fp32_master_v1'

    def __init__(self, student, total_steps, global_batch):
        super().__init__()
        if total_steps <= 0 or global_batch <= 0:
            raise ValueError('EMA requires a fixed positive schedule and actual accumulated global batch')
        self.vision = copy.deepcopy(student.model.vision_model)
        self.adapter = copy.deepcopy(student.planning_register_adapter)
        self.requires_grad_(False)
        source = visual_parameters(student)
        self.names = [n for n,p in source.items() if p.requires_grad]
        if not self.names:
            raise ValueError('EMA has no trainable visual parameters to track')
        for i,name in enumerate(self.names):
            self.register_buffer('master_%04d'%i,source[name].detach().float().clone())
        self.register_buffer('updates',torch.zeros((),dtype=torch.long))
        self.register_buffer('total_steps',torch.tensor(total_steps,dtype=torch.long))
        self.register_buffer('global_batch',torch.tensor(global_batch,dtype=torch.long))
        self.last_diagnostics = {}
        self.train(False)

    def get_extra_state(self):
        return dict(schema=self.SCHEMA, names=self.names, dtype='float32', schedule_version=1)

    def set_extra_state(self,state):
        if state['schema'] != self.SCHEMA or state['names'] != self.names or state['dtype'] != 'float32':
            raise ValueError('Incompatible EMA schema/name/dtype map')

    def _apply(self,fn,recurse=True):
        masters = {n:b for n,b in self._buffers.items() if n.startswith('master_')}
        super()._apply(fn,recurse)
        for name,value in masters.items():
            self._buffers[name] = value.to(device=self._buffers[name].device,dtype=torch.float32)
        return self

    def train(self,mode=True):
        super().train(False)
        # InternViT gradient checkpointing must not introduce teacher training mode.
        return self

    def momentum(self):
        step,total = int(self.updates),int(self.total_steps)
        if step >= total:
            raise RuntimeError('EMA schedule exhausted: extension requires an explicit new schedule version')
        progress = step/max(1,total-1)
        reference = .9999-(.9999-.996)*.5*(1+math.cos(math.pi*progress))
        return reference**(int(self.global_batch)/16.)

    @torch.no_grad()
    def update(self, student, diagnostics=False):
        source, target = visual_parameters(student), dict(self.named_parameters())
        momentum = self.momentum()
        update_sq = drift_sq = 0.
        unchanged = copy_changed = total_elements = 0
        for i,name in enumerate(self.names):
            master = getattr(self,'master_%04d'%i)
            value = source[name].detach().float()
            before = master.clone() if diagnostics else None
            copy_before = target[name].clone() if diagnostics else None
            master.add_(value-master,alpha=1-momentum)
            target[name].copy_(master)
            if diagnostics:
                update_sq += float((master-before).square().sum())
                drift_sq += float((master-value).square().sum())
                unchanged += int((master == before).sum())
                copy_changed += int((target[name] != copy_before).sum())
                total_elements += master.numel()
        self.updates.add_(1)
        self.last_diagnostics = dict(momentum=momentum, updates=int(self.updates))
        if diagnostics:
            self.last_diagnostics.update(master_update_norm=math.sqrt(update_sq),
                teacher_student_drift=math.sqrt(drift_sq),master_zero_update_fraction=unchanged/max(1,total_elements),
                forward_copy_changed_fraction=copy_changed/max(1,total_elements))

    @torch.no_grad()
    def forward(self,pixels):
        encoded,_ = self.adapter._encode_with_registers(self.vision,pixels)
        return self.adapter.register_projection(self.adapter.register_norm(encoded.float()))
