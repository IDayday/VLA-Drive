"""New contracts; the historical World and DDP validators remain unchanged."""
from dataclasses import dataclass
from typing import Any, Mapping, Tuple
import torch

ARCHITECTURES = ('A', 'B', 'C', 'D')
SCHEMA = 'qwen_world_four_frameworks_v1'
LOSS_WEIGHTS = dict(x=10., dino=1., wa_pre=.5, object=.1,
                    future_track=.2, wa_post=.25, evaluation=.2)
METRICS = ('NC', 'DAC', 'TTC', 'EP', 'Comfort', 'DDC')

@dataclass(frozen=True)
class CurrentContext:
    scene: Any
    valid: Any
    ego: torch.Tensor
    encoded: Any
    scene_identity: Tuple[str, ...]
    policy_version: str
    execution_identity: str
    position_mask_identity: str

@dataclass(frozen=True)
class ArchitecturePrefix:
    architecture_id: str
    scene_identity: Tuple[str, ...]
    policy_version: str
    execution_identity: str
    position_mask_identity: str
    tensors: Any

    def index_select(self, indices):
        ids = indices.detach().cpu().tolist()
        return ArchitecturePrefix(self.architecture_id, tuple(self.scene_identity[i] for i in ids),
            self.policy_version, self.execution_identity, self.position_mask_identity,
            self.tensors.index_select(indices))

def validate_action(value):
    if value.ndim != 3 or value.shape[1:] != (8, 3) or not torch.isfinite(value).all():
        raise ValueError('Finite normalized [B,8,3] action required')

def first_event_class(seconds):
    """(0,.5]→0, ..., (3.5,4]→7; none→8. t0 maps to first bin."""
    import math
    if seconds is None:
        return 8
    if not math.isfinite(seconds) or seconds < 0 or seconds > 4.000001:
        raise ValueError('An official event must be within the physical window')
    return max(0, min(7, math.ceil((seconds-1e-8)/.5)-1))
