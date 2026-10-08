"""Current road/occupancy and time-specific semantics read the same planner BEV."""
import torch
from torch import nn


class StructuredSemantics(nn.Module):
    def __init__(self, mode, *, channels=256, distance_scale_m=10.):
        super().__init__()
        if mode not in ('none', 'full', 'event') or distance_scale_m <= 0:
            raise ValueError('Explicit registered semantic mode and physical scale required')
        self.mode, self.distance_scale_m = mode, float(distance_scale_m)
        self.current = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(channels, 2, 1))
        # Same output dimensions/parameter count, including G0's unlabelled readout.
        self.future = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(channels, 2, 1))

    def forward(self, B0, Bt):
        batch, times = Bt.shape[:2]
        current = self.current(B0)
        future = self.future(Bt.flatten(0, 1)).reshape(batch, times, 2, *B0.shape[-2:])
        p0 = current[:, 1:2].sigmoid()
        road = current[:, :1].tanh()*self.distance_scale_m
        if self.mode == 'full':
            pt = future.softmax(2)[:, :, 1:2]
            enter, release = (1-p0[:, None])*pt, p0[:, None]*(1-pt)
            conditional = None
        else:
            arrival, release_condition = future.sigmoid().unbind(2)
            arrival, release_condition = arrival[:, :, None], release_condition[:, :, None]
            # Predicted p0 is used both in training and deployment, never GT O0.
            enter = (1-p0[:, None])*arrival
            release = p0[:, None]*release_condition
            pt = enter+p0[:, None]*(1-release_condition)
            conditional = (arrival, release_condition)
        return {'road_distance_m': road, 'current_logits': current[:, 1:2], 'p0': p0,
                'future_logits': future, 'pt': pt, 'p_enter': enter, 'p_release': release,
                'conditional': conditional}


def endpoint_events(O0, Ot, valid0, validt, *, ignore_index=255):
    if O0.dtype != torch.uint8 or Ot.dtype != torch.uint8 or valid0.dtype != torch.bool or validt.dtype != torch.bool:
        raise ValueError('Occupancy and validity types are explicit')
    valid = valid0[:, None] & validt & (O0[:, None] != ignore_index) & (Ot != ignore_index)
    enter = (O0[:, None] == 0) & (Ot == 1) & valid
    release = (O0[:, None] == 1) & (Ot == 0) & valid
    return {'enter': enter, 'release': release, 'valid': valid,
            'arrival_mask': valid & (O0[:, None] == 0),
            'release_mask': valid & (O0[:, None] == 1)}
