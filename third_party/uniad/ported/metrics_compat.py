"""Small offline state container replacing the removed Lightning Metric API.

UniAD's collision, L2, update and compute implementations are unchanged. This
adapter does not alter masks, coordinates, geometry, or time aggregation.
Offline scene records are aggregated explicitly; no hidden distributed sync.
"""
import torch


class Metric(torch.nn.Module):
    def __init__(self, compute_on_step=False):
        super().__init__()
        self.compute_on_step = compute_on_step

    def add_state(self, name, default, dist_reduce_fx):
        if dist_reduce_fx != 'sum': raise ValueError('Only original sum states supported')
        self.register_buffer(name, default.clone())

    def forward(self, *args, **kwargs):
        return self.update(*args, **kwargs)
