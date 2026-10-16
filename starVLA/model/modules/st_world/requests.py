"""Label-independent requests sampled once before optimizer-batch slicing."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from .config import FUTURE_TIMES_S


def _indices(indices, batch, views, positions):
    if indices.dtype != torch.long or indices.ndim != 3 or indices.shape[:2] != (batch, views):
        raise ValueError('Expected B,V,K integer patch addresses')
    if not indices.shape[-1] or ((indices < 0) | (indices >= positions)).any():
        raise ValueError('Requested patch does not exist in the declared grid')


@dataclass(frozen=True)
class SampledRequests:
    patch_indices: torch.Tensor  # B,V,K; shared across requested physical times
    grid_hw: tuple
    future_pair: torch.Tensor | None = None  # B,2; indexes .5,1,2,4 seconds
    future_times_s: tuple = FUTURE_TIMES_S

    def validate(self):
        if not isinstance(self.grid_hw, tuple) or len(self.grid_hw) != 2 or any(type(n) is not int or n <= 0 for n in self.grid_hw):
            raise ValueError('Actual rectangular grid required')
        if self.patch_indices.ndim != 3 or min(self.patch_indices.shape) < 1:
            raise ValueError('Nonempty B,V,K requests required')
        batch, views, _ = self.patch_indices.shape
        _indices(self.patch_indices, batch, views, self.grid_hw[0] * self.grid_hw[1])
        if self.future_times_s != FUTURE_TIMES_S:
            raise ValueError('Registered physical time table changed')
        if self.future_pair is not None:
            p = self.future_pair
            if p.dtype != torch.long or p.shape != (batch, 2):
                raise ValueError('Expected B,2 adjacent future requests')
            if ((p[:, 0] < 0) | (p[:, 1] >= len(self.future_times_s))).any() or not (p[:, 1] == p[:, 0] + 1).all():
                raise ValueError('Only an adjacent pair of registered future times is valid')
            if p.device != self.patch_indices.device:
                raise ValueError('Request tensors must share a device')
        return self

    def slice(self, start, stop):
        """Slice an already sampled optimizer batch; never resample a microbatch."""
        self.validate()
        return SampledRequests(self.patch_indices[start:stop], self.grid_hw,
            None if self.future_pair is None else self.future_pair[start:stop],
            self.future_times_s).validate()

    def query_layout(self, *, current=True, future=False, device=None):
        """Return view, xy, seconds in physical-time → view → patch order."""
        self.validate()
        if not isinstance(current, bool) or not isinstance(future, bool) or not (current or future):
            raise ValueError('Explicit nonempty objective request required')
        if future and self.future_pair is None:
            raise ValueError('Future objective has no sampled time pair')
        index = self.patch_indices.to(device=device)
        batch, views, count = index.shape
        height, width = self.grid_hw
        xy = torch.stack(((index.remainder(width).float() + .5) / width * 2 - 1,
                          (torch.div(index, width, rounding_mode='floor').float() + .5) / height * 2 - 1), -1)
        time_parts = []
        if current:
            time_parts.append(torch.zeros((batch, 1), device=index.device))
        if future:
            table = torch.tensor(self.future_times_s, device=index.device, dtype=torch.float32)
            time_parts.append(table[self.future_pair.to(index.device)])
        times = torch.cat(time_parts, dim=1)
        num_times = times.shape[1]
        ids = torch.arange(views, device=index.device)[None, None, :, None].expand(batch, num_times, views, count)
        coords = xy[:, None].expand(batch, num_times, views, count, 2)
        times = times[:, :, None, None].expand(batch, num_times, views, count)
        return ids.reshape(batch, -1), coords.reshape(batch, -1, 2), times.reshape(batch, -1)


def sample_requests(batch_size, views, grid_hw, positions_per_view, *, patch_generator,
                    future=False, horizon_generator=None):
    """Uniform sampling without replacement, independent of target validity.

    CPU streams are intentionally explicit and distinct from FM/dropout streams.
    Full-grid current controls request every position without consuming patch RNG.
    """
    if not isinstance(grid_hw, (tuple, list)) or len(grid_hw) != 2:
        raise ValueError('Actual height,width grid required')
    for n in (batch_size, views, positions_per_view, *grid_hw):
        if type(n) is not int or n < 1:
            raise ValueError('Positive integral batch/view/grid/request sizes required')
    total = grid_hw[0] * grid_hw[1]
    if positions_per_view > total:
        raise ValueError('Sampling cannot create nonexistent teacher positions')
    if not isinstance(patch_generator, torch.Generator) or patch_generator.device.type != 'cpu':
        raise ValueError('Dedicated CPU patch generator required')
    if not isinstance(future, bool):
        raise ValueError('Future request must be Boolean')
    if future and (not isinstance(horizon_generator, torch.Generator) or horizon_generator.device.type != 'cpu'):
        raise ValueError('Dedicated CPU future-pair generator required')
    if future and horizon_generator is patch_generator:
        raise ValueError('Patch and physical-horizon RNG streams must be separate')
    if positions_per_view == total:
        indices = torch.arange(total).expand(batch_size, views, total).clone()
    else:
        indices = torch.stack([torch.randperm(total, generator=patch_generator)[:positions_per_view]
                               for _ in range(batch_size * views)]).reshape(batch_size, views, positions_per_view)
    pair = None
    if future:
        first = torch.randint(len(FUTURE_TIMES_S) - 1, (batch_size,), generator=horizon_generator)
        pair = torch.stack((first, first + 1), dim=1)
    return SampledRequests(indices, tuple(grid_hw), pair).validate()


def gather_current(features, valid, requests):
    """B,V,H,W,C → B,V,K,C. Selection does not change the request RNG."""
    requests.validate()
    if features.ndim != 5 or valid.dtype != torch.bool or valid.shape != features.shape[:-1]:
        raise ValueError('Current targets require B,V,H,W,C and Boolean spatial validity')
    batch, views, height, width, channels = features.shape
    if (height, width) != requests.grid_hw or requests.patch_indices.shape[:2] != (batch, views):
        raise ValueError('Requested addresses belong to a different target layout')
    if features.device != valid.device or not features.is_floating_point() or channels < 1:
        raise ValueError('Floating target and mask must share a device')
    index = requests.patch_indices.to(features.device)
    selected = features.flatten(2, 3).gather(2, index[..., None].expand(-1, -1, -1, channels))
    mask = valid.flatten(2, 3).gather(2, index)
    return selected.detach(), mask


def gather_future(features, valid, requests):
    """Select the pre-sampled pair and identical view/patch addresses at both times."""
    requests.validate()
    if requests.future_pair is None:
        raise ValueError('No registered future pair was requested')
    if features.ndim != 6 or valid.dtype != torch.bool or valid.shape != features.shape[:-1]:
        raise ValueError('Future targets require B,T,V,H,W,C and Boolean validity')
    if features.shape[1] != len(requests.future_times_s) or len(features) != len(requests.patch_indices):
        raise ValueError('Future time table or scene population changed')
    if features.device != valid.device:
        raise ValueError('Future targets and mask must share a device')
    pair = requests.future_pair.to(features.device)
    rows = torch.arange(len(features), device=features.device)
    values, masks = [], []
    for endpoint in range(2):
        chosen, chosen_mask = gather_current(features[rows, pair[:, endpoint]], valid[rows, pair[:, endpoint]], requests)
        values.append(chosen)
        masks.append(chosen_mask)
    return torch.stack(values, 1), torch.stack(masks, 1)
