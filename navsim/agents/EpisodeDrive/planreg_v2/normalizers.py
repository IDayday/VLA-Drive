"""Physical-coordinate contracts. Statistics are measured, never learned."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from . import NORMALIZER_SCHEMA


def wrap_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def unwrap_heading(x,valid_mask=None):
    if valid_mask is not None:
        # Compact in time logically, without allowing a padded NaN into cumsum.
        valid_mask=valid_mask.bool()
        clean=torch.where(valid_mask,x,0.)
        previous=torch.zeros_like(clean[...,0]); unwrapped=previous; seen=valid_mask[...,0]&False
        outputs=[]
        for i in range(clean.shape[-1]):
            ok=valid_mask[...,i]
            candidate=torch.where(seen,unwrapped+wrap_angle(clean[...,i]-previous),wrap_angle(clean[...,i]))
            unwrapped=torch.where(ok,candidate,unwrapped)
            previous=torch.where(ok,clean[...,i],previous);seen=seen|ok
            outputs.append(torch.where(ok,unwrapped,0.))
        return torch.stack(outputs,-1)
    first = wrap_angle(x[..., :1])
    return torch.cat((first, first + wrap_angle(x[..., 1:] - x[..., :-1]).cumsum(-1)), -1)


class FP32Statistics(nn.Module):
    def _apply(self, fn, recurse=True):
        # Preserve original FP32 values even when an AMP plugin requests BF16.
        original = {n: b for n, b in self._buffers.items() if b.is_floating_point()}
        super()._apply(fn, recurse)
        for name, value in original.items():
            self._buffers[name] = value.to(device=self._buffers[name].device, dtype=torch.float32)
        return self


class TrajectoryNormalizer(FP32Statistics):
    def __init__(self, mean, std, metadata, std_floor=1e-3, mode=None):
        super().__init__()
        mode=mode or metadata.get('mode','stepwise_zscore')
        if mode not in ('stepwise_zscore','global_zscore','legacy_fixed_affine'): raise ValueError('Unknown normalization mode')
        mean, std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()
        if mode!='stepwise_zscore' and mean.shape==(3,) and std.shape==(3,):
            mean,std=mean.expand(8,3).clone(),std.expand(8,3).clone()
        if mean.shape != (8, 3) or std.shape != (8, 3):
            raise ValueError("Trajectory statistics must have shape [8,3]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError("Non-finite/negative statistics")
        if std_floor <= 0 or metadata.get("split") in ("test", "navtest", "validation", "val"):
            raise ValueError("Positive std floor and training-only provenance required")
        if not metadata.get("token_sha256") or int(metadata.get("count", 0)) <= 0:
            raise ValueError("Missing measured training statistics provenance")
        self.register_buffer("mean", mean)
        self.register_buffer("raw_std", std)
        self.register_buffer("std", std.clamp_min(std_floor))
        self.register_buffer("times", torch.arange(1, 9).float() * .5)
        self.metadata = dict(metadata, mode=mode,std_floor=std_floor,
                             floor_fraction=float((std < std_floor).float().mean()))

    def get_extra_state(self):
        return self.metadata

    def set_extra_state(self, state):
        self.metadata = state

    def indices(self, times):
        times = torch.as_tensor(times, device=self.times.device, dtype=torch.float32)
        match = torch.isclose(times[..., None], self.times, atol=1e-6, rtol=0)
        if not (match.sum(-1) == 1).all():
            raise ValueError("Unsupported statistics timestamp; rounding is forbidden")
        return match.long().argmax(-1)

    def normalize(self, physical, times=None, valid_mask=None):
        idx = self.indices(times) if times is not None else slice(0, physical.shape[-2])
        clean=physical if valid_mask is None else torch.where(valid_mask[...,None],physical,0.)
        value = torch.cat((clean[..., :2], unwrap_heading(clean[..., 2],valid_mask)[..., None]), -1)
        result = (value - self.mean[idx]) / self.std[idx]
        return result if valid_mask is None else torch.where(valid_mask[...,None],result,0.)

    def inverse(self, normalized, times=None, valid_mask=None):
        idx = self.indices(times) if times is not None else slice(0, normalized.shape[-2])
        clean=normalized if valid_mask is None else torch.where(valid_mask[...,None],normalized,0.)
        value = clean * self.std[idx] + self.mean[idx]
        result = torch.cat((value[..., :2], wrap_angle(value[..., 2:3])), -1)
        return result if valid_mask is None else torch.where(valid_mask[...,None],result,0.)

    def save(self, path):
        Path(path).write_text(json.dumps(dict(mean=self.mean.cpu().tolist(),
            std=self.raw_std.cpu().tolist(), metadata=self.metadata), indent=2))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(data["mean"], data["std"], data["metadata"], data["metadata"].get("std_floor", 1e-3))


class MotionConditionNormalizer(FP32Statistics):
    """Declared physical scales, not estimated trajectory statistics."""
    def __init__(self, scales=(30., 10., 1., 1., 15., 15., 8., 8.)):
        super().__init__()
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        if self.scales.shape != (8,) or not torch.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError("Eight positive motion scales required")

    def get_extra_state(self):
        return dict(kind='configured_fixed_physical_scales',units=['m','m','unitless','unitless','m/s','m/s','m/s^2','m/s^2'],source='V2.2 declared engineering scales; not fitted Z-score')

    def set_extra_state(self,state):
        if state!=self.get_extra_state(): raise ValueError('Motion scale provenance mismatch')

    def forward(self, x):
        return x / self.scales

    def inverse(self, x):
        return x * self.scales


class EgoStateNormalizer(FP32Statistics):
    def __init__(self,scales=(1.,1.,1.,1.,15.,15.,8.,8.)):
        super().__init__()
        self.register_buffer("scales", torch.tensor(scales,dtype=torch.float32))
        if self.scales.shape!=(8,) or not torch.isfinite(self.scales).all() or (self.scales<=0).any():
            raise ValueError('Eight finite positive ego scales required')

    def get_extra_state(self):
        return dict(kind='configured_fixed_physical_scales',units=['unitless']*4+['m/s']*2+['m/s^2']*2,source='V2.2 declared ego scales; not fitted Z-score')

    def set_extra_state(self,state):
        if state!=self.get_extra_state(): raise ValueError('Ego scale provenance mismatch')

    def forward(self, status):
        if status.shape[-1] != 8:
            raise ValueError("Ego status must be navigation[4], vx/vy, ax/ay")
        return status / self.scales


def measured_statistics(records, split, data_version, std_floor=1e-3, mode='stepwise_zscore'):
    """Records: (unique scene token, raw physical [8,3], valid [8])."""
    if split not in ("train", "trainval_final_fit"):
        raise ValueError("Statistics require train or declared final-fit trainval")
    unique, values, masks = {}, [], []
    for token, trajectory, valid in records:
        trajectory, valid = torch.as_tensor(trajectory).double().clone(), torch.as_tensor(valid).bool()
        if trajectory.shape != (8, 3) or valid.shape != (8,):
            raise ValueError("Invalid statistics record shape")
        if token in unique:
            old,old_valid=unique[token]
            if not torch.equal(torch.where(valid[:,None],old,0.),torch.where(valid[:,None],trajectory,0.)) or not torch.equal(old_valid, valid):
                raise ValueError("Conflicting duplicate training token")
            continue
        unique[token] = (trajectory.clone(), valid.clone())
        valid = valid & torch.isfinite(trajectory).all(-1)
        trajectory[valid, 2] = unwrap_heading(trajectory[valid, 2])
        values.append(torch.where(valid[:, None], trajectory, 0.))
        masks.append(valid)
    if not values:
        raise ValueError("No unique training scenes")
    x, mask = torch.stack(values), torch.stack(masks)
    n = mask.sum(0)
    if (n == 0).any():
        raise ValueError("Every trajectory timestep needs measured statistics")
    mean = x.sum(0) / n[:, None]
    std = (((x - mean).square() * mask[..., None]).sum(0) / n[:, None]).sqrt()
    if mode=='global_zscore':
        flat=x[mask];mean=flat.mean(0).expand(8,3).clone();std=flat.std(0,unbiased=False).expand(8,3).clone()
    elif mode!='stepwise_zscore': raise ValueError('Fixed affine constants must be explicitly supplied, never called measured statistics')
    raw_hash=hashlib.sha256()
    for token,(raw,valid) in sorted(unique.items()):
        raw_hash.update(token.encode());raw_hash.update(valid.numpy().tobytes())
        raw_hash.update(torch.where(valid[:,None],raw,0.).float().contiguous().numpy().tobytes())
    metadata = dict(count=len(unique), valid_count=n.tolist(), split=split,
        token_sha256=hashlib.sha256("\n".join(sorted(unique)).encode()).hexdigest(),
        dt_seconds=.5, heading="current-anchor-relative, temporally unwrapped radians",
        coordinates="current rear axle, metres", data_version=data_version,
        schema=NORMALIZER_SCHEMA,mode=mode,raw_gt_sha256=raw_hash.hexdigest(),
        statistics_contract='unique_raw_gt_only_8x3_masked_unwrap_population_std_v2',
        valid_mask=(n>0).tolist())
    return TrajectoryNormalizer(mean.float(), std.float(), metadata, std_floor,mode)
