"""Physical ego(t0) grid: x forward, y left; tensors use [y row, x column]."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class GridSpec:
    xmin: float = -20.
    xmax: float = 80.
    ymin: float = -40.
    ymax: float = 40.
    dx: float = .5
    dy: float = .5

    def __post_init__(self):
        values = (self.xmin, self.xmax, self.ymin, self.ymax, self.dx, self.dy)
        if not all(math.isfinite(x) for x in values) or self.xmax <= self.xmin or self.ymax <= self.ymin or min(self.dx, self.dy) <= 0:
            raise ValueError('Invalid physical grid')
        for extent, step in ((self.xmax-self.xmin, self.dx), (self.ymax-self.ymin, self.dy)):
            if abs(extent/step-round(extent/step)) > 1e-6:
                raise ValueError('Grid bounds must contain complete cells')

    @property
    def width(self):
        return round((self.xmax-self.xmin)/self.dx)

    @property
    def height(self):
        return round((self.ymax-self.ymin)/self.dy)

    def centers(self, *, device=None, dtype=torch.float32):
        x = self.xmin + (torch.arange(self.width, device=device, dtype=dtype)+.5)*self.dx
        y = self.ymin + (torch.arange(self.height, device=device, dtype=dtype)+.5)*self.dy
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack((xx, yy), -1)

    def normalized_reference(self, xy):
        """MSDeformAttn coordinates; never clamp an out-of-range position."""
        low = xy.new_tensor([self.xmin, self.ymin])
        high = xy.new_tensor([self.xmax, self.ymax])
        uv = (xy-low)/(high-low)
        invalid = ((uv < 0) | (uv >= 1) | ~torch.isfinite(uv)).any(-1)
        return uv, invalid

    def prepare_sample(self, xy):
        """Ephemeral coordinates for repeated reads of the same XY tensor.

        The returned tensors retain the coordinate gradient. They must only be
        reused within one forward, with the same grid and unchanged XY.
        """
        uv, invalid = self.normalized_reference(xy)
        safe = torch.where(torch.isfinite(uv), uv, torch.full_like(uv, -2))
        return (2*safe-1).unsqueeze(2), ~invalid

    def sample(self, fields, xy, *, mode='bilinear', _prepared=None):
        """Returns values plus explicit validity; padding is never a free label."""
        if fields.ndim != 4 or fields.shape[-2:] != (self.height, self.width) or xy.ndim != 3 or len(fields) != len(xy):
            raise ValueError('Expected B,C,H,W and B,K,2 on the registered grid')
        from torch.nn.functional import grid_sample
        coordinates, supported = self.prepare_sample(xy) if _prepared is None else _prepared
        if coordinates.shape != (*xy.shape[:2], 1, 2) or supported.shape != xy.shape[:2]:
            raise ValueError('Prepared coordinates must match the current XY tensor')
        values = grid_sample(fields, coordinates, mode=mode,
                             padding_mode='zeros', align_corners=False).squeeze(-1).transpose(1, 2)
        return values, supported

    def resworld_config(self, *, depth=(1., 100., 1.)):
        return {'x': [self.xmin, self.xmax, self.dx], 'y': [self.ymin, self.ymax, self.dy],
                'z': [-2., 2., 4.], 'depth': list(depth)}
