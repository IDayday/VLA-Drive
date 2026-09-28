"""Remove hidden futures BEFORE any learned encoding or arithmetic."""
from dataclasses import dataclass
import torch


@dataclass
class TeacherInputs:
    current: torch.Tensor       # B,N,8: ego(t0) xy,z,l,w,h,sin,cos
    active: torch.Tensor        # B,N
    future: torch.Tensor        # B,N,T,2 in COMMON ego(t0), meters
    visible: torch.Tensor      # B,N,T, never includes the hidden target
    target: torch.Tensor       # B, actor index
    navigation: torch.Tensor   # B, current command
    ego_state: torch.Tensor    # B,4, legal current/history state

    def sanitized(self):
        b,n,d = self.current.shape
        if d != 8 or self.active.shape != (b,n) or self.active.dtype != torch.bool or not self.active[:,0].all():
            raise ValueError('Invalid teacher current/active contract')
        if self.future.ndim != 4 or self.future.shape[:2] != (b,n) or self.future.shape[-1] != 2:
            raise ValueError('Invalid future shape')
        if self.visible.shape != self.future.shape[:-1] or self.visible.dtype != torch.bool:
            raise ValueError('Invalid visible point mask')
        if (self.visible & ~self.active[:,:,None]).any(): raise ValueError('Visible inactive actor')
        if self.target.shape != (b,) or self.target.dtype != torch.long or ((self.target<0)|(self.target>=n)).any():
            raise ValueError('Invalid target actor')
        rows = torch.arange(b, device=self.current.device)
        if not self.active[rows,self.target].all() or self.visible[rows,self.target].any():
            raise ValueError('Target must be active with its ENTIRE future masked')
        if self.navigation.shape != (b,) or self.navigation.dtype != torch.long or ((self.navigation<0)|(self.navigation>3)).any():
            raise ValueError('Invalid current navigation')
        if self.ego_state.shape != (b,4) or not torch.isfinite(self.ego_state).all():
            raise ValueError('Invalid current ego state')
        if not torch.isfinite(self.current[self.active]).all() or (self.current[self.active][:,3:6]<=0).any():
            raise ValueError('Invalid active current boxes')
        if not torch.isfinite(self.future[self.visible]).all(): raise ValueError('Invalid visible trajectory')
        return (torch.where(self.active[...,None],self.current,0.),
                torch.where(self.visible[...,None],self.future,0.))
