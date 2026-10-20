"""Conservative, detached numeric support check before a learned effect query.

Passing this gate is NOT an interpolation certificate, simulator execution or
safety label. It only excludes endpoint proxies outside the declared small G0
intervention family. Thresholds are fixed engineering bounds, not fit on DEV.
"""
from dataclasses import asdict,dataclass
import math
import numpy as np
import torch

from .intervention_basis import BasisSpec,basis_matrix,reconstruct,wrap_angle
from .trajectory_constraints import ConstraintSpec,geometric_screen


@dataclass(frozen=True)
class EndpointSupportSpec:
    longitudinal_coefficient_m: float = 2.
    lateral_coefficient_m: float = .6
    reconstruction_xy_tolerance_m: float = 2e-5
    reconstruction_yaw_tolerance_rad: float = 2e-5
    coefficient_tolerance_m: float = 2e-5
    # Additional conservative numeric envelope, not an official NAVSIM metric.
    maximum_heading_change_rad: float = .5
    active_coefficients: tuple[bool,...] = (True,False,True,True,True,False,True,True)

    def validate(self):
        for k,v in asdict(self).items():
            if k=='active_coefficients':continue
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0:
                raise ValueError('Finite fixed positive support bounds required')
        if self.active_coefficients!=(True,False,True,True,True,False,True,True):
            raise ValueError('Do not invent exposure for G0 untested intervention windows')
        return self


@torch.no_grad()
def endpoint_support(base_raw,endpoint_raw,*,spec=EndpointSupportSpec()):
    """Return detached bool [B] plus auditable reasons; never invoke a teacher.

Only current Base and predicted endpoint are accepted. No GT, feedback, learned
threshold, scene label or metric cache enters this check. The small least-square
fit is CPU float64 and checks reconstruction, not a finite-difference gradient.
Native initial velocity/controller tracking are unavailable here and never
claimed verified. Caller must apply this mask BEFORE the learned teacher call.
"""
    from starVLA.dataloader.foresight_dataset import decode_ego
    spec.validate()
    if (base_raw.ndim!=3 or base_raw.shape[1:]!=(8,4) or not len(base_raw)
            or endpoint_raw.shape!=base_raw.shape or endpoint_raw.device!=base_raw.device
            or base_raw.dtype!=torch.float32 or endpoint_raw.dtype!=torch.float32):
        raise ValueError('Canonical same-device FP32 raw8x4 Base/endpoint required')
    if not torch.isfinite(base_raw).all() or not torch.isfinite(endpoint_raw).all():
        raise FloatingPointError('A nonfinite model prediction cannot be masked away as unsupported')
    if (base_raw[...,2:4].square().sum(-1)<1e-12).any():
        raise ValueError('Canonical Base has an undefined heading')
    defined=(endpoint_raw[...,2:4].square().sum(-1)>=1e-12).all(-1).cpu().numpy()
    base=decode_ego(base_raw.detach().cpu()).double().numpy()
    endpoint=decode_ego(endpoint_raw.detach().cpu()).double().numpy()
    basis=BasisSpec();limits=ConstraintSpec();active=np.asarray(spec.active_coefficients)
    bounds=np.asarray([spec.longitudinal_coefficient_m]*4+[spec.lateral_coefficient_m]*4)
    rows=[];mask=[]
    for i,(a,b) in enumerate(zip(base,endpoint)):
        reasons=[];coeff=np.zeros(8);xy_error=yaw_error=None
        if not defined[i]:reasons.append('undefined_endpoint_heading')
        else:
            matrix=basis_matrix(a,basis).reshape(-1,8)[:,active]
            fitted=np.linalg.lstsq(matrix,(b[:,:2]-a[:,:2]).reshape(-1),rcond=None)[0]
            coeff[active]=fitted;rebuilt=reconstruct(a,coeff,basis)
            displacement=b[:,:2]-a[:,:2]
            along=np.stack((np.cos(a[:,2]),np.sin(a[:,2])),-1)
            lateral=np.stack((-np.sin(a[:,2]),np.cos(a[:,2])),-1)
            if np.abs((displacement*along).sum(-1)).max()>spec.longitudinal_coefficient_m+spec.coefficient_tolerance_m:
                reasons.append('longitudinal_displacement_radius')
            if np.abs((displacement*lateral).sum(-1)).max()>spec.lateral_coefficient_m+spec.coefficient_tolerance_m:
                reasons.append('lateral_displacement_radius')
            xy_error=float(np.linalg.norm(rebuilt[:,:2]-b[:,:2],axis=-1).max())
            yaw_error=float(np.abs(wrap_angle(rebuilt[:,2]-b[:,2])).max())
            if np.any(np.abs(coeff)>bounds+spec.coefficient_tolerance_m):reasons.append('coefficient_radius')
            if xy_error>spec.reconstruction_xy_tolerance_m:reasons.append('outside_exposed_smooth_basis')
            if yaw_error>spec.reconstruction_yaw_tolerance_rad:reasons.append('yaw_not_from_same_intervention')
            if np.abs(wrap_angle(b[:,2]-a[:,2])).max()>spec.maximum_heading_change_rad:
                reasons.append('heading_change_envelope')
            geometry=geometric_screen(b,basis,limits)
            if not geometry['geometrically_admissible']:reasons.append('geometric_screen')
        accepted=not reasons;mask.append(accepted)
        rows.append({'applicable':accepted,'reasons':reasons,'fitted_coefficients_m':coeff.tolist(),
            'reconstruction_xy_error_m':xy_error,'reconstruction_yaw_error_rad':yaw_error})
    count=sum(mask)
    return {'mask':torch.tensor(mask,dtype=torch.bool,device=base_raw.device),
        'statistics':{'requested_rows':len(rows),'applicable_rows':count,'outside_rows':len(rows)-count,
            'applicable_fraction':count/len(rows),'status':'NUMERIC_ENVELOPE_AVAILABLE' if count else 'NOT_IDENTIFIABLE'},
        'rows':rows,'support_spec':asdict(spec),'initial_velocity_observed':False,
        'interpolation_certified':False,'real_executability':'UNTESTED_REQUIRES_REAL_CONTROLLER_REPLAY',
        'is_safety_label':False}
