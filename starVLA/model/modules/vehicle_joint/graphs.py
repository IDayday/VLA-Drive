"""Small current-only VEHICLE set, shared by training and camera prediction."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class VehicleGraphConfig:
    max_vehicles: int = 8
    max_context: int = 16
    existence_floor: float = .2
    radius_m: float = 50.
    neighbor_radius_m: float = 15.
    turn_radius_m: float = 12.
    corridor_half_width_m: float = 2.
    unknown_motion_margin_m: float = 2.


def select_vehicles(boxes, probabilities, geometric_support, speed, navigation, config=VehicleGraphConfig()):
    """No labels, track IDs, logged vehicle speeds or future arguments exist.

    Unknown vehicle speed stays unknown. Geometric FOV is not visibility.
    Returns source query indices; caller gathers differentiable predicted values.
    """
    n = len(boxes)
    if boxes.shape != (n, 8) or probabilities.shape != (n,) or geometric_support.shape != (n,):
        raise ValueError("Invalid current vehicle query contract")
    if navigation not in (0, 1, 2, 3) or not math.isfinite(float(speed)) or speed < 0:
        raise ValueError("Invalid current ego speed/navigation")
    if not torch.isfinite(boxes).all() or not torch.isfinite(probabilities).all() or (boxes[:, 3:6] <= 0).any():
        raise ValueError("Nonfinite current prediction or illegal vehicle dimensions")
    if geometric_support.dtype != torch.bool or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid existence probability/geometric support")
    if config.max_vehicles < 1 or config.max_context < 0:
        raise ValueError("Invalid vehicle capacity")
    boxes, probabilities = boxes.detach(), probabilities.detach()
    length = min(45., max(12., float(speed)*4+8))
    s = torch.linspace(0, length, 49, device=boxes.device, dtype=boxes.dtype)
    theta = (s/config.turn_radius_m).clamp(max=math.pi/2)
    left = torch.stack((config.turn_radius_m*theta.sin(),
                        config.turn_radius_m*(1-theta.cos())+(s-config.turn_radius_m*math.pi/2).clamp_min(0)), -1)
    straight = torch.stack((s, torch.zeros_like(s)), -1)
    right = left*left.new_tensor([1., -1.])
    path = (left, straight, right, torch.cat((left, straight, right)))[navigation]
    xy = boxes[:, :2]
    distance = xy.norm(dim=-1)
    clearance = torch.cdist(xy, path).amin(-1)-boxes[:, 3:5].norm(dim=-1)/2
    clearance = clearance-config.corridor_half_width_m-config.unknown_motion_margin_m
    exists = (probabilities >= config.existence_floor) & (distance <= config.radius_m)
    ordered = torch.where(exists)[0].tolist()
    ordered.sort(key=lambda i: (-float(probabilities[i]), float(distance[i]), i))
    unique, duplicate = [], {}
    for i in ordered:
        same = next((j for j in unique if float((xy[i]-xy[j]).norm()) < .6
                     and bool(((boxes[i, 3:6]/boxes[j, 3:6] >= .75) &
                               (boxes[i, 3:6]/boxes[j, 3:6] <= 1.3334)).all())), None)
        if same is None: unique.append(i)
        else: duplicate[i] = same
    direct = [i for i in unique if geometric_support[i] and clearance[i] <= 0]
    direct.sort(key=lambda i: (float(clearance[i])+.05*float(distance[i]), -float(probabilities[i]), i))
    primary = direct[:config.max_vehicles]
    secondary = [i for i in unique if geometric_support[i] and i not in primary and primary
                 and float((xy[primary]-xy[i]).norm(dim=-1).min()) <= config.neighbor_radius_m]
    secondary.sort(key=lambda i: (float((xy[primary]-xy[i]).norm(dim=-1).min()), float(distance[i]), i))
    selected = (primary+secondary)[:config.max_vehicles]
    context_pool = [i for i in unique if i not in selected]
    context_pool.sort(key=lambda i: (float(clearance[i])+.05*float(distance[i]), i))
    context = context_pool[:config.max_context]
    audit = {"source_vehicle_queries": n, "selected_vehicles": len(selected), "ego_only": not selected,
             "context_vehicles": len(context), "context_overflow": max(0, len(context_pool)-len(context)),
             "geometric_support_not_occlusion": True, "vehicle_velocity_available": False,
             "objects": [{"query": i, "probability": float(probabilities[i]),
               "distance_m": float(distance[i]), "geometric_support": bool(geometric_support[i]),
               "selected": i in selected, "context": i in context,
               "reason": "selected" if i in selected else "duplicate" if i in duplicate else
               "below_existence_or_range" if not exists[i] else "low_geometric_support" if not geometric_support[i]
               else "capacity_or_outside_current_corridor"} for i in range(n)]}
    return selected, context, audit
