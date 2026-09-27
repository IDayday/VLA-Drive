"""Training labels only: current matching determines future track identity once."""
import torch
from starVLA.model.modules.structured_world.rehab import match_reference


@torch.no_grad()
def joint_targets(prediction, world_targets, ego_xy):
    b, k, t, _ = prediction['future_xy'].shape
    if ego_xy.shape != (b, t, 2) or len(world_targets) != b:
        raise ValueError('Ego/agent target horizons must agree')
    xy = ego_xy.new_zeros(b, k + 1, t, 2)
    valid = torch.zeros(b, k + 1, t, device=ego_xy.device, dtype=torch.bool)
    xy[:, 0] = ego_xy
    valid[:, 0] = torch.isfinite(ego_xy).all(-1)
    assignments = []
    for batch, target in enumerate(world_targets):
        pred = {key: value[batch] for key, value in prediction.items()}
        rows, cols = match_reference(pred, target)
        xy[batch, rows + 1] = target.future_xy_in_ego_t0[cols]
        valid[batch, rows + 1] = target.future_valid_mask[cols]
        assignments.append((rows, cols))
    xy = xy.masked_fill(~valid[..., None], 0.)
    return xy, valid, assignments
