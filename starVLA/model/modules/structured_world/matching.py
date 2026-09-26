import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def match_current(prediction, target, centre_scale=20.):
    """A single current-time assignment; its target indices also select future tracks."""
    valid = target.current_supervision_mask.bool()
    indices = torch.where(valid)[0]
    if not bool(target.annotation_valid_mask) or len(indices) == 0:
        empty = torch.empty(0,device=prediction['boxes'].device,dtype=torch.long)
        return empty, empty
    boxes = target.current_boxes[indices]
    pred = prediction['boxes']
    mask = target.box_valid_mask[indices]
    diff = (pred[:,None] - boxes[None]).abs()
    scale = pred.new_tensor([centre_scale]*3 + [5.]*3 + [1.]*2)
    cost = (diff / scale * mask[None]).sum(-1)
    cost -= prediction['logits'].softmax(-1)[:,target.current_classes[indices]]
    if not torch.isfinite(cost).all():
        raise ValueError('Nonfinite Hungarian matching cost')
    row, col = linear_sum_assignment(cost.float().cpu().numpy())
    return torch.as_tensor(row,device=pred.device),indices[torch.as_tensor(col,device=indices.device)]
