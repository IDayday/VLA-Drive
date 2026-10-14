"""Fixed teacher decoder diagnostic; not enabled by first-round registrations."""
import torch
from torch.nn import functional as F


def frozen_teacher_functional_loss(teacher, latent, anchor, truth, valid, scale=20.):
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Teacher must be frozen before functional supervision')
    if scale <= 0:
        raise ValueError('Invalid physical loss scale')
    # No no_grad: teacher parameters stay frozen, input latent retains its gradient.
    prediction = teacher.reconstruct(latent) * scale + anchor[:, None, :2]
    if prediction.shape != truth.shape or valid.shape != truth.shape[:-1]:
        raise ValueError('Functional trajectory shape')
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(truth[valid]).all():
        raise ValueError('Nonfinite valid functional target')
    p = torch.where(valid[..., None], prediction, 0.).float() / scale
    t = torch.where(valid[..., None], truth.detach(), 0.).float() / scale
    return (F.smooth_l1_loss(p, t, reduction='none') * valid[..., None]).sum() / max(2 * int(valid.sum()), 1)
