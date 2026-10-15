"""Natural-scene cost CE: retain absolute error cost across scenes."""
import torch
from iqe.contracts import require
from iqe.losses import LossTerm


def cost_weights(scores, valid):
    r, v = scores.detach().float(), valid.bool()
    require(r.ndim == 2 and r.shape == v.shape and v.any(-1).all().item(), 'valid quality vector per scene')
    require(torch.isfinite(r[v]).all().item() and ((r[v] >= 0) & (r[v] <= 1)).all().item(), 'official score scale [0,1]')
    minimum = r.masked_fill(~v, torch.inf).amin(-1, keepdim=True)
    return torch.where(v, r-minimum, torch.zeros_like(r))


def fixed_normalizer(scores, valid):
    w = cost_weights(scores, valid)
    return float(w.sum(-1).double().mean())


def cost_ce(logits, scores, valid, z_train, scene_valid=None):
    require(float(z_train) > 0, 'positive fit-only fixed Z_train required')
    v = valid.bool()
    require(logits.shape == scores.shape == v.shape, 'decision quality/mask shape')
    require(torch.isfinite(logits[v]).all().item(), 'nonfinite decision logits')
    w = cost_weights(scores, v)
    # Invalid logits become finite zero before multiplication; never 0 * -inf.
    logp = torch.log_softmax(logits.float().masked_fill(~v, -torch.inf), -1)
    safe_logp = torch.where(v, logp, torch.zeros_like(logp))
    per_scene = -(w*safe_logp).sum(-1)/float(z_train)
    mask = torch.ones_like(per_scene, dtype=torch.bool) if scene_valid is None else scene_valid.bool()
    require(mask.shape == per_scene.shape, 'actual scene-count denominator')
    return LossTerm((per_scene*mask).sum(), mask.sum().float())


def denominators(batch):
    return {'decision': batch['scene_valid'].sum().float()}
