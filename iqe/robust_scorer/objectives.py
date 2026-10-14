"""Natural-distribution expected reward; unchanged global gap comparison stream."""
import torch
import torch.nn.functional as F
from iqe.contracts import require
from iqe.losses import LossTerm
from iqe.specialist_study.objectives import masks


def denominators(batch, objective):
    v, c, r = masks(batch)
    result = {"value": v.any(-1).sum().float(),
              "components": c.flatten(1).any(-1).sum().float(), "ranking": r.sum().float()}
    if objective == "zero_hurdle":
        result["nonzero"] = result["value"].clone()
    return result


def terms(pred, batch, objective):
    v, c, r = masks(batch)
    legal = batch["valid"].bool()
    for key in ("values", "value_logits", "component_logits"):
        require(torch.isfinite(pred[key][legal]).all().item(), "nonfinite legal Scorer prediction")
    truth = torch.nan_to_num(batch["scores"].detach().float())
    component_truth = torch.nan_to_num(batch["components"].detach().float())
    require(((truth[v] >= 0)&(truth[v] <= 1)).all().item(), "reward outside [0,1]")
    require(((component_truth[c] >= 0)&(component_truth[c] <= 1)).all().item(), "component outside [0,1]")
    logits = torch.where(legal, pred["value_logits"].float(), 0.)
    cp = torch.where(legal[..., None], pred["component_logits"].float(), 0.)
    value = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    components = F.binary_cross_entropy_with_logits(cp, component_truth, reduction="none")
    # Per natural scene, each legal candidate has mass 1/K. Comparison samples
    # are never used for pointwise value or component regression.
    count = v.sum(-1).clamp_min(1)
    if objective == "zero_hurdle":
        require(torch.isfinite(pred["nonzero_logits"][legal]).all().item(), "nonfinite zero-risk prediction")
        nonzero = truth > 1e-8
        gate = torch.where(legal, pred["nonzero_logits"].float(), 0.)
        gate_loss = F.binary_cross_entropy_with_logits(gate, nonzero.float(), reduction="none")
        # E[1(R>0) * BCE(conditional_mean,R)] under the SAME natural measure.
        # Dividing by the number of positive candidates inside each scene would
        # silently give rare-positive scenes a different weight. Do not do that.
        value = value*nonzero
    elif objective != "mean_bce":
        raise ValueError("unsupported reward estimand")
    p = torch.where(legal, pred["values"].float(), 0.)
    pair = F.softplus(-(truth[:, 0]-truth[:, 1]).sign()*(p[:, 0]-p[:, 1])/.1) if p.shape[1] > 1 else p[:, 0]*0
    result = {
        "value": LossTerm(((value*v).sum(-1)/count).sum(), v.any(-1).sum().float()),
        "components": LossTerm(((components*c).flatten(1).sum(-1)/c.flatten(1).sum(-1).clamp_min(1)).sum(),
                               c.flatten(1).any(-1).sum().float()),
        "ranking": LossTerm((pair*r).sum(), r.sum().float()),
    }
    if objective == "zero_hurdle":
        result["nonzero"] = LossTerm(((gate_loss*v).sum(-1)/count).sum(), v.any(-1).sum().float())
    return result
