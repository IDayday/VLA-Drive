"""Calibrated Base fallback. Tie-breaking uses stable registration identity."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from .contracts import require


@dataclass(frozen=True)
class SelectionRule:
    delta: float
    eta: dict[str, float]
    epsilon: dict[str, float]
    always_base: bool = False
    calibration_role: str = "selector_cal"
    dependency_hash: str = ""

    def __post_init__(self):
        require(self.calibration_role == "selector_cal", "selection thresholds need selector_cal")
        require(0 <= self.delta <= 1 and set(self.eta) == set(self.epsilon), "selection threshold schema")
        require(all(0 <= x <= 1 for x in (*self.eta.values(), *self.epsilon.values())), "safety threshold range")

    def __call__(self, candidates, prediction):
        ids = candidates.expert_ids
        require(ids[0] == "expert_0" and len(set(ids)) == len(ids), "Base identity/duplicate IDs")
        require(bool(candidates.valid[:, 0].all()), "Base invalid: no fabricated fallback")
        if self.always_base:
            rows = torch.zeros(len(candidates.trajectories), dtype=torch.long, device=candidates.trajectories.device)
            return candidates.trajectories[:, 0], ["expert_0"] * len(rows), ["always_base"] * len(rows)
        require(prediction.values.shape == candidates.valid.shape, "prediction/candidate mismatch")
        require(bool(prediction.valid[:, 0].all()), "Base score prediction invalid")
        scores = prediction.values
        allowed = candidates.valid & prediction.valid & (scores - scores[:, :1] > self.delta)
        for c, eta in self.eta.items():
            require(c in prediction.components, f"missing protected head: {c}")
            p = prediction.components[c]
            allowed &= torch.isfinite(p) & (p >= eta) & (p >= p[:, :1] - self.epsilon[c])
        allowed[:, 0] = True
        masked = scores.masked_fill(~allowed, -torch.inf)
        # stable numeric expert IDs are validated by the registry; current tensor order is irrelevant.
        order = sorted(range(len(ids)), key=lambda j: (ids[j] != "expert_0", int(ids[j].split("_")[-1])))
        ordered = masked[:, order]
        selected = torch.tensor(order, device=scores.device)[ordered.argmax(-1)]
        rows = torch.arange(len(selected), device=scores.device)
        return candidates.trajectories[rows, selected], [ids[j] for j in selected.cpu().tolist()], \
            ["base_fallback" if j == 0 else "predicted_gain_and_safety_guards" for j in selected.cpu().tolist()]


@dataclass(frozen=True)
class RouterRule:
    logit_margin: float = 0.0
    probability_margin: float = 0.0
    always_base: bool = False
    dependency_hash: str = ""
    calibration_role: str = "selector_cal"

    def __post_init__(self):
        require(self.logit_margin >= 0 and 0 <= self.probability_margin <= 1 and self.calibration_role == "selector_cal", "invalid Router calibration")

    def __call__(self, logits, expert_ids):
        require(len(expert_ids) == logits.shape[-1] and expert_ids[0] == "expert_0", "Router registry mismatch")
        require(bool(torch.isfinite(logits).all()), "invalid Router logits")
        order = sorted(range(len(expert_ids)), key=lambda j: int(expert_ids[j].split("_")[-1]))
        idx = torch.tensor(order, device=logits.device)[logits[:, order].argmax(-1)]
        row = torch.arange(len(idx), device=idx.device)
        prob = logits.softmax(-1)
        use = (logits[row, idx] - logits[:, 0] > self.logit_margin) & \
              (prob[row, idx] - prob[:, 0] > self.probability_margin)
        return torch.where(use & (not self.always_base), idx, torch.zeros_like(idx))
