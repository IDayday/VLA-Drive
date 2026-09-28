"""Independent resumable role scheduler; auxiliary never replaces main FM."""
from collections import Counter
import torch


class VehicleRoleScheduler:
    def __init__(self, seed):
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.cursor = 0
        self.counts = Counter()

    def known_mask(self, active, feature_valid):
        if active.dtype != torch.bool or feature_valid.dtype != torch.bool:
            raise ValueError("Role eligibility masks must be boolean")
        if feature_valid.shape[:2] != active.shape or feature_valid.shape[-1] != 4:
            raise ValueError("Role shape mismatch")
        known = torch.zeros_like(feature_valid)
        tasks = []
        for b in range(len(active)):
            nominal_neighbor = bool(self.cursor % 2)
            self.cursor += 1
            self.counts["nominal_neighbor" if nominal_neighbor else "nominal_ego"] += 1
            # A complete xy point needs BOTH coordinates; partial horizons allowed.
            eligible = torch.where(active[b, 1:] & feature_valid[b, 1:, :, :2].all(-1).any(-1))[0]+1
            if not len(eligible):
                tasks.append({"role": "all_hidden_fallback", "target": None})
                self.counts["fallback"] += 1
                continue
            target = int(eligible[torch.randint(len(eligible), (1,), generator=self.generator).item()]) if nominal_neighbor else 0
            known[b] = feature_valid[b] & active[b, :, None, None]
            known[b, target] = False
            role = "neighbor" if target else "ego"
            tasks.append({"role": role, "target": target})
            self.counts["actual_"+role] += 1
            self.counts[role+"_hidden_xy_coordinates"] += int(feature_valid[b, target, :, :2].sum())
        return known, tasks

    def state_dict(self):
        return {"cursor": self.cursor, "rng": self.generator.get_state(), "counts": dict(self.counts)}

    def load_state_dict(self, state):
        self.cursor = int(state["cursor"])
        self.generator.set_state(state["rng"])
        self.counts = Counter(state["counts"])


def auxiliary_due(completed_updates, all_hidden_start, cadence=4):
    if cadence != 4 or completed_updates < 0 or all_hidden_start < 0:
        raise ValueError("This campaign fixes auxiliary cadence at four updates")
    return completed_updates < all_hidden_start and (completed_updates+1) % cadence == 0
