"""Update-counted LR and fixed-SFT KL feedback; no changes to behavior data.

The clipped proportional KL feedback follows TRL v0.11.4 AdaptiveKLController
(trainer/utils.py, citing Ziegler et al. 2019). Our measured quantity is the
dimension-mean conditional Gaussian transition KL, NOT a text sequence KL.
"""
import math


def validate_schedule(spec):
    if spec is None:
        return
    if spec.get("type") != "warmup_cosine":
        raise ValueError("unsupported learning-rate schedule")
    total, warmup = spec.get("total_updates"), spec.get("warmup_updates")
    if (type(total) is not int or type(warmup) is not int
            or not 1 <= warmup < total or not 0 <= spec.get("min_lr_ratio", -1) < 1):
        raise ValueError("invalid fixed optimizer-update schedule")


def lr_multiplier(completed_updates, spec=None):
    """LR for the NEXT update; update 1 is peak/warmup, not a zero-LR Adam step.

    Update warmup reaches the peak; update total reaches the specified floor.
    The horizon is immutable config, independent of a pilot's --max-updates.
    """
    validate_schedule(spec)
    if spec is None:
        return 1.0
    upcoming = completed_updates + 1
    if upcoming <= spec["warmup_updates"]:
        return upcoming / spec["warmup_updates"]
    phase = min(1., (upcoming - spec["warmup_updates"]) /
                (spec["total_updates"] - spec["warmup_updates"]))
    floor = spec["min_lr_ratio"]
    return floor + (1. - floor) * .5 * (1. + math.cos(math.pi * phase))


class ReferenceKLController:
    """Checkpointed scalar updated once from globally scene-weighted real KL.

    No gradients through feedback, no reference synchronization, no changing
    old probabilities/advantages. Bounds keep retention active even at KL=0.
    """
    def __init__(self, initial, spec):
        self.spec = dict(spec)
        required = {"target", "horizon_scenes", "min_coefficient", "max_coefficient"}
        if set(spec) != required or not all(math.isfinite(v) and v > 0 for v in spec.values()):
            raise ValueError("invalid adaptive reference KL settings")
        if not spec["min_coefficient"] <= initial <= spec["max_coefficient"]:
            raise ValueError("initial reference coefficient outside bounds")
        self.value = float(initial)
        self.updates = 0

    def advance(self, update, measured_kl, scenes):
        if update != self.updates + 1:
            raise ValueError("KL controller must advance once per optimizer update")
        if (not math.isfinite(measured_kl) or measured_kl < 0
                or not 0 < scenes <= self.spec["horizon_scenes"]):
            raise ValueError("invalid global KL observation/scope")
        error = min(.2, max(-.2, measured_kl / self.spec["target"] - 1.))
        value = self.value * (1. + error * scenes / self.spec["horizon_scenes"])
        self.value = min(self.spec["max_coefficient"], max(self.spec["min_coefficient"], value))
        self.updates = update
        return self.value

    def state_dict(self):
        return {"schema": 1, "spec": self.spec, "value": self.value, "updates": self.updates}

    def load_state_dict(self, state):
        if state.get("schema") != 1 or state.get("spec") != self.spec:
            raise ValueError("exact resume KL controller configuration changed")
        value, updates = state["value"], state["updates"]
        if (not math.isfinite(value) or not self.spec["min_coefficient"] <= value <= self.spec["max_coefficient"]
                or type(updates) is not int or updates < 0):
            raise ValueError("corrupt KL controller state")
        self.value, self.updates = value, updates
