"""High-quality replay preserves the original velocity on shared noise/time."""
from .effect_distillation import velocity_loss


def preservation_loss(model, condition, pair, valid_high_quality, **reduction):
    pair.validate()
    state = pair.state('minus')
    base = model.base_velocity(condition, state, pair.times)
    adapted = model.velocity(condition, state, pair.times)
    return velocity_loss(adapted, base, valid_high_quality, **reduction)
