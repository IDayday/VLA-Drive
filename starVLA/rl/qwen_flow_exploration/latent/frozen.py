"""Qualification helpers for frozen generation and wrapper equivalence."""
import torch
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.model.modules.qwen_sourceflow.cached_action_runner import expand_condition


def freeze_generator(model):
    model.requires_grad_(False).eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Generator closure did not freeze')
    return model


def frozen_digest(model):
    """Parameters AND buffers; use around finite qualification, not each update."""
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Root-policy generator must be permanently frozen/eval')
    # state_dict alone omits nonpersistent buffers (e.g. rotary frequencies).
    return tensor_digest(dict(parameters=dict(model.named_parameters()), buffers=dict(model.named_buffers())))


@torch.no_grad()
def decode_roots(model, condition, roots):
    if roots.ndim != 4 or roots.shape[2:] != (8, 3):
        raise ValueError('Root-policy samples must be [scene,candidate,8,3]')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('The root-KL interpretation requires a frozen generator')
    b, g = roots.shape[:2]
    # Same native source ODE; no new latent scale, clamp, or action postprocess.
    output = model.policy.sample_ode(expand_condition(condition, g), roots.reshape(b*g, 8, 3))
    return output.reshape(b, g, 8, 3)
