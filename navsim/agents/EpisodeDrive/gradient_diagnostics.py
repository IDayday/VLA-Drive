"""Isolated same-batch gradients: never touch optimizer .grad or DDP hooks."""
from contextlib import contextmanager
import random
import numpy as np
import torch
from torch import nn
from torch.func import functional_call


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def compare_loss_gradients(plan_loss, wm_loss, groups, weight):
    """Both losses must be from this single forward, before any clipping."""
    all_tensors, spans = [], {}
    for name, tensors in groups.items():
        tensors = [value for value in tensors if value.requires_grad]
        spans[name] = (len(all_tensors), len(all_tensors) + len(tensors))
        all_tensors.extend(tensors)
    plan = torch.autograd.grad(plan_loss, all_tensors, retain_graph=True, allow_unused=True)
    wm = torch.autograd.grad(wm_loss, all_tensors, retain_graph=True, allow_unused=True)
    results = {}
    with torch.no_grad():
        for name, (start, end) in spans.items():
            p2 = plan_loss.new_zeros((), dtype=torch.float32)
            w2, dot = p2.clone(), p2.clone()
            for pg, wg in zip(plan[start:end], wm[start:end]):
                if pg is not None:
                    p2 += pg.detach().float().square().sum()
                if wg is not None:
                    w2 += wg.detach().float().square().sum()
                if pg is not None and wg is not None:
                    dot += (pg.detach().float() * wg.detach().float()).sum()
            pn, wn = p2.sqrt(), w2.sqrt()
            results[name] = {'plan_norm': float(pn), 'unweighted_wm_norm': float(wn),
                             'weighted_wm_norm': float(abs(weight) * wn),
                             'weighted_wm_to_plan_ratio': float(abs(weight)*wn/pn.clamp_min(1e-30)),
                             'cosine': float(dot/(pn*wn).clamp_min(1e-30)),
                             'cosine_defined': bool(pn > 0 and wn > 0)}
    return results


class _LossForward(nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def forward(self, features, targets):
        predictions = self.agent(features)
        losses = self.agent.compute_loss(features, targets, predictions)
        # Backward must run while functional replacements are still installed:
        # checkpoint recomputation otherwise observes the original DDP leaves.
        clones = {name: p for name, p in self.agent.named_parameters() if p.requires_grad}
        groups = {
            'vision_qv_lora': [p for name, p in clones.items() if '.q_lora_' in name or '.v_lora_' in name],
            'planning_registers': [p for name, p in clones.items() if name.endswith('.planning_registers')],
            'planning_readout': [p for name, p in clones.items() if '.planning_register_adapter.' in name and not name.endswith('.planning_registers')],
            'readout_output': [predictions['planning_registers']],
        }
        weight = float(losses['wm_weight_current'].detach())
        results = compare_loss_gradients(losses['plan_loss'], losses['wm_loss'], groups, weight)
        return {'scope': 'same_batch_rank_local_unclipped', 'wm_weight': weight, 'groups': results}


def isolated_same_batch_audit(agent, features, targets):
    """Functional cloned leaves have no DDP reducer/optimizer hooks attached.

    This intentionally pays one additional forward/backward at low frequency.
    The result is rank-local; all ranks can audit independently with no extra
    collectives, and a report must retain that scope when comparing norms.
    """
    if not agent.training or not agent.world_model_enabled:
        raise ValueError('Same-batch WM audit requires a training agent with WM enabled')
    clones = {'agent.' + name: p.detach().clone().requires_grad_(True)
              for name, p in agent.named_parameters() if p.requires_grad}
    wrapper = _LossForward(agent)
    # InternViT's remote encoder uses reentrant checkpointing, incompatible
    # with autograd.grad. Only during this diagnostic use non-reentrant block
    # checkpointing, preserving recomputation and the same block function.
    # Do not globally monkey-patch torch or materialize B*tiles full attention.
    checkpoint_modules = [(m, m.gradient_checkpointing) for m in agent.modules()
                          if isinstance(getattr(m, 'gradient_checkpointing', None), bool)]
    forwards = []
    try:
        for module, _ in checkpoint_modules:
            module.gradient_checkpointing = False
        vision = getattr(getattr(getattr(agent, 'backbone', None), 'model', None), 'vision_model', None)
        encoder = getattr(vision, 'encoder', None)
        if encoder is not None and any(m is encoder and flag for m, flag in checkpoint_modules):
            from torch.utils.checkpoint import checkpoint
            for layer in encoder.layers:
                original = layer.forward
                forwards.append((layer, original))
                def recomputed(*args, _forward=original, **kwargs):
                    return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                layer.forward = recomputed
        with preserve_rng():
            return functional_call(wrapper, clones, (features, targets), strict=False)
    finally:
        for module, forward in forwards:
            module.forward = forward
        for module, flag in checkpoint_modules:
            module.gradient_checkpointing = flag
