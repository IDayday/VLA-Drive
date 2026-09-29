"""Sparse, read-only ZeRO-aware gradient/master-update observations."""
import torch


def parameters(model):
    selected={'action_decoder':model.action_model.action_decoder.layer2.weight}
    if hasattr(model,'foresight_queries'):selected['W']=model.foresight_queries
    layers=model.qwen_vl_interface.model.model.language_model.layers
    selected['Qwen_first_q']=layers[0].self_attn.q_proj.weight
    selected['Qwen_last_q']=layers[-1].self_attn.q_proj.weight
    return selected


def before_step(model):
    from deepspeed.utils import safe_get_full_grad,safe_get_full_fp32_param
    rows={};snapshots={}
    for name,p in parameters(model).items():
        grad=safe_get_full_grad(p);master=safe_get_full_fp32_param(p)
        if grad is None or master is None:raise RuntimeError('Missing declared trainable gradient/master: '+name)
        if not torch.isfinite(grad).all() or not torch.isfinite(master).all():raise FloatingPointError('Invalid observed parameter: '+name)
        rows[name]={'gradient_norm':float(grad.float().norm()),'gradient_nonzero':int(torch.count_nonzero(grad)),
                    'gradient_dtype':str(grad.dtype),'elements':p.numel()}
        snapshots[name]=master.detach().clone()
    return rows,snapshots


def after_step(model,observation):
    from deepspeed.utils import safe_get_full_fp32_param
    rows,snapshots=observation
    for name,p in parameters(model).items():
        difference=safe_get_full_fp32_param(p).detach()-snapshots[name]
        if not torch.isfinite(difference).all():raise FloatingPointError('Invalid observed update: '+name)
        rows[name].update(update_norm=float(difference.norm()),changed_elements=int(torch.count_nonzero(difference)),
                          max_abs_update=float(difference.abs().max()))
    return rows
