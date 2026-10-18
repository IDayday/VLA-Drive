"""Registered heterogeneous IL parameter groups and the bounded interface stage."""
import math
import torch


def interface_parameter(name):
    return (name.startswith('provider.') or
            name.startswith('policy.runner.mixtures.action.action_encoder.') or
            name.startswith('policy.runner.mixtures.action.head.'))


def set_stage(model, update, cfg):
    full = update >= cfg['interface_updates']
    for name, param in model.named_parameters():
        param.requires_grad_(full or interface_parameter(name))
    model.train()
    if not full:
        model.qwen.eval()
    return 'full' if full else 'interface'


def parameter_groups(model, cfg):
    groups = {key: dict(params=[], lr=value, name=key) for key,value in cfg['learning_rates'].items()}
    for name,param in model.named_parameters():
        if name.startswith('qwen.model.visual.'):
            group = 'qwen_language_merger' if 'merger' in name else 'qwen_vision'
        elif name.startswith('qwen.'):
            group = 'qwen_language_merger'
        else:
            group = 'action_interfaces'
        groups[group]['params'].append(param)
    return list(groups.values())


def lr_scale(step, cfg):
    warm=max(1,math.ceil(cfg['max_updates']*cfg['warmup_fraction']))
    if step<warm:return (step+1)/warm
    progress=min(1.,(step-warm)/max(1,cfg['max_updates']-warm))
    return .5*(1+math.cos(math.pi*progress))


def validate_optimizer(model, optimizer, scheduler, state, cfg):
    updates=state['updates']
    if state['phase']!='READY' or state['cursor']!=updates*cfg['global_batch']:
        raise ValueError('Uncommitted IL exposure boundary')
    if scheduler.last_epoch != updates:
        raise ValueError('IL scheduler differs from optimizer progress')
    for name,param in model.named_parameters():
        expected=max(0,updates-(0 if interface_parameter(name) else cfg['interface_updates']))
        value=optimizer.state.get(param)
        if expected==0:
            if value and int(value['step'])!=0:raise ValueError('Frozen parameter received Adam update: '+name)
        elif value is None or int(value['step'])!=expected:
            raise ValueError('Adam progress changed for '+name)
