"""Execution-only tradeoff: keep long Qwen checkpointing, retain short graphs.

No tensor, precision, objective, mask or candidate selection is changed. Flags
are ordinary execution attributes and do not introduce checkpoint parameters.
"""
def short_stream_checkpointing(model,enabled):
    architecture=model.architecture_id;decision=model.decision
    if architecture in('A','C'):
        decision.actor.activation_checkpointing=bool(enabled)
        decision.world_stream.activation_checkpointing=bool(enabled)
    elif architecture=='B':decision.activation_checkpointing=bool(enabled)
    elif architecture=='D':decision.checkpoint_short_suffix=bool(enabled)
    else:raise ValueError('Unknown architecture')
    return dict(architecture=architecture,short_stream_checkpointing=bool(enabled),
                long_Qwen_checkpointing_unchanged=True,parameters_and_precision_unchanged=True)


def checkpoint_flags(model):
    """Only execution booleans; neither a weight nor an optimizer setting."""
    names=('activation_checkpointing','checkpoint_short_suffix','gradient_checkpointing')
    return {(name,attr):getattr(module,attr)for name,module in model.named_modules()
            for attr in names if isinstance(getattr(module,attr,None),bool)}


def apply_checkpoint_policy(model,policy,source_flags):
    if policy not in('source','short_off','all_off'):raise ValueError('Unknown graph-retention policy')
    modules=dict(model.named_modules())
    for (name,attr),value in source_flags.items():setattr(modules[name],attr,value)
    if policy=='short_off':short_stream_checkpointing(model,False)
    elif policy=='all_off':
        for name,attr in source_flags:setattr(modules[name],attr,False)
    return dict(policy=policy,flags={name+':'+attr:getattr(modules[name],attr)for name,attr in source_flags},
        parameters_and_precision_unchanged=True)
