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
