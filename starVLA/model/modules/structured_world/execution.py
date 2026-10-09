"""Execution changes only: same operators, objectives, precision and RNG streams."""


def configure_execution(model, mode):
    if mode not in ('reference', 'loss_preserving_v1'):
        raise ValueError('Unregistered execution mode')
    enabled = mode == 'loss_preserving_v1'
    language = model.qwen_vl_interface.model.model.language_model
    if enabled:
        language.gradient_checkpointing_disable()
    else:
        language.gradient_checkpointing_enable()
    model.action_model.model.gradient_checkpointing = not enabled
    model.defer_activation_metrics = enabled
    # Do not change geometry checkpointing: recomputation involves mutable BN
    # statistics, unlike the language/DiT LayerNorm blocks above.
    return {'mode': mode, 'Qwen_language_activation_checkpointing': not enabled,
            'DiT_activation_checkpointing': not enabled,
            'geometry_checkpointing_and_BN': 'unchanged',
            'deferred_activation_metrics': enabled, 'ordered_CPU_prefetch': enabled,
            'precision_loss_noise_and_proposal_steps': 'unchanged'}
