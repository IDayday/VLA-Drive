"""Execution changes only: same operators, objectives, precision and RNG streams."""


def configure_execution(model, mode):
    if mode not in ('reference', 'io_preserving_v1', 'loss_preserving_v1', 'engineering_fast', 'precision_fast', 'engineering_timevec', 'engineering_host_fast'):
        raise ValueError('Unregistered execution mode')
    enabled = mode != 'reference'
    checkpointed = mode != 'loss_preserving_v1'
    language = model.qwen_vl_interface.model.model.language_model
    if not checkpointed:
        language.gradient_checkpointing_disable()
    else:
        language.gradient_checkpointing_enable()
    model.action_model.model.gradient_checkpointing = checkpointed
    model.defer_activation_metrics = enabled
    fast = mode in ('engineering_fast', 'precision_fast', 'engineering_timevec')
    prepared = fast or mode == 'engineering_host_fast'
    # Input preparation/logging is independent of mathematical reuse. The
    # host-only pilot must retain the reference future/query/loss operators.
    model.execution_prepare_current = prepared
    model.execution_fast = fast
    model.future_space.execution_fast = fast
    model.geometry.conv_autocast_bf16 = mode == 'precision_fast'
    model.future_space.future_time_vectorized = mode == 'engineering_timevec'
    model.refiner.fgtr_time_vectorized = mode == 'engineering_timevec'
    from .qwen_execution import install_qwen_vision_lengths, uninstall_qwen_vision_lengths
    vision = model.qwen_vl_interface.model.model.visual
    qwen_metadata = install_qwen_vision_lengths(vision) if prepared else uninstall_qwen_vision_lengths(vision)
    # Do not change geometry checkpointing: recomputation involves mutable BN
    # statistics, unlike the language/DiT LayerNorm blocks above.
    return {'mode': mode, 'Qwen_language_activation_checkpointing': checkpointed,
            'DiT_activation_checkpointing': checkpointed,
            'geometry_checkpointing_and_BN': 'unchanged',
            'deferred_activation_metrics': enabled, 'ordered_CPU_prefetch': enabled,
            'precision_loss_noise_and_proposal_steps': (
                'Original FM/proposal/loss unchanged; BF16 geometry convolution activations only'
                if mode == 'precision_fast' else 'unchanged'),
            'future_shared_current_spatial_mix_and_action_MLN': fast,
            'typed_current_CPU_processor_and_pinned_batch_transfer': prepared,
            'query_and_body_execution_reuse': fast,
            'Qwen_vision_metadata_adapter': qwen_metadata,
            'formal_acceptance_required': prepared,
            'numerical_level': 'E3' if mode == 'precision_fast' else 'E2',
            'geometry_convolution_BF16_autocast': mode == 'precision_fast',
            'DeepSpeed_parameter_and_optimizer_precision': 'FP32',
            'DeepSpeed_gradient_communication_precision': 'FP32',
            'TF32': False,
            'canonical_inference_precision': 'FP32',
            'time_vectorization': mode == 'engineering_timevec',
            'projection_depthnet_grid_loss_master_communication': 'FP32 unchanged'}
