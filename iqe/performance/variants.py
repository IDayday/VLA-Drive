"""Execution changes apply only to a newly constructed, strictly loaded model."""
import torch
from ..contracts import require
from ..training.checkpoint import configure_language_checkpointing


def configure_model(model, options):
    visual=model.qwen_vl_interface.model.model.visual
    require(not visual.training and all(not p.requires_grad for p in visual.parameters()),'visual must be frozen eval including mergers')
    if options.visual_precision in {'bf16_compute','bf16_compute_fp32_mergers'}:
        original=visual.forward
        def bf16_visual(*args,**kwargs):
            require(not visual.training,'frozen visual mode changed')
            with torch.autocast('cuda',dtype=torch.bfloat16):return original(*args,**kwargs)
        visual.forward=bf16_visual
    if options.visual_precision=='bf16_compute_fp32_mergers':
        # Keep the four output projections at the original precision; the tower
        # remains BF16-compute. Distinct B1 policy, never relabel the failed B1 run.
        for merger in [visual.merger,*visual.deepstack_merger_list]:
            original_merger=merger.forward
            def fp32_merger(x, _forward=original_merger):
                with torch.autocast('cuda',enabled=False):return _forward(x.float())
            merger.forward=fp32_merger
    configure_language_checkpointing(model)
    language=model.qwen_vl_interface.model.model.language_model
    if options.checkpoint_policy=='none': language.gradient_checkpointing_disable()
    elif options.checkpoint_policy=='alternating':
        # Qwen3-VL uses GradientCheckpointingLayer flags; reject other implementations.
        require(all(hasattr(layer,'gradient_checkpointing') for layer in language.layers),'selective checkpoint layer flags unsupported by installed Qwen')
        for i,layer in enumerate(language.layers): layer.gradient_checkpointing = i % 2 == 0
    if options.query_training_semantics=='native_repeat1':
        from .native_query import install_native_query
        install_native_query(model)
    if options.execution_weights=='bf16_master':
        # Frozen vision stays exactly at the explicitly selected B1 precision.
        for name,p in model.named_parameters():
            if p.requires_grad:p.data=p.data.to(torch.bfloat16)
        # Source geometry/query buffers remain unchanged; autocast handles readouts.
    return model


def adam(model,cfg,contract,options):
    params=[p for p in model.parameters() if p.requires_grad]
    kwargs={'lr':cfg['lr'],'weight_decay':cfg['weight_decay'],'betas':tuple(contract['source_config']['trainer']['optimizer']['betas']),
            'eps':contract['source_config']['trainer']['optimizer'].get('eps',1e-8)}
    if options.optimizer=='fused':kwargs['fused']=True
    if options.execution_weights=='bf16_master':
        from .master_optimizer import MasterAdamW
        return MasterAdamW(list((n,p) for n,p in model.named_parameters() if p.requires_grad),**kwargs)
    return torch.optim.AdamW(params,**kwargs)


def load_adam(optimizer,state,options):
    optimizer.load_state_dict(state)
    # load_state_dict restores saved flags too: explicitly enact the declared implementation.
    for group in optimizer.param_groups:
        group['fused']=True if options.optimizer=='fused' else None
        group['foreach']=None
        group['capturable']=False
    if options.optimizer=='fused':
        for p,values in optimizer.state.items():
            if 'step' in values: values['step']=values['step'].to(p.device)
