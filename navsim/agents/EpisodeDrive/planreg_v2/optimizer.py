import math
import torch

PEAK_LRS = dict(planning_adapter=2e-4,fusion=2e-4,action_head=2e-4,scorer=2e-4,
                future_predictor=1e-4,semantic_qformer=1e-4,vision_qv_lora=3e-5,language_lora=1e-5)
CAPS = dict(planning_adapter=3e-4,fusion=3e-4,action_head=3e-4,scorer=3e-4,
            future_predictor=3e-4,semantic_qformer=1.5e-4,vision_qv_lora=5e-5,language_lora=1e-5)


def logical_group(name):
    if name.startswith('backbone.model.vision_model.') and '_lora_' in name: return 'vision_qv_lora'
    if name.startswith('backbone.model.language_model.') and '.lora_' in name: return 'language_lora'
    if name.startswith('backbone.planning_register_adapter.'): return 'planning_adapter'
    if name.startswith('backbone.task_queries.'): return 'semantic_qformer'
    if name.startswith('scene_memory.'): return 'fusion'
    if name.startswith('wm_predictor.'): return 'future_predictor'
    if name.startswith(('action_head.scorer.','action_head.scorer_attention.','action_head.pos_embed.')): return 'scorer'
    if name.startswith(('action_head.hist_encoding.','action_head.init_feature.','action_head.attention.','action_head.trajectory_head.')): return 'action_head'
    raise ValueError('Unclassified V2 trainable parameter: '+name)


def multiplier(step,total,warmup_ratio=.05):
    warmup = max(1,round(total*warmup_ratio))
    if step <= warmup: return .01+.99*step/warmup
    progress = min(1.,(step-warmup)/max(1,total-1-warmup))
    return .1+.9*.5*(1+math.cos(math.pi*progress))


def build_optimizer(agent,total_steps,learning_rates=None):
    rates = dict(PEAK_LRS,**(learning_rates or {}))
    groups = {}
    no_decay_ids = set()
    for module in agent.modules():
        if isinstance(module,(torch.nn.LayerNorm,torch.nn.Embedding)) or 'RMSNorm' in type(module).__name__:
            no_decay_ids.update(id(p) for p in module.parameters(recurse=False))
    seen = set()
    for name,p in agent.named_parameters():
        if not p.requires_grad: continue
        if p.dtype != torch.float32 or id(p) in seen:
            raise ValueError('Trainable must occur once and be FP32: '+name)
        seen.add(id(p))
        logical = logical_group(name)
        leaf = name.rsplit('.',1)[-1]
        no_decay = id(p) in no_decay_ids or p.ndim < 2 or 'lora' in name.lower() or leaf in ('queries','slot_queries','planning_registers','semantic_gate','tile_gate')
        key = logical+('/no_decay' if no_decay else '/decay')
        if rates[logical] > CAPS[logical]: raise ValueError('LR exceeds declared cap: '+logical)
        group = groups.setdefault(key,dict(params=[],lr=rates[logical],weight_decay=0. if no_decay else .01,
                                          name=key,logical_name=logical))
        group['params'].append(p)
    summary = [dict(name=n,parameters=sum(p.numel() for p in g['params']),lr=g['lr'],weight_decay=g['weight_decay']) for n,g in sorted(groups.items())]
    print('V2_OPTIMIZER_GROUPS',summary)
    optimizer = torch.optim.AdamW([g for _,g in sorted(groups.items())],betas=(.9,.999),eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:multiplier(step,total_steps))
    return optimizer,scheduler,summary


def audit_adam_state(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.dtype != torch.float32: raise AssertionError('Trainable storage is not FP32')
            for name in ('exp_avg','exp_avg_sq'):
                if name in optimizer.state[p] and optimizer.state[p][name].dtype != torch.float32:
                    raise AssertionError('AdamW '+name+' is not FP32')
