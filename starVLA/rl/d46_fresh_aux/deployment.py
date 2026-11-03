"""Separate native training closure from policy-only deployment weights."""
import torch
from .protocol import PROTOCOL_SHA256

R_NAME='action_expert.decision.r_embeddings'


def policy_payload(adapters,scope):
    if scope.get('fresh_auxiliary_protocol_sha256')!=PROTOCOL_SHA256:
        raise ValueError('Not the declared fresh auxiliary training closure')
    if set(adapters)!=set(scope['trainable_names'])or set(adapters)!=set(scope['trainable_specs']):
        raise ValueError('Native payload omits/adds a training parameter')
    for name,x in adapters.items():
        expected=scope['trainable_specs'][name]
        if (not torch.is_tensor(x)or x.requires_grad or x.grad_fn is not None or x.dtype!=torch.float32
                or str(x.dtype)!=expected['dtype']or list(x.shape)!=expected['shape']or not torch.isfinite(x).all()):
            raise ValueError('Invalid detached native training tensor '+name)
    drop={name for name in adapters if name==R_NAME or name.startswith('action_expert.readouts.')}
    if drop!=set(scope['fresh_parameter_names'])or R_NAME not in drop:
        raise ValueError('Fresh-only tensors are not explicitly separated from deployment')
    policy={name.removeprefix('action_expert.'):x for name,x in adapters.items()if name not in drop}
    if not policy or any(not name.endswith(('.lora_A','.lora_B'))for name in policy):
        raise ValueError('D4 fresh deployment must retain only its text LoRA deltas')
    return policy,dict(schema='d46_fresh_auxiliary_deployment_split_v1',training_tensors=len(adapters),
        deployment_tensors=len(policy),excluded_training_only=sorted(drop),R_and_readouts_excluded=True)


def restore_policy(model,adapters,scope):
    """Caller strictly loads the IL46 policy and installs the declared text LoRA."""
    policy,receipt=policy_payload(adapters,scope)
    if hasattr(model,'readouts')or model.decision.r_embeddings is not None:
        raise ValueError('Deployment must already be pruned; never load fresh R into it')
    params=dict(model.named_parameters());expected={n for n in params if n.endswith(('.lora_A','.lora_B'))}
    if set(policy)!=expected:raise ValueError('Deployment LoRA closure changed')
    with torch.no_grad():
        for name,x in policy.items():
            if params[name].shape!=x.shape or params[name].dtype!=x.dtype:raise ValueError('Deployment tensor shape/dtype changed')
            params[name].copy_(x)
    return receipt
