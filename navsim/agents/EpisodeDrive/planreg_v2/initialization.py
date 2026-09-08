"""Versioned named initialization bank; shared values survive control topology changes."""
import hashlib
import torch
from . import ARCHITECTURE_VERSION,SHARED_INIT_SCHEMA,RECIPE_VERSION


def initialization_identity(config):
    return dict(architecture=ARCHITECTURE_VERSION,register_init_std=config.get('register_init_std',.02),
        semantic_query_init_std=.02,recipe_version=RECIPE_VERSION,
        predictor_layers=config.get('predictor_layers',2),visual_rank=32,language_rank=32)


def tensor_manifest(state):
    return {n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(
        v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for n,v in sorted(state.items())}


def shared_artifact(agent,seed):
    state=agent.trainable_state()
    return dict(schema=SHARED_INIT_SCHEMA,identity=initialization_identity(agent.config),
                tensors=tensor_manifest(state),trainable_state=state,seed=seed,config=agent.config)


def load_shared_bank(agent,artifact):
    if artifact.get('schema')!=SHARED_INIT_SCHEMA or artifact.get('identity')!=initialization_identity(agent.config):
        raise ValueError('Stale shared initialization: require V2.2 FP32/std0.02 named parameter bank, not old 1e-6 tensors')
    state=artifact['trainable_state']
    if tensor_manifest(state)!=artifact['tensors']: raise ValueError('Shared parameter bank fingerprint mismatch')
    target={n:p for n,p in agent.named_parameters() if p.requires_grad}
    missing=set(target)-set(state)
    extra=set(state)-set(target)
    allowed_extra={n for n in extra if n.startswith('wm_predictor.') and not agent.world_model_enabled}
    if missing or extra!=allowed_extra: raise ValueError('Shared init topology mismatch: '+str(missing|extra-allowed_extra))
    with torch.no_grad():
        for n,p in target.items():
            if p.shape!=state[n].shape or p.dtype!=torch.float32 or state[n].dtype!=torch.float32:
                raise ValueError('Shared initialization shape/dtype mismatch: '+n)
            p.copy_(state[n])
    return dict(loaded=len(target),filtered_training_only=len(extra),identity=artifact['identity'])
